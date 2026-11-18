"""LAN link to ELEGOO printers over SDCP (the protocol SatelLite itself uses).

Discovery   UDP broadcast "M99999" -> port 3000, every printer answers with a JSON description.
Commands    WebSocket  ws://<ip>:3030/websocket  (JSON, topic sdcp/request/<MainboardID>).
Upload      HTTP POST  http://<ip>:3030/uploadFile/upload  (multipart, 1 MiB chunks, MD5 of the whole file).
"""
import hashlib
import json
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

DISCOVERY_PORT = 3000
SDCP_PORT = 3030
WS_PORT = SDCP_PORT          # separate names so tests can point them at a local mock
HTTP_PORT = SDCP_PORT
CHUNK = 1024 * 1024

CMD_STATUS, CMD_ATTRIBUTES = 0, 1
CMD_START, CMD_PAUSE, CMD_STOP, CMD_RESUME = 128, 129, 130, 131
CMD_FILES, CMD_DELETE, CMD_HISTORY = 258, 259, 320

# SDCP "CurrentStatus" machine states and "PrintInfo.Status" print states
MACHINE_STATUS = {0: "Libera", 1: "In stampa", 2: "Trasferimento file", 3: "Verifica esposizione", 4: "Calibrazione",
                  5: "Homing"}
PRINT_STATUS = {0: "Ferma", 1: "Homing", 2: "Abbasso la piastra", 3: "Esposizione", 4: "Sollevamento",
                5: "Pausa in corso", 6: "In pausa", 7: "Arresto in corso", 8: "Arrestata", 9: "Completata",
                10: "Verifica file", 13: "In stampa", 14: "Preparazione", 15: "Pausa", 16: "Ripresa"}


class PrinterError(Exception):
    pass


@dataclass
class Printer:
    ip: str
    mainboard_id: str = ""
    name: str = ""
    machine: str = ""
    brand: str = ""
    firmware: str = ""
    protocol: str = ""
    msg_id: str = ""            # the "Id" field of the discovery answer (used in every command envelope)

    @property
    def label(self):
        return f"{self.name or self.machine or 'Stampante'} — {self.ip}"


def _local_broadcasts():
    """Directed broadcast addresses of this PC's IPv4 interfaces (255.255.255.255 alone misses some adapters)."""
    out = {"255.255.255.255"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                out.add(ip.rsplit(".", 1)[0] + ".255")
    except OSError:
        pass
    return sorted(out)


def discover(timeout=3.0):
    """Broadcast M99999 and collect the answers: list[Printer]."""
    found = {}
    for target in _local_broadcasts():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.settimeout(0.3)
            s.sendto(b"M99999", (target, DISCOVERY_PORT))
            end = time.time() + timeout / max(len(_local_broadcasts()), 1)
            while time.time() < end:
                try:
                    data, addr = s.recvfrom(65535)
                except socket.timeout:
                    continue
                p = parse_discovery(data, addr[0])
                if p:
                    found[p.ip] = p
        except OSError:
            pass
        finally:
            s.close()
    return list(found.values())


def parse_discovery(data: bytes, ip: str):
    try:
        j = json.loads(data.decode("utf-8", "replace"))
        d = j.get("Data", j)
    except ValueError:
        return None
    return Printer(ip=d.get("MainboardIP") or ip, mainboard_id=d.get("MainboardID", ""), name=d.get("Name", ""),
                   machine=d.get("MachineName", ""), brand=d.get("BrandName", ""),
                   firmware=d.get("FirmwareVersion", ""), protocol=d.get("ProtocolVersion", ""),
                   msg_id=j.get("Id", ""))


def probe(ip, timeout=2.0):
    """Ask one printer directly (unicast M99999), for networks that block broadcasts."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(timeout)
        s.sendto(b"M99999", (ip, DISCOVERY_PORT))
        data, addr = s.recvfrom(65535)
        return parse_discovery(data, addr[0])
    except OSError:
        return None
    finally:
        s.close()


def file_md5(path, progress=None):
    h = hashlib.md5()
    size = Path(path).stat().st_size
    done = 0
    with open(path, "rb") as f:
        while chunk := f.read(4 << 20):
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, size)
    return h.hexdigest()


def upload(printer: Printer, path, remote_name=None, progress=None, cancelled=lambda: False, timeout=30):
    """Send a file to the printer's storage, 1 MiB at a time. progress(sent_bytes, total_bytes).
    Same request SatelLite makes: form fields TotalSize, Uuid, Offset, Check=0, S-File-MD5, then the File part;
    every chunk is retried up to 3 times."""
    path = Path(path)
    total = path.stat().st_size
    name = remote_name or path.name
    md5 = file_md5(path)
    uid = "{" + uuid.uuid4().hex + "}"           # SatelLite sends QUuid::toString() without the dashes, braces kept
    url = f"http://{printer.ip}:{HTTP_PORT}/uploadFile/upload"
    sent = 0
    with open(path, "rb") as f, requests.Session() as http:
        while sent < total:
            if cancelled():
                raise PrinterError("Invio annullato")
            f.seek(sent)
            chunk = f.read(CHUNK)
            fields = [("TotalSize", str(total)), ("Uuid", uid), ("Offset", str(sent)), ("Check", "0"),
                      ("S-File-MD5", md5)]
            err = None
            for _ in range(3):
                try:
                    r = http.post(url, files=[(k, (None, v)) for k, v in fields] +
                                  [("File", (name, chunk, "application/octet-stream"))], timeout=timeout)
                    err = None if r.ok else f"HTTP {r.status_code}: {r.text[:200]}"
                    if err is None:
                        break
                except requests.RequestException as e:
                    err = f"connessione persa: {e}"
            if err:
                raise PrinterError(f"Invio fallito a {sent} di {total} byte ({err})")
            sent += len(chunk)
            if progress:
                progress(sent, total)
    return md5


class Link:
    """One WebSocket session with a printer (sync API; use from a worker thread)."""

    def __init__(self, printer: Printer, timeout=8.0):
        from websockets.sync.client import connect
        self.p = printer
        try:
            self.ws = connect(f"ws://{printer.ip}:{WS_PORT}/websocket", open_timeout=timeout, max_size=None)
        except Exception as e:
            raise PrinterError(f"Non riesco a collegarmi a {printer.ip}: {e}") from e

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def send(self, cmd, data=None, wait=6.0):
        """Send a command and return the matching response payload (or None if none arrived in `wait` s)."""
        rid = uuid.uuid4().hex
        msg = {"Id": self.p.msg_id, "Data": {"Cmd": cmd, "Data": data or {}, "RequestID": rid,
                                             "MainboardID": self.p.mainboard_id, "TimeStamp": int(time.time() * 1000),
                                             "From": 0},
               "Topic": f"sdcp/request/{self.p.mainboard_id}"}
        self.ws.send(json.dumps(msg))
        end = time.time() + wait
        while time.time() < end:
            try:
                raw = self.ws.recv(timeout=max(0.1, end - time.time()))
            except TimeoutError:
                return None
            try:
                j = json.loads(raw)
            except ValueError:
                continue
            d = j.get("Data", {})
            if d.get("RequestID") == rid:
                return d
        return None

    def status(self):
        """Wait for a status report: dict with Status (machine) and PrintInfo."""
        end = time.time() + 6
        self.send(CMD_STATUS, wait=0.2)
        while time.time() < end:
            try:
                j = json.loads(self.ws.recv(timeout=max(0.1, end - time.time())))
            except TimeoutError:
                break
            except ValueError:
                continue
            if "Status" in j:
                return j["Status"]
        return None

    def start_print(self, filename, start_layer=0):
        ans = self.send(CMD_START, {"Filename": filename, "StartLayer": start_layer})
        ack = (ans or {}).get("Data", {}).get("Ack", None)
        if ans is None:
            raise PrinterError("Nessuna risposta dalla stampante al comando di avvio")
        if ack not in (0, None):
            raise PrinterError(f"La stampante non ha avviato la stampa (Ack={ack})")
        return ans

    def pause(self):
        return self.send(CMD_PAUSE)

    def resume(self):
        return self.send(CMD_RESUME)

    def stop(self):
        return self.send(CMD_STOP)

    def files(self, url="/local"):
        ans = self.send(CMD_FILES, {"Url": url})
        return (ans or {}).get("Data", {}).get("FileList", [])

    def delete(self, *names):
        """Delete files from the printer (full names as listed, e.g. '/local/x.goo')."""
        return self.send(CMD_DELETE, {"FileList": list(names), "FolderList": []})
