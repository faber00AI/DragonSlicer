"""'Stampante' tab: find an ELEGOO printer on the LAN, send the sliced job over Wi-Fi, start / pause / stop it."""
import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from core import printer_link as pl
from core.export import export_goo
from .appsettings import app_settings


class Job(QThread):
    """One blocking printer operation off the GUI thread. fn(job) may call job.phase(text) / job.progress.emit(i, n)."""
    phase = Signal(str)
    progress = Signal(int, int)
    done = Signal(object, object)           # result, exception

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._stop = False

    def cancel(self):
        self._stop = True

    @property
    def cancelled(self):
        return self._stop

    def run(self):
        try:
            self.done.emit(self.fn(self), None)
        except Exception as e:
            self.done.emit(None, e)


def _norm(s):
    return "".join(ch for ch in s.lower().replace("elegoo", "") if ch.isalnum())


class PrinterPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.job = None
        self.printers: list[pl.Printer] = []
        self.settings = app_settings()
        lay = QVBoxLayout(self)

        g = QGroupBox("Collegamento Wi-Fi")
        f = QFormLayout(g)
        self.combo = QComboBox()
        self.combo.setPlaceholderText("Nessuna stampante trovata")
        self.combo.currentIndexChanged.connect(self._selected_changed)
        row = QHBoxLayout()
        row.addWidget(self.combo, 1)
        self.b_scan = QPushButton("Cerca in rete")
        self.b_scan.clicked.connect(self.scan)
        row.addWidget(self.b_scan)
        f.addRow("Stampante", row)
        row = QHBoxLayout()
        self.ip = QLineEdit(self.settings.value("printer/ip", "", str))
        self.ip.setPlaceholderText("es. 192.168.1.50")
        b = QPushButton("Collega")
        b.setToolTip("Usa questo indirizzo se la ricerca automatica non la trova")
        b.clicked.connect(self.connect_ip)
        row.addWidget(self.ip, 1)
        row.addWidget(b)
        f.addRow("Indirizzo IP", row)
        self.info = QLabel("Stampante e PC devono essere nella stessa rete Wi-Fi.")
        self.info.setWordWrap(True)
        f.addRow(self.info)
        lay.addWidget(g)

        g = QGroupBox("Invio")
        f = QFormLayout(g)
        self.resin = QComboBox()                     # the same items as the Resina tab: the profile name only
        self.resin.setModel(win.resin_panel.combo.model())
        self.resin.setCurrentIndex(win.resin_panel.combo.currentIndex())
        self.resin.activated.connect(win.resin_panel.pick_index)
        win.resin_panel.changed.connect(self._sync_resin)
        f.addRow("Resina", self.resin)
        self.name = QLineEdit()
        self.name.setPlaceholderText("nome del file sulla stampante")
        f.addRow("Nome file", self.name)
        self.auto_start = QCheckBox("Avvia la stampa dopo l'invio")
        f.addRow(self.auto_start)
        self.b_send = QPushButton("Esporta la scena e invia")
        self.b_send.clicked.connect(self.send_scene)
        f.addRow(self.b_send)
        self.b_file = QPushButton("Invia un file .goo esistente…")
        self.b_file.clicked.connect(self.send_file)
        f.addRow(self.b_file)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        f.addRow(self.progress)
        self.b_cancel = QPushButton("Annulla")
        self.b_cancel.setVisible(False)
        self.b_cancel.clicked.connect(lambda: self.job and self.job.cancel())
        f.addRow(self.b_cancel)
        lay.addWidget(g)

        g = QGroupBox("Controllo stampa")
        gl = QVBoxLayout(g)
        self.status = QLabel("—")
        self.status.setWordWrap(True)
        gl.addWidget(self.status)
        row = QHBoxLayout()
        self.ctl = []
        for text, fn in (("Stato", self.read_status), ("Pausa", lambda: self.command("pause")),
                         ("Riprendi", lambda: self.command("resume")), ("Stop", lambda: self.command("stop"))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
            self.ctl.append(b)
        gl.addLayout(row)
        self.b_files = QPushButton("File sulla stampante")
        self.b_files.clicked.connect(self.list_files)
        gl.addWidget(self.b_files)
        self.b_start_remote = QPushButton("Avvia un file già sulla stampante")
        self.b_start_remote.clicked.connect(self.start_remote)
        gl.addWidget(self.b_start_remote)
        lay.addWidget(g)
        lay.addStretch(1)
        self._enable(False)

    # ---- helpers ----
    def _sync_resin(self):
        self.resin.blockSignals(True)
        self.resin.setCurrentIndex(self.win.resin_panel.combo.currentIndex())
        self.resin.blockSignals(False)

    def _enable(self, on):
        for w in (self.b_send, self.b_file, self.b_files, self.b_start_remote, *self.ctl):
            w.setEnabled(on and self.job is None)

    def _printer(self):
        i = self.combo.currentIndex()
        return self.printers[i] if 0 <= i < len(self.printers) else None

    def _selected_changed(self, _):
        p = self._printer()
        self._enable(p is not None)
        if p:
            self.info.setText(f"{p.label}\nModello {p.machine or '?'} · firmware {p.firmware or '?'} · "
                              f"protocollo {p.protocol or '?'}")
            self.settings.setValue("printer/ip", p.ip)

    def _start(self, fn, on_done, busy_text=None):
        if self.job is not None:
            return
        self.job = Job(fn)
        self.job.phase.connect(lambda t: self.status.setText(t))
        self.job.progress.connect(self._progress)
        self.job.done.connect(lambda r, e: self._finished(on_done, r, e))
        self.progress.setVisible(False)
        self.b_cancel.setVisible(False)
        self.b_scan.setEnabled(False)
        self._enable(False)
        if busy_text:
            self.status.setText(busy_text)
        self.job.start()

    def _progress(self, i, n):
        self.progress.setVisible(True)
        self.b_cancel.setVisible(True)
        self.progress.setMaximum(max(n, 1))
        self.progress.setValue(i)

    def _finished(self, on_done, result, err):
        self.job.wait()
        self.job = None
        self.progress.setVisible(False)
        self.b_cancel.setVisible(False)
        self.b_scan.setEnabled(True)
        self._enable(self._printer() is not None)
        on_done(result, err)

    def _error(self, title, err):
        self.status.setText(f"{title}: {err}")
        QMessageBox.warning(self, title, str(err))

    # ---- discovery ----
    def scan(self):
        def done(res, err):
            if err:
                return self._error("Ricerca non riuscita", err)
            self._set_printers(res or [])
            if not res:
                self.status.setText("Nessuna stampante ha risposto. Controlla che sia accesa e sulla stessa rete, "
                                    "oppure inserisci l'indirizzo IP.")
        self._start(lambda job: pl.discover(3.0), done, "Cerco le stampanti in rete…")

    def connect_ip(self):
        ip = self.ip.text().strip()
        if not ip:
            return

        def done(res, err):
            if err:
                return self._error("Collegamento non riuscito", err)
            if res is None:
                return self._error("Collegamento non riuscito", f"{ip} non risponde come stampante ELEGOO")
            self._set_printers([res] + [p for p in self.printers if p.ip != res.ip])
        self._start(lambda job: pl.probe(ip), done, f"Provo {ip}…")

    def _set_printers(self, printers):
        last = self.settings.value("printer/ip", "", str)
        self.printers = printers
        self.combo.blockSignals(True)
        self.combo.clear()
        for p in printers:
            self.combo.addItem(p.label)
        self.combo.blockSignals(False)
        if printers:
            idx = next((i for i, p in enumerate(printers) if p.ip == last), 0)
            self.combo.setCurrentIndex(idx)
            self._selected_changed(idx)
            self.status.setText(f"Trovate {len(printers)} stampanti")
        else:
            self._enable(False)

    # ---- sending ----
    def _check_model(self, p):
        want = self.win.machine.name
        if p.machine and _norm(p.machine) != _norm(want):
            r = QMessageBox.question(
                self, "Modello diverso",
                f"La scena è tagliata per «{want}», ma la stampante collegata è «{p.machine}».\n"
                "Un file per un altro modello può essere rifiutato o stampare male. Continuare?")
            return r == QMessageBox.Yes
        return True

    def _name(self, default):
        n = self.name.text().strip() or default
        return n if n.lower().endswith(".goo") else n + ".goo"

    def send_scene(self):
        p = self._printer()
        mesh = self.win.scene.combined(machine=self.win.machine)
        if p is None or mesh is None:
            return self.status.setText("Aggiungi un oggetto e scegli una stampante")
        if not self._check_model(p):
            return
        machine, ps, preview = self.win.machine, self.win._print_settings(), self.win._viewer_preview()
        name = self._name("job.goo")
        start = self.auto_start.isChecked()
        if start and not self._confirm_start(p, name):
            return
        tmp = Path(tempfile.gettempdir()) / f"dragonslicer_{os.getpid()}.goo"

        def work(job):
            job.phase.emit("Taglio i layer e creo il file…")
            ok = export_goo(str(tmp), mesh, machine, ps, preview=preview,
                            progress=job.progress.emit, cancelled=lambda: job.cancelled)
            if not ok:
                raise pl.PrinterError("Operazione annullata")
            try:
                return self._transfer(job, p, tmp, name, start)
            finally:
                tmp.unlink(missing_ok=True)
        self._start(work, lambda r, e: self._sent(name, start, r, e))

    def send_file(self):
        p = self._printer()
        if p is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "File da inviare", "", "ELEGOO GOO (*.goo)")
        if not path:
            return
        name = self._name(Path(path).name)
        start = self.auto_start.isChecked()
        if start and not self._confirm_start(p, name):
            return
        self._start(lambda job: self._transfer(job, p, Path(path), name, start),
                    lambda r, e: self._sent(name, start, r, e))

    def send_generated(self, index, path, name, start, remove=False):
        """Send an already generated .goo to the printer at `index` of the list (used by 'Genera GOO')."""
        self.combo.setCurrentIndex(index)
        p = self._printer()
        path = Path(path)

        def work(job):
            try:
                return self._transfer(job, p, path, self._name(name), start)
            finally:
                if remove:
                    path.unlink(missing_ok=True)
        self._start(work, lambda r, e: self._sent(self._name(name), start, r, e))

    def confirm_wifi(self, index, name, start):
        """Checks before generating: a printer is chosen, the model matches, a start is confirmed."""
        self.combo.setCurrentIndex(index)
        p = self._printer()
        if p is None:
            QMessageBox.warning(self, "Invio", "Nessuna stampante Wi-Fi selezionata.")
            return False
        return self._check_model(p) and (not start or self._confirm_start(p, self._name(name)))

    def _confirm_start(self, p, name):
        return QMessageBox.question(
            self, "Avviare la stampa?",
            f"Dopo l'invio «{name}» partirà subito su {p.label}.\n"
            "Controlla che ci sia resina nella vasca e che il piatto sia montato.") == QMessageBox.Yes

    @staticmethod
    def _transfer(job, p, path, name, start):
        job.phase.emit("Invio alla stampante…")
        pl.upload(p, path, name, progress=lambda i, n: job.progress.emit(i, n), cancelled=lambda: job.cancelled)
        if start:
            job.phase.emit("Avvio la stampa…")
            with pl.Link(p) as link:
                link.start_print(name)
        return name

    def _sent(self, name, start, res, err):
        if err:
            return self._error("Invio non riuscito", err)
        self.status.setText(f"«{name}» inviato" + (" e stampa avviata" if start else ""))
        self.win.statusBar().showMessage(self.status.text(), 8000)

    # ---- control ----
    def _link_job(self, fn, on_done, busy):
        p = self._printer()
        if p is None:
            return

        def work(job):
            with pl.Link(p) as link:
                return fn(link)
        self._start(work, on_done, busy)

    def read_status(self):
        def done(st, err):
            if err:
                return self._error("Stato non disponibile", err)
            if not st:
                return self.status.setText("La stampante non ha inviato lo stato")
            info = st.get("PrintInfo", {})
            cur = st.get("CurrentStatus", [])
            cur = cur[0] if isinstance(cur, list) and cur else cur
            txt = pl.MACHINE_STATUS.get(cur, f"stato {cur}")
            if info:
                txt += (f" — {pl.PRINT_STATUS.get(info.get('Status'), info.get('Status'))}, layer "
                        f"{info.get('CurrentLayer', '?')}/{info.get('TotalLayer', '?')} · {info.get('Filename', '')}")
            self.status.setText(txt)
        self._link_job(lambda link: link.status(), done, "Leggo lo stato…")

    def command(self, what):
        if what == "stop" and QMessageBox.question(self, "Fermare la stampa?",
                                                    "La stampa in corso verrà interrotta.") != QMessageBox.Yes:
            return
        names = {"pause": "Pausa", "resume": "Ripresa", "stop": "Stop"}

        def done(res, err):
            if err:
                return self._error(f"{names[what]} non riuscito", err)
            self.status.setText(f"{names[what]} inviato" if res else f"{names[what]}: nessuna risposta")
        self._link_job(lambda link: getattr(link, what)(), done, "Invio il comando…")

    def list_files(self):
        def done(files, err):
            if err:
                return self._error("Elenco non disponibile", err)
            names = [x.get("name", str(x)) if isinstance(x, dict) else str(x) for x in files or []]
            self.status.setText("File sulla stampante:\n" + ("\n".join(names[:30]) or "(nessuno)"))
        self._link_job(lambda link: link.files(), done, "Leggo i file…")

    def start_remote(self):
        p = self._printer()
        name = self.name.text().strip()
        if p is None or not name:
            return self.status.setText("Scrivi in «Nome file» il file da avviare")
        name = self._name(name)
        if not self._confirm_start(p, name):
            return

        def done(res, err):
            self._error("Avvio non riuscito", err) if err else self.status.setText(f"Stampa di «{name}» avviata")
        self._link_job(lambda link: link.start_print(name), done, "Avvio la stampa…")
