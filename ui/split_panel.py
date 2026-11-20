"""Dividi tab: taglio a pennello. Si dipinge sul modello la fascia dove tagliare, poi "Anteprima del taglio" mostra i pezzi
(chiusi con coperchio, con i connettori) e "Taglia" li mette sul piatto.
Dimensioni e orientamento si sistemano prima, in Modella: qui si usa il modello cosi' com'e' sul piatto."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
                               QListWidget, QListWidgetItem, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from core import connectors, split, splitcut

from .appsettings import app_settings

GREEN = ("QPushButton{background:#16a34a;color:white;font-size:14px;font-weight:bold;border-radius:6px}"
         "QPushButton:hover{background:#22c55e}QPushButton:disabled{background:#3a3f47;color:#888}")
MASK_COLOR = (1.0, 0.15, 0.15)
CONN_COLOR = (0.98, 0.9, 0.2)
MODEL_GRAY = (0.72, 0.74, 0.78)

PALETTE = [(0.12, 0.47, 0.71), (1.0, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16), (0.58, 0.40, 0.74),
           (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81),
           (0.68, 0.78, 0.91), (1.0, 0.73, 0.47), (0.60, 0.87, 0.54), (1.0, 0.60, 0.59), (0.77, 0.69, 0.84),
           (0.77, 0.61, 0.58), (0.97, 0.71, 0.82), (0.78, 0.78, 0.78), (0.86, 0.86, 0.55), (0.62, 0.85, 0.90)]


def _col(i):
    return PALETTE[i % len(PALETTE)]


def _spin(lo, hi, val, step, dec=1, suffix="", tip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(dec)
    if suffix:
        s.setSuffix(suffix)
    if tip:
        s.setToolTip(tip)
    return s


def _fits(ext, m):
    """Il pezzo sta nel piatto nella posa attuale o ruotato di 90° sul piano."""
    ex, ey, ez = (float(v) for v in ext)
    return ez <= m.bed_z and ((ex <= m.bed_x and ey <= m.bed_y) or (ey <= m.bed_x and ex <= m.bed_y))


class SplitPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._obj = None            # oggetto a cui si riferisce la maschera
        self._src = None            # (mesh dell'oggetto, matrice) al momento della preparazione
        self._small = None          # mesh ridotta in coordinate mondo (solo per dipingere e calcolare)
        self._tree = None
        self._painted = None        # maschera: bool per faccia della mesh ridotta
        self._lab = None            # etichette del taglio simulato (None = nessuna anteprima)
        self._pieces = None         # pezzi chiusi dell'anteprima
        self._base = None
        self._view = []             # [(mesh, colore, offset)] cio' che sostituisce l'oggetto nel viewer
        self._conn_view = []
        self._pending_view = QTimer(self)
        self._pending_view.setSingleShot(True)
        self._pending_view.setInterval(40)
        self._pending_view.timeout.connect(self._update_view)

        lay = QVBoxLayout(self)
        lay.addWidget(win.model_panel.g_cut)                 # 1. taglio lineare (con un piano), vive nel pannello Modella
        g3 = QGroupBox("Taglio a pennello")
        v3 = QVBoxLayout(g3)
        t3 = QLabel("1. Seleziona il modello (già orientato e in scala, da Modella) e attiva il <b>Pennello</b>.<br>"
                    "2. Dipingi sul modello una fascia dove vuoi il taglio: il taglio segue la linea di mezzo della fascia. "
                    "La fascia deve fare tutto il giro. Puoi dipingere più fasce.<br>"
                    "3. Premi <b>Anteprima del taglio</b>, poi <b>Taglia</b>.<br>"
                    "<i>Clic sinistro sul modello = dipingi · Ctrl+clic = cancella · trascina fuori dal modello (o Alt+trascina) = ruota la vista.</i>")
        t3.setWordWrap(True)
        v3.addWidget(t3)
        self.b_paint = QPushButton("Pennello")
        self.b_paint.setCheckable(True)
        self.b_paint.setMinimumHeight(34)
        self.b_paint.setStyleSheet(GREEN + "QPushButton:checked{background:#f59e0b;color:black}")
        self.b_paint.toggled.connect(self._paint_toggled)
        v3.addWidget(self.b_paint)
        f3 = QFormLayout()
        self.brush = _spin(0.5, 60, 4, 0.5, 1, " mm", tip="Raggio del pennello")
        f3.addRow("Raggio", self.brush)
        v3.addLayout(f3)
        self.b_erase = QPushButton("Cancella la maschera")
        self.b_erase.clicked.connect(self.clear_mask)
        v3.addWidget(self.b_erase)
        self.b_prev = QPushButton("Anteprima del taglio")
        self.b_prev.setMinimumHeight(34)
        self.b_prev.setStyleSheet(GREEN)
        self.b_prev.setToolTip("Simula il taglio lungo le fasce dipinte e mostra i pezzi")
        self.b_prev.clicked.connect(self.preview)
        v3.addWidget(self.b_prev)
        self.b_back = QPushButton("Torna alla maschera")
        self.b_back.setToolTip("Chiude l'anteprima e riattiva il pennello per correggere la maschera")
        self.b_back.clicked.connect(self.back_to_mask)
        v3.addWidget(self.b_back)
        v3.addWidget(QLabel("Vista esplosa"))
        self.explode = QSlider(Qt.Horizontal)
        self.explode.setRange(0, 100)
        self.explode.setToolTip("Allontana i pezzi tra loro per vedere i tagli")
        self.explode.valueChanged.connect(lambda _: self._update_view())
        v3.addWidget(self.explode)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        v3.addWidget(self.info)
        self.b_cut = QPushButton("Taglia")
        self.b_cut.setMinimumHeight(38)
        self.b_cut.setStyleSheet(GREEN)
        self.b_cut.setToolTip("Taglia il modello nei pezzi dell'anteprima, chiude ogni pezzo e li mette sul piatto appoggiati sul taglio. "
                              "Si può annullare")
        self.b_cut.clicked.connect(self.cut)
        v3.addWidget(self.b_cut)
        self.cut_info = QLabel("")
        self.cut_info.setWordWrap(True)
        v3.addWidget(self.cut_info)

        lay.addWidget(g3)

        g4 = QGroupBox("Connettori")
        v4 = QVBoxLayout(g4)
        t4 = QLabel("Valgono per il taglio lineare e per quello a pennello. Aggiunge ai tagli perni e fori (o tasche per magneti, o fori per spinotti separati) così i pezzi si incastrano. "
                    "Le misure sono in mm; il gioco va aggiunto al foro. Nell'anteprima compaiono in giallo.")
        t4.setWordWrap(True)
        v4.addWidget(t4)
        self.cmode = QComboBox()
        self.cmode.addItems(["Nessuno", "Perni e fori", "Tasche per magneti", "Fori + spinotti separati"])
        v4.addWidget(self.cmode)
        f4 = QFormLayout()
        self.cdiam = _spin(1, 30, 4.0, 0.5, 1, " mm", tip="Diametro del perno / del magnete / dello spinotto")
        self.clen = _spin(2, 40, 6.0, 0.5, 1, " mm", tip="Lunghezza del perno, profondità della tasca (per parte) o lunghezza totale dello spinotto")
        self.cclr = _spin(0.0, 1.5, 0.2, 0.05, 2, " mm", tip="Gioco aggiunto al diametro dei fori")
        self.cn = _spin(1, 8, 3, 1, 0, "", tip="Numero massimo di connettori per ogni taglio")
        self.cmar = _spin(0.5, 15, 2.0, 0.5, 1, " mm", tip="Distanza minima dal bordo del taglio")
        for lbl, wd in (("Diametro", self.cdiam), ("Lunghezza", self.clen), ("Gioco", self.cclr), ("Per taglio (max)", self.cn),
                        ("Dal bordo", self.cmar)):
            f4.addRow(lbl, wd)
        v4.addLayout(f4)
        lay.addWidget(g4)
        lay.addStretch(1)

        for wd in (self.cdiam, self.clen, self.cclr, self.cn, self.cmar):
            wd.valueChanged.connect(self._conn_changed)
        self.cmode.currentIndexChanged.connect(self._conn_changed)
        win.viewer.paint_ray.connect(self._on_paint)
        self.load_settings()
        self._sync()

    # ---- impostazioni ----
    def _fields(self):
        return dict(brush=self.brush, cmode=self.cmode, cdiam=self.cdiam, clen=self.clen, cclr=self.cclr, cn=self.cn, cmar=self.cmar)

    def save_settings(self):
        st = app_settings()
        for k, wd in self._fields().items():
            st.setValue(f"split/{k}", wd.currentIndex() if isinstance(wd, QComboBox) else wd.value())
        st.sync()

    def load_settings(self):
        st = app_settings()
        for k, wd in self._fields().items():
            val = st.value(f"split/{k}")
            if val is None:
                continue
            try:
                wd.blockSignals(True)
                if isinstance(wd, QComboBox):
                    wd.setCurrentIndex(max(0, min(int(val), wd.count() - 1)))
                else:
                    wd.setValue(float(val))
            except (TypeError, ValueError):
                pass
            finally:
                wd.blockSignals(False)

    def _sync(self):
        have = self._small is not None and self._painted is not None
        prev = self._lab is not None
        any_paint = have and bool(self._painted.any())
        self.b_erase.setEnabled(any_paint)
        self.b_prev.setEnabled(any_paint and not prev)
        self.b_back.setEnabled(prev)
        self.explode.setEnabled(prev)
        self.b_cut.setEnabled(prev)

    # ---- aggancio al viewer ----
    def preview_for(self, row, obj):
        """[(mesh, colore, offset)] da disegnare al posto dell'oggetto, oppure None (disegno normale)."""
        if obj is not self._obj or self._src is None or not self._view:
            return None
        if obj.mesh is not self._src[0] or not np.allclose(obj.matrix(), self._src[1]):
            return None                                      # l'oggetto e' stato mosso/cambiato: anteprima non piu' valida
        return self._view

    def clear(self):
        self.b_paint.setChecked(False)
        self._obj = self._src = self._small = self._tree = self._painted = None
        self._lab = self._pieces = self._base = None
        self._view = []
        self._conn_view = []
        self.info.setText("")
        self._sync()
        self.win._refresh(light=True)

    def _row(self):
        try:
            return self.win.scene.objects.index(self._obj)
        except ValueError:
            return -1

    # ---- preparazione (silenziosa) ----
    def _valid(self):
        o = self._obj
        return (o is not None and self._src is not None and self._row() >= 0 and o.mesh is self._src[0]
                and np.allclose(o.matrix(), self._src[1]))

    def _ensure(self):
        """Prepara il modello selezionato per dipingere (mesh ridotta); non mostra nulla. True se pronto."""
        if self._valid():
            return True
        w = self.win
        rows = w.selected_rows()
        if not rows:
            self.info.setText("Seleziona il modello da tagliare.")
            return False
        o = w.scene.objects[rows[0]]
        world = o.world_mesh()
        try:
            small = w.run_busy("Preparo il modello…", lambda: split.decimate(world, 30000))
        except Exception as e:
            self.info.setText(f"Errore: {e}")
            return False
        if small is None or not len(small.faces):
            self.info.setText("Modello non utilizzabile.")
            return False
        self._obj = o
        self._small = small
        self._painted = np.zeros(len(small.faces), bool)
        self._tree = cKDTree(small.triangles_center)
        self._src = (o.mesh, o.matrix().copy())
        self._lab = self._pieces = self._base = None
        self._view = []
        self._conn_view = []
        self.brush.setValue(float(np.clip(world.extents.max() / 40.0, 0.5, 60)))
        return True

    # ---- pennello ----
    def _paint_toggled(self, on):
        v = self.win.viewer
        if on:
            if not self._ensure():
                self.b_paint.blockSignals(True)
                self.b_paint.setChecked(False)
                self.b_paint.blockSignals(False)
                return
            if self._lab is not None:
                self._drop_preview()
                self._update_view()
            v.paint_probe = self._probe
            v.paint_mode = True
            self.win.statusBar().showMessage("Pennello: dipingi la fascia dove vuoi il taglio (trascina fuori dal modello per ruotare)", 8000)
        else:
            v.paint_mode = False
            v.paint_probe = None
        self._sync()

    def _probe(self, o, d):
        """Il raggio colpisce il modello? (altrimenti il clic ruota la vista)"""
        if self._small is None:
            return False
        try:
            return bool(self._small.ray.intersects_any([o], [d])[0])
        except Exception:
            return False

    def _on_paint(self, o, d, erase):
        if self._small is None or self._painted is None or not self.b_paint.isChecked():
            return
        s = self._small
        try:
            loc, _ray, tri = s.ray.intersects_location([o], [d], multiple_hits=False)
        except Exception:
            return
        if not len(loc):
            return
        hit = loc[0]
        nrm = s.face_normals[tri[0]]
        idx = np.array(self._tree.query_ball_point(hit, self.brush.value()), int)
        if not len(idx):
            return
        idx = idx[s.face_normals[idx] @ nrm > 0.2]          # solo facce rivolte come quella colpita (non il lato opposto)
        self._painted[idx] = not erase
        self.cut_info.setText("")
        if not self._pending_view.isActive():
            self._pending_view.start()

    def clear_mask(self):
        if self._painted is not None:
            self._painted[:] = False
            self._drop_preview()
            self._update_view()

    def _drop_preview(self):
        self._lab = self._pieces = self._base = None
        self._conn_view = []

    def back_to_mask(self):
        self._drop_preview()
        self._update_view()
        self.b_paint.setChecked(True)

    # ---- anteprima del taglio ----
    def preview(self):
        if self._small is None or self._painted is None or not self._painted.any():
            self.info.setText("Dipingi prima una fascia sul modello.")
            return
        s = self._small

        def work():
            lab0 = np.zeros(len(s.faces), int)
            new, cut, open_ = splitcut.apply_mask(s, lab0, self._painted, 0.002)
            if not cut:
                return None, None, None
            new = split.smooth_labels(s, new, 60)
            new = splitcut.clean_labels(s, new)
            pieces, base = splitcut.cut_pieces(s, new, 0.7, 4, return_base=True)
            return new, pieces, base

        try:
            new, pieces, base = self.win.run_busy("Simulo il taglio…", work)
        except Exception as e:
            self.info.setText(f"Anteprima non riuscita: {e}")
            return
        if new is None or len(np.unique(new)) < 2:
            self.info.setText("La fascia non separa il modello: deve fare tutto il giro (anche dietro). Completala e riprova.")
            return
        self.b_paint.setChecked(False)
        self._lab, self._pieces, self._base = new, pieces, base
        self.save_settings()
        self._update_view()

    def _conn_changed(self, *_):
        if self._lab is not None:
            self._update_view()

    def _conn_args(self):
        mode = (None, "peg", "magnet", "dowel")[self.cmode.currentIndex()]
        return mode, (self.cdiam.value(), self.clen.value(), self.cclr.value(), int(self.cn.value()), self.cmar.value())

    def _update_view(self):
        if self._small is None:
            return
        s = self._small
        view = []                                            # (mesh, colore, gruppo)
        conn_txt = ""
        self._conn_view = []
        if self._lab is not None and self._pieces:
            for p in self._pieces:
                view.append((p["mesh"], _col(p["label"]), p["label"]))
            mode, args = self._conn_args()
            if mode:
                try:
                    cyl, ok, skip = connectors.preview(self._pieces, self._base, self._lab, mode, *args)
                    conn_txt = f"\nConnettori: {ok} previsti (circa)" + (f", {skip} tagli troppo piccoli per averne" if skip else "")
                    self._conn_view = [(m, CONN_COLOR, g) for m, g in cyl]
                except Exception as e:
                    conn_txt = f"\nAnteprima connettori non riuscita: {e}"
        elif self._painted is not None and self._painted.any():
            f = np.asarray(s.faces)
            V = np.asarray(s.vertices)
            for sel, col in ((~self._painted, MODEL_GRAY), (self._painted, MASK_COLOR)):
                if sel.any():
                    sub = trimesh.Trimesh(V, f[sel], process=False)
                    sub.remove_unreferenced_vertices()
                    view.append((sub, col, 0))
        ex = self.explode.value() / 100.0 if self._lab is not None else 0.0
        ctr = s.bounds.mean(axis=0)
        allv = view + self._conn_view
        cen = {}
        for m, col, grp in allv:
            cen.setdefault(grp, []).append(m.bounds.mean(axis=0))
        cen = {g: np.mean(c, axis=0) for g, c in cen.items()}
        self._view = [(m, col, (cen[grp] - ctr) * ex * 0.8) for m, col, grp in allv]
        if self._lab is not None and self._pieces:
            self._report(conn_txt)
        elif not self._view:
            self.info.setText("")
        self._sync()
        self.win._refresh(light=True)

    def _report(self, extra=""):
        mach = self.win.machine
        bad = []
        for p in self._pieces:
            ext = np.ptp(p["mesh"].bounds, axis=0)
            if not _fits(ext, mach):
                bad.append(f"{ext[0]:.0f}×{ext[1]:.0f}×{ext[2]:.0f} mm")
        txt = f"{len(self._pieces)} pezzi risultanti"
        if bad:
            txt += f"\nNon stanno nella stampante: {len(bad)} (" + "; ".join(bad[:4]) + ("…" if len(bad) > 4 else "") + ")"
        else:
            txt += "\nTutti i pezzi stanno nella stampante."
        self.info.setText(txt + extra)

    # ---- taglio vero ----
    def cut(self):
        w = self.win
        if self._lab is None or not self._valid():
            self.cut_info.setText("Fai prima l'anteprima del taglio (il modello non deve essere stato mosso).")
            return
        o = self._obj
        row = self._row()
        small, lab_small = self._small, self._lab
        world = o.world_mesh()
        flat = 0.7
        mode, (cd, cl, cc, cn, cm) = self._conn_args()

        def work():
            m = world.copy()
            m.merge_vertices()
            _, j = cKDTree(small.triangles_center).query(m.triangles_center)
            full = splitcut.clean_labels(m, lab_small[j])
            res, base = splitcut.cut_pieces(m, full, flat, 8, return_base=True)
            rep, extra = None, []
            if mode and len(res) > 1:
                seams = connectors.seam_loops(base, full)
                res, extra, rep = connectors.add_connectors(res, seams, mode, cd, cl, cc, cn, cm)
            return res, extra, rep

        try:
            pieces, extra, rep = w.run_busy("Taglio il modello e chiudo i pezzi…", work)
        except Exception as e:
            self.cut_info.setText(f"Taglio non riuscito: {e}")
            return
        mach = w.machine
        w.push_undo()
        name = o.name
        w.remove_rows([row])
        bx, by = mach.bed_x / 2, mach.bed_y / 2
        x, y, rowh, gap = -bx + 5, -by + 5, 0.0, 5.0
        added, open_n, big = 0, 0, []
        for k, p in enumerate(sorted(pieces, key=lambda q: -len(q["mesh"].faces)), 1):
            m = p["mesh"].copy()
            m.apply_transform(splitcut.lay_on_cut(p))
            ex = np.ptp(m.bounds, axis=0)
            if x + ex[0] > bx - 5 and x > -bx + 5:
                x, y, rowh = -bx + 5, y + rowh + gap, 0.0
            m.apply_translation([x - m.bounds[0][0], y - m.bounds[0][1], -m.bounds[0][2]])
            x += ex[0] + gap
            rowh = max(rowh, ex[1])
            w.add_mesh_object(m, f"{name} – parte {k}", select=False)
            added += 1
            open_n += 0 if p["watertight"] else 1
            if not _fits(ex, mach):
                big.append(f"parte {k}")
        for k, d in enumerate(extra or []):                 # spinotti: uno per foro, in fila lungo il bordo del piatto
            m = d.copy()
            ex = np.ptp(m.bounds, axis=0)
            if x + ex[0] > bx - 5 and x > -bx + 5:
                x, y, rowh = -bx + 5, y + rowh + gap, 0.0
            m.apply_translation([x - m.bounds[0][0], y - m.bounds[0][1], -m.bounds[0][2]])
            x += ex[0] + gap
            rowh = max(rowh, ex[1])
            w.add_mesh_object(m, f"{name} – spinotto {k + 1}", select=False)
        w.reselect(len(w.scene.objects) - 1)
        self.clear()
        msg = f"Tagliato in {added} pezzi, appoggiati sul coperchio più grande."
        if rep:
            msg += f" Connettori: {rep['made']} messi"
            msg += f", {rep['skipped']} saltati (zona troppo piccola o sottile)" if rep["skipped"] else ""
            msg += f", {rep['failed']} pezzi senza connettori (superficie non valida: usa «Ripara» prima di tagliare)" if rep["failed"] else ""
            msg += "."
        if open_n:
            msg += f" {open_n} pezzi hanno ancora fori (il modello d'origine aveva difetti: usa «Ripara» prima di tagliare)."
        if big:
            msg += " Non stanno nel piatto: " + ", ".join(big) + "."
        self.cut_info.setText(msg)
        w.statusBar().showMessage("Taglio eseguito (Ctrl+Z per annullare)", 6000)
