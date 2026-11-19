import io
import json
import os
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QEventLoop, QItemSelectionModel, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QFrame, QGridLayout, QScrollArea, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QListWidget, QMainWindow, QMenu, QMessageBox, QProgressBar, QProgressDialog, QPushButton, QSlider,
                               QSpinBox, QSplitter, QTabBar, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from core import modeling as md
from core.export import export_goo, export_zip
from core import resins, updater
from core.version import __version__
from core.machine import PrintSettings, load_machines, load_support_presets
from core.project import DSL_EXT, TRANSFORM, ProjectError, place_objects, read_dsl, write_dsl
from core.threemf import write_3mf
from core.satellite import is_satellite, read_satellite
from core.scene import Scene
from core.slicer import layer_count, slice_layers
from core.supports import SupportParams, build_supports, build_supports_each, drop_pillars, split_pillars
from .modeling_panel import ModelingPanel
from .generate_dialog import GenerateDialog
from .no_wheel import install as install_no_wheel
from .appsettings import app_settings
from .printer_panel import PrinterPanel
from .mold_panel import MoldPanel
from .roller_panel import RollerPanel
from .resin_panel import ResinPanel
from .scale_dialog import SMALL_MM, ScaleDialog
from .viewer import Drawable, Geometry, Viewer

class _UpdateCheck(QThread):
    done = Signal(object, str)

    def run(self):
        try:
            self.done.emit(updater.check(), "")
        except Exception as ex:
            self.done.emit(None, str(ex))


MODEL_EXTS = {".stl", ".obj", ".3mf", ".ply", ".off"}
MODEL_COLOR = (0.78, 0.80, 0.86)
SEL_COLOR = (0.45, 0.70, 1.0)
SUPPORT_COLOR = (0.95, 0.60, 0.20)
SUPPORT_SEL_COLOR = (1.0, 0.15, 0.15)


def dspin(lo, hi, val, step=0.1, dec=2, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setSuffix(suffix)
    return s


class ExportThread(QThread):
    progress = Signal(int, int)
    done = Signal(bool, str)

    def __init__(self, path, mesh, machine, ps, preview=None):
        super().__init__()
        self.path, self.mesh, self.machine, self.ps, self.preview = path, mesh, machine, ps, preview
        self._stop = False

    def cancel(self):
        self._stop = True

    def run(self):
        try:
            if self.path.lower().endswith(".goo"):
                ok = export_goo(self.path, self.mesh, self.machine, self.ps, preview=self.preview,
                                progress=self.progress.emit, cancelled=lambda: self._stop)
            else:
                ok = export_zip(self.path, self.mesh, self.machine, self.ps,
                                progress=self.progress.emit, cancelled=lambda: self._stop)
            self.done.emit(ok, "")
        except Exception as e:  # surfaced in the UI
            self.done.emit(False, str(e))


class Worker(QThread):
    """Runs one blocking function off the GUI thread; result or exception comes back through `done`."""
    done = Signal(object, object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), None)
        except Exception as e:  # re-raised on the GUI thread
            self.done.emit(None, e)


class MoveUndo(NamedTuple):
    """Undo entry of 'move to another plate': the source plate's snapshot plus the destination's, restored together."""
    snap: tuple
    dest: dict
    dest_snap: tuple


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DragonSlicer")
        self.setAcceptDrops(True)
        self.resize(1500, 900)
        self._no_wheel = install_no_wheel()   # spin boxes / drop-downs never change with the wheel
        self.machines = load_machines()
        self.machine = self.machines[0]
        self.presets = load_support_presets()
        self.scene = Scene()
        self.ps = PrintSettings()
        self._building = False
        self._geo = {}                       # id(mesh) -> (mesh, Geometry): GPU data shared by every redraw
        self._thread = None
        self._undo = []
        self._redo = []
        self.plates = [dict(name="Piatto 1", scene=self.scene, undo=self._undo, redo=self._redo, info="")]   # build plates; self.scene / _undo are the active one's
        self.plate_i = 0
        self.project_file = None             # the .dsl the scene was last saved to / opened from
        self.ask_scale = ScaleDialog.ask      # replaceable (tests)
        self.ask_repair = self._ask_repair    # replaceable (tests)

        self.viewer = Viewer()
        self.viewer.set_bed((self.machine.bed_x, self.machine.bed_y, self.machine.bed_z))
        self.preview = QLabel("Premi “Anteprima layer”")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#000;color:#888")
        self.preview.setMinimumSize(300, 200)
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.setEnabled(False)
        self.layer_slider.valueChanged.connect(self._layer_changed)
        self.layer_label = QLabel("-")

        views = QTabWidget()
        v3d = QWidget()                      # the 3D view with a vertical layer slider beside it
        h3d = QHBoxLayout(v3d)
        h3d.setContentsMargins(0, 0, 0, 0)
        h3d.addWidget(self.viewer, 1)
        col = QVBoxLayout()
        self.clip_label = QLabel("Tutto")
        self.clip_label.setAlignment(Qt.AlignHCenter)
        self.clip_label.hide()                  # no text beside the slider: it only feeds the tooltip
        self.clip_label.setToolTip("Taglia il modello a questo strato e mostra la sezione (anche l'interno di un pezzo svuotato).\n"
                                   "Tutto in alto = nessun taglio")
        self.clip_slider = QSlider(Qt.Vertical)
        self.clip_slider.setRange(1, 1)
        self.clip_slider.setToolTip(self.clip_label.toolTip())
        self.clip_slider.setValue(1)
        self.clip_slider.setEnabled(False)
        self.clip_slider.valueChanged.connect(self._clip_changed)
        col.addWidget(self.clip_slider, 1)
        h3d.addLayout(col)
        views.addTab(v3d, "3D")
        pv = QWidget()
        lay = QVBoxLayout(pv)
        lay.addWidget(self.preview, 1)
        row = QHBoxLayout()
        b_prev = QPushButton("Anteprima layer")
        b_prev.clicked.connect(self.preview_layers)
        row.addWidget(b_prev)
        row.addWidget(self.layer_slider, 1)
        row.addWidget(self.layer_label)
        lay.addLayout(row)
        views.addTab(pv, "Layer")
        self.views = views

        left = QWidget()                     # plate tabs above the 3D / layer views
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        prow = QHBoxLayout()
        self.plate_bar = QTabBar()
        self.plate_bar.setExpanding(False)
        self.plate_bar.setDrawBase(False)
        self.plate_bar.setStyleSheet(               # the active plate stands out in the accent colour
            "QTabBar::tab{background:#2a2d34;color:#aaa;padding:5px 14px;margin-right:2px;"
            "border:1px solid #3a3f47;border-radius:4px}"
            "QTabBar::tab:hover{background:#363a43;color:#e6e6e6}"
            "QTabBar::tab:selected{background:#3b82f6;color:white;font-weight:bold;border-color:#60a5fa}")
        self.plate_bar.addTab("Piatto 1")
        self.plate_bar.currentChanged.connect(self.switch_plate)
        self.plate_bar.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plate_bar.customContextMenuRequested.connect(self._plate_menu)
        b_plate = QToolButton()
        b_plate.setText("+")
        b_plate.setToolTip("Nuovo piatto di stampa")
        b_plate.clicked.connect(lambda: self.add_plate())
        prow.addWidget(self.plate_bar)
        prow.addWidget(b_plate)
        prow.addStretch(1)
        lv.addLayout(prow)
        lv.addWidget(views, 1)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self._side_panel())
        split.setStretchFactor(0, 1)
        self.setCentralWidget(split)
        self._menus()
        self.statusBar().showMessage("Importa un modello (STL / OBJ / 3MF / PLY)")

    def run_busy(self, text, fn):
        """Run a slow function in a worker thread behind a busy dialog, keeping the window alive."""
        dlg = QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle("Attendere")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()
        loop, out = QEventLoop(), {}
        th = Worker(fn)
        th.done.connect(lambda r, e: (out.update(r=r, e=e), loop.quit()))
        th.start()
        loop.exec()
        th.wait()
        dlg.close()
        if out.get("e") is not None:
            raise out["e"]
        return out.get("r")

    # ---------------- drag & drop from the file manager ----------------
    @staticmethod
    def _dropped(e):
        return [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]

    def dragEnterEvent(self, e):
        if self._dropped(e):
            e.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e):
        paths = self._dropped(e)
        if paths:
            e.acceptProposedAction()
            self.views.setCurrentIndex(0)
            projects = [p for p in paths if p.lower().endswith(DSL_EXT) or is_satellite(p)]
            if projects:
                self.open_project(projects[0])
            else:
                self.import_paths(paths)

    # ---------------- UI construction ----------------
    def _menus(self):
        m = self.menuBar().addMenu("File")
        for text, fn, key in (("Apri progetto…", self.open_project, "Ctrl+Shift+O"),
                              ("Salva progetto", self.save_project, "Ctrl+S"),
                              ("Salva progetto con nome…", lambda: self.save_project(ask=True), "Ctrl+Shift+S"),
                              ("Importa modello…", self.import_model, "Ctrl+O"),
                              ("Esporta scena come 3MF…", self.export_3mf, "Ctrl+Shift+E"),
                              ("Genera GOO…", self.export, "Ctrl+E")):
            a = QAction(text, self)
            a.setShortcut(key)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        m.addAction("Esporta impostazioni resina/supporti…", self.export_settings_file)
        m.addAction("Importa impostazioni resina/supporti…", self.import_settings_file)
        e = self.menuBar().addMenu("Modifica")
        for text, fn, key in (("Annulla", self.undo, "Ctrl+Z"), ("Ripeti", self.redo, "Ctrl+Y"),
                              ("Duplica", self.duplicate_selected, "Ctrl+D"), ("Array…", self.array_selected, ""),
                              ("Elimina", self.remove_selected, "Ctrl+Del")):
            a = QAction(text, self)
            if text == "Ripeti":
                a.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
            elif key:
                a.setShortcut(key)
            a.triggered.connect(fn)
            e.addAction(a)
        self.move_menu = e.addMenu("Sposta selezione su piatto")
        self.move_menu.aboutToShow.connect(lambda: self._fill_move_menu(self.move_menu))
        v = self.menuBar().addMenu("Vista")
        a = QAction("Reimposta vista", self)
        a.triggered.connect(self.viewer.view_fit)
        v.addAction(a)
        h = self.menuBar().addMenu("Aiuto")
        h.addAction("Cerca aggiornamenti…", lambda: self.check_updates(True))
        h.addAction("Associa i file .dsl a DragonSlicer…", self.associate_dsl)
        h.addAction(f"Informazioni (v{__version__})", lambda: QMessageBox.information(
            self, "DragonSlicer", f"DragonSlicer v{__version__}\nhttps://github.com/faber00AI/DragonSlicer"))
        QTimer.singleShot(4000, lambda: self.check_updates(False))

    def associate_dsl(self):
        from core import fileassoc
        try:
            fileassoc.register()
        except Exception as e:
            return QMessageBox.warning(self, "Associazione file", f"Non sono riuscito a registrare i file .dsl: {e}")
        QMessageBox.information(self, "Associazione file", "Fatto: i file .dsl ora hanno la loro icona e con un doppio click si aprono "
                                "in DragonSlicer.\nSe l'icona non cambia subito, riavvia Esplora risorse o attendi qualche istante.\n"
                                "Se sposti il programma in un'altra cartella, ripeti questa operazione.")

    def check_updates(self, manual):
        if getattr(self, "_upd_busy", False):
            return
        self._upd_busy = True
        t = _UpdateCheck(self)
        t.done.connect(lambda info, err: self._update_result(info, err, manual))
        t.finished.connect(lambda: setattr(self, "_upd_busy", False))
        self._upd_thread = t
        t.start()

    def _update_result(self, info, err, manual):
        if err:
            if manual:
                QMessageBox.warning(self, "Aggiornamenti", "Impossibile controllare gli aggiornamenti:" + chr(10) + err)
            return
        if not info:
            if manual:
                QMessageBox.information(self, "Aggiornamenti", f"Hai già l'ultima versione (v{__version__}).")
            return
        nl = chr(10)
        exe = Path(sys.executable).name if updater.can_self_update() else None
        url = info["assets"].get("DragonSlicer.exe")
        auto = bool(exe and url)
        txt = f"È disponibile la versione {info['version']} (hai la {__version__})." + nl * 2 + info["notes"][:800]
        txt += nl * 2 + ("Scaricare e installare ora? L'app si riavvierà." if auto else "Aprire la pagina di download?")
        if QMessageBox.question(self, "Aggiornamento disponibile", txt) != QMessageBox.Yes:
            return
        if not auto:
            QDesktopServices.openUrl(QUrl(info["page"]))
            return
        dlg = QProgressDialog("Download aggiornamento…", None, 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()
        try:
            def prog(p):
                dlg.setValue(p)
                QApplication.processEvents()
            files = {updater.download(url, prog): Path(sys.executable)}
            updater.apply_and_restart(files)
        except Exception as ex:
            dlg.close()
            QMessageBox.warning(self, "Aggiornamento", "Download non riuscito:" + chr(10) + str(ex))
            return
        dlg.close()
        QApplication.quit()

    def _side_panel(self):
        w = QWidget()
        w.setMinimumWidth(400)
        w.setMaximumWidth(520)
        outer = QVBoxLayout(w)
        bar = QHBoxLayout()
        for text, fn in (("Apri progetto", self.open_project), ("Salva progetto", self.save_project)):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, f=fn: f())
            bar.addWidget(btn)
        outer.addLayout(bar)
        self.b_generate = QPushButton("GENERA GOO")
        self.b_generate.setMinimumHeight(48)
        self.b_generate.setStyleSheet("QPushButton{background:#16a34a;color:white;font-size:16px;font-weight:bold;"
                                      "border-radius:6px}QPushButton:hover{background:#22c55e}"
                                      "QPushButton:disabled{background:#3a3f47;color:#888}")
        self.b_generate.clicked.connect(self.export)
        outer.addWidget(self.b_generate)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        outer.addWidget(self.progress)
        self.est = QLabel("")
        self.est.setWordWrap(True)
        outer.addWidget(self.est)
        tabs = QTabWidget()
        self.side_tabs = tabs

        def page():
            pw = QWidget()
            pl = QVBoxLayout(pw)
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(pw)
            return sa, pl

        page_scene, lay = page()

        g = QGroupBox("Stampante")
        f = QFormLayout(g)
        self.machine_box = QComboBox()
        self.machine_box.addItems([m.name for m in self.machines])
        pref = next((i for i, m in enumerate(self.machines) if m.name == "ELEGOO Saturn 4 Ultra"), 0)
        self.machine_box.setCurrentIndex(pref)
        self.machine_box.currentIndexChanged.connect(self._machine_changed)
        f.addRow("Modello", self.machine_box)
        self.machine_info = QLabel()
        f.addRow(self.machine_info)
        lay.addWidget(g)
        self._machine_changed(self.machine_box.currentIndex())

        g = QGroupBox("Oggetti")
        v = QVBoxLayout(g)
        self.obj_list = QListWidget()
        self.obj_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.obj_list.setMaximumHeight(130)
        self.obj_list.currentRowChanged.connect(self._select)
        self.obj_list.itemSelectionChanged.connect(self._refresh)
        self.viewer.clicked.connect(self._viewer_click)
        self.viewer.box_selected.connect(self._viewer_box)
        self.viewer.gizmo_begin.connect(lambda _: self.push_undo())
        self.viewer.gizmo_delta.connect(self._gizmo_delta)
        self.viewer.gizmo_end.connect(self._gizmo_end)
        v.addWidget(self.obj_list)
        b = QHBoxLayout()
        for text, fn in (("Importa", self.import_model), ("Duplica", self.duplicate_selected),
                         ("Array", self.array_selected), ("Rimuovi", self.remove_selected)):
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            b.addWidget(btn)
        btn = QPushButton("Sposta ▾")
        btn.setToolTip("Sposta gli oggetti selezionati su un altro piatto di stampa")
        self.move_btn_menu = QMenu(btn)
        self.move_btn_menu.aboutToShow.connect(lambda: self._fill_move_menu(self.move_btn_menu))
        btn.setMenu(self.move_btn_menu)
        b.addWidget(btn)
        v.addLayout(b)
        self.obj_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.obj_list.customContextMenuRequested.connect(
            lambda pos: self._context_menu(self.obj_list.row(self.obj_list.itemAt(pos)) if self.obj_list.itemAt(pos) else -1,
                                           self.obj_list.mapToGlobal(pos)))
        self.viewer.context_menu.connect(self._context_menu)
        outer.addWidget(g)

        g = QGroupBox("Trasformazione")
        f = QFormLayout(g)
        self.t_x, self.t_y = dspin(-500, 500, 0, 1, 2, ""), dspin(-500, 500, 0, 1, 2, "")
        self.t_z = dspin(0, 500, 0, 1, 2, "")
        self.t_z.setToolTip("Quota: altezza dal piatto")
        self.t_rx, self.t_ry, self.t_rz = (dspin(-360, 360, 0, 5, 1, "°") for _ in range(3))
        self.t_s = dspin(0.01, 100, 1, 0.05, 3)
        self.t_sx, self.t_sy, self.t_sz = (dspin(0.01, 100, 1, 0.05, 3) for _ in range(3))
        self.t_prop = QCheckBox("Proporzionale")
        self.t_prop.setChecked(True)
        self.t_prop.setToolTip("Attivo: la scala è una sola e tutti gli assi si ridimensionano insieme (campi, maniglie 3D e "
                               "dimensioni finali). Spento: X, Y e Z si scalano separatamente")
        self.t_prop.toggled.connect(lambda on: self._nonuni_toggled(not on))
        f.addRow(QLabel("Dimensioni finali"))
        self.t_dims = []
        for label, ax in (("Larghezza X", 0), ("Profondità Y", 1), ("Altezza Z", 2)):
            wd = dspin(0.01, 2000, 1, 1, 3, " mm")
            wd.setKeyboardTracking(False)                # apply on Enter / focus out, not on every digit
            wd.valueChanged.connect(lambda v, a=ax: self._dim_edited(a, v))
            f.addRow(label, wd)
            self.t_dims.append(wd)
        self.scale_rows = []
        for wd in (self.t_x, self.t_y, self.t_z):        # these three live in the Allinea box, on one row
            wd.valueChanged.connect(self._transform_edited)
        for label, wd in (("Rot X", self.t_rx), ("Rot Y", self.t_ry), ("Rot Z", self.t_rz)):
            row = QHBoxLayout()
            row.addWidget(wd, 1)
            for deg in (45, 90):                         # fixed values: they set the angle, they do not add to it
                b = QPushButton(str(deg))
                b.setFixedWidth(34)
                b.setToolTip(f"Imposta {label} a {deg}°")
                b.clicked.connect(lambda _=False, w_=wd, d=deg: w_.setValue(d))
                row.addWidget(b)
            f.addRow(label, row)
            wd.valueChanged.connect(self._transform_edited)
        for label, wd, uni in (("Scala", self.t_s, True), ("Scala X", self.t_sx, False), ("Scala Y", self.t_sy, False),
                               ("Scala Z", self.t_sz, False)):
            lab = QLabel(label)
            f.addRow(lab, wd)
            wd.valueChanged.connect(self._transform_edited)
            self.scale_rows.append((lab, wd, uni))
        self._scale_fields()
        f.addRow(self.t_prop)
        row = QHBoxLayout()
        btn = QPushButton("Posa sul piatto")
        btn.setToolTip("Posa sul piatto (stabile): ruota l'oggetto nella posizione più stabile, appoggiando sul piatto la "
                       "faccia piana più grande (azzera le rotazioni manuali e cancella i supporti di quell'oggetto)")
        btn.clicked.connect(self._lay_flat)
        row.addWidget(btn)
        btn = QPushButton("Orienta per stampa")
        btn.setToolTip("Prova molte inclinazioni e sceglie la migliore per la resina: meno sbalzi da supportare, meno isole, "
                       "sezione per strato più piccola (meno forza di distacco). Cancella i supporti dell'oggetto")
        btn.clicked.connect(lambda: self._orient(False))
        row.addWidget(btn)
        self.b_orient_next = QPushButton("Altra ▸")
        self.b_orient_next.setToolTip("Prossimo orientamento in classifica (dopo «Orienta per stampa»)")
        self.b_orient_next.setEnabled(False)
        self.b_orient_next.clicked.connect(lambda: self._orient(True))
        row.addWidget(self.b_orient_next)
        f.addRow(row)
        self.orient_info = QLabel("")
        self.orient_info.setWordWrap(True)
        f.addRow(self.orient_info)
        self._orient_state = None
        g_transform = g                                  # shown on the Modella tab, see below

        g = QGroupBox("Posizione e allineamento")
        gl_ = QGridLayout(g)
        pos = QHBoxLayout()
        pos.addWidget(QLabel("Pos. mm"))
        for axis, wd in (("X", self.t_x), ("Y", self.t_y), ("Z", self.t_z)):
            pos.addWidget(QLabel(axis))
            pos.addWidget(wd, 1)
        gl_.addLayout(pos, 0, 0, 1, 4)
        self.align_btns = []
        names = {0: ("Sinistra", "Centro", "Destra"), 1: ("Fronte", "Centro", "Retro"), 2: ("Base", "Centro", "Cima")}
        for r, axis in enumerate("XYZ"):
            gl_.addWidget(QLabel(axis), r + 1, 0)
            for c, how in enumerate(("min", "center", "max")):
                b = QPushButton(names[r][c])
                b.setToolTip("Allinea (servono 2 o più oggetti selezionati)")
                b.clicked.connect(lambda _=False, a=r, h=how: self.align_selected(a, h))
                gl_.addWidget(b, r + 1, c + 1)
                self.align_btns.append((b, 2))
        row = QHBoxLayout()
        for c, axis in enumerate("XY"):
            b = QPushButton(f"Distr. {axis}")
            b.setToolTip(f"Distribuisci {axis}: spazi uguali tra gli oggetti (servono 3 o più oggetti)")
            b.clicked.connect(lambda _=False, a=c: self.distribute_selected(a))
            row.addWidget(b)
            self.align_btns.append((b, 3))
        b = QPushButton("A su B")
        b.setToolTip("Metti A sopra B: A = oggetto corrente (selezionato per ultimo/evidenziato), B = l'altro. A poggia sulla cima di B")
        b.clicked.connect(self.stack_on_top)
        row.addWidget(b)
        self.align_btns.append((b, 2))
        gl_.addLayout(row, 4, 0, 1, 4)
        self.stack_center = QCheckBox("Centra A su B")
        self.stack_center.setChecked(True)
        gl_.addWidget(self.stack_center, 5, 1, 1, 3)
        g_align = g
        tabs.addTab(page_scene, "Scena")
        self.model_panel = ModelingPanel(self)
        lay.addWidget(self.model_panel.g_add)            # Scena: Stampante, Aggiungi oggetto, Libreria personale
        lay.addWidget(self.model_panel.g_lib)
        lay.addStretch(1)
        self.model_panel.layout().insertWidget(0, g_transform)   # Modella: Trasformazione, Allinea, then its own tools
        lp = self.model_panel.layout()
        lp.insertWidget(lp.indexOf(self.model_panel.g_bool) + 1, g_align)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setWidget(self.model_panel)
        self.model_page = sa
        tabs.addTab(sa, "Modella")
        self.mold_panel = MoldPanel(self)
        ms = QScrollArea()
        ms.setWidgetResizable(True)
        ms.setFrameShape(QFrame.NoFrame)
        ms.setWidget(self.mold_panel)
        tabs.addTab(ms, "Stampo")
        self.roller_panel = RollerPanel(self)
        rr = QScrollArea()
        rr.setWidgetResizable(True)
        rr.setFrameShape(QFrame.NoFrame)
        rr.setWidget(self.roller_panel)
        tabs.addTab(rr, "Rullo")
        tabs.currentChanged.connect(lambda _: self.model_panel._show_plane())
        page_sup, lay = page()

        g = QGroupBox("Supporti")
        f = QFormLayout(g)
        self.sup_preset = QComboBox()
        self.sup_preset.addItems(self.presets)
        self.sup_preset.currentTextChanged.connect(self._preset_changed)
        f.addRow("Preset", self.sup_preset)
        self.s_angle = dspin(0, 89, 30, 1, 0, "°")
        self.s_space = dspin(0.5, 20, 3, 0.5, 1, " mm")
        self.s_tip = dspin(0.05, 2, 0.3, 0.05, 2, " mm")
        self.s_body = dspin(0.2, 4, 0.7, 0.1, 2, " mm")
        self.s_lift = dspin(0, 30, 5, 0.5, 1, " mm")
        self.s_raft = dspin(0, 5, 1, 0.1, 1, " mm")
        self.s_rhole = dspin(0, 10, 2, 0.5, 1, " mm")
        self.s_cross = dspin(0, 100, 30, 5, 0, " %")
        self.s_cross.setToolTip("Traversini a X tra i pilastri vicini: 0 = nessuno, 10 = circa 3 livelli ogni 100 mm di altezza, 100 = circa 30. Funziona solo coi preset che li prevedono (Tree_Medium, Tree_Large).")
        for label, wd in (("Angolo critico", self.s_angle), ("Spaziatura", self.s_space), ("Raggio punta", self.s_tip),
                          ("Raggio fusto", self.s_body), ("Rialzo dal piatto", self.s_lift), ("Raft (0 = nessuno)", self.s_raft),
                          ("Fori raft (0 = pieno)", self.s_rhole), ("Traversini", self.s_cross)):
            f.addRow(label, wd)
        row = QHBoxLayout()
        btn = QPushButton("Genera")
        btn.setMinimumHeight(34)
        btn.setStyleSheet("QPushButton{background:#16a34a;color:white;font-size:14px;font-weight:bold;"
                          "border-radius:6px}QPushButton:hover{background:#22c55e}"
                          "QPushButton:disabled{background:#3a3f47;color:#888}")
        btn.clicked.connect(self.generate_supports)
        row.addWidget(btn, 3)
        btn = QPushButton("Rimuovi")
        btn.setMinimumHeight(28)
        btn.clicked.connect(self.clear_supports)
        row.addWidget(btn, 2)
        btn = QPushButton("Salva impostazioni")
        btn.setMinimumHeight(28)
        btn.setToolTip("Memorizza preset e valori: verranno ripristinati a ogni avvio del programma")
        btn.clicked.connect(self._save_support_settings)
        row.addWidget(btn, 3)
        f.addRow(row)
        self.sup_info = QLabel("")
        f.addRow(self.sup_info)
        lay.addWidget(g)

        g = QGroupBox("Modifica supporti")
        v = QVBoxLayout(g)
        self.sup_edit = QPushButton("Seleziona supporti nel 3D")
        self.sup_edit.setCheckable(True)
        self.sup_edit.setToolTip("Clic su un supporto per selezionarlo (Ctrl+clic per più di uno), poi Elimina o Canc")
        self.sup_edit.toggled.connect(self._support_edit_toggled)
        v.addWidget(self.sup_edit)
        row = QHBoxLayout()
        self.sup_del = QPushButton("Elimina selezionati")
        self.sup_del.clicked.connect(self.delete_selected_supports)
        b_none = QPushButton("Deseleziona")
        b_none.clicked.connect(lambda: self._set_support_sel(set()))
        row.addWidget(self.sup_del)
        row.addWidget(b_none)
        v.addLayout(row)
        self.sup_sel_info = QLabel("")
        self.sup_sel_info.setWordWrap(True)
        v.addWidget(self.sup_sel_info)
        lay.addWidget(g)
        self.viewer.ray_clicked.connect(self._support_click)
        self.viewer.delete_pressed.connect(self._delete_key)
        self.viewer.screenshot_requested.connect(self.save_screenshot)
        sc = QShortcut(QKeySequence(Qt.Key_Delete), self.obj_list)     # Canc with the object list focused
        sc.setContext(Qt.WidgetShortcut)
        sc.activated.connect(self.remove_selected)
        self._preset_changed(self.sup_preset.currentText())
        self._load_support_settings()

        lay.addStretch(1)
        tabs.addTab(page_sup, "Supporti")
        self.resin_panel = ResinPanel(self.machine)
        self.resin_panel.changed.connect(self._print_settings_changed)
        rs = QScrollArea()
        rs.setWidgetResizable(True)
        rs.setFrameShape(QFrame.NoFrame)
        rs.setWidget(self.resin_panel)
        tabs.addTab(rs, "Resina")
        self.printer_panel = PrinterPanel(self)
        sp = QScrollArea()
        sp.setWidgetResizable(True)
        sp.setFrameShape(QFrame.NoFrame)
        sp.setWidget(self.printer_panel)
        tabs.addTab(sp, "Stampante")
        outer.addWidget(tabs, 1)
        return w

    # ---------------- scene handling ----------------
    def _machine_changed(self, i):
        self.machine = self.machines[max(i, 0)]
        m = self.machine
        self.machine_info.setText(f"{m.bed_x:g} × {m.bed_y:g} × {m.bed_z:g} mm — {m.res_x}×{m.res_y} px")
        if hasattr(self, "viewer"):
            self.viewer.set_bed((m.bed_x, m.bed_y, m.bed_z))
        if hasattr(self, "resin_panel"):
            self.resin_panel.set_machine(m)            # also resets the print settings to the machine defaults
            self.ps = self.resin_panel.settings()
            self._preview_mesh = None
            self.layer_slider.setEnabled(False)

    def _sel(self):
        i = self.obj_list.currentRow()
        return self.scene.objects[i] if 0 <= i < len(self.scene.objects) else None

    # ---- object list helpers (shared with the Modella tab) ----
    def selected_rows(self):
        """Selected rows, the current object first (it is 'A' in a boolean)."""
        rows = sorted({self.obj_list.row(i) for i in self.obj_list.selectedItems()})
        cur = self.obj_list.currentRow()
        if cur in rows:
            rows.remove(cur)
            rows.insert(0, cur)
        elif cur >= 0 and not rows:
            rows = [cur]
        return rows

    def push_undo(self):
        self._undo.append(self.scene.snapshot())
        del self._undo[:-30]
        self._redo.clear()                                   # a new edit ends the redo chain

    def _swap_state(self, entry):
        """Restore `entry` (an undo/redo record) and return the record that undoes that restore."""
        if isinstance(entry, MoveUndo):                       # a move between plates is restored on both
            inverse = MoveUndo(self.scene.snapshot(), entry.dest, entry.dest["scene"].snapshot())
            entry.dest["scene"].restore(entry.dest_snap)
            entry.dest["info"] = ""
            self.scene.restore(entry.snap)
        else:
            inverse = self.scene.snapshot()
            self.scene.restore(entry)
        self._rebuild_list(min(max(self.obj_list.currentRow(), 0), len(self.scene.objects) - 1))
        return inverse

    def undo(self):
        if not self._undo:
            return self.statusBar().showMessage("Niente da annullare", 3000)
        self._redo.append(self._swap_state(self._undo.pop()))
        self.statusBar().showMessage("Annullato", 3000)

    def redo(self):
        if not self._redo:
            return self.statusBar().showMessage("Niente da ripetere", 3000)
        self._undo.append(self._swap_state(self._redo.pop()))
        self.statusBar().showMessage("Ripetuto", 3000)

    def _rebuild_list(self, select=None):
        self.obj_list.blockSignals(True)
        self.obj_list.clear()
        self.obj_list.addItems([o.name for o in self.scene.objects])
        if select is not None and 0 <= select < len(self.scene.objects):
            self.obj_list.setCurrentRow(select)
        self.obj_list.blockSignals(False)
        self._select(self.obj_list.currentRow())

    def reselect(self, row):
        self._rebuild_list(row)

    def remove_rows(self, rows):
        for r in sorted(rows, reverse=True):
            self.scene.remove(r)
        self._rebuild_list(min(max(rows[0] - 1, 0), len(self.scene.objects) - 1) if self.scene.objects else None)

    def add_mesh_object(self, mesh, name, select=True, keep_pos=True):
        """Add a mesh given in world coordinates, keeping where it is (x, y and height)."""
        o = self.scene.add(mesh, name)
        b = mesh.bounds
        o.x, o.y = ((b[0][:2] + b[1][:2]) / 2).tolist()
        o.lift = float(b[0][2]) if keep_pos else 0.0
        self.obj_list.blockSignals(True)
        self.obj_list.addItem(name)
        self.obj_list.blockSignals(False)
        if select:
            self.reselect(len(self.scene.objects) - 1)
        return o

    def duplicate_selected(self):
        rows = self.selected_rows()
        if not rows:
            return
        self.push_undo()
        for r in rows:
            o = self.scene.objects[r]
            m = o.world_mesh().copy()
            m.apply_translation([m.extents[0] + 5, 0, 0])
            self.add_mesh_object(m, o.name + " copia", select=False)
        self.reselect(len(self.scene.objects) - 1)

    def array_selected(self):
        """Ask for a number N and make N copies of the selected objects in a row along X (one undo step)."""
        rows = self.selected_rows()
        if not rows:
            return self.statusBar().showMessage("Seleziona uno o più oggetti", 4000)
        n, ok = QInputDialog.getInt(self, "Array", "Numero di copie:", 2, 1, 200)
        if not ok:
            return
        self.push_undo()
        objs = [self.scene.objects[r] for r in rows]
        lo = min(o.bounds()[0][0] for o in objs)
        hi = max(o.bounds()[1][0] for o in objs)
        step = hi - lo + 5                                   # the whole selection moves as one block
        for k in range(1, n + 1):
            for o in objs:
                m = o.world_mesh().copy()
                m.apply_translation([step * k, 0, 0])
                self.add_mesh_object(m, o.name + " copia", select=False)
        self.reselect(len(self.scene.objects) - 1)
        out = sum(1 for o in self.scene.objects if not o.fits(self.machine))
        self.statusBar().showMessage(f"Create {n * len(objs)} copie" + (f" — {out} fuori dal piatto: usa «Disponi tutti sul piatto»" if out else ""), 8000)

    # ---------------- build plates ----------------
    # Every plate has its own Scene (objects + supports) and undo stack; printer, resin and support settings are shared.
    def _sync_plate(self):
        pl = self.plates[self.plate_i]
        pl["scene"], pl["undo"], pl["redo"], pl["info"] = self.scene, self._undo, self._redo, self.sup_info.text()

    def switch_plate(self, i):
        if 0 <= i < len(self.plates) and i != self.plate_i:
            self._sync_plate()
            self._show_plate(i)

    def _show_plate(self, i):
        self.plate_i = i
        pl = self.plates[i]
        self.scene, self._undo = pl["scene"], pl["undo"]
        self._redo = pl.setdefault("redo", [])
        self._preview_mesh = None
        self.layer_slider.setEnabled(False)
        self.plate_bar.blockSignals(True)
        self.plate_bar.setCurrentIndex(i)
        self.plate_bar.blockSignals(False)
        self.b_generate.setText("GENERA GOO" if len(self.plates) == 1 else f"GENERA GOO — {pl['name']}")
        self._rebuild_list(0 if self.scene.objects else None)
        self.sup_info.setText(pl["info"])

    def add_plate(self, name=None, show=True):
        self._sync_plate()
        name = name or f"Piatto {len(self.plates) + 1}"
        self.plates.append(dict(name=name, scene=Scene(), undo=[], info=""))
        self.plate_bar.blockSignals(True)
        self.plate_bar.addTab(name)
        self.plate_bar.blockSignals(False)
        if show:
            self._show_plate(len(self.plates) - 1)

    def rename_plate(self, i):
        name, ok = QInputDialog.getText(self, "Rinomina piatto", "Nome:", text=self.plates[i]["name"])
        if ok and name.strip():
            self.plates[i]["name"] = name.strip()
            self.plate_bar.setTabText(i, name.strip())
            self._show_plate(self.plate_i)

    def delete_plate(self, i):
        if len(self.plates) == 1:
            return self.statusBar().showMessage("Serve almeno un piatto", 4000)
        self._sync_plate()
        n = len(self.plates[i]["scene"].objects)
        if n and QMessageBox.question(self, "Elimina piatto", f"«{self.plates[i]['name']}» contiene {n} oggetti. Eliminarlo?") != QMessageBox.Yes:
            return
        active = self.plate_i - (i < self.plate_i)            # the plate that stays on screen, if it is not the deleted one
        del self.plates[i]
        self.plate_bar.blockSignals(True)
        self.plate_bar.removeTab(i)
        self.plate_bar.blockSignals(False)
        self._show_plate(min(i, len(self.plates) - 1) if i == self.plate_i else active)

    def _plate_menu(self, pos):
        i = self.plate_bar.tabAt(pos)
        if i < 0:
            return
        m = QMenu(self)
        m.addAction("Rinomina…", lambda: self.rename_plate(i))
        m.addAction("Elimina piatto", lambda: self.delete_plate(i))
        if i != self.plate_i:
            m.addSeparator()
            m.addAction("Sposta qui la selezione", lambda: self.move_selected_to(i))
        m.exec(self.plate_bar.mapToGlobal(pos))

    def _fill_move_menu(self, menu):
        menu.clear()
        for i, pl in enumerate(self.plates):
            a = menu.addAction(pl["name"])
            a.setEnabled(i != self.plate_i)
            a.triggered.connect(lambda _=False, i=i: self.move_selected_to(i))
        menu.addSeparator()
        menu.addAction("Nuovo piatto con la selezione", self.move_to_new_plate)

    def move_to_new_plate(self):
        rows = self.selected_rows()
        if not rows:
            return self.add_plate()
        self.add_plate(show=False)
        self.move_selected_to(len(self.plates) - 1)

    def arrange_all(self):
        """Lay all the objects of this plate flat side by side, centred (the same packing as the CLI's `arrange`)."""
        if not self.scene.objects:
            return
        self.push_undo()
        place_objects(self.scene.objects, self.machine)
        self._rebuild_list(max(self.obj_list.currentRow(), 0))

    def _context_menu(self, row, pos):
        """Right-click menu (viewer or object list). A click on an unselected object selects it first."""
        if row >= 0 and not self.obj_list.item(row).isSelected():
            self._viewer_click(row, False)
        mp = getattr(self, "model_panel", None)
        m = QMenu(self)
        if row >= 0:
            m.addAction("Duplica", self.duplicate_selected)
            m.addAction("Rimuovi", self.remove_selected)
            m.addSeparator()
            mv = m.addMenu("Sposta su piatto")
            self._fill_move_menu(mv)
            m.addSeparator()
        else:
            m.addAction("Importa modello…", self.import_model)
            m.addAction("Seleziona tutto", self.obj_list.selectAll)
        m.addAction("Disponi tutti sul piatto", self.arrange_all)
        if row >= 0 and mp:
            m.addAction("Esporta selezione…", mp.export_selected)
        m.addSeparator()
        m.addAction("Nuovo piatto", lambda: self.add_plate())
        m.exec(pos)

    def move_selected_to(self, i):
        """Move the selected objects to plate `i` (same position, supports included)."""
        rows = self.selected_rows()
        if not rows or i == self.plate_i:
            return self.statusBar().showMessage("Seleziona un oggetto e scegli un altro piatto", 4000)
        self._sync_plate()
        dest = self.plates[i]
        self._undo.append(MoveUndo(self.scene.snapshot(), dest, dest["scene"].snapshot()))
        del self._undo[:-30]
        self._redo.clear()
        for r in rows:
            dest["scene"].objects.append(replace(self.scene.objects[r], _cache=None))
        dest["info"] = ""
        self.remove_rows(rows)
        self.statusBar().showMessage(f"{len(rows)} oggetti spostati su «{dest['name']}»", 5000)

    # ---------------- project file (.dsl) ----------------
    def _default_stem(self, fallback="progetto"):
        """File name proposed for every save: the object names joined by '_' (or just the first object's name
        when that gets too long); the fallback when the scene is empty."""
        names = []
        for o in self.scene.objects:
            n = re.sub(r"[^\w.-]", "_", Path(str(o.name)).stem).strip("_")
            if n and n not in names:
                names.append(n)
        if not names:
            return fallback
        joined = "_".join(names)
        return joined if len(joined) <= 60 else names[0]

    def save_project(self, ask=False, path=None):
        """Save machine, print/support settings and objects (mesh + transform) to a .dsl; same format as the CLI/MCP."""
        path = path or (None if ask else self.project_file)
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Salva progetto", self.project_file or (self._default_stem() + ".dsl"),
                                                  "Progetto DragonSlicer (*.dsl)")
            if not path:
                return False
        if not path.lower().endswith(DSL_EXT):
            path += DSL_EXT
        self._sync_plate()
        sp = self._support_params()

        def pack(scene, tag, sup_name):
            """(objects, supports record, {rel: ply bytes}) of one plate; `tag` keeps file names unique between plates."""
            objects, meshes = [], {}
            for i, o in enumerate(scene.objects):
                stem = re.sub(r"[^\w.-]", "_", o.name)
                rel = f"meshes/{tag}{i:03d}_{stem}.ply"
                meshes[rel] = o.mesh.export(file_type="ply")
                objects.append(dict(name=o.name, mesh=rel, lift=o.lift, **({'text': o.text} if o.text else {}),
                                    **{k: float(getattr(o, k)) for k in TRANSFORM}))
            sup = dict(enabled=scene.supports is not None, lift=sp.lift_height if scene.supports is not None else 0.0)
            if scene.supports is not None:                    # the generated supports, stored as they are (regenerating can differ slightly)
                sup["mesh"] = sup_name
                meshes[sup_name] = scene.supports.export(file_type="ply")
            return objects, sup, meshes

        objects, sup, meshes = pack(self.plates[0]["scene"], "", "meshes/supports.ply")
        p = dict(machine=self.machine.name, print=self._print_settings().as_dict(), objects=objects,
                 supports=dict(preset=self.sup_preset.currentText(),
                               overrides=dict(critical_angle=sp.critical_angle, spacing=sp.spacing, tip_radius=sp.tip_radius,
                                              body_radius=sp.body_radius, lift_height=sp.lift_height,
                                              raft_height=sp.raft_height, raft_hole=sp.raft_hole,
                                              cross_percent=sp.cross_percent), **sup))
        if len(self.plates) > 1:
            p["plate_name"] = self.plates[0]["name"]
            p["extra_plates"] = []
            for j, pl in enumerate(self.plates[1:], 1):
                o, sup_j, m = pack(pl["scene"], f"p{j}_", f"meshes/supports_p{j}.ply")
                meshes.update(m)
                p["extra_plates"].append(dict(name=pl["name"], objects=o, supports=sup_j))
        try:
            self.run_busy("Salvo il progetto…", lambda: write_dsl(path, p, meshes))
        except Exception as e:
            QMessageBox.critical(self, "Salvataggio fallito", str(e))
            return False
        self._set_project_file(path)
        self._mark_clean()
        self.statusBar().showMessage(f"Progetto salvato: {path}", 6000)
        return True

    def open_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Apri progetto", "", "Progetti (*.dsl *.3mf);;Progetto DragonSlicer (*.dsl);;Progetto SatelLite (*.goo.3mf *.3mf)")
            if not path:
                return False
        if any(pl["scene"].objects for pl in self.plates) and QMessageBox.question(
                self, "Apri progetto", "La scena attuale verrà sostituita. Continuare?") != QMessageBox.Yes:
            return False
        try:
            satellite = is_satellite(path)               # a SatelLite .goo.3mf is converted to the same (project, meshes) pair
            if satellite:
                p, raw, notes = self.run_busy("Leggo il progetto SatelLite…", lambda: read_satellite(path))
            else:
                p, raw = self.run_busy("Apro il progetto…", lambda: read_dsl(path))
            meshes = {rel: trimesh.load(io.BytesIO(data), file_type="ply", force="mesh", process=False)
                      for rel, data in raw.items()}
            idx = next((i for i, m in enumerate(self.machines) if m.name == p["machine"]), None)
            if idx is None:
                raise ProjectError(f"stampante sconosciuta: {p['machine']!r}")
        except Exception as e:
            QMessageBox.critical(self, "Apertura fallita", str(e))
            return False
        self.machine_box.setCurrentIndex(idx)             # also resets the slicing fields to the machine defaults
        self.machine_box.currentIndexChanged.emit(idx)
        ps = PrintSettings.from_machine(self.machine)
        for k, v in p["print"].items():
            if k in PrintSettings.__dataclass_fields__:
                setattr(ps, k, type(getattr(ps, k))(v))
        known = {r["name"] for r in resins.list_profiles(self.machine.name)}
        self.resin_panel.set_settings(ps, ps.resin if ps.resin in known else None)
        self.ps = ps
        s = p["supports"]
        if s["preset"] in self.presets:
            self.sup_preset.setCurrentText(s["preset"])
        for wd, k in ((self.s_angle, "critical_angle"), (self.s_space, "spacing"), (self.s_tip, "tip_radius"),
                      (self.s_body, "body_radius"), (self.s_lift, "lift_height"),
                      (self.s_raft, "raft_height"), (self.s_rhole, "raft_hole"), (self.s_cross, "cross_percent")):
            if k in s["overrides"]:
                wd.setValue(s["overrides"][k])

        def scene_from(records, srec):
            sc, info = Scene(), ""
            for rec in records:
                o = sc.add(meshes[rec["mesh"]], rec["name"])
                for k in TRANSFORM:
                    setattr(o, k, rec[k])
                o.lift = rec.get("lift", srec["lift"] if srec["enabled"] else 0.0)
                o.text = rec.get("text")
            if srec["enabled"] and sc.objects:
                if srec.get("mesh"):
                    sc.supports = trimesh.load(io.BytesIO(raw[srec["mesh"]]), file_type="ply", force="mesh", process=False)
                    info = "Supporti caricati dal progetto"
                else:
                    sp, total = self._support_params(), 0
                    for o in sc.printable(self.machine):
                        sup, n = build_supports(o.world_mesh(), sp)
                        sc.set_object_supports(o, sup)
                        total += n
                    if sc.printable(self.machine):
                        info = f"{total} supporti" if total else "Nessun supporto necessario"
            return sc, info

        sc, info = scene_from(p["objects"], s)
        self.plates = [dict(name=p.get("plate_name", "Piatto 1"), scene=sc, undo=[], info=info)]
        for pl in p.get("extra_plates", []):
            sc, info = scene_from(pl["objects"], pl["supports"])
            self.plates.append(dict(name=pl["name"], scene=sc, undo=[], info=info))
        self.plate_bar.blockSignals(True)
        while self.plate_bar.count():
            self.plate_bar.removeTab(0)
        for pl in self.plates:
            self.plate_bar.addTab(pl["name"])
        self.plate_bar.blockSignals(False)
        self.plate_i = 0
        self._show_plate(0)
        self.viewer.view_fit()
        if satellite:                                     # saving must not offer to overwrite the SatelLite file
            self.project_file = None
            self.setWindowTitle(f"DragonSlicer — {Path(path).name}")
            self.statusBar().showMessage(f"Progetto SatelLite importato: {'; '.join(notes)}", 12000)
        else:
            self._set_project_file(path)
            self.statusBar().showMessage(f"Progetto aperto: {path}", 6000)
        self._mark_clean()
        return True

    def _set_project_file(self, path):
        self.project_file = path
        self.setWindowTitle(f"DragonSlicer — {Path(path).name}")

    # ---------------- unsaved changes ----------------
    def _state_sig(self):
        """Cheap fingerprint of everything a project file stores: used to tell whether there is unsaved work."""
        self._sync_plate()
        plates = []
        for pl in self.plates:
            objs = tuple((o.name, id(o.mesh), o.x, o.y, o.rx, o.ry, o.rz, o.sx, o.sy, o.sz, o.lift, id(o.sup))
                         for o in pl["scene"].objects)
            plates.append((pl["name"], objs))
        try:
            ps = repr(self.resin_panel.settings().as_dict())
        except Exception:
            ps = ""
        return (tuple(plates), ps, self.machine.name, self.sup_preset.currentText(), repr(self._support_params()))

    def _mark_clean(self):
        self._clean_sig = self._state_sig()

    def _has_unsaved(self):
        if getattr(self, "_clean_sig", None) is None:
            return any(pl["scene"].objects for pl in self.plates)          # never saved: only an empty scene is "clean"
        return self._state_sig() != self._clean_sig

    def closeEvent(self, e):
        if self._thread is None and self._has_unsaved():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Modifiche non salvate")
            box.setText("Ci sono modifiche non salvate al progetto.")
            box.setInformativeText("Vuoi salvarle prima di chiudere?")
            b_save = box.addButton("Salva", QMessageBox.AcceptRole)
            b_drop = box.addButton("Chiudi senza salvare", QMessageBox.DestructiveRole)
            box.addButton("Annulla", QMessageBox.RejectRole)
            box.setDefaultButton(b_save)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_save:
                if not self.save_project():
                    return e.ignore()
            elif clicked is not b_drop:
                return e.ignore()
        super().closeEvent(e)

    def import_model(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Importa", "", "Modelli (*.stl *.obj *.3mf *.ply *.off);;Tutti (*)")
        self.import_paths(paths)

    @staticmethod
    def _ask_repair(parent, name, diag, batch):
        """True / False for this file; 'all' / 'none' to answer for the rest of the batch too."""
        parts = []
        if diag["open_edges"]:
            parts.append(f"{diag['open_edges']} spigoli aperti")
        if diag["nonmanifold_edges"]:
            parts.append(f"{diag['nonmanifold_edges']} spigoli non-manifold")
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Mesh aperta")
        box.setText(f"«{name}» non è a tenuta" + (f" ({', '.join(parts)})" if parts else "") + ".\n"
                    "Una mesh aperta può dare slicing e supporti sbagliati. Vuoi ripararla ora?")
        yes = box.addButton("Ripara", QMessageBox.AcceptRole)
        no = box.addButton("Lascia com'è", QMessageBox.RejectRole)
        yes_all = box.addButton("Ripara tutte", QMessageBox.AcceptRole) if batch else None
        no_all = box.addButton("Nessuna", QMessageBox.RejectRole) if batch else None
        box.exec()
        c = box.clickedButton()
        return {yes: True, no: False, yes_all: "all", no_all: "none"}.get(c, False)

    def import_paths(self, paths):
        """Import files; folders are searched recursively for model files."""
        files = []
        for p in map(Path, paths):
            if p.is_dir():
                files += sorted(f for f in p.rglob("*") if f.suffix.lower() in MODEL_EXTS)
            else:
                files.append(p)
        paths = [str(f) for f in files]
        projects = [p for p in paths if is_satellite(p)]
        if projects:
            return self.open_project(projects[0])
        if not paths:
            return self.statusBar().showMessage("Nessun modello trovato (STL / OBJ / 3MF / PLY / OFF)", 5000)
        added = False
        repair_all = None                                # None = ask, True/False = answer given for the whole batch
        for p in paths:
            try:
                mesh = self.run_busy(f"Carico {Path(p).name}…", lambda p=p: trimesh.load(p, force="mesh"))
                if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                    raise ValueError("nessuna mesh valida")
            except Exception as e:
                QMessageBox.warning(self, "Importazione fallita", f"{Path(p).name}: {e}")
                continue
            if mesh.extents.max() < SMALL_MM:
                m = self.machine
                k = self.ask_scale(self, Path(p).name, mesh.extents, (m.bed_x, m.bed_y, m.bed_z))
                if k:
                    mesh.apply_scale(k)
                    self.statusBar().showMessage(f"{Path(p).name}: scalato ×{k:g}", 6000)
            if not mesh.is_watertight:
                answer = repair_all if repair_all is not None else self.ask_repair(self, Path(p).name, self.run_busy("Controllo la mesh…", lambda: md.diagnose(mesh)),
                                                                                   len(paths) > 1)
                if answer in ("all", "none"):
                    repair_all = answer == "all"
                if answer in (True, "all"):
                    mesh, info = self.run_busy(f"Riparo {Path(p).name}…", lambda m=mesh: md.repair(m))
                    self.statusBar().showMessage(f"{Path(p).name}: {ModelingPanel._problems(info['before'])} → "
                                                 f"{ModelingPanel._problems(info['after'])}", 10000)
                else:
                    self.statusBar().showMessage(f"{Path(p).name}: mesh non a tenuta (Modella > Ripara mesh)", 8000)
            if not added:
                self.push_undo()
            added = True
            self.scene.add(mesh, Path(p).name)
            self.obj_list.blockSignals(True)
            self.obj_list.addItem(Path(p).name)
            self.obj_list.blockSignals(False)
        if added:
            self.reselect(len(self.scene.objects) - 1)

    def _delete_key(self):
        """Canc in the 3D view: selected support points while editing supports, otherwise the selected objects."""
        if self.viewer.pick_rays:
            self.delete_selected_supports()
        else:
            self.remove_selected()

    def remove_selected(self):
        rows = self.selected_rows()
        if rows:
            self.push_undo()
            self.remove_rows(rows)

    def _scale_fields(self):
        non = (not self.t_prop.isChecked())
        for lab, wd, uni in self.scale_rows:
            lab.setVisible(uni != non)
            wd.setVisible(uni != non)

    def _nonuni_toggled(self, on):
        """Proportional on/off only changes how FUTURE edits behave: the object keeps its current shape."""
        self._scale_fields()
        o = self._sel()
        if o is not None and not self._building:
            self._load_fields(o)

    def _load_fields(self, o):
        self._building = True
        for wd, v in ((self.t_x, o.x), (self.t_y, o.y), (self.t_z, o.lift), (self.t_rx, o.rx), (self.t_ry, o.ry),
                      (self.t_rz, o.rz), (self.t_s, o.scale), (self.t_sx, o.sx), (self.t_sy, o.sy), (self.t_sz, o.sz)):
            wd.setValue(v)
        self._building = False

    def _select(self, _):
        o = self._sel()
        if o:
            self._load_fields(o)
        self._refresh()
        if hasattr(self, "model_panel"):
            self.model_panel.on_selection_changed()

    def _update_dims(self, o):
        ext = np.ptp(o.bounds(), axis=0)
        was, self._building = self._building, True
        for wd, v in zip(self.t_dims, ext):
            wd.setValue(max(float(v), wd.minimum()))
        self._building = was

    def _dim_edited(self, axis, value):
        """Set the final size (mm) of the selected object along a world axis."""
        o = self._sel()
        if self._building or o is None:
            return
        cur = float(np.ptp(o.bounds(), axis=0)[axis])
        if cur < 1e-9 or value <= 0:
            return
        k = value / cur
        self.push_undo()
        if self.t_prop.isChecked():
            o.sx, o.sy, o.sz = o.sx * k, o.sy * k, o.sz * k
        else:
            local = int(np.argmax(np.abs(o.rotation()[axis])))   # object axis lying closest to that world axis
            setattr(o, ("sx", "sy", "sz")[local], getattr(o, ("sx", "sy", "sz")[local]) * k)
        o.sup = None                                     # a resize invalidates the supports
        o.invalidate()
        self._select(0)

    def _transform_edited(self):
        o = self._sel()
        if self._building or o is None:
            return
        o.x, o.y, o.lift = self.t_x.value(), self.t_y.value(), self.t_z.value()
        rot = (self.t_rx.value(), self.t_ry.value(), self.t_rz.value())
        old = (o.rx, o.ry, o.rz, o.sx, o.sy, o.sz)
        o.rx, o.ry, o.rz = rot
        # a scale field only counts when the user changed it (compared with what it displays), so moving or rotating
        # never re-applies a rounded scale value
        if self.t_prop.isChecked():
            if abs(self.t_s.value() - round(o.scale, 3)) > 1e-9:
                k = self.t_s.value() / o.scale
                o.sx, o.sy, o.sz = o.sx * k, o.sy * k, o.sz * k
        else:
            for attr, wd in (("sx", self.t_sx), ("sy", self.t_sy), ("sz", self.t_sz)):
                if abs(wd.value() - round(getattr(o, attr), 3)) > 1e-9:
                    setattr(o, attr, wd.value())
        if old != (o.rx, o.ry, o.rz, o.sx, o.sy, o.sz):
            o.sup = None                                 # rotation or scale invalidates the supports
        o.invalidate()
        self._refresh()

    # ---- mouse selection and gizmo ----
    def _viewer_click(self, row, additive):
        lst = self.obj_list
        if row < 0:
            if not additive:
                lst.clearSelection()
                lst.setCurrentRow(-1)
            return
        item = lst.item(row)
        if additive:
            if item.isSelected():
                item.setSelected(False)
            else:
                lst.setCurrentRow(row, QItemSelectionModel.NoUpdate)
                item.setSelected(True)
        else:
            lst.clearSelection()
            lst.setCurrentRow(row)
            item.setSelected(True)

    def _viewer_box(self, rows, additive):
        lst = self.obj_list
        if not additive:
            lst.clearSelection()
        for r in sorted(rows):
            if 0 <= r < lst.count():
                if not additive and lst.currentRow() < 0:
                    lst.setCurrentRow(r, QItemSelectionModel.NoUpdate)
                lst.item(r).setSelected(True)
        if rows and (lst.currentRow() not in rows):
            lst.setCurrentRow(min(rows), QItemSelectionModel.NoUpdate)

    @staticmethod
    def _rotate_world(o, axis, deg):
        a = np.zeros(3)
        a[axis] = 1
        M = trimesh.transformations.rotation_matrix(np.radians(deg), a)
        M[:3, :3] = M[:3, :3] @ o.rotation()
        o.rx, o.ry, o.rz = np.degrees(trimesh.transformations.euler_from_matrix(M, "sxyz")).tolist()

    # ---- align / distribute ----
    def _shift(self, o, axis, delta):
        if axis == 0:
            o.x += delta
        elif axis == 1:
            o.y += delta
        else:
            o.lift = max(0.0, o.lift + delta)
        o.invalidate()

    def align_selected(self, axis, how):
        rows = self.selected_rows()
        if len(rows) < 2:
            return self.statusBar().showMessage("Seleziona almeno due oggetti", 3000)
        objs = [self.scene.objects[r] for r in rows]
        b = np.array([o.world_mesh().bounds[:, axis] for o in objs])
        lo, hi = b[:, 0].min(), b[:, 1].max()
        self.push_undo()
        for o, (a0, a1) in zip(objs, b):
            if how == "min":
                d = lo - a0
            elif how == "max":
                d = hi - a1
            else:
                d = (lo + hi) / 2 - (a0 + a1) / 2
            self._shift(o, axis, d)
        if self._sel() is not None:
            self._load_fields(self._sel())
        self._refresh()

    def stack_on_top(self):
        rows = self.selected_rows()
        if len(rows) != 2:
            return self.statusBar().showMessage("Seleziona esattamente due oggetti: A (corrente) e B", 4000)
        a, b = (self.scene.objects[r] for r in rows)
        top = b.world_mesh().bounds
        self.push_undo()
        if self.stack_center.isChecked():
            ba = a.world_mesh().bounds
            a.x += (top[0][0] + top[1][0]) / 2 - (ba[0][0] + ba[1][0]) / 2
            a.y += (top[0][1] + top[1][1]) / 2 - (ba[0][1] + ba[1][1]) / 2
        a.lift = float(top[1][2])
        a.invalidate()
        self.statusBar().showMessage(f"{a.name} appoggiato sopra {b.name}", 4000)
        if self._sel() is not None:
            self._load_fields(self._sel())
        self._refresh()

    def distribute_selected(self, axis):
        rows = self.selected_rows()
        if len(rows) < 3:
            return self.statusBar().showMessage("Servono almeno tre oggetti", 3000)
        objs = sorted((self.scene.objects[r] for r in rows), key=lambda o: o.world_mesh().bounds[:, axis].mean())
        b = [o.world_mesh().bounds[:, axis].copy() for o in objs]
        gap = ((b[-1][0] - b[0][1]) - sum(x[1] - x[0] for x in b[1:-1])) / (len(objs) - 1)
        self.push_undo()
        pos = b[0][1] + gap
        for o, (a0, a1) in zip(objs[1:-1], b[1:-1]):
            self._shift(o, axis, pos - a0)
            pos += (a1 - a0) + gap
        self._refresh()

    def _gizmo_delta(self, mode, axis, v):
        for r in self.selected_rows():
            o = self.scene.objects[r]
            if mode == "move":
                if axis == 2:
                    o.lift = max(0.0, o.lift + v)
                else:
                    o.x, o.y = (o.x + v, o.y) if axis == 0 else (o.x, o.y + v)
            elif mode == "rotate":
                self._rotate_world(o, axis, v)
                o.sup = None
            else:
                f = float(np.clip(v, 0.01, 100))
                if not self.t_prop.isChecked():
                    s = [o.sx, o.sy, o.sz]
                    s[axis] = float(np.clip(s[axis] * f, 0.01, 100))
                else:                                    # proportional: all three factors, shape preserved
                    s = [float(np.clip(c * f, 0.01, 100)) for c in (o.sx, o.sy, o.sz)]
                o.sx, o.sy, o.sz = s
                o.sup = None
            o.invalidate()
        cur = self._sel()
        if cur is not None:
            self._load_fields(cur)
        self._refresh(light=True)

    def _gizmo_end(self):
        self._refresh()
        if hasattr(self, "model_panel"):
            self.model_panel.on_selection_changed()

    def _orient(self, nxt):
        """Rank print orientations of the selected object and apply the best (or, with `nxt`, the next one)."""
        o = self._sel()
        if o is None:
            return self.statusBar().showMessage("Seleziona un oggetto", 4000)
        st = self._orient_state
        if not (nxt and st and st["obj"] is o and o in self.scene.objects):
            from core import orient
            try:
                res, base = self.run_busy("Analizzo gli orientamenti…",
                                          lambda: orient.analyse(o.world_mesh(), self.s_angle.value()))
            except Exception as e:
                return self.statusBar().showMessage(f"Analisi non riuscita: {e}", 6000)
            self.push_undo()
            st = self._orient_state = dict(obj=o, res=res, base=base, idx=-1, rot=o.rotation().copy())
        st["idx"] = (st["idx"] + 1) % min(5, len(st["res"]))
        c, base = st["res"][st["idx"]], st["base"]
        M = np.eye(4)
        M[:3, :3] = c["R"] @ st["rot"]
        o.rx, o.ry, o.rz = np.degrees(trimesh.transformations.euler_from_matrix(M, "sxyz")).tolist()
        o.lift = 0.0
        o.sup = None
        o.invalidate()
        self.b_orient_next.setEnabled(True)
        self._select(0)
        self.orient_info.setText(
            f"Orientamento {st['idx'] + 1} di {min(5, len(st['res']))}: sbalzi {c['support']:.0f} mm² (prima {base['support']:.0f}) · "
            f"isole {c['islands']} (prima {base['islands']}) · sezione max {c['peel']:.0f} mm² (prima {base['peel']:.0f}) · "
            f"altezza {c['height']:.0f} mm")
        self.statusBar().showMessage("Orientamento applicato — «Altra ▸» per le alternative", 6000)

    def _lay_flat(self):
        rows = self.selected_rows()
        if not rows:
            return self.statusBar().showMessage("Seleziona un oggetto", 4000)
        self.push_undo()
        ok = 0
        for r in rows:
            o = self.scene.objects[r]
            if o.lay_flat():
                o.sup = None
                ok += 1
        self._select(0)
        self.statusBar().showMessage("Posato sul piatto" if ok == len(rows) else
                                     f"Posa riuscita su {ok} di {len(rows)} oggetti", 5000)

    def _geometry(self, mesh, live):
        """Geometry for `mesh`, built once and reused (a move/rotate/scale only changes the object's matrix)."""
        key = id(mesh)
        ent = self._geo.get(key)
        if ent is None or ent[0] is not mesh:
            ent = (mesh, Geometry(mesh))
        live[key] = ent
        return ent[1]

    def _clip_layers(self):
        """Number of layers of what is on the plate (objects + supports), or 0 when it is empty."""
        tops = [o.bounds()[1][2] for o in self.scene.printable(self.machine)]
        if tops and self.scene.supports is not None and len(self.scene.supports.vertices):
            tops.append(float(self.scene.supports.bounds[1][2]))
        return max(1, int(np.ceil(max(tops) / self.resin_panel.settings().layer_height - 1e-6))) if tops else 0

    def _update_clip_range(self):
        """Keep the 3D layer slider in step with the scene; at the top it means 'no cut'."""
        n = self._clip_layers()
        s = self.clip_slider
        was_top = s.value() >= s.maximum()
        s.blockSignals(True)
        s.setRange(1, max(n, 1))
        s.setValue(s.maximum() if was_top else min(s.value(), s.maximum()))
        s.blockSignals(False)
        s.setEnabled(n > 1)
        self._clip_changed(s.value())

    def _clip_changed(self, v):
        top = v >= self.clip_slider.maximum()
        lh = self.resin_panel.settings().layer_height
        self.clip_label.setText("Tutto" if top else f"Strato {v}/{self.clip_slider.maximum()}\n{(v - 0.5) * lh:.2f} mm")
        self.clip_slider.setToolTip(self.clip_label.text().replace("\n", " - ") + "\n" + self.clip_label.toolTip())
        self.viewer.set_clip_z(None if top else (v - 0.5) * lh)      # the plane the slicer uses for layer v

    def _refresh(self, light=False):
        items = []
        sel = {self.obj_list.row(i) for i in self.obj_list.selectedItems()} | {self.obj_list.currentRow()}
        sel.discard(-1)
        live = {}
        for k, o in enumerate(self.scene.objects):
            items.append(Drawable(self._geometry(o.mesh, live), SEL_COLOR if k in sel else MODEL_COLOR,
                                  pick=k, matrix=o.matrix()))
        has_sup = any(o.sup is not None for o in self.scene.objects)
        if has_sup:
            pl = self._pillars() if getattr(self, "sup_edit", None) is not None and self.sup_edit.isChecked() else dict(sel=())
            if pl["sel"]:                                            # picked pillars in their own colour
                if pl.get("view_key") != frozenset(pl["sel"]):
                    chosen = np.isin(pl["lab"][np.asarray(self.scene.supports.faces)[:, 0]], pl["ids"][sorted(pl["sel"])])
                    sup = self.scene.supports
                    pl["view"] = [trimesh.Trimesh(np.asarray(sup.vertices), np.asarray(sup.faces)[m], process=False)
                                  for m in (~chosen, chosen)]
                    pl["view_key"] = frozenset(pl["sel"])
                rest, picked = pl["view"]
                items.append(Drawable(self._geometry(rest, live), SUPPORT_COLOR))
                items.append(Drawable(self._geometry(picked, live), SUPPORT_SEL_COLOR))
            else:                                                    # each object's supports ride on its matrix: cheap while dragging
                for o in self.scene.objects:
                    if o.sup is not None:
                        items.append(Drawable(self._geometry(o.sup, live), SUPPORT_COLOR, matrix=o.matrix()))
        elif getattr(self, "sup_edit", None) is not None and self.sup_edit.isChecked():
            self.sup_edit.setChecked(False)
        for key, (_, g) in self._geo.items():
            if key not in live:
                self.viewer.discard(g)
        self._geo = live
        self.viewer.set_drawables(items)
        if hasattr(self, "resin_panel"):
            self._update_clip_range()
        for b, need in getattr(self, "align_btns", ()):
            b.setEnabled(len(sel & set(range(len(self.scene.objects)))) >= need)
        valid = sorted(k for k in sel if k < len(self.scene.objects))
        if valid:
            b = np.array([self.scene.objects[k].bounds() for k in valid])
            cur = self._sel() or self.scene.objects[valid[0]]
            self._update_dims(cur)
            self.viewer.set_gizmo((b[:, 0].min(axis=0) + b[:, 1].max(axis=0)) / 2, cur.rotation())
        else:
            self.viewer.set_gizmo(None)
        if light:
            return
        self._update_estimate()

    def _update_estimate(self):
        out = len(self.scene.objects) - len(self.scene.printable(self.machine))
        tops = [o.bounds()[1][2] for o in self.scene.printable(self.machine)]
        if tops and self.scene.supports is not None and len(self.scene.supports.vertices):
            tops.append(float(self.scene.supports.bounds[1][2]))
        if not tops and out:
            self.est.setText(f"Nessun oggetto dentro l'area di stampa ({out} fuori area ignorati)")
        elif tops:
            ps = self.resin_panel.settings()
            n = max(1, int(np.ceil(max(tops) / ps.layer_height - 1e-6)))
            t = ps.estimate_seconds(n)
            price = ""
            if ps.cost > 0:
                try:                                  # resin of the part (supports included), no waste
                    vol = sum(abs(o.world_mesh().volume) for o in self.scene.printable(self.machine))
                    if self.scene.supports is not None and len(self.scene.supports.vertices):
                        vol += abs(self.scene.supports.volume)
                    price = f" — resina {vol / 1000:.1f} ml ≈ {vol / 1e6 * ps.cost:.2f} €"
                except Exception:
                    price = ""
            self.est.setText(f"{n} layer — altezza {max(tops):.1f} mm — stampa stimata {int(t // 3600)}h {int(t % 3600 // 60)}m"
                            + price + (f" — {out} fuori area ignorati" if out else ""))

    # ---------------- supports ----------------
    def _preset_changed(self, name):
        d = self.presets.get(name)
        if not d:
            return
        self.s_angle.setValue(d["critical_angle"]); self.s_space.setValue(d["spacing"])
        self.s_tip.setValue(d["tip_radius"]); self.s_body.setValue(d["body_radius"])
        self.s_lift.setValue(d["lift_height"]); self.s_raft.setValue(d.get("raft_height", 0.0))
        self.s_rhole.setValue(d.get("raft_hole", 0.0))
        self.s_cross.setValue(d.get("cross_percent", 0.0) if d.get("cross_radius", 0) > 0 else 0.0)

    _SUPPORT_FIELDS = (("s_angle", "critical_angle"), ("s_space", "spacing"), ("s_tip", "tip_radius"),
                       ("s_body", "body_radius"), ("s_lift", "lift_height"), ("s_raft", "raft_height"), ("s_rhole", "raft_hole"),
                       ("s_cross", "cross_percent"))

    def _load_support_settings(self):
        """Restore the support preset and values saved with the 'Salva impostazioni' button (QSettings)."""
        st = app_settings()
        preset = st.value("supports/preset", type=str)
        if preset in self.presets:
            self.sup_preset.blockSignals(True)
            self.sup_preset.setCurrentText(preset)
            self.sup_preset.blockSignals(False)
            self._preset_changed(preset)
        for attr, key in self._SUPPORT_FIELDS:
            v = st.value(f"supports/{key}")
            if v is not None:
                try:
                    getattr(self, attr).setValue(float(v))
                except (TypeError, ValueError):
                    pass

    def _save_support_settings(self):
        st = app_settings()
        st.setValue("supports/preset", self.sup_preset.currentText())
        for attr, key in self._SUPPORT_FIELDS:
            st.setValue(f"supports/{key}", getattr(self, attr).value())
        st.sync()
        self.statusBar().showMessage("Impostazioni supporti salvate: verranno ripristinate a ogni avvio", 5000)

    def export_settings_file(self):
        """Dump resin profiles + the values now in the Resina and Supporti tabs to a JSON file, to survive a rebuild."""
        path, _ = QFileDialog.getSaveFileName(self, "Esporta impostazioni", "dragonslicer_impostazioni.json", "JSON (*.json);;Testo (*.txt)")
        if not path:
            return
        data = dict(format="dragonslicer-settings", version=1, machine=self.machine.name,
                    resin_profiles=resins._user(),
                    current_print=self.resin_panel.settings().as_dict(),
                    supports=dict(preset=self.sup_preset.currentText(),
                                  **{key: getattr(self, attr).value() for attr, key in self._SUPPORT_FIELDS}))
        try:
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "Esportazione fallita", str(e))
            return
        self.statusBar().showMessage(f"Impostazioni esportate in {path}", 5000)

    def import_settings_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Importa impostazioni", "", "Impostazioni (*.json *.txt);;Tutti (*)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("format") not in ("dragonslicer-settings", "dragonslide-settings"):
                raise ValueError("non è un file di impostazioni DragonSlicer")
            n = resins.import_user_profiles(data.get("resin_profiles", {}))
        except (OSError, ValueError, AttributeError, resins.ResinError) as e:
            QMessageBox.critical(self, "Importazione fallita", str(e))
            return
        sup = data.get("supports") or {}
        if sup.get("preset") in self.presets:
            self.sup_preset.blockSignals(True)
            self.sup_preset.setCurrentText(sup["preset"])
            self.sup_preset.blockSignals(False)
        for attr, key in self._SUPPORT_FIELDS:
            try:
                getattr(self, attr).setValue(float(sup[key]))
            except (KeyError, TypeError, ValueError):
                pass
        self._save_support_settings()
        cp = data.get("current_print")
        self.resin_panel._fill_combo()
        if cp and data.get("machine") == self.machine.name:
            ps = PrintSettings.from_machine(self.machine)
            for k, v in cp.items():
                if k in resins.FIELDS:
                    setattr(ps, k, type(getattr(ps, k))(v))
            self.resin_panel.set_settings(ps, cp.get("resin") or None)
            self._print_settings_changed()
        self.statusBar().showMessage(f"Importati {n} profili resina e le impostazioni supporti", 5000)

    def _support_params(self):
        d = dict(self.presets[self.sup_preset.currentText()])
        d.update(critical_angle=self.s_angle.value(), spacing=self.s_space.value(), tip_radius=self.s_tip.value(),
                 body_radius=self.s_body.value(), lift_height=self.s_lift.value(), raft_height=self.s_raft.value(),
                 raft_hole=self.s_rhole.value(), cross_percent=self.s_cross.value())
        return SupportParams.from_dict(d)

    def _support_targets(self):
        """Objects the support buttons act on: the selected ones, or all when nothing is selected."""
        rows = [r for r in self.selected_rows() if r < len(self.scene.objects)]
        return [self.scene.objects[r] for r in rows] if rows else list(self.scene.objects)

    def _support_count_text(self, extra=""):
        n = sum(len(split_pillars(o.sup)[1]) for o in self.scene.objects if o.sup is not None)
        return (f"{n} supporti" if any(o.sup is not None for o in self.scene.objects) else "") + extra

    def generate_supports(self):
        """Build supports for the selected objects only (all when none is selected) with the current parameters;
        the other objects keep the supports they already have, so one plate can mix different support types."""
        if not self.scene.objects:
            return
        p = self._support_params()
        targets = self._support_targets()
        inside = [o for o in targets if o.fits(self.machine)]     # objects outside the print area get no supports
        skipped = len(targets) - len(inside)
        for o in targets:
            o.sup = None
        for o in inside:
            o.lift = p.lift_height
            o.invalidate()
        total = 0
        for o in inside:
            sup, n = build_supports(o.world_mesh(), p)
            self.scene.set_object_supports(o, sup)
            total += n
        if not inside:
            txt = "Nessun oggetto dentro l'area di stampa"
        else:
            txt = f"{total} supporti su {len(inside)} oggetti" if total else "Nessun supporto necessario"
        self.sup_info.setText(txt + (f" ({skipped} fuori area ignorati)" if skipped else ""))
        self._refresh()

    # ---- single supports: pick and delete ----
    def _pillars(self):
        """The pillars of the current support mesh (cached by mesh identity), with the set of picked ones."""
        sup = self.scene.supports
        c = getattr(self, "_pillar_cache", None)
        if c is None or c["mesh"] is not sup:
            lab, ids, a, b = split_pillars(sup)
            c = self._pillar_cache = dict(mesh=sup, lab=lab, ids=ids, a=a, b=b, sel=set())
        return c

    def _support_edit_toggled(self, on):
        self.viewer.pick_rays = on
        if on and self.scene.supports is None:
            self.sup_edit.setChecked(False)
            return self.statusBar().showMessage("Genera prima i supporti", 4000)
        self._set_support_sel(set())

    def _set_support_sel(self, sel):
        if self.scene.supports is None:
            self.sup_sel_info.setText("")
            return
        pl = self._pillars()
        pl["sel"] = set(sel)
        n = len(pl["ids"])
        self.sup_sel_info.setText(f"{len(sel)} selezionati su {n} supporti" if self.sup_edit.isChecked() else "")
        self._refresh(light=True)

    def _support_click(self, o, d, additive):
        if self.scene.supports is None:
            return
        pl = self._pillars()
        if not len(pl["ids"]):
            return
        a, b = pl["a"], pl["b"]
        e, w0 = b - a, a - o
        A, B, D, E = (e * e).sum(1), e @ d, (e * w0).sum(1), w0 @ d
        den = A - B * B
        u = np.clip(np.where(den > 1e-9, (B * E - D) / np.where(den > 1e-9, den, 1), 0.5), 0, 1)
        s = E + u * B
        dist = np.linalg.norm(a + u[:, None] * e - (o + s[:, None] * d), axis=1)
        dist[s <= 0] = np.inf
        k = int(dist.argmin())
        tol = max(1.0, float(s[k]) * 0.012)                          # a little forgiving: the pillars are thin
        sel = set(pl["sel"]) if additive else set()
        if dist[k] <= tol:
            sel ^= {k}
        self._set_support_sel(sel)

    def delete_selected_supports(self):
        if self.scene.supports is None or not self.sup_edit.isChecked():
            return
        pl = self._pillars()
        if not pl["sel"]:
            return self.statusBar().showMessage("Seleziona prima i supporti da eliminare", 4000)
        n, left = len(pl["sel"]), len(pl["ids"]) - len(pl["sel"])
        self.push_undo()
        if left <= 0:
            self.sup_edit.setChecked(False)
            self.clear_supports(everything=True)
            return self.statusBar().showMessage("Eliminati tutti i supporti", 4000)
        self.scene.supports = drop_pillars(self.scene.supports, pl["lab"], pl["ids"][sorted(pl["sel"])])
        self.sup_info.setText(f"{left} supporti")
        self._refresh()
        self._set_support_sel(set())
        self.statusBar().showMessage(f"{n} supporti eliminati (Annulla per ripristinarli)", 5000)

    def clear_supports(self, everything=False):
        """Remove the supports of the selected objects (all when none is selected) and put them back on the plate."""
        for o in (self.scene.objects if everything else self._support_targets()):
            o.sup = None
            o.lift = 0
            o.invalidate()
        self.sup_info.setText(self._support_count_text())
        self._refresh()

    # ---------------- slicing ----------------
    def _print_settings(self):
        self.ps = self.resin_panel.settings()
        return self.ps

    def _print_settings_changed(self):
        self.ps = self.resin_panel.settings()
        self._preview_mesh = None
        self.layer_slider.setEnabled(False)
        self._update_estimate()
        self._update_clip_range()

    def preview_layers(self):
        mesh = self.scene.combined(machine=self.machine)
        if mesh is None:
            return
        ps = self._print_settings()
        n = layer_count(mesh, ps.layer_height)
        self._preview_mesh, self._preview_n = mesh, n
        self.layer_slider.blockSignals(True)
        self.layer_slider.setRange(1, n)
        self.layer_slider.setValue(max(1, n // 2))
        self.layer_slider.setEnabled(True)
        self.layer_slider.blockSignals(False)
        self.views.setCurrentIndex(1)
        self._layer_changed(self.layer_slider.value())
        t = ps.estimate_seconds(n)
        self.est.setText(f"{n} layer — {mesh.bounds[1][2]:.1f} mm — stampa stimata {int(t // 3600)}h {int(t % 3600 // 60)}m")

    def _layer_changed(self, v):
        mesh = getattr(self, "_preview_mesh", None)
        if mesh is None:
            return
        W = 1280
        H = int(W * self.machine.res_y / self.machine.res_x)
        (_, mask), = slice_layers(mesh, self.machine, self.ps.layer_height, W, H, indices=[v - 1])
        mask = np.ascontiguousarray(mask)
        img = QImage(mask.data, W, H, W, QImage.Format_Grayscale8).copy()
        self.preview.setPixmap(QPixmap.fromImage(img).scaled(self.preview.size(), Qt.KeepAspectRatio,
                                                             Qt.SmoothTransformation))
        z = (v - 0.5) * self.ps.layer_height
        self.layer_label.setText(f"{v}/{self._preview_n}  ({z:.2f} mm)")
        self.viewer.set_slice_z(z)

    def export_3mf(self):
        """Standard 3MF of the plate (no supports), laid out like a SatelLite project: origin at the plate corner."""
        if not self.scene.objects:
            return
        stem = self._default_stem("scena")
        path, _ = QFileDialog.getSaveFileName(self, "Esporta scena come 3MF", stem + ".3mf", "3MF (*.3mf)")
        if not path:
            return
        parts = []
        for o in self.scene.objects:
            w = o.world_mesh().copy()
            w.apply_translation([self.machine.bed_x / 2, self.machine.bed_y / 2, -o.lift])
            parts.append((o.name, w))
        try:
            out = self.run_busy("Scrivo il 3MF…", lambda: write_3mf(path, parts))
        except Exception as e:
            return QMessageBox.critical(self, "Esportazione fallita", str(e))
        self.statusBar().showMessage(f"3MF salvato: {out}", 8000)

    def export(self):
        mesh = self.scene.combined(machine=self.machine)
        if mesh is None or self._thread is not None:
            return
        folder = Path(self.project_file).parent if self.project_file else Path.home()
        stem = self._default_stem("job")
        dlg = GenerateDialog(self, self.machines, self.machine.name, None, str(folder), stem + ".goo",
                             printer_panel=self.printer_panel)
        if not dlg.exec():
            return
        machine, resin, path, send, start, wifi = dlg.result_values()
        if not path and not send:
            return
        if machine != self.machine.name:
            self.machine_box.setCurrentText(machine)           # resets the print settings to that printer's defaults
        if resin:
            self.resin_panel.select_profile(resin)
        name = Path(path).name if path else f"{stem}.goo"
        if send and not self.printer_panel.confirm_wifi(wifi, name, start):
            return
        temp = not path
        if temp:
            path = str(Path(tempfile.gettempdir()) / f"dragonslicer_{os.getpid()}.goo")
        self._thread = ExportThread(path, mesh, self.machine, self._print_settings(), self._viewer_preview())
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.b_generate.setEnabled(False)
        self._thread.progress.connect(lambda i, n: (self.progress.setMaximum(n), self.progress.setValue(i)))
        self._thread.done.connect(lambda ok, err: self._export_done(
            ok, err, send=(wifi, path, name, start, temp) if send else None))
        self._thread.start()

    def save_screenshot(self):
        """Image of the 3D view (build plate and objects, without the tool buttons): saved as PNG and copied."""
        self.views.setCurrentIndex(0)
        img = self.viewer.grabFramebuffer()
        start = (Path(self.project_file).parent if self.project_file else Path.home() / "Pictures") / (self._default_stem("piatto") + ".png")
        path, _ = QFileDialog.getSaveFileName(self, "Salva screenshot del piatto", str(start), "Immagine PNG (*.png)")
        QApplication.clipboard().setImage(img)
        if not path:
            self.statusBar().showMessage("Screenshot copiato negli appunti", 6000)
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if img.save(path, "PNG"):
            self.statusBar().showMessage(f"Screenshot salvato (e copiato negli appunti): {path}", 8000)
        else:
            QMessageBox.critical(self, "Screenshot", f"Impossibile salvare {path}")

    def _viewer_preview(self):
        """Current 3D view as a PIL image (used for the .goo thumbnails)."""
        try:
            self.views.setCurrentIndex(0)
            qi = self.viewer.grabFramebuffer().convertToFormat(QImage.Format_RGB888)
            return Image.frombuffer("RGB", (qi.width(), qi.height()), bytes(qi.constBits()),
                                    "raw", "RGB", qi.bytesPerLine(), 1).copy()
        except Exception:
            return None

    def _export_done(self, ok, err, send=None):
        self.progress.setVisible(False)
        self.b_generate.setEnabled(True)
        self._thread = None
        if send:
            wifi, path, name, start, temp = send
        if err:
            QMessageBox.critical(self, "Esportazione fallita", err)
        elif ok:
            self.statusBar().showMessage("Esportazione completata", 8000)
            if send:
                self.printer_panel.send_generated(wifi, path, name, start, remove=temp)
                return
        if send and temp:
            Path(path).unlink(missing_ok=True)
