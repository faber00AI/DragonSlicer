"""Stampo tab: silicone mold around the selected model. Two kinds: a shell that hugs the model (default) or a plain box."""
import numpy as np
import trimesh

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from core import moldbox, moldshell

from .appsettings import app_settings

GREEN = ("QPushButton{background:#16a34a;color:white;font-size:14px;font-weight:bold;border-radius:6px}"
         "QPushButton:hover{background:#22c55e}QPushButton:disabled{background:#3a3f47;color:#888}")


def _spin(lo, hi, val, step, dec=1, suffix=" mm", tip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(dec)
    s.setSuffix(suffix)
    if tip:
        s.setToolTip(tip)
    return s


def _fits_alone(ext, m):
    """True if a part of this size fits on the plate by itself (either orientation on the plate, height included)."""
    ex, ey, ez = (float(v) for v in ext)
    return ez <= m.bed_z and ((ex <= m.bed_x and ey <= m.bed_y) or (ey <= m.bed_x and ex <= m.bed_y))


class MoldPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)

        g = QGroupBox("Stampo in silicone")
        v = QVBoxLayout(g)
        self.kind = QComboBox()
        self.kind.addItems(["Guscio sagomato sul modello", "Scatola rettangolare"])
        self.kind.setToolTip("Il guscio segue la forma del modello (meno silicone, si divide in due metà); "
                             "la scatola è un parallelepipedo attorno al modello (più semplice)")
        v.addWidget(self.kind)
        self.txt = QLabel("")
        self.txt.setWordWrap(True)
        v.addWidget(self.txt)

        self.f = f = QFormLayout()
        self.rows = {}                                   # name -> (widget, shown for kinds)

        def row(name, label, wd, kinds=("shell", "box")):
            f.addRow(label, wd)
            self.rows[name] = (label, wd, kinds)
        self.silicone = _spin(3, 60, 10, 1, tip="Spessore del silicone attorno al modello. Sotto gli 8 mm il calco può strapparsi "
                                                 "sui modelli con molti dettagli")
        self.top = _spin(3, 60, 12, 1, tip="Silicone sopra il punto più alto del modello")
        self.wall = _spin(1, 12, 5, 0.5, tip="Spessore delle pareti stampate")
        self.smooth = _spin(0, 30, 6, 1, tip="Levigatura: riempie scanalature e rientranze più strette del doppio di questo valore. "
                                              "Più alta = guscio più semplice e meno sottosquadri; 0 = segue ogni dettaglio")
        self.axis = QComboBox()
        self.axis.addItems(["X", "Y"])
        self.axis.setToolTip("Il guscio si divide in due con un piano per il centro del modello, perpendicolare a questo asse")
        self.spout = _spin(6, 80, 14, 1, tip="Diametro del foro di colata, sul punto più alto della cavità")
        row("silicone", "Spessore silicone", self.silicone)
        row("top", "Silicone sopra", self.top, ("box",))
        row("wall", "Parete", self.wall)
        row("smooth", "Levigatura", self.smooth, ("shell",))
        row("axis", "Divisione perpendicolare a", self.axis, ("shell",))
        row("spout", "Foro di colata Ø", self.spout, ("shell",))
        v.addLayout(f)

        self.adv = QGroupBox("Avanzate")
        self.adv.setCheckable(True)
        self.adv.setChecked(False)
        fa = QFormLayout(self.adv)
        self.clear = _spin(0, 1.5, 0.25, 0.05, 2, tip="Gioco tra guscio/scatola e scanalatura della base, per lato. Aumentalo se l'incastro è stretto")
        self.base = _spin(2, 20, 6, 0.5, tip="Spessore della base")
        self.groove = _spin(1, 15, 3, 0.5, tip="Profondità della scanalatura: quanto il bordo del guscio entra nella base")
        self.margin = _spin(2, 30, 8, 1, tip="Bordo della base oltre il guscio")
        self.pocket = _spin(0, 10, 2, 0.5, tip="Incavo al centro della base con la sagoma del modello originale: serve da guida per "
                                                "appoggiarlo nel punto giusto. Profondità dell'incavo; 0 = nessun incavo. Il gioco è "
                                                "quello dell'incastro")
        self.seam_w = _spin(4, 40, 10, 1, tip="Larghezza delle linguette (flangia) lungo il taglio, misurata dalla parete del guscio verso "
                                               "l'esterno. È la superficie dove agganci mollette o morsetti: più larga = presa più comoda, "
                                               "ma pezzi più grandi")
        self.seam_t = _spin(2, 25, 3, 0.5, tip="Spessore delle linguette su ciascuna metà (a guscio chiuso il doppio). Più spesso = più "
                                                "rigido e più presa per le mollette; 2 mm bastano per delle mollette leggere. "
                                                "Intorno a ogni perno la linguetta si ispessisce da sola con un piccolo rinforzo "
                                                "tondo, il resto resta sottile")
        self.vents = QCheckBox("Sfiati d'aria")
        self.vents.setChecked(True)
        self.vents.setToolTip("Piccoli fori sugli altri punti alti della cavità, dove resterebbe l'aria")
        self.pins = QCheckBox("Perni di allineamento")
        self.pins.setToolTip("Perni e fori sulle linguette, alternati tra le due metà, per chiuderle centrate e senza scivolamenti")
        self.pins.setChecked(True)
        self.pin_d = _spin(1.5, 8, 2.5, 0.5, tip="Diametro dei perni di allineamento (lunghezza = 1,5 volte, minimo 5 mm). Il foro ha 0,2 mm di gioco")
        self.pin_n = QSpinBox()
        self.pin_n.setRange(1, 12)
        self.pin_n.setValue(4)
        self.pin_n.setToolTip("Quanti perni (e relativi fori) mettere lungo il taglio. Ne mette meno se non c'è posto sulle linguette")
        self.res = QComboBox()
        self.res.addItems(list(moldshell.RES))
        self.res.setCurrentText("Normale")
        self.res.setToolTip("Risoluzione del calcolo: Fine è più preciso ma più lento e usa più memoria")
        self.arows = {}
        for name, label, wd, kinds in (("clear", "Gioco incastro", self.clear, ("shell", "box")),
                                       ("base", "Spessore base", self.base, ("shell", "box")),
                                       ("groove", "Profondità incastro", self.groove, ("shell", "box")),
                                       ("margin", "Bordo base", self.margin, ("shell", "box")),
                                       ("pocket", "Incavo per il modello", self.pocket, ("shell",)),
                                       ("seam_w", "Larghezza linguette", self.seam_w, ("shell",)),
                                       ("seam_t", "Spessore linguette", self.seam_t, ("shell",)),
                                       ("vents", "", self.vents, ("shell",)),
                                       ("pins", "", self.pins, ("shell",)),
                                       ("pin_d", "Ø perni", self.pin_d, ("shell",)),
                                       ("pin_n", "N. perni", self.pin_n, ("shell",)),
                                       ("res", "Risoluzione", self.res, ("shell",))):
            if label:
                fa.addRow(label, wd)
            else:
                fa.addRow(wd)
            self.arows[name] = (fa, wd, kinds)
        v.addWidget(self.adv)

        self.adv.setToolTip("Impostazioni di dettaglio: incastro con la base, linguette di chiusura, perni, sfiati e risoluzione")
        self.btn = QPushButton("Genera stampo")
        self.btn.setToolTip("Crea base e guscio (o scatola) attorno al modello selezionato e li aggiunge al piatto")
        self.btn.setMinimumHeight(34)
        self.btn.setStyleSheet(GREEN)
        self.btn.clicked.connect(self.generate)
        v.addWidget(self.btn)
        self.b_save = QPushButton("Salva impostazioni")
        self.b_save.setToolTip("Memorizza tutte le impostazioni di questa tab (un unico salvataggio, senza profili): "
                               "verranno ripristinate a ogni avvio del programma")
        self.b_save.clicked.connect(self.save_settings)
        v.addWidget(self.b_save)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        lay.addWidget(g)
        lay.addStretch(1)
        for fl, wds in ((self.f, [r[1] for r in self.rows.values()]), (None, None)):
            if fl is None:
                continue
            for wd in wds:
                lbl = fl.labelForField(wd)
                if lbl is not None and wd.toolTip():
                    lbl.setToolTip(wd.toolTip())
        for fa, wd, _k in self.arows.values():
            lbl = fa.labelForField(wd)
            if lbl is not None and wd.toolTip():
                lbl.setToolTip(wd.toolTip())
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.load_settings()
        self._kind_changed(self.kind.currentIndex())

    def _fields(self):
        return dict(kind=self.kind, silicone=self.silicone, top=self.top, wall=self.wall, smooth=self.smooth, axis=self.axis,
                    spout=self.spout, clear=self.clear, base=self.base, groove=self.groove, margin=self.margin, pocket=self.pocket,
                    seam_w=self.seam_w, seam_t=self.seam_t, vents=self.vents, pins=self.pins, pin_d=self.pin_d,
                    pin_n=self.pin_n, res=self.res, adv=self.adv)

    def save_settings(self):
        st = app_settings()
        for key, wd in self._fields().items():
            if isinstance(wd, QComboBox):
                st.setValue(f"mold/{key}", wd.currentIndex())
            elif isinstance(wd, (QCheckBox, QGroupBox)):
                st.setValue(f"mold/{key}", wd.isChecked())
            else:
                st.setValue(f"mold/{key}", wd.value())
        st.sync()
        self.win.statusBar().showMessage("Impostazioni stampo salvate: verranno ripristinate a ogni avvio", 5000)

    def load_settings(self):
        st = app_settings()
        for key, wd in self._fields().items():
            v = st.value(f"mold/{key}")
            if v is None:
                continue
            try:
                if isinstance(wd, QComboBox):
                    wd.setCurrentIndex(max(0, min(int(v), wd.count() - 1)))
                elif isinstance(wd, (QCheckBox, QGroupBox)):
                    wd.setChecked(str(v).lower() in ("true", "1"))
                else:
                    wd.setValue(float(v))
            except (TypeError, ValueError):
                pass

    def _is_shell(self):
        return self.kind.currentIndex() == 0

    def _kind_changed(self, _):
        k = "shell" if self._is_shell() else "box"
        self.txt.setText(
            "Seleziona il modello da clonare e genera: ottieni una <b>base</b> con scanalatura e un <b>guscio</b> che segue la forma "
            "del modello, diviso in due metà con flangia e perni, con foro di colata e sfiati. Incolla il modello al centro della "
            "base, chiudi le metà (mollette o elastici), cola dall'alto." if k == "shell" else
            "Seleziona il modello da clonare e genera: ottieni una <b>base</b> con scanalatura e una <b>scatola</b> rettangolare che "
            "si incastra nella base. Incolla il modello al centro, inserisci la scatola, cola il silicone dall'alto.")
        for lab, wd, kinds in self.rows.values():
            wd.setVisible(k in kinds)
            lbl = self.f.labelForField(wd)
            if lbl is not None:
                lbl.setVisible(k in kinds)
        for fa, wd, kinds in self.arows.values():
            wd.setVisible(k in kinds)
            lbl = fa.labelForField(wd)
            if lbl is not None:
                lbl.setVisible(k in kinds)

    def generate(self):
        w = self.win
        rows = w.selected_rows()
        if not rows:
            return self._fail("Seleziona il modello da clonare.")
        objs = [w.scene.objects[r] for r in rows]
        master = trimesh.util.concatenate([o.world_mesh().copy() for o in objs])
        name = objs[0].name if len(objs) == 1 else "stampo"
        try:
            if self._is_shell():
                sp = moldshell.ShellParams(
                    silicone=self.silicone.value(), smooth=self.smooth.value(), wall=self.wall.value(),
                    clearance=self.clear.value(), base=self.base.value(), groove=self.groove.value(),
                    margin=self.margin.value(), pocket=self.pocket.value(), seam_w=self.seam_w.value(), seam_t=self.seam_t.value(), spout=self.spout.value(),
                    vents=self.vents.isChecked(), split_axis=self.axis.currentText(), pins=self.pins.isChecked(),
                    pin_d=self.pin_d.value(), pin_n=self.pin_n.value(), res=self.res.currentText())
                res = w.run_busy("Genero il guscio sagomato…", lambda: moldshell.build(master, sp))
                parts = [("base", res["base"])] + [(f"guscio {k}", m) for k, m in enumerate(res["shells"], 1)]
            else:
                bp = moldbox.MoldParams(silicone=self.silicone.value(), top=self.top.value(), wall=self.wall.value(),
                                        clearance=self.clear.value(), base=self.base.value(), groove=self.groove.value(),
                                        flange=self.margin.value())
                res = w.run_busy("Genero la scatola…", lambda: moldbox.build(master, bp))
                parts = [("base", res["base"]), ("scatola", res["frame"])]
        except Exception as e:
            return self._fail(e)
        info = res["info"]
        b = master.bounds
        cy = (b[0][1] + b[1][1]) / 2
        w.push_undo()
        x = b[1][0] + 10                                           # parts side by side to the right of the master, 10 mm apart
        placed = []
        for label, m in parts:
            m = m.copy()
            m.apply_translation([0, 0, -m.bounds[0][2]])           # every part rests on the plate
            cx = (m.bounds[0][0] + m.bounds[1][0]) / 2
            ccy = (m.bounds[0][1] + m.bounds[1][1]) / 2
            m.apply_translation([x + m.extents[0] / 2 - cx, cy - ccy, 0])
            placed.append(w.add_mesh_object(m, f"{name} – {label}", select=False))
            x += m.extents[0] + 10
        w.reselect(len(w.scene.objects) - 1)
        mach = w.machine
        too_big = [o.name for o in placed if not _fits_alone(np.ptp(o.bounds(), axis=0), mach)]
        out = [o.name for o in placed if not o.fits(mach)]
        ml = info["silicone_ml"]
        text = f"Silicone necessario ≈ {ml:.0f} ml (con il 10% di scarto: {ml * 1.1:.0f} ml)"
        if self._is_shell():
            text += (f"\n{info['parts']} metà · {info['pins']} perni · {info['vents']} sfiati · foro di colata Ø{self.spout.value():.0f} mm"
                     f"\nBase {info['base'][0]:.0f}×{info['base'][1]:.0f} mm · guscio {info['shell'][0]:.0f}×{info['shell'][1]:.0f}×"
                     f"{info['shell'][2]:.0f} mm")
            if info.get("pocket", 0) > 0:
                text += f"\nIncavo al centro della base ({info['pocket']:.1f} mm): appoggia lì il modello originale."
            if self.pins.isChecked() and info["pins"] == 0:
                text += "\nNessun perno: la flangia non ha spazio. Aumenta la larghezza della flangia o riduci il Ø dei perni."
        else:
            cav = info["cavity"]
            text += (f"\nCavità {cav[0]:.0f}×{cav[1]:.0f}×{cav[2]:.0f} mm · base {info['base'][0]:.0f}×{info['base'][1]:.0f} mm · "
                     f"scatola alta {info['frame'][2]:.0f} mm")
        if not self._is_shell() and not info["master_watertight"]:
            text += "\nAttenzione: il modello non è chiuso, il volume del silicone è approssimato (usa «Ripara» in Modella)."
        if too_big:
            text += ("\nTroppo grandi per il piatto: " + ", ".join(too_big) + ". Riduci gli spessori, usa la scatola oppure "
                     "una stampante più grande.")
        elif out:
            text += "\nAlcuni pezzi sono fuori dall'area di stampa: usa «Disponi tutti sul piatto» (tasto destro) per sistemarli."
        self.info.setText(text)
        w.statusBar().showMessage("Stampo generato", 6000)

    def _fail(self, e):
        self.info.setText(f"{e}")
        self.win.statusBar().showMessage(str(e), 6000)
