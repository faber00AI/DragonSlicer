"""3D viewport: orbit camera (Z up), build plate, meshes, mouse picking and a move/rotate/scale gizmo.
Fixed-function OpenGL via PyOpenGL."""
import math
import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QButtonGroup, QToolButton

AXIS_COLORS = ((0.92, 0.25, 0.25), (0.35, 0.80, 0.30), (0.30, 0.50, 0.95))
MODES = ("select", "transform", "scale")      # "transform" = move arrows + rotation rings together
RING = 0.62                                         # ring radius as a fraction of the arrow length


CAP_COLOR = (0.96, 0.82, 0.30)      # the cut face of the layer view


def _perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def _look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _axis_line_param(o, d, a, e):
    """Parameter u of the point a+u*e (unit e) of the axis line that is closest to the ray o+s*d (unit d)."""
    w0 = a - o
    B, D, E = e @ d, e @ w0, d @ w0
    den = 1.0 - B * B
    return None if den < 1e-6 else (B * E - D) / den


def _ray_segment(o, d, a, b):
    """(distance, ray parameter) of the closest approach between the ray and the segment a-b."""
    e = b - a
    w0 = a - o
    A, B, D, E = e @ e, e @ d, e @ w0, d @ w0
    den = A - B * B
    u = 0.0 if den < 1e-9 else float(np.clip((B * E - D) / den, 0, 1))
    s = E + u * B
    return float(np.linalg.norm(a + u * e - (o + s * d))), s


def _arc(axes, i, count):
    """Unit points of the quarter circle around axis i that runs from axis i+1 to axis i+2 (right-handed)."""
    a = np.linspace(0, math.pi / 2, count)[:, None]
    return np.cos(a) * axes[:, (i + 1) % 3] + np.sin(a) * axes[:, (i + 2) % 3]


def _perp(n):
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    return u / np.linalg.norm(u)


def _mode_icon(kind, size=40):
    """Tool icon drawn in code (no image files, so it works inside the exe)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = size / 2
    col = QColor(235, 238, 245)
    p.setPen(QPen(col, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(col)

    def arrow(x0, y0, x1, y1, h=5.5):
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        a = math.atan2(y1 - y0, x1 - x0)
        pts = [QPointF(x1, y1)] + [QPointF(x1 - h * math.cos(a + d), y1 - h * math.sin(a + d)) for d in (0.5, -0.5)]
        p.drawPolygon(QPolygonF(pts))

    if kind == "select":                          # cursor arrow
        pts = [(13, 8), (13, 30), (19, 25), (23, 33), (27, 31), (23, 23), (31, 23)]
        p.setBrush(col)
        p.drawPolygon(QPolygonF([QPointF(*q) for q in pts]))
    elif kind == "transform":                     # four arrows
        r = size * 0.36
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            arrow(c + dx * 3, c + dy * 3, c + dx * r, c + dy * r)
    elif kind == "camera":                        # camera body, viewfinder bump and lens
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(6, 13, 28, 19), 3, 3)
        p.setBrush(col)
        p.drawPolygon(QPolygonF([QPointF(*q) for q in ((14, 13), (16.5, 8), (23.5, 8), (26, 13))]))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(20, 22.5), 5.5, 5.5)
    else:                                         # square with a diagonal arrow
        p.setBrush(Qt.NoBrush)
        p.drawRect(QRectF(9, 17, 14, 14))
        p.setBrush(col)
        arrow(18, 22, 31, 9, 6)
    p.end()
    return QIcon(pm)


class Geometry:
    """Triangle soup of one mesh in its own (local) space, uploaded once to GPU buffers and reused every frame."""

    def __init__(self, mesh):
        tri = mesh.triangles.astype(np.float32)
        self.verts = np.ascontiguousarray(tri.reshape(-1, 3))
        self.norms = np.ascontiguousarray(np.repeat(mesh.face_normals.astype(np.float32), 3, axis=0))
        self.count = len(self.verts)
        self.lo, self.hi = (mesh.bounds[0], mesh.bounds[1]) if len(mesh.faces) else (np.zeros(3), np.zeros(3))
        self.buffers = None                  # (vertex VBO, normal VBO) once uploaded, False if VBOs are unavailable
        self._edges = None

    def upload(self):
        try:
            ids = gl.glGenBuffers(2)
            for buf, data in zip(ids, (self.verts, self.norms)):
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            self.buffers = tuple(int(i) for i in ids)
            self.norms = None                # only the GPU needs the normals from here on
        except Exception:
            self.buffers = False

    def draw(self):
        if self.buffers is None:
            self.upload()
        if self.buffers:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffers[0])
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.buffers[1])
            gl.glNormalPointer(gl.GL_FLOAT, 0, None)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        else:
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.verts)
            gl.glNormalPointer(gl.GL_FLOAT, 0, self.norms)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.count)

    def edges(self):
        if self._edges is None:
            v = self.verts.reshape(-1, 3, 3)
            self._edges = (v[:, 0], v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        return self._edges


class Drawable:
    """A Geometry placed in the scene by a 4x4 matrix: moving an object only changes the matrix."""

    def __init__(self, geom, color, alpha=1.0, pick=None, matrix=None):
        self.geom = geom if isinstance(geom, Geometry) else Geometry(geom)
        self.matrix = np.eye(4) if matrix is None else np.asarray(matrix, float)
        self.color = color
        self.alpha = alpha
        self.pick = pick                     # scene row this drawable stands for (None = not selectable)

    def ray_hit(self, o, d):
        """Distance along the world ray (unit `d`) to the first triangle hit, or None. Moller-Trumbore, vectorised."""
        g = self.geom
        inv = np.linalg.inv(self.matrix)
        o, d = inv[:3, :3] @ o + inv[:3, 3], inv[:3, :3] @ d    # ray in local space; t keeps its world meaning
        with np.errstate(divide="ignore", invalid="ignore"):    # slab test on the bounding box first
            t1, t2 = (g.lo - o) / d, (g.hi - o) / d
        if np.nanmin(np.maximum(t1, t2)) < max(np.nanmax(np.minimum(t1, t2)), 0):
            return None
        v0, e1, e2 = g.edges()
        o, d = o.astype(np.float32), d.astype(np.float32)
        pv = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) > 1e-12
        inv_det = 1.0 / np.where(ok, det, 1.0)
        tv = o - v0
        u = np.einsum("ij,ij->i", tv, pv) * inv_det
        qv = np.cross(tv, e1)
        w = (qv @ d) * inv_det
        t = np.einsum("ij,ij->i", e2, qv) * inv_det
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 0)
        return float(t[hit].min()) if hit.any() else None


class Viewer(QOpenGLWidget):
    context_menu = Signal(int, object)      # right click without dragging: scene row under the cursor (-1 = empty), global position
    paint_ray = Signal(object, object, bool)     # in paint_mode: world ray under the cursor while the left button is down, erase (Ctrl)
    ray_clicked = Signal(object, object, bool)   # in pick_rays mode: world ray (origin, direction) of a click, additive
    delete_pressed = Signal()
    clicked = Signal(int, bool)             # scene row under the cursor (-1 = empty space), additive (Ctrl/Shift)
    gizmo_begin = Signal(str)               # a handle drag starts (mode)
    gizmo_delta = Signal(str, int, float)   # mode, axis, mm (move) / degrees (rotate) / factor (scale)
    gizmo_end = Signal()
    mode_changed = Signal(str)
    screenshot_requested = Signal()
    box_selected = Signal(object, bool)     # middle-button rectangle: set of scene rows inside it, additive (Ctrl/Shift)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)                 # the cut face (cap) is drawn with the stencil buffer
        self.setFormat(fmt)
        self.bed = (277.76, 156.24, 300.0)
        self.drawables: list[Drawable] = []
        self._trash: list[int] = []
        self.target = np.array([0.0, 0.0, 30.0])
        self.dist, self.yaw, self.pitch = 420.0, -60.0, 28.0
        self.slice_z: float | None = None
        self.clip_z: float | None = None   # everything above this height is cut away and the section is shown
        self.cut_plane = None            # (point, normal, size)
        self.mode = "transform"
        self.gizmo = None                # (origin, frame 3x3 whose columns are the axes used in scale mode)
        self._hover = None
        self._drag = None
        self._last = self._press = None
        self._moved = False
        self._labels = []
        self._box = None                            # (start, current) widget positions of the rubber-band rectangle
        self._build_mode_bar()
        self.paint_mode = False                     # left drag on the model paints; off the model (or Alt+drag) it orbits
        self.paint_probe = None                     # callable(origin, direction) -> True if the ray hits the model being painted
        self._painting = False
        self.pick_rays = False                      # clicks are reported as rays (support editing) instead of picking objects

    def _build_mode_bar(self):
        """Selection / move-rotate / scale tools as icons floating over the top-left corner of the view."""
        self.mode_btns = {}
        grp = QButtonGroup(self)
        for i, (mode, name, key) in enumerate((("select", "Seleziona", "Q"), ("transform", "Sposta / Ruota", "W"),
                                               ("scale", "Scala", "R"))):
            b = QToolButton(self)
            b.setIcon(_mode_icon(mode))
            b.setIconSize(QSize(34, 34))
            b.setFixedSize(44, 44)
            b.setCheckable(True)
            b.setFocusPolicy(Qt.NoFocus)
            b.setToolTip(f"{name} ({key})")
            b.setStyleSheet("QToolButton{background:rgba(24,27,33,200);border:1px solid #3a3f47;border-radius:6px}"
                            "QToolButton:hover{background:rgba(60,66,78,220)}"
                            "QToolButton:checked{background:#2563eb;border-color:#60a5fa}")
            b.move(10, 10 + i * 50)
            grp.addButton(b)
            b.clicked.connect(lambda _=False, m=mode: self.set_mode(m))
            self.mode_btns[mode] = b
        shot = QToolButton(self)                      # not a mode: one click saves an image of the build plate
        shot.setIcon(_mode_icon("camera"))
        shot.setIconSize(QSize(34, 34))
        shot.setFixedSize(44, 44)
        shot.setFocusPolicy(Qt.NoFocus)
        shot.setToolTip("Screenshot del piatto di stampa (salva PNG e copia negli appunti)")
        shot.setStyleSheet("QToolButton{background:rgba(24,27,33,200);border:1px solid #3a3f47;border-radius:6px}"
                           "QToolButton:hover{background:rgba(60,66,78,220)}"
                           "QToolButton:pressed{background:#2563eb;border-color:#60a5fa}")
        shot.move(10, 10 + 3 * 50 + 8)
        shot.clicked.connect(self.screenshot_requested)
        self.shot_btn = shot
        self.mode_btns[self.mode].setChecked(True)
        self.mode_changed.connect(lambda m: self.mode_btns[m].setChecked(True))

    def set_bed(self, bed):
        self.bed = bed
        self.update()

    def discard(self, geom):
        """Forget a Geometry: its GPU buffers are deleted on the next paint (needs the GL context)."""
        if geom.buffers:
            self._trash.extend(geom.buffers)
            geom.buffers = None

    def set_drawables(self, items):
        self.drawables = items
        self.update()

    def set_slice_z(self, z):
        self.slice_z = z
        self.update()

    def set_clip_z(self, z):
        self.clip_z = z
        self.update()

    def set_cut_plane(self, spec):
        self.cut_plane = spec
        self.update()

    def set_gizmo(self, origin, frame=None):
        self.gizmo = None if origin is None else (np.asarray(origin, float),
                                                  np.eye(3) if frame is None else np.asarray(frame, float))
        self.update()

    def set_mode(self, mode):
        if mode in MODES and mode != self.mode:
            self.mode = mode
            self._hover = None
            self.update()
            self.mode_changed.emit(mode)

    def view_fit(self):
        self.target = np.array([0.0, 0.0, 30.0])
        self.dist, self.yaw, self.pitch = max(self.bed[:2]) * 1.5, -60.0, 28.0
        self.update()

    # ---- GL ----
    def initializeGL(self):
        gl.glClearColor(0.11, 0.12, 0.14, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, 1)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.85, 0.85, 0.85, 1])
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, [0.30, 0.32, 0.38, 1])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.25, 0.25, 0.28, 1])

    def _eye(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return self.target + self.dist * np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])

    def _matrices(self):
        w, h = max(self.width(), 1), max(self.height(), 1)
        P = _perspective(40, w / h, self.dist * 0.02, self.dist * 10).astype(np.float64)
        V = _look_at(self._eye(), self.target, np.array([0.0, 0.0, 1.0])).astype(np.float64)
        return P, V

    def paintGL(self):
        w, h = max(self.width(), 1), max(self.height(), 1)
        gl.glViewport(0, 0, int(w * self.devicePixelRatio()), int(h * self.devicePixelRatio()))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT | gl.GL_STENCIL_BUFFER_BIT)
        P, V = self._matrices()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixf(P.astype(np.float32).T)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        eye = self._eye()
        gl.glLoadMatrixf(V.astype(np.float32).T)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [*(eye - self.target), 0])
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, [-eye[0], -eye[1], 200, 0])
        self._draw_bed()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
        if self._trash:
            gl.glDeleteBuffers(len(self._trash), self._trash)
            self._trash = []
        clip = self.clip_z is not None
        if clip:
            gl.glClipPlane(gl.GL_CLIP_PLANE0, [0.0, 0.0, -1.0, float(self.clip_z)])   # keeps z <= clip_z
            gl.glEnable(gl.GL_CLIP_PLANE0)
        self._draw_all()
        if clip:
            self._draw_cap(float(self.clip_z), eye)
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        if self.slice_z is not None:
            self._draw_plane(self.slice_z)
        if self.cut_plane is not None:
            self._draw_cut_plane(*self.cut_plane)
        if self.gizmo is not None and self.mode != "select":
            self._draw_gizmo()
        if self._box is not None:
            self._draw_box()
        if self._labels and self.gizmo is not None and self.mode != "select":
            self._draw_labels()

    _GLYPHS = {"X": (((-1, -1), (1, 1)), ((-1, 1), (1, -1))),
               "Y": (((-1, 1), (0, 0)), ((1, 1), (0, 0)), ((0, 0), (0, -1))),
               "Z": (((-1, 1), (1, 1)), ((1, 1), (-1, -1)), ((-1, -1), (1, -1)))}

    def _draw_labels(self):
        """Axis letters drawn with plain GL lines (a QPainter inside paintGL would leave the GL state altered):
        white inside the square heads (scale), coloured next to the arrows (move)."""
        P, V = self._matrices()
        M = P @ V
        w, h = max(self.width(), 1), max(self.height(), 1)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, w, 0, h, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(2.0)
        k = 5.0
        for pos, letter, inside, i, on in self._labels:
            q = M @ np.array([*pos, 1.0])
            if q[3] <= 1e-6:
                continue
            x, y = (q[0] / q[3] + 1) * w / 2, (q[1] / q[3] + 1) * h / 2
            if inside:
                gl.glColor3f(1, 1, 1)
            else:
                gl.glColor3f(*((1.0, 0.9, 0.3) if on else AXIS_COLORS[i]))
            gl.glBegin(gl.GL_LINES)
            for (x0, y0), (x1, y1) in self._GLYPHS[letter]:
                gl.glVertex2f(x + k * x0, y + k * y0)
                gl.glVertex2f(x + k * x1, y + k * y1)
            gl.glEnd()
        gl.glLineWidth(1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def _draw_box(self):
        w, h = max(self.width(), 1), max(self.height(), 1)
        (x0, y0), (x1, y1) = ((p.x(), p.y()) for p in self._box)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, w, h, 0, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glColor4f(0.35, 0.65, 1.0, 0.18)
        gl.glBegin(gl.GL_QUADS)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            gl.glVertex2f(x, y)
        gl.glEnd()
        gl.glColor4f(0.35, 0.65, 1.0, 0.95)
        gl.glBegin(gl.GL_LINE_LOOP)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            gl.glVertex2f(x, y)
        gl.glEnd()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)

    def rows_in_rect(self, a, b):
        """Scene rows whose projected bounding box overlaps the screen rectangle a-b."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        P, V = self._matrices()
        M = P @ V
        lo = (min(a.x(), b.x()), min(a.y(), b.y()))
        hi = (max(a.x(), b.x()), max(a.y(), b.y()))
        rows = set()
        for d in self.drawables:
            if d.pick is None:
                continue
            g = d.geom
            c = np.array([[x, y, z, 1.0] for x in (g.lo[0], g.hi[0]) for y in (g.lo[1], g.hi[1])
                          for z in (g.lo[2], g.hi[2])])
            q = (M @ d.matrix @ c.T).T
            if (q[:, 3] <= 1e-6).any():                      # corner behind the camera: skip the unreliable projection
                continue
            sx = (q[:, 0] / q[:, 3] + 1) * w / 2
            sy = (1 - q[:, 1] / q[:, 3]) * h / 2
            if sx.max() >= lo[0] and sx.min() <= hi[0] and sy.max() >= lo[1] and sy.min() <= hi[1]:
                rows.add(d.pick)
        return rows

    def _draw_all(self, colored=True):
        for d in self.drawables:
            if colored:
                gl.glColor4f(*d.color, d.alpha)
            gl.glPushMatrix()
            gl.glMultMatrixf(d.matrix.T.astype(np.float32))
            d.geom.draw()
            gl.glPopMatrix()

    def _draw_cap(self, z, eye):
        """Fill the cut face of the (clipped) solids: per pixel, the front/back faces behind the plane are counted in the
        stencil buffer; a non-zero count means the pixel is inside material (a hollow's cavity counts zero). Only valid
        from above the plane, where nothing in front of the cut is left to count."""
        if gl.glGetIntegerv(gl.GL_STENCIL_BITS) > 0 and eye[2] > z:
            gl.glEnable(gl.GL_STENCIL_TEST)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
            gl.glStencilFunc(gl.GL_ALWAYS, 0, 0xFF)
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_FRONT)                                   # back faces: +1
            gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_INCR_WRAP)
            self._draw_all(False)
            gl.glCullFace(gl.GL_BACK)                                    # front faces: -1
            gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_DECR_WRAP)
            self._draw_all(False)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glStencilFunc(gl.GL_NOTEQUAL, 0, 0xFF)
            gl.glStencilOp(gl.GL_KEEP, gl.GL_KEEP, gl.GL_KEEP)
            gl.glDisable(gl.GL_CLIP_PLANE0)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
            gl.glDisable(gl.GL_LIGHTING)
            gl.glColor4f(*CAP_COLOR, 1.0)
            hx, hy = self.bed[0], self.bed[1]
            gl.glBegin(gl.GL_QUADS)
            for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
                gl.glVertex3f(x, y, z)
            gl.glEnd()
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glDisable(gl.GL_STENCIL_TEST)
        gl.glDisable(gl.GL_CLIP_PLANE0)

    def _draw_bed(self):
        bx, by, bz = self.bed
        hx, hy = bx / 2, by / 2
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor4f(0.18, 0.20, 0.24, 1)
        gl.glBegin(gl.GL_QUADS)
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            gl.glVertex3f(x, y, -0.05)
        gl.glEnd()
        gl.glColor4f(0.30, 0.33, 0.38, 1)
        gl.glBegin(gl.GL_LINES)
        step = 10
        for x in np.arange(-hx, hx + 0.01, step):
            gl.glVertex3f(x, -hy, 0); gl.glVertex3f(x, hy, 0)
        for y in np.arange(-hy, hy + 0.01, step):
            gl.glVertex3f(-hx, y, 0); gl.glVertex3f(hx, y, 0)
        gl.glEnd()
        gl.glColor4f(0.35, 0.55, 0.95, 1)
        gl.glBegin(gl.GL_LINES)
        for a, b in (((-hx, -hy), (hx, -hy)), ((hx, -hy), (hx, hy)), ((hx, hy), (-hx, hy)), ((-hx, hy), (-hx, -hy))):
            gl.glVertex3f(*a, 0); gl.glVertex3f(*b, 0)
            gl.glVertex3f(*a, bz); gl.glVertex3f(*b, bz)
            gl.glVertex3f(*a, 0); gl.glVertex3f(*a, bz)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

    def _draw_cut_plane(self, point, normal, size):
        n = np.asarray(normal, float)
        helper = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        S = size * 0.75 + 5
        p = np.asarray(point, float)
        quad = [p - u * S - v * S, p + u * S - v * S, p + u * S + v * S, p - u * S + v * S]
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glColor4f(0.95, 0.25, 0.25, 0.22)
        gl.glBegin(gl.GL_QUADS)
        for q in quad:
            gl.glVertex3f(*q)
        gl.glEnd()
        gl.glDepthMask(gl.GL_TRUE)
        gl.glColor4f(0.95, 0.25, 0.25, 0.9)
        gl.glBegin(gl.GL_LINE_LOOP)
        for q in quad:
            gl.glVertex3f(*q)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

    def _draw_plane(self, z):
        hx, hy = self.bed[0] / 2, self.bed[1] / 2
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glColor4f(0.3, 0.8, 0.5, 0.25)
        gl.glBegin(gl.GL_QUADS)
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            gl.glVertex3f(x, y, z)
        gl.glEnd()
        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_LIGHTING)

    def _draw_gizmo(self):
        origin, S = self.gizmo[0], self._gsize()
        kinds = ("move", "rotate") if self.mode == "transform" else ("scale",)
        active = (self._drag["mode"], self._drag["axis"]) if self._drag else self._hover
        self._labels = []                                # (world point, letter, inside the cube head?) drawn after the GL pass
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        for kind in kinds:
            axes = self._axes(kind)
            for i in range(3):
                n = axes[:, i]
                on = active == (kind, i)
                gl.glColor4f(*((1.0, 0.9, 0.3) if on else AXIS_COLORS[i]), 1)
                gl.glLineWidth(4 if on else 3)
                u = _perp(n)
                v = np.cross(n, u)
                if kind == "rotate":                     # quarter arc joining the arrows of the other two axes
                    gl.glBegin(gl.GL_LINE_STRIP)
                    for q in _arc(axes, i, 24):
                        gl.glVertex3f(*(origin + S * RING * q))
                    gl.glEnd()
                    continue
                tip = origin + n * S
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(*origin)
                gl.glVertex3f(*tip)
                gl.glEnd()
                r = S * (0.05 if kind == "move" else 0.075)
                self._labels.append((tip if kind == "scale" else tip + n * S * 0.14, "XYZ"[i], kind == "scale", i, on))
                if kind == "move":                       # cone head
                    base = origin + n * S * 0.80
                    gl.glBegin(gl.GL_TRIANGLE_FAN)
                    gl.glVertex3f(*(tip + n * S * 0.04))
                    for a in np.linspace(0, 2 * math.pi, 17):
                        gl.glVertex3f(*(base + r * (u * math.cos(a) + v * math.sin(a))))
                    gl.glEnd()
                else:                                    # cube head
                    gl.glBegin(gl.GL_QUADS)
                    for sgn in (-1, 1):
                        for ax, p, q in ((n, u, v), (u, v, n), (v, n, u)):
                            for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                                gl.glVertex3f(*(tip + sgn * r * ax + r * cu * p + r * cv * q))
                    gl.glEnd()
        gl.glLineWidth(1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)

    # ---- rays, picking, gizmo hit-testing ----
    def _ray(self, pos):
        """World-space ray (origin, unit direction) through a widget position."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        P, V = self._matrices()
        inv = np.linalg.inv(P @ V)
        nx, ny = 2 * pos.x() / w - 1, 1 - 2 * pos.y() / h
        a, b = inv @ [nx, ny, -1, 1], inv @ [nx, ny, 1, 1]
        a, b = a[:3] / a[3], b[:3] / b[3]
        d = b - a
        return a, d / np.linalg.norm(d)

    def pick(self, pos):
        """Scene row of the nearest selectable object under `pos`, or -1."""
        o, d = self._ray(pos)
        best, row = None, -1
        for dr in self.drawables:
            if dr.pick is None:
                continue
            t = dr.ray_hit(o, d)
            if t is not None and (best is None or t < best):
                best, row = t, dr.pick
        return row

    def _gsize(self):
        return self.dist * 0.16

    def _axes(self, kind):
        return self.gizmo[1] if kind == "scale" else np.eye(3)

    def _hit_handle(self, o, d):
        """(kind, axis) of the handle the ray passes closest to, or None."""
        origin, S = self.gizmo[0], self._gsize()
        tol, best = S * 0.09, None
        for kind in (("move", "rotate") if self.mode == "transform" else ("scale",)):
            axes = self._axes(kind)
            for i in range(3):
                n = axes[:, i]
                if kind == "rotate":                     # distance from the ray to the sampled ring
                    q = origin + S * RING * _arc(axes, i, 48) - o
                    s_all = q @ d
                    perp = np.where(s_all > 0, np.linalg.norm(q - s_all[:, None] * d, axis=1), np.inf)
                    k = int(perp.argmin())
                    dist, s = perp[k], s_all[k]
                else:
                    dist, s = _ray_segment(o, d, origin, origin + n * S)
                if s > 0 and dist < tol and (best is None or dist < best[0]):
                    best = (dist, kind, i)
        return None if best is None else best[1:]

    def _drag_value(self, o, d):
        """Active drag scalar: axis parameter (move/scale) or angle in degrees (rotate)."""
        g = self._drag
        n = g["axes"][:, g["axis"]]
        if g["mode"] == "rotate":
            den = d @ n
            if abs(den) < 1e-6:
                return None
            p = o + (((g["origin"] - o) @ n) / den) * d - g["origin"]
            return math.degrees(math.atan2(p @ g["v"], p @ g["u"]))
        return _axis_line_param(o, d, g["origin"], n)

    # ---- interaction ----
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            return self.delete_pressed.emit()
        key = {Qt.Key_Q: "select", Qt.Key_W: "transform", Qt.Key_E: "transform", Qt.Key_R: "scale"}.get(e.key())
        if key:
            self.set_mode(key)
        else:
            super().keyPressEvent(e)

    def mousePressEvent(self, e):
        self.setFocus()
        self._last = self._press = e.position()
        self._moved = False
        if e.button() == Qt.MiddleButton:
            self._box = (e.position(), e.position())
            return
        if e.button() == Qt.LeftButton and self.paint_mode and not (e.modifiers() & Qt.AltModifier):
            ray = self._ray(e.position())
            if self.paint_probe is None or self.paint_probe(*ray):
                self._painting = True
                self.paint_ray.emit(*ray, bool(e.modifiers() & Qt.ControlModifier))
                return
        if e.button() == Qt.LeftButton and self.gizmo is not None and self.mode != "select":
            o, d = self._ray(e.position())
            hit = self._hit_handle(o, d)
            if hit is not None:
                kind, axis = hit
                axes = self._axes(kind)
                u = _perp(axes[:, axis])
                self._drag = dict(mode=kind, axis=axis, origin=self.gizmo[0].copy(), axes=axes, u=u,
                                  v=np.cross(axes[:, axis], u))
                self._drag["last"] = self._drag_value(o, d)
                self.gizmo_begin.emit(kind)
                self.update()

    def mouseMoveEvent(self, e):
        if self._painting:
            self.paint_ray.emit(*self._ray(e.position()), bool(e.modifiers() & Qt.ControlModifier))
            return
        g = self._drag
        if g is not None:
            val = self._drag_value(*self._ray(e.position()))
            if val is None:
                return
            if g["last"] is not None:
                S = self._gsize() * 0.05
                if g["mode"] == "move":
                    self.gizmo_delta.emit("move", g["axis"], float(val - g["last"]))
                elif g["mode"] == "rotate":
                    self.gizmo_delta.emit("rotate", g["axis"], float((val - g["last"] + 180) % 360 - 180))
                elif abs(val) > S and abs(g["last"]) > S and val * g["last"] > 0:
                    self.gizmo_delta.emit("scale", g["axis"], float(np.clip(val / g["last"], 0.5, 2.0)))
                else:
                    return
            g["last"] = val
            return
        if self._last is None:                                  # hover: highlight the handle under the cursor
            if self.gizmo is not None and self.mode != "select":
                h = self._hit_handle(*self._ray(e.position()))
                if h != self._hover:
                    self._hover = h
                    self.update()
            return
        if (e.position() - self._press).manhattanLength() > 4:
            self._moved = True
        dx, dy = e.position().x() - self._last.x(), e.position().y() - self._last.y()
        self._last = e.position()
        if self._box is not None:
            self._box = (self._box[0], e.position())
        elif e.buttons() & Qt.LeftButton:
            self.yaw -= dx * 0.4
            self.pitch = max(-89, min(89, self.pitch + dy * 0.4))
        elif e.buttons() & Qt.RightButton:
            y = math.radians(self.yaw)
            right = np.array([-math.sin(y), math.cos(y), 0.0])
            k = self.dist * 0.0012
            self.target = self.target - right * dx * k + np.array([0, 0, 1.0]) * dy * k
        self.update()

    def mouseReleaseEvent(self, e):
        if self._painting and e.button() == Qt.LeftButton:
            self._painting = False
            self._last = self._press = None
            return
        if e.button() == Qt.MiddleButton and self._box is not None:
            a, b = self._box
            self._box = None
            if self._moved and not self.pick_rays:
                self.box_selected.emit(self.rows_in_rect(a, b), bool(e.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)))
            self._last = self._press = None
            self.update()
            return
        if self._drag is not None:
            self._drag = None
            self.gizmo_end.emit()
            self.update()
        elif e.button() == Qt.LeftButton and self._press is not None and not self._moved and self.pick_rays:
            self.ray_clicked.emit(*self._ray(e.position()), bool(e.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)))
        elif e.button() == Qt.LeftButton and self._press is not None and not self._moved and not self.paint_mode:
            self.clicked.emit(self.pick(e.position()), bool(e.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)))
        elif e.button() == Qt.RightButton and self._press is not None and not self._moved:
            self.context_menu.emit(self.pick(e.position()), e.globalPosition().toPoint())
        self._last = self._press = None

    def wheelEvent(self, e):
        new = max(20.0, min(3000.0, self.dist * (0.9 if e.angleDelta().y() > 0 else 1.1)))
        f = new / self.dist
        # zoom towards the cursor: the point under it, on the plane through the target, stays under it
        o, d = self._ray(e.position())
        fwd = self.target - self._eye()
        fwd = fwd / np.linalg.norm(fwd)
        den = float(d @ fwd)
        if abs(den) > 1e-6:
            p = o + d * (float((self.target - o) @ fwd) / den)
            self.target = p + (self.target - p) * f
        self.dist = new
        self.update()
