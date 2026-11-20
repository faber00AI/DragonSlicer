"""Taglio vero di un modello gia' diviso in parti (etichette per faccia): lisciatura/spianatura delle cuciture,
chiusura di ogni pezzo con un coperchio, e maschera "pennello" che calcola la curva di taglio.

Tutto lavora su mesh con vertici saldati (trimesh.Trimesh con merge_vertices) in coordinate mondo (mm).
"""
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


# ---------------------------------------------------------------------------
# pulizia etichette
# ---------------------------------------------------------------------------
def clean_labels(mesh, lab, min_faces=40):
    """Elimina le isole: ogni etichetta resta connessa; i frammenti piccoli passano al vicino piu' lungo."""
    lab = lab.copy()
    adj = mesh.face_adjacency
    n = len(lab)
    for _ in range(3):
        same = lab[adj[:, 0]] == lab[adj[:, 1]]
        g = coo_matrix((np.ones(same.sum()), (adj[same, 0], adj[same, 1])), shape=(n, n))
        nc, cc = connected_components(g, directed=False)
        size = np.bincount(cc, minlength=nc)
        owner = np.zeros(nc, int); owner[cc] = lab
        main = {}
        for c in np.argsort(-size):                        # componente piu' grande di ogni etichetta
            main.setdefault(int(owner[c]), int(c))
        small = np.array([size[c] < min_faces or main[int(owner[c])] != c for c in range(nc)])
        # un'etichetta con una sola componente grande resta; i frammenti (non principali) vanno riassorbiti
        frag = small[cc]
        if not frag.any():
            break
        # riassegna per maggioranza dei vicini non-frammento (iterando)
        for _ in range(50):
            fa, fb = frag[adj[:, 0]], frag[adj[:, 1]]
            m1 = fa & ~fb
            m2 = fb & ~fa
            if not (m1.any() or m2.any()):
                break
            tgt = np.r_[adj[m1, 0], adj[m2, 1]]
            val = np.r_[lab[adj[m1, 1]], lab[adj[m2, 0]]]
            k = lab.max() + 1
            votes = np.zeros((n, 1))
            cnt = {}
            for t, v in zip(tgt, val):
                cnt.setdefault(int(t), {}).setdefault(int(v), 0)
                cnt[int(t)][int(v)] += 1
            for t, d in cnt.items():
                lab[t] = max(d, key=d.get)
                frag[t] = False
    _, inv = np.unique(lab, return_inverse=True)
    return inv


# ---------------------------------------------------------------------------
# cuciture: lisciatura della curva e spianatura
# ---------------------------------------------------------------------------
def _seam_edges(mesh, lab):
    adj = mesh.face_adjacency
    cut = lab[adj[:, 0]] != lab[adj[:, 1]]
    e = mesh.face_adjacency_edges[cut]
    pa = lab[adj[cut, 0]]
    pb = lab[adj[cut, 1]]
    return e, np.minimum(pa, pb), np.maximum(pa, pb)


def flatten_seams(mesh, lab, flat=0.7, smooth=8):
    """Restituisce nuove posizioni dei vertici: la curva di ogni cucitura viene lisciata (`smooth` passate lungo la curva) e
    spinta verso il suo piano medio (`flat` 0..1: 0 = segue la superficie, 1 = piano). Le due facce del taglio condividono
    i vertici, quindi i pezzi combaciano esattamente."""
    V = np.array(mesh.vertices, float)
    e, pa, pb = _seam_edges(mesh, lab)
    if not len(e):
        return V
    nV = len(V)
    # 1) lisciatura della curva (media con i due vicini lungo la cucitura)
    if smooth > 0:
        A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nV, nV)).tocsr()
        A.data[:] = 1.0
        deg = np.asarray(A.sum(axis=1)).ravel()
        on = deg > 0
        for _ in range(smooth):
            avg = (A @ V) / np.maximum(deg, 1)[:, None]
            V[on] = 0.5 * V[on] + 0.5 * avg[on]
    # 2) spianatura: piano ai minimi quadrati di ogni coppia di parti
    if flat > 0:
        disp = np.zeros_like(V)
        cnt = np.zeros(nV)
        key = pa.astype(np.int64) * (lab.max() + 1) + pb
        for kk in np.unique(key):
            sel = e[key == kk]
            vi = np.unique(sel)
            if len(vi) < 3:                                  # cucitura minuscola: niente piano da calcolare
                continue
            P = V[vi]
            c = P.mean(axis=0)
            u, s, vt = np.linalg.svd(P - c, full_matrices=False)
            nrm = vt[2]
            d = (P - c) @ nrm
            disp[vi] += -flat * d[:, None] * nrm
            cnt[vi] += 1
        m = cnt > 0
        V[m] += disp[m] / cnt[m][:, None]
    return V


# ---------------------------------------------------------------------------
# chiusura e separazione dei pezzi
# ---------------------------------------------------------------------------
def _loops(faces):
    """Anelli di bordo di una mesh aperta, come liste di indici di vertice, nell'ordine di percorrenza delle facce."""
    f = np.asarray(faces)
    a = np.r_[f[:, 0], f[:, 1], f[:, 2]]
    b = np.r_[f[:, 1], f[:, 2], f[:, 0]]
    V = int(f.max()) + 1
    key = a.astype(np.int64) * V + b
    rev = b.astype(np.int64) * V + a
    bd = ~np.isin(rev, key)
    a, b = a[bd], b[bd]
    nxt = {}
    for x, y in zip(a.tolist(), b.tolist()):
        nxt.setdefault(x, []).append(y)
    loops = []
    used = set()
    for x0 in list(nxt):
        for y0 in list(nxt[x0]):
            if (x0, y0) in used:
                continue
            loop = [x0]
            cur, nx = x0, y0
            used.add((x0, y0))
            guard = 0
            while nx != x0 and guard < 10_000_000:
                loop.append(nx)
                cand = [z for z in nxt.get(nx, []) if (nx, z) not in used]
                if not cand:
                    break
                used.add((nx, cand[0]))
                cur, nx = nx, cand[0]
                guard += 1
            if nx == x0 and len(loop) >= 3:
                loops.append(loop)
    return loops


def _cap_loop(V, loop):
    """Triangoli che chiudono l'anello (earcut sul piano medio, ripiego: ventaglio dal baricentro).
    Ritorna (vertici extra Nx3, triangoli come indici in [vertici esistenti ... extra]) con orientamento opposto all'anello."""
    P = V[loop]
    c = P.mean(axis=0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    ax, ay = vt[0], vt[1]
    xy = np.c_[(P - c) @ ax, (P - c) @ ay]
    tris = None
    try:
        import mapbox_earcut as ec
        idx = ec.triangulate_float64(xy, np.array([len(xy)], np.uint32))
        t = np.asarray(idx, int).reshape(-1, 3)
        # area triangolata vs area poligono: se non coincidono l'anello si autointerseca in proiezione
        poly = 0.5 * abs(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
        tarea = 0.5 * np.abs(np.cross(xy[t[:, 1]] - xy[t[:, 0]], xy[t[:, 2]] - xy[t[:, 0]])).sum()
        if len(t) and abs(tarea - poly) < 1e-6 * max(poly, 1e-9) + 1e-9:
            tris = np.asarray(loop)[t]
    except Exception:
        tris = None
    extra = np.zeros((0, 3))
    if tris is None:                                       # ventaglio dal baricentro (nuovo vertice)
        n = len(loop)
        k = len(V)
        extra = c[None, :]
        L = np.asarray(loop)
        tris = np.c_[L, np.roll(L, -1), np.full(n, k)]
    # orientamento: ogni lato (x->y) dell'anello deve comparire come (y->x) nel coperchio
    x0, y0 = loop[0], loop[1]
    fl = False
    for tri in tris:
        for i in range(3):
            if tri[i] == x0 and tri[(i + 1) % 3] == y0:
                fl = True
            if tri[i] == y0 and tri[(i + 1) % 3] == x0:
                fl = False
    if fl:
        tris = tris[:, ::-1]
    return extra, tris


def close_piece(sub):
    """Chiude con coperchi i fori di una mesh aperta. Ritorna (mesh chiusa, info coperchio principale)."""
    V = np.asarray(sub.vertices, float)
    F = np.asarray(sub.faces)
    loops = _loops(F)
    verts = [V]
    faces = [F]
    n = len(V)
    caps = []
    for lp in loops:
        extra, tris = _cap_loop(np.vstack(verts), lp)
        if len(extra):
            verts.append(extra)
            tris = tris.copy()
        faces.append(tris)
        n += len(extra)
        a = 0.5 * np.linalg.norm(np.cross(np.vstack(verts)[tris[:, 1]] - np.vstack(verts)[tris[:, 0]],
                                          np.vstack(verts)[tris[:, 2]] - np.vstack(verts)[tris[:, 0]]), axis=1).sum()
        caps.append((a, tris))
    out = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
    info = None
    if caps:
        a, tris = max(caps, key=lambda x: x[0])
        vv = np.asarray(out.vertices)
        nn = np.cross(vv[tris[:, 1]] - vv[tris[:, 0]], vv[tris[:, 2]] - vv[tris[:, 0]])
        nv = nn.sum(axis=0)
        info = dict(area=float(a), normal=(nv / (np.linalg.norm(nv) + 1e-12)).tolist(),
                    center=vv[np.unique(tris)].mean(axis=0).tolist())
    return out, info


def open_edge_count(m):
    """Lati di bordo (usati una volta sola): 0 = nessun foro, anche se la mesh d'origine ha lati non-manifold."""
    f = np.sort(np.asarray(m.edges), axis=1)
    k = f[:, 0].astype(np.int64) * len(m.vertices) + f[:, 1]
    _, cnt = np.unique(k, return_counts=True)
    return int((cnt == 1).sum())


def cut_pieces(mesh, lab, flat=0.7, smooth=8, return_base=False):
    """Taglia `mesh` secondo le etichette: ritorna lista di dict(mesh=pezzo chiuso, label=int, cap=info|None, watertight=bool).
    Un modello con una sola etichetta non viene tagliato (ritorna se stesso)."""
    mesh = mesh.copy()
    mesh.merge_vertices()
    lab = np.asarray(lab)
    if len(np.unique(lab)) < 2:
        return [dict(mesh=mesh, label=int(lab[0]) if len(lab) else 0, cap=None, watertight=open_edge_count(mesh) == 0)]
    V = flatten_seams(mesh, lab, flat, smooth)
    base = trimesh.Trimesh(V, np.asarray(mesh.faces), process=False)
    out = []
    for p in np.unique(lab):
        f = np.asarray(base.faces)[lab == p]
        sub = trimesh.Trimesh(np.asarray(base.vertices), f, process=False)
        sub.remove_unreferenced_vertices()
        closed, cap = close_piece(sub)
        closed.merge_vertices()
        out.append(dict(mesh=closed, label=int(p), cap=cap, watertight=open_edge_count(closed) == 0))
    return (out, base) if return_base else out


def lay_on_cut(piece):
    """Matrice 4x4 che appoggia il pezzo sul coperchio piu' grande (normale verso il basso) e lo porta a quota 0."""
    m = piece["mesh"]
    cap = piece.get("cap")
    M = np.eye(4)
    if cap is not None:
        M = trimesh.geometry.align_vectors(np.asarray(cap["normal"]), [0, 0, -1])
    v = np.asarray(m.vertices) @ M[:3, :3].T
    M[2, 3] = -v[:, 2].min()
    return M


# ---------------------------------------------------------------------------
# maschera: la zona dipinta diventa una curva di taglio
# ---------------------------------------------------------------------------
def apply_mask(mesh, lab, painted, min_frac=0.01, grow_max=12):
    """Taglia seguendo la maschera. Per ogni parte attraversata da una fascia dipinta che la separa in due o piu' regioni
    significative, le facce dipinte vengono distribuite alla regione piu' vicina (geodetica): la curva di taglio cade nel
    mezzo della fascia e ne segue la forma. Ritorna (nuove etichette, [parti tagliate], [parti dove la fascia non chiude il giro])."""
    lab = lab.copy()
    painted = np.asarray(painted, bool)
    if not painted.any():
        return lab, [], []
    adj = mesh.face_adjacency
    c = mesh.triangles_center
    area = mesh.area_faces
    total = area.sum()
    d = np.linalg.norm(c[adj[:, 0]] - c[adj[:, 1]], axis=1)
    n = len(lab)
    cut, open_ = [], []
    nxt = int(lab.max()) + 1
    A = coo_matrix((np.ones(2 * len(adj)), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n)).tocsr()
    for p in np.unique(lab[painted]):
        F = lab == p
        sig = []
        if (F & painted).sum() < 12:                   # qualche faccia dipinta per sbaglio: ignorata
            continue
        # se la fascia ha buchi (zone nascoste, sottosquadri) la ispessisce finche' separa la parte, al minimo necessario
        band0 = F & painted
        for grow in range(0, grow_max + 1):
            band = band0
            for _ in range(grow):
                band = band | (F & ((A @ band.astype(np.float32)) > 0))
            rest = F & ~band
            if not rest.any():
                break
            ok = rest[adj[:, 0]] & rest[adj[:, 1]]
            g = coo_matrix((np.ones(ok.sum()), (adj[ok, 0], adj[ok, 1])), shape=(n, n))
            nc, cc = connected_components(g, directed=False)
            ids = np.unique(cc[rest])
            asz = np.array([area[rest & (cc == i)].sum() for i in ids])
            sig = ids[asz >= min_frac * total]
            if len(sig) >= 2:
                break
        if len(sig) < 2:
            open_.append(int(p))
            continue
        # distanza geodetica dentro la parte, da ogni regione significativa
        inside = F[adj[:, 0]] & F[adj[:, 1]]
        w = np.where(inside, d, 0)
        G = coo_matrix((np.r_[w[inside], w[inside]], (np.r_[adj[inside, 0], adj[inside, 1]], np.r_[adj[inside, 1], adj[inside, 0]])),
                       shape=(n, n)).tocsr()
        srcs = np.flatnonzero(rest & np.isin(cc, sig))
        dist, pred, src = dijkstra(G, indices=srcs, return_predecessors=True, min_only=True)
        owner = np.full(n, -1)
        reg_of_face = np.where(rest & np.isin(cc, sig), cc, -1)
        valid = src >= 0
        owner[valid] = reg_of_face[src[valid]]
        asg = np.full(n, -1)
        for k, r in enumerate(sig):
            asg[(owner == r) & F] = (p if k == 0 else nxt)
            if k > 0:
                nxt += 1
        m = F & (asg >= 0)
        lab[m] = asg[m]
        cut.append(int(p))
    _, inv = np.unique(lab, return_inverse=True)
    return inv, cut, open_
