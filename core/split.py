"""Segmentazione di un modello in parti (prototipo, ispirato a Cleave).

Idea: seed = punti piu' lontani (geodetica) sul grafo duale delle facce;
crescita multi-sorgente (Dijkstra) con costo alto sugli spigoli concavi,
cosi' i confini tra le parti cadono nelle valli/pieghe del modello.
"""
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components


def decimate(mesh, target=30000):
    mesh = mesh.copy()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)   # STL/GLB: vertici duplicati, serve la connettivita'
    if len(mesh.faces) <= target:
        return mesh
    import fast_simplification
    v, f = fast_simplification.simplify(
        np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32),
        target_count=target)
    return trimesh.Trimesh(v, f, process=False)


def dual_graph(mesh, concave_penalty=8.0, angle_gain=2.0, axis_gain=0.0, axis=(0, 0, 1)):
    """Grafo duale pesato. Spigoli concavi => costo alto (barriere)."""
    adj = mesh.face_adjacency                       # (E,2)
    ang = mesh.face_adjacency_angles                # 0 = piano
    convex = mesh.face_adjacency_convex
    c = mesh.triangles_center
    d = np.linalg.norm(c[adj[:, 0]] - c[adj[:, 1]], axis=1)
    dv = c[adj[:, 0]] - c[adj[:, 1]]
    along = np.abs(dv @ np.asarray(axis, float)) / np.maximum(d, 1e-12)   # 1 = spigolo parallelo all'asse
    d = d * (1.0 + axis_gain * along)
    w = d * (1.0 + angle_gain * ang)
    w = np.where(convex, w, d * (1.0 + concave_penalty * (0.3 + ang)))
    n = len(mesh.faces)
    g = coo_matrix((np.r_[w, w], (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n)).tocsr()
    return g


def _geo(g, i):
    d = dijkstra(g, indices=i)
    d[~np.isfinite(d)] = 0
    return d


def pick_seeds(g, k):
    """Farthest point sampling sul grafo (distanze geodetiche), sulla componente principale."""
    ncomp, cc = connected_components(g, directed=False)
    main = np.argmax(np.bincount(cc))
    start = int(np.flatnonzero(cc == main)[0])
    first = int(np.argmax(_geo(g, start)))
    seeds = [first]
    dist = _geo(g, first)
    while len(seeds) < k:
        s = int(np.argmax(dist))
        seeds.append(s)
        dist = np.minimum(dist, _geo(g, s))
    return seeds


def segment(mesh, k=8, concave_penalty=8.0, seeds=None, axis_gain=0.0):
    from scipy.spatial import cKDTree
    g = dual_graph(mesh, concave_penalty, axis_gain=axis_gain)
    if seeds is None:
        seeds = pick_seeds(g, k)
    ncomp, cc = connected_components(g, directed=False)
    sizes = np.bincount(cc)
    covered = set(cc[seeds])
    for comp in np.argsort(-sizes):                 # componenti staccate ma grandi: un seed ciascuna
        if comp not in covered and sizes[comp] > 0.003 * len(cc):
            seeds = list(seeds) + [int(np.flatnonzero(cc == comp)[0])]
    dist, _, src = dijkstra(g, indices=seeds, return_predecessors=True, min_only=True)
    lab = np.full(len(mesh.faces), -1, int)
    for i, s in enumerate(seeds):
        lab[src == s] = i
    bad = lab < 0                                   # frammenti minuscoli irraggiungibili: label del vicino piu' prossimo
    if bad.any():
        c = mesh.triangles_center
        _, j = cKDTree(c[~bad]).query(c[bad])
        lab[bad] = lab[~bad][j]
    return lab, seeds


# ---------------------------------------------------------------------------
# Segmentazione per "colli": curve di livello geodetiche dalla base, albero dei
# componenti, taglio dove la sezione si stringa (collo, polsi, vita, base cappello...)
# ---------------------------------------------------------------------------
def segment_necks(mesh, k=8, bins=160, root=None, smooth=1, min_frac=0.03, axis=(0, 0, 1)):
    n = len(mesh.faces)
    adj = mesh.face_adjacency
    c = mesh.triangles_center
    area = mesh.area_faces
    d = np.linalg.norm(c[adj[:, 0]] - c[adj[:, 1]], axis=1)
    g = coo_matrix((np.r_[d, d], (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n)).tocsr()
    ncomp, cc = connected_components(g, directed=False)
    main = np.argmax(np.bincount(cc))
    ax = np.asarray(axis, float); ax /= np.linalg.norm(ax)
    f = c @ ax
    f = f - f.min()
    h = f.max() / bins
    b = np.minimum(np.floor(f / h).astype(int), bins - 1)

    same = (b[adj[:, 0]] == b[adj[:, 1]]) & (b[adj[:, 0]] >= 0)
    gs = coo_matrix((np.ones(same.sum()), (adj[same, 0], adj[same, 1])), shape=(n, n))
    _, node = connected_components(gs, directed=False)  # (banda, componente) per ogni faccia
    nn = node.max() + 1
    nb = np.zeros(nn, int); nb[node] = b
    narea = np.bincount(node, weights=area, minlength=nn)
    valid = np.zeros(nn, bool); valid[node[b >= 0]] = True
    w = narea / h                                    # ~ perimetro della sezione

    # genitore = nodo adiacente nella banda sotto, con area maggiore
    na, nb2 = node[adj[:, 0]], node[adj[:, 1]]
    diff = na != nb2
    sw = nb[na] > nb[nb2]
    lo = np.where(sw, nb2, na)[diff]
    hi = np.where(sw, na, nb2)[diff]
    ok2 = valid[lo] & valid[hi]
    lo, hi = lo[ok2], hi[ok2]
    parent = np.full(nn, -1)
    best = np.full(nn, -1.0)
    for l, hh in zip(lo, hi):
        if narea[l] > best[hh]:
            best[hh] = narea[l]; parent[hh] = l
    order = np.argsort(nb, kind='stable')
    order = order[valid[order]]
    # larghezza lisciata sul percorso (media con genitore)
    ws = w.copy()
    for _ in range(smooth):
        t = ws.copy()
        for x in order:
            p = parent[x]
            if p >= 0:
                t[x] = (ws[x] * 2 + ws[p]) / 3
        ws = t
    up = np.zeros(nn)
    for x in order:
        p = parent[x]
        up[x] = ws[p] if p < 0 else max(up[p], ws[p])
    down = ws.copy()
    for x in order[::-1]:
        p = parent[x]
        if p >= 0:
            down[p] = max(down[p], down[x])
    sub = narea.copy()
    for x in order[::-1]:
        if parent[x] >= 0:
            sub[parent[x]] += sub[x]
    total = narea[valid].sum()
    kids = [[] for _ in range(nn)]
    for x in order:
        if parent[x] >= 0:
            kids[parent[x]].append(x)
    cands = []
    for x in order:
        p = parent[x]
        if p < 0 or not kids[x]:
            continue                                  # niente tagli sulle punte
        if sub[x] < min_frac * total or total - sub[x] < min_frac * total:
            continue                                  # pezzi troppo piccoli
        if ws[x] <= ws[p] and all(ws[x] <= ws[ch] for ch in kids[x]):
            cands.append((ws[x] / max(min(up[x], down[x]), 1e-9), x))
    cands.sort()
    # non-max suppression: un taglio per volta, non vicino ad uno gia' scelto sullo stesso ramo
    chosen = []
    anc = lambda x: _ancestors(parent, x)
    for r, x in cands:
        if len(chosen) >= k - 1:
            break
        ax = set(anc(x))
        if any(abs(nb[x] - nb[y]) < 4 and (y in ax or x in set(anc(y))) for _, y in chosen):
            continue
        chosen.append((r, x))
    cutset = {x for _, x in chosen}
    lab_node = np.full(nn, -1)
    nxt = 1
    cutid = {}
    for x in order:
        p = parent[x]
        if x in cutset:
            lab_node[x] = nxt; nxt += 1
        elif p >= 0:
            lab_node[x] = lab_node[p]
        else:
            lab_node[x] = 0
    lab = np.where(b >= 0, lab_node[node], -1)
    bad = lab < 0
    if bad.any():
        from scipy.spatial import cKDTree
        _, j = cKDTree(c[~bad]).query(c[bad]); lab[bad] = lab[~bad][j]
    return lab, [x for _, x in chosen]


def _ancestors(parent, x):
    out = []
    while parent[x] >= 0:
        x = parent[x]; out.append(x)
    return out


def auto_up(mesh, min_frac=0.02):
    """Direzione 'su' del modello: se c'e' una grande faccia piana (basetta) il su e' opposto alla sua normale
    (la basetta guarda in basso); altrimenti Z. Restituisce una matrice 4x4 che porta il su in +Z."""
    try:
        fa, fn = mesh.facets_area, mesh.facets_normal
        if len(fa):
            i = int(np.argmax(fa))
            if fa[i] > min_frac * mesh.area:
                n = fn[i]
                p = mesh.vertices @ n
                c = mesh.triangles_center[mesh.facets[i][0]] @ n
                up = n if c < p.mean() else -n      # la faccia e' sotto il resto lungo n => n punta in su
                return trimesh.geometry.align_vectors(up, [0, 0, 1])
    except Exception:
        pass
    # nessuna basetta: asse piu' lungo della bounding box; la base e' l'estremo con la sezione piu' larga
    ext = mesh.bounds[1] - mesh.bounds[0]
    ax = int(np.argmax(ext))
    v = mesh.vertices
    t = (v[:, ax] - mesh.bounds[0][ax]) / max(ext[ax], 1e-12)
    others = [i for i in range(3) if i != ax]
    def span(sel):
        if sel.sum() < 3:
            return 0.0
        e = v[sel][:, others]
        return float(np.prod(e.max(axis=0) - e.min(axis=0)))
    low, high = span(t < 0.1), span(t > 0.9)
    up = np.zeros(3); up[ax] = 1.0 if low >= high else -1.0
    return trimesh.geometry.align_vectors(up, [0, 0, 1])


# ---------------------------------------------------------------------------
# Lisciatura dei confini e gestione delle parti
# ---------------------------------------------------------------------------
def smooth_labels(mesh, lab, iters=25, crease_keep=0.5, protect=None):
    """Lisciatura dei confini tra le parti (flusso di curvatura sul campo delle etichette).

    Diffonde le etichette one-hot sul grafo delle facce e prende l'argmax: i confini frastagliati
    si accorciano e diventano curve morbide. Gli spigoli concavi (pieghe) diffondono meno
    (`crease_keep` = peso relativo), cosi' il confine non si stacca dalle pieghe naturali.
    `protect`: indici di facce che non possono cambiare etichetta (es. i seed).
    Ritorna le nuove etichette (le parti non spariscono: ogni parte tiene almeno il suo seed).
    """
    n = len(mesh.faces)
    adj = mesh.face_adjacency
    convex = mesh.face_adjacency_convex
    w = np.where(convex, 1.0, crease_keep)
    A = coo_matrix((np.r_[w, w], (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel() + 1.0
    k = lab.max() + 1
    P = np.zeros((n, k)); P[np.arange(n), lab] = 1.0
    P0 = P.copy()
    for _ in range(iters):
        P = (A @ P + P) / deg[:, None]
        if protect is not None:
            P[protect] = P0[protect]
    out = P.argmax(axis=1)
    return out


def boundary_length(mesh, lab):
    """Lunghezza totale dei confini tra le parti (per misurare la lisciatura)."""
    adj = mesh.face_adjacency
    cut = lab[adj[:, 0]] != lab[adj[:, 1]]
    return float(mesh.face_adjacency_edges_len[cut].sum()) if hasattr(mesh, 'face_adjacency_edges_len') else float(
        np.linalg.norm(mesh.vertices[mesh.face_adjacency_edges[cut][:, 0]] - mesh.vertices[mesh.face_adjacency_edges[cut][:, 1]], axis=1).sum())


def part_adjacency(mesh, lab):
    """{(a,b): lunghezza del confine condiviso} per a<b."""
    adj = mesh.face_adjacency
    e = mesh.face_adjacency_edges
    ln = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    a, b = lab[adj[:, 0]], lab[adj[:, 1]]
    m = a != b
    out = {}
    for x, y, l in zip(np.minimum(a[m], b[m]), np.maximum(a[m], b[m]), ln[m]):
        out[(int(x), int(y))] = out.get((int(x), int(y)), 0.0) + float(l)
    return out


def merge_parts(lab, a, b):
    """Unisce la parte b dentro a. Le etichette restano 0..k-1 compatte."""
    lab = lab.copy(); lab[lab == b] = a
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def strip_scores(mesh, lab, elong_min=1.6, bins=10, gap_deg=110.0):
    """Per ogni parte: punteggio 'striscia' 0..1 (0 se non e' allungata). Una striscia (pieghe di un cappotto) e' un pezzo di
    superficie che NON fa il giro di se stesso: in ogni sezione trasversale e' solo un arco. Una testa, un braccio, una gamba,
    un cappello fanno il giro. Punteggio = frazione di superficie (pesata per area) delle sezioni lungo l'asse principale in cui
    il buco angolare piu' grande attorno al centro della sezione supera `gap_deg`."""
    c = mesh.triangles_center
    area = mesh.area_faces
    out = {}
    for p in np.unique(lab):
        idx = np.flatnonzero(lab == p)
        out[int(p)] = 0.0
        if len(idx) < 60:
            continue
        mu, ax, ev = _part_axes(c[idx], area[idx])
        d = c[idx] - mu
        t = d @ ax[0]
        L = float(np.ptp(t))
        wd = max(float(np.sqrt(np.ptp(d @ ax[1]) * np.ptp(d @ ax[2]))), 1e-9)
        if L / wd < elong_min:
            continue
        b = np.clip(((t - t.min()) / max(L, 1e-12) * bins).astype(int), 0, bins - 1)
        op = tot = 0.0
        for k in range(bins):
            m = b == k
            if m.sum() < 12:
                continue
            xy = np.c_[d[m] @ ax[1], d[m] @ ax[2]]
            w = area[idx][m]
            xy = xy - (xy * w[:, None]).sum(0) / w.sum()
            ang = np.sort(np.arctan2(xy[:, 1], xy[:, 0]))
            gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
            tot += w.sum()
            if np.degrees(gaps.max()) > gap_deg:
                op += w.sum()
        out[int(p)] = op / tot if tot > 0 else 0.0
    return out


def merge_strips(mesh, lab, thr=0.6, protect=None, min_parts=2):
    """Riassorbe le strisce (vedi strip_scores) nel vicino con cui hanno il confine piu' lungo, la peggiore per prima,
    finche' nessuna supera `thr`. Il cappotto torna un pezzo solo e poi lo taglia un piano orizzontale."""
    lab = lab.copy()
    protect = set(int(p) for p in (protect or []))
    for _ in range(200):
        if len(np.unique(lab)) <= min_parts:
            break
        sc = {p: v for p, v in strip_scores(mesh, lab).items() if p not in protect}
        if not sc:
            break
        p = max(sc, key=sc.get)
        if sc[p] <= thr:
            break
        adj = part_adjacency(mesh, lab)
        nb = [(l, (a if b == p else b)) for (a, b), l in adj.items() if p in (a, b)]
        if not nb:
            break
        lab[lab == p] = max(nb)[1]
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def apply_selection(mesh, lab, cut_parts):
    """Tiene staccate solo le parti in `cut_parts` (insieme di indici); le altre vengono
    riassorbite nella parte confinante con cui condividono il confine piu' lungo.
    Ritorna (etichette, mappa vecchia->nuova etichetta)."""
    keep = set(int(p) for p in cut_parts)
    lab = lab.copy()
    guard = 0
    while guard < 1000:
        guard += 1
        drop = [p for p in np.unique(lab) if p not in keep]
        if not drop:
            break
        adj = part_adjacency(mesh, lab)
        merged = False
        for p in drop:
            nb = [(l, (a if b == p else b)) for (a, b), l in adj.items() if p in (a, b)]
            if nb:
                tgt = max(nb)[1]
                lab[lab == p] = tgt
                merged = True
                break
        if not merged:
            break
    return lab


# ---------------------------------------------------------------------------
# Taglio piano delle parti lunghe + pipeline completa
# ---------------------------------------------------------------------------
def _part_axes(c, a):
    """Assi principali (pesati per area) di un gruppo di facce: (centro, assi 3x3 per righe, autovalori decrescenti)."""
    w = a / a.sum()
    mu = (c * w[:, None]).sum(axis=0)
    d = c - mu
    cov = (d * w[:, None]).T @ d
    ev, evec = np.linalg.eigh(cov)
    o = np.argsort(-ev)
    return mu, evec[:, o].T, ev[o]


def split_long_parts(mesh, lab, elong=2.0, max_len=0.0, min_frac=0.03, zone=0.25, bins=48, max_depth=3, protect_ids=None):
    """Divide con un piano le parti lunghe e strette (gambe, braccia, gonna, cappotto) nel punto piu' stretto.

    elong      allungamento minimo (lunghezza / larghezza) per tagliare una parte
    max_len    se > 0, taglia anche le parti piu' lunghe di cosi' (mm), per farle stare nella stampante
    min_frac   un pezzo non puo' essere piu' piccolo di questa frazione dell'area totale
    zone       il taglio cade tra zone e 1-zone della lunghezza (lontano dalle estremita')
    Ritorna (nuove etichette, lista di piani [(punto, normale, etichetta_sopra, etichetta_sotto)]).
    """
    lab = lab.copy()
    c = mesh.triangles_center
    area = mesh.area_faces
    total = area.sum()
    planes = []
    queue = [(int(p), 0) for p in np.unique(lab)]
    nxt = int(lab.max()) + 1
    while queue:
        p, depth = queue.pop(0)
        idx = np.flatnonzero(lab == p)
        if len(idx) < 50 or area[idx].sum() < 2 * min_frac * total:
            continue
        mu, ax, ev = _part_axes(c[idx], area[idx])
        t = (c[idx] - mu) @ ax[0]
        L = float(np.ptp(t))
        wd = max(float(np.sqrt(np.ptp((c[idx] - mu) @ ax[1]) * np.ptp((c[idx] - mu) @ ax[2]))), 1e-9)
        too_long = max_len > 0 and L > max_len
        if not (L / wd >= elong or too_long) or depth >= max_depth:
            continue
        lo, hi = float(t.min()), float(t.max())
        edges = np.linspace(lo, hi, bins + 1)
        which = np.clip(np.digitize(t, edges) - 1, 0, bins - 1)
        prof = np.bincount(which, weights=area[idx], minlength=bins) / ((hi - lo) / bins)
        prof = np.convolve(np.pad(prof, 2, mode='edge'), np.ones(5) / 5, mode='valid')
        ctr = (edges[:-1] + edges[1:]) / 2
        frac = (ctr - lo) / (hi - lo)
        ok = (frac >= zone) & (frac <= 1 - zone)
        if too_long:                                       # entrambi i pezzi devono poter stare entro max_len, se possibile
            ok2 = ok & ((ctr - lo) <= max_len) & ((hi - ctr) <= max_len)
            if ok2.any():
                ok = ok2
        if not ok.any():
            continue
        score = np.where(ok, prof * (1 + 0.6 * np.abs(frac - 0.5) * 2), np.inf)
        cut = float(ctr[int(np.argmin(score))])
        up = idx[t > cut]; dn = idx[t <= cut]
        if area[up].sum() < min_frac * total or area[dn].sum() < min_frac * total:
            continue
        lab[up] = nxt
        planes.append((mu + ax[0] * cut, ax[0].copy(), nxt, p))
        queue += [(p, depth + 1), (nxt, depth + 1)]
        nxt += 1
    _, inv = np.unique(lab, return_inverse=True)
    remap = {int(o): int(n) for o, n in zip(np.unique(lab), range(len(np.unique(lab))))}
    planes = [(pt, nv, remap[a], remap[b]) for pt, nv, a, b in planes]
    return inv, planes


def auto_split(mesh, parts=8, penalty=8.0, smooth=400, elong=2.0, max_len=0.0, min_frac=0.03, target=30000, do_planes=True, strips=0.6):
    """Pipeline completa. `mesh` e' in coordinate mondo (mm, gia' orientata e a scala).
    Ritorna dict: labels (una per faccia della mesh data), small (mesh ridotta), small_labels, planes."""
    from scipy.spatial import cKDTree
    s = decimate(mesh, target)
    lab, seeds = segment(s, parts, penalty)
    if smooth > 0:
        lab = smooth_labels(s, lab, smooth, protect=np.array(seeds))
    if strips:
        lab = merge_strips(s, lab, strips)
    planes = []
    if do_planes:
        lab, planes = split_long_parts(s, lab, elong=elong, max_len=max_len, min_frac=min_frac)
    if s is mesh:
        full = lab
    else:
        _, j = cKDTree(s.triangles_center).query(mesh.triangles_center)
        full = lab[j]
    return dict(labels=full, small=s, small_labels=lab, planes=planes)
