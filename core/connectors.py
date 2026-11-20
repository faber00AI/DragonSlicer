"""Connettori tra i pezzi tagliati: perni+fori, tasche per magneti, spinotti separati.
Algoritmo proprio: per ogni coppia di pezzi adiacenti si prende l'anello di taglio piu' grande, lo si proietta sul suo piano
medio e si scelgono i punti piu' lontani dal bordo (e tra loro); in ognuno si scava/aggiunge un cilindro lungo la normale."""
import numpy as np
import trimesh

from . import splitcut


def seam_loops(base, lab):
    """Per ogni coppia (a, b) di etichette vicine: lista di anelli ordinati (array Nx3) lungo la cucitura."""
    e, pa, pb = splitcut._seam_edges(base, lab)
    V = np.asarray(base.vertices)
    out = {}
    for key in set(zip(pa.tolist(), pb.tolist())):
        sel = (pa == key[0]) & (pb == key[1])
        ed = e[sel]
        nxt = {}
        for x, y in ed.tolist():
            nxt.setdefault(x, []).append(y)
            nxt.setdefault(y, []).append(x)
        seen = set()
        loops = []
        for s in list(nxt):
            if s in seen:
                continue
            loop = [s]
            seen.add(s)
            cur, prev = s, None
            while True:
                c = [z for z in nxt[cur] if z != prev and z not in seen]
                if not c:
                    break
                prev, cur = cur, c[0]
                loop.append(cur)
                seen.add(cur)
            if len(loop) >= 6:
                loops.append(V[loop])
        if loops:
            out[key] = loops
    return out


def _inside(poly, pts):
    """Punti (N,2) dentro il poligono (M,2): test del raggio, senza matplotlib (non e' nell'exe)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cross & (x < xint)
        j = i
    return inside


def _plane(P):
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0], vt[1], vt[2]


def _poly_area(xy):
    return 0.5 * abs(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def _edge_dist(pts, poly):
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    L2 = (ab ** 2).sum(1) + 1e-12
    d = np.full(len(pts), np.inf)
    for i in range(len(a)):
        t = np.clip(((pts - a[i]) @ ab[i]) / L2[i], 0, 1)
        q = a[i] + t[:, None] * ab[i]
        d = np.minimum(d, np.linalg.norm(pts - q, axis=1))
    return d


def pick_points(loop3, diam, margin, max_n, spacing):
    """Punti (3D) dove mettere i connettori sull'anello + (centro, u, v, normale del piano)."""
    c, u, v, n = _plane(loop3)
    xy = np.c_[(loop3 - c) @ u, (loop3 - c) @ v]
    if len(xy) > 400:                                   # poligono piu' leggero per le distanze
        xy = xy[np.linspace(0, len(xy) - 1, 400).astype(int)]
    need = diam / 2 + margin
    lo, hi = xy.min(0), xy.max(0)
    step = max(diam * 0.4, (hi - lo).max() / 60)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + step, step), np.arange(lo[1], hi[1] + step, step))
    pts = np.c_[gx.ravel(), gy.ravel()]
    pts = pts[_inside(xy, pts)]
    if len(pts) == 0:
        return [], (c, u, v, n)
    d = _edge_dist(pts, xy)
    ok = d >= need
    pts, d = pts[ok], d[ok]
    if len(pts) == 0:
        return [], (c, u, v, n)
    chosen = [int(np.argmax(d))]
    while len(chosen) < max_n:
        dm = np.min(np.linalg.norm(pts[:, None, :] - pts[chosen][None, :, :], axis=2), axis=1)
        dm[dm < spacing] = -1
        j = int(np.argmax(dm))
        if dm[j] < 0:
            break
        chosen.append(j)
    P2 = pts[chosen]
    return [c + p[0] * u + p[1] * v for p in P2], (c, u, v, n)


def _cyl(p, n, r, a, b):
    """Cilindro di raggio r lungo la direzione n, dal punto p+n*a al punto p+n*b."""
    h = abs(b - a)
    cyl = trimesh.creation.cylinder(radius=r, height=h, sections=40)
    T = trimesh.geometry.align_vectors([0, 0, 1], n)
    cyl.apply_transform(T)
    cyl.apply_translation(p + n * (a + b) / 2)
    return cyl


def outer_loops(loops):
    """Toglie gli anelli interni (buchi della sezione): un anello il cui baricentro cade dentro un anello piu' grande."""
    if len(loops) < 2:
        return list(loops)
    info = []
    for L in loops:
        c, u, v, n = _plane(L)
        info.append((L, c, _poly_area(np.c_[(L - c) @ u, (L - c) @ v])))
    keep = []
    for i, (L, c, ar) in enumerate(info):
        inside = False
        for j, (M, cm, am) in enumerate(info):
            if j != i and am > ar:
                cc, u, v, n = _plane(M)
                if _inside(np.c_[(M - cc) @ u, (M - cc) @ v], [[(c - cc) @ u, (c - cc) @ v]])[0]:
                    inside = True
                    break
        if not inside:
            keep.append(L)
    return keep


def _thick(mesh, p, d, skip=1.0):
    """Spessore del pezzo dal punto p (sul taglio) verso l'interno, lungo la direzione d: distanza del primo bordo oltre `skip` mm
    (i colpi entro `skip` sono il coperchio stesso, che e' un po' curvo). inf se non trova nulla."""
    try:
        loc, _, _ = mesh.ray.intersects_location([p - d * 0.2], [d])
        if len(loc):
            t = (loc - p) @ d
            t = t[t > skip]
            if len(t):
                return float(t.min())
    except Exception:
        pass
    return np.inf


def _orient(A, B, p, n):
    """Sceglie il verso di n che esce da A ed entra in B (A e B sono i due pezzi del taglio nel punto p). None se nessuno dei due va bene."""
    best, score = None, 0
    for nn in (n, -n):
        s = int(bool(A.contains([p - nn * 1.0])[0])) + int(bool(B.contains([p + nn * 1.0])[0]))
        if s > score:
            best, score = nn, s
    return best


def add_connectors(pieces, seams, mode="peg", diam=4.0, length=6.0, clear=0.2, max_n=3, margin=2.0, dry=False):
    """Modifica i pezzi (lista di dict da cut_pieces) aggiungendo i connettori. Ritorna (pezzi, extra, rapporto) dove `extra` e'
    una lista di mesh in piu' (spinotti) e il rapporto e' un dict con conteggi. Ogni anello di ogni taglio riceve i suoi connettori."""
    by = {p["label"]: p for p in pieces}
    add_to = {k: [] for k in by}
    cut_from = {k: [] for k in by}
    extra, made, skipped, marks = [], 0, 0, []
    why = {}
    for (a, b), loops in seams.items():
        if a not in by or b not in by:
            continue
        A, B = by[a]["mesh"], by[b]["mesh"]
        for loop in sorted(loops, key=lambda L: -_poly_area(np.c_[(L - L.mean(0)) @ _plane(L)[1], (L - L.mean(0)) @ _plane(L)[2]])):
            pts, (c, u, v, n0) = pick_points(loop, diam + clear, margin, max_n, (diam + clear) * 2.4)
            if not pts:
                skipped += 1
                why.setdefault((a, b), 'zona piccola')
                continue
            r_h = (diam + clear) / 2
            for p in pts:
                n = _orient(A, B, np.asarray(p), n0)
                if n is None:
                    skipped += 1
                    why.setdefault((a, b), 'orientamento incerto')
                    continue
                ta = _thick(A, p, -n)
                tb = _thick(B, p, n)
                if mode == "peg":
                    hole = min(length + 0.5, 0.85 * tb)
                    emb = min(length, 0.8 * ta)
                    if hole < 0.6 * length or emb < 1.2:
                        why.setdefault((a, b), f'sottile ta={ta:.1f} tb={tb:.1f}')
                        skipped += 1
                        continue
                    add_to[a].append(_cyl(p, n, diam / 2, -emb, hole - 0.4))
                    cut_from[b].append(_cyl(p, n, r_h, -0.3, hole))
                else:                                       # magnete o spinotto: foro da entrambe le parti
                    da = min(length / 2 if mode == "dowel" else length, 0.85 * ta)
                    db = min(length / 2 if mode == "dowel" else length, 0.85 * tb)
                    if da < 0.5 * length / (2 if mode == "dowel" else 1) or db < 0.5 * length / (2 if mode == "dowel" else 1):
                        why.setdefault((a, b), f'sottile ta={ta:.1f} tb={tb:.1f}')
                        skipped += 1
                        continue
                    cut_from[a].append(_cyl(p, n, r_h, -da, 0.3))
                    cut_from[b].append(_cyl(p, n, r_h, -0.3, db))
                    if mode == "dowel":
                        extra.append(_cyl(np.zeros(3), np.array([1.0, 0, 0]), diam / 2, -(da + db - 0.6) / 2, (da + db - 0.6) / 2))
                made += 1
                marks.append((_cyl(p, n, diam / 2, -length / 2, length / 2), a))
    if dry:
        return pieces, extra, dict(made=made, skipped=skipped, failed=0, marks=marks, why=why)
    failed = 0
    for k, p in by.items():
        try:
            if add_to[k] or cut_from[k]:
                p["mesh"] = prepare(p["mesh"])
            if add_to[k]:
                m = trimesh.boolean.union([p["mesh"]] + add_to[k], engine="manifold")
                p["mesh"] = m
            if cut_from[k]:
                m = trimesh.boolean.difference([p["mesh"]] + cut_from[k], engine="manifold")
                p["mesh"] = m
            p["watertight"] = splitcut.open_edge_count(p["mesh"]) == 0
        except Exception as e:
            failed += 1
            p['error'] = repr(e)
    return pieces, extra, dict(made=made, skipped=skipped, failed=failed, why=why)


def _nonmanifold_faces(F):
    """Indici delle facce che toccano un lato usato da piu' di due facce."""
    e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)
    key = e[:, 0].astype(np.int64) * (int(F.max()) + 1) + e[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bad_e = cnt[inv.ravel()] > 2
    return np.flatnonzero(bad_e.reshape(3, -1).any(axis=0))


def preview(pieces, base, lab, mode, diam=4.0, length=6.0, clear=0.2, max_n=3, margin=2.0):
    """Segnaposto dei connettori con gli stessi controlli di add_connectors (spessore, zone piccole), ma senza booleane.
    Ritorna ([(cilindro, etichetta)], quanti, quanti saltati)."""
    seams = seam_loops(base, lab)
    _, _, rep = add_connectors(pieces, seams, mode, diam, length, clear, max_n, margin, dry=True)
    return rep["marks"], rep["made"], rep["skipped"]


def prepare(mesh):
    """Rende il pezzo adatto alle booleane: toglie facce doppie (le coppie opposte si annullano), le facce sui lati
    non-manifold ereditati dal modello d'origine, e richiude i buchi che ne risultano."""
    m = mesh.copy()
    F = np.asarray(m.faces)
    key = np.sort(F, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    keep = np.ones(len(F), bool)
    for g in np.where(cnt > 1)[0]:
        idx = np.where(inv == g)[0]
        # orientamento relativo: permutazione ciclica uguale = stessa direzione
        def canon(f):
            k = int(np.argmin(f))
            return tuple(np.roll(f, -k))
        dirs = {}
        for i in idx:
            dirs.setdefault(canon(F[i]), []).append(i)
        if len(dirs) == 1:
            keep[idx[1:]] = False                       # copie identiche: ne resta una
        else:
            keep[idx] = False                           # facce opposte: sliver interno, si elimina
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    for _ in range(3):                                  # lati non-manifold: via le facce coinvolte, poi si richiudono i buchi
        bad = _nonmanifold_faces(np.asarray(m.faces))
        if not len(bad):
            break
        k = np.ones(len(m.faces), bool)
        k[bad] = False
        m.update_faces(k)
        m.remove_unreferenced_vertices()
        m, _ = splitcut.close_piece(m)
        m.merge_vertices()
    if not m.is_volume:                                 # briciole (gusci minuscoli o rovesciati, residui dei buchi): via
        parts = m.split(only_watertight=False)
        good = [c for c in parts if c.is_volume and len(c.faces) >= 20]
        if good and len(good) < len(parts):
            m = trimesh.util.concatenate(good)
            m.merge_vertices()
    return m
