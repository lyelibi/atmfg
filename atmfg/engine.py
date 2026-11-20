"""
Numba engine: face pool, lazy max-heap and the local expansion loop.

A face is a set of s = clique_size - 1 nodes (s = 3 for the TMFG: triangles).
Inserting a node v into a face creates the clique {v} + face and s new faces,
each obtained by replacing one node of the parent face with v. Each face is
used once.

State layout (U = universe size, K = max candidates per face, s = face size)
---------------------------------------------------------------------------
    faces        int32   (U, s)   sorted node IDs of the face in each slot
    scores_pool  float32 (U*K,)   candidate scores, face i at [i*K : i*K + counts[i]]
    nbs_pool     int32   (U*K,)   candidate node IDs, same layout
    counts       int32   (U,)     number of stored candidates per face
    active       uint8   (U,)     1 if the slot holds a live face
    birth        int64   (U,)     creation clock of the face in the slot: pruning order,
                                  and the generation tag of heap entries
    free_stack   int32   (U,)     LIFO stack of free slots
    is_dead      uint8   (N,)     1 once a node is in the graph

Heap (max-heap on score; ties: smaller slot, then smaller node ID):
    hk float32 score, hf int32 slot, hn int32 node, hb int64 birth of the face when pushed.
An entry is stale if its slot is inactive or now holds a younger face (hb != birth).

Scalars live in the int64 array `meta` (indices below).
"""
import numpy as np
from numba import njit

STACK, ACTIVE, CLOCK, HSIZE, NEDGE, NADDED, NCLIQUE = 0, 1, 2, 3, 4, 5, 6
META_SIZE = 7

SCORE_SUM, SCORE_CENTROID = 0, 1
STATUS_DONE, STATUS_RESCUE, STATUS_GROW, STATUS_CHUNK = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Heap
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _before(hk, hf, hn, i, j):
    """True if entry i must pop before entry j."""
    if hk[i] != hk[j]:
        return hk[i] > hk[j]
    if hf[i] != hf[j]:
        return hf[i] < hf[j]
    return hn[i] < hn[j]


@njit(cache=True, inline="always")
def _swap(hk, hf, hn, hb, i, j):
    hk[i], hk[j] = hk[j], hk[i]
    hf[i], hf[j] = hf[j], hf[i]
    hn[i], hn[j] = hn[j], hn[i]
    hb[i], hb[j] = hb[j], hb[i]


@njit(cache=True)
def heap_push(hk, hf, hn, hb, meta, score, face, node, birth):
    i = meta[HSIZE]
    hk[i] = score; hf[i] = face; hn[i] = node; hb[i] = birth
    meta[HSIZE] = i + 1
    while i > 0:
        p = (i - 1) >> 1
        if _before(hk, hf, hn, i, p):
            _swap(hk, hf, hn, hb, i, p)
            i = p
        else:
            break


@njit(cache=True)
def heap_pop(hk, hf, hn, hb, meta):
    score, face, node, birth = hk[0], hf[0], hn[0], hb[0]
    size = meta[HSIZE] - 1
    meta[HSIZE] = size
    hk[0] = hk[size]; hf[0] = hf[size]; hn[0] = hn[size]; hb[0] = hb[size]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        m = i
        if l < size and _before(hk, hf, hn, l, m):
            m = l
        if r < size and _before(hk, hf, hn, r, m):
            m = r
        if m == i:
            break
        _swap(hk, hf, hn, hb, i, m)
        i = m
    return score, face, node, birth


# ---------------------------------------------------------------------------
# Face pool
# ---------------------------------------------------------------------------

@njit(cache=True)
def delete_face(idx, active, counts, free_stack, meta):
    if active[idx] == 0:
        return
    active[idx] = 0
    counts[idx] = 0
    if meta[STACK] < active.shape[0]:
        free_stack[meta[STACK]] = idx
        meta[STACK] += 1
    meta[ACTIVE] -= 1


@njit(cache=True)
def prune_universe(active, counts, birth, free_stack, meta):
    """Deletes the oldest half of the active faces (by birth time)."""
    U = active.shape[0]
    if meta[ACTIVE] < U:
        return
    idxs = np.empty(U, dtype=np.int64)
    ages = np.empty(U, dtype=np.int64)
    n = 0
    for i in range(U):
        if active[i] == 1:
            idxs[n] = i
            ages[n] = birth[i]
            n += 1
    if n == 0:
        return
    num_to_delete = n - n // 2
    order = np.argsort(ages[:n])          # birth times are unique
    for t in range(num_to_delete):
        delete_face(idxs[order[t]], active, counts, free_stack, meta)


@njit(cache=True)
def force_delete_oldest(active, counts, birth, free_stack, meta):
    oldest, min_age = -1, np.iinfo(np.int64).max
    for i in range(active.shape[0]):
        if active[i] == 1 and birth[i] < min_age:
            min_age = birth[i]
            oldest = i
    if oldest != -1:
        delete_face(oldest, active, counts, free_stack, meta)


@njit(cache=True)
def refill_face(f, scores_pool, nbs_pool, counts, active, birth, is_dead,
                hk, hf, hn, hb, meta):
    """Pushes the face's best ALIVE candidate (if any)."""
    if active[f] == 0:
        return
    K = scores_pool.shape[0] // active.shape[0]
    off = f * K
    best_s, best_n = np.float32(-1.0), -1
    for i in range(counts[f]):
        nb = nbs_pool[off + i]
        if is_dead[nb] == 0 and scores_pool[off + i] > best_s:
            best_s = scores_pool[off + i]
            best_n = nb
    if best_n != -1:
        heap_push(hk, hf, hn, hb, meta, best_s, f, best_n, birth[f])


@njit(cache=True)
def add_face(nodes, scores, nbs, faces, scores_pool, nbs_pool, counts, active, birth,
             free_stack, is_dead, hk, hf, hn, hb, meta):
    """Allocates a slot (pruning if the universe is full), stores the candidates
    and pushes the face's best candidate. Returns the slot."""
    U = active.shape[0]
    K = scores_pool.shape[0] // U
    if meta[ACTIVE] >= U:
        prune_universe(active, counts, birth, free_stack, meta)
        if meta[ACTIVE] >= U:
            force_delete_oldest(active, counts, birth, free_stack, meta)
    meta[STACK] -= 1
    idx = free_stack[meta[STACK]]
    k = min(len(nbs), K)
    off = idx * K
    for i in range(k):
        scores_pool[off + i] = scores[i]
        nbs_pool[off + i] = nbs[i]
    counts[idx] = k
    active[idx] = 1
    birth[idx] = meta[CLOCK]
    meta[CLOCK] += 1
    meta[ACTIVE] += 1
    for j in range(faces.shape[1]):
        faces[idx, j] = nodes[j]
    best_s, best_n = np.float32(-1.0), -1
    for i in range(k):
        if scores_pool[off + i] > best_s:
            best_s = scores_pool[off + i]
            best_n = nbs_pool[off + i]
    if best_n != -1 and is_dead[best_n] == 0:
        heap_push(hk, hf, hn, hb, meta, best_s, idx, best_n, birth[idx])
    return idx


@njit(cache=True)
def alive_candidates(f, nbs_pool, counts, active, is_dead):
    if active[f] == 0 or counts[f] == 0:
        return np.empty(0, dtype=np.int32)
    K = nbs_pool.shape[0] // active.shape[0]
    off = f * K
    out = np.empty(counts[f], dtype=np.int32)
    n = 0
    for i in range(counts[f]):
        nb = nbs_pool[off + i]
        if is_dead[nb] == 0:
            out[n] = nb
            n += 1
    return out[:n]


@njit(cache=True)
def pop_best_edge(scores_pool, nbs_pool, counts, active, birth, is_dead,
                  hk, hf, hn, hb, meta):
    """Lazy deletion: skips stale entries (deleted face, or an older face that used
    the same slot) and refills the face when its node is already taken."""
    while meta[HSIZE] > 0:
        s, f, v, b = heap_pop(hk, hf, hn, hb, meta)
        if active[f] == 0 or b != birth[f]:
            continue
        if is_dead[v] == 1:
            refill_face(f, scores_pool, nbs_pool, counts, active, birth, is_dead,
                        hk, hf, hn, hb, meta)
            continue
        return f, v
    return -1, -1


# ---------------------------------------------------------------------------
# Scoring and candidates
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def face_vector(nodes, X, mode):
    """mode SCORE_SUM: sum of the face's rows.  SCORE_CENTROID: standardized mean."""
    D = X.shape[1]
    s = len(nodes)
    v = np.zeros(D, dtype=np.float32)
    for j in range(s):
        node = nodes[j]
        for f in range(D):
            v[f] += X[node, f]
    if mode == SCORE_CENTROID:
        for f in range(D):
            v[f] /= s
        mean = 0.0
        for f in range(D):
            mean += v[f]
        mean /= D
        var = 0.0
        for f in range(D):
            d = v[f] - mean
            var += d * d
        var /= D
        std = np.sqrt(var) + 1e-12
        for f in range(D):
            v[f] = (v[f] - mean) / std
    return v


@njit(cache=True)
def face_vectors(slots, faces, X, mode):
    out = np.empty((len(slots), X.shape[1]), dtype=np.float32)
    for i in range(len(slots)):
        out[i] = face_vector(faces[slots[i]], X, mode)
    return out


@njit(cache=True, fastmath=True)
def score_candidates(fv, cands, X, shift):
    """f . x_v / D + shift. Sum mode: sum of the s correlations; centroid mode:
    correlation with the centroid. shift = s keeps every score above -1, the
    "no candidate" sentinel."""
    D = X.shape[1]
    out = np.empty(len(cands), dtype=np.float32)
    for i in range(len(cands)):
        t = cands[i]
        dot = np.float32(0.0)
        for f in range(D):
            dot += fv[f] * X[t, f]
        out[i] = dot / D + shift
    return out


@njit(cache=True)
def valid_union(nodes, indptr, indices, is_dead):
    """Sorted unique alive kNN neighbours of the face's nodes."""
    size = 0
    for j in range(len(nodes)):
        size += indptr[nodes[j] + 1] - indptr[nodes[j]]
    buf = np.empty(size, dtype=np.int32)
    n = 0
    for j in range(len(nodes)):
        node = nodes[j]
        for t in range(indptr[node], indptr[node + 1]):
            nb = indices[t]
            if is_dead[nb] == 0:
                buf[n] = nb
                n += 1
    if n == 0:
        return np.empty(0, dtype=np.int32)
    srt = np.sort(buf[:n])
    out = np.empty(n, dtype=np.int32)
    out[0] = srt[0]
    u = 1
    for i in range(1, n):
        if srt[i] != srt[i - 1]:
            out[u] = srt[i]
            u += 1
    return out[:u]


@njit(cache=True)
def keep_top(scores, cands, K):
    """The K best candidates by score (stable: ties keep node-ID order)."""
    if len(cands) <= K:
        return scores, cands
    order = np.argsort(-scores, kind="mergesort")[:K]
    return scores[order], cands[order]


@njit(cache=True)
def children_of(face, v):
    """The s child faces of inserting v into `face` (sorted). Child r drops node
    s-1-r; for triangles this gives (a,b,v), (a,c,v), (b,c,v)."""
    s = len(face)
    out = np.empty((s, s), dtype=np.int32)
    for r in range(s):
        drop = s - 1 - r
        c = 0
        placed = False
        for j in range(s):
            if j == drop:
                continue
            if not placed and v < face[j]:
                out[r, c] = v
                c += 1
                placed = True
            out[r, c] = face[j]
            c += 1
        if not placed:
            out[r, c] = v
    return out


@njit(cache=True)
def add_children(parent, children, X, indptr, indices, mode, shift, faces, scores_pool,
                 nbs_pool, counts, active, birth, free_stack, is_dead, hk, hf, hn, hb, meta):
    """Registers new faces. The parent's alive candidates are read and the parent is
    deleted BEFORE the children are allocated; a child with no kNN candidates
    inherits them."""
    K = scores_pool.shape[0] // active.shape[0]
    parent_nbs = np.empty(0, dtype=np.int32)
    if parent >= 0:
        parent_nbs = alive_candidates(parent, nbs_pool, counts, active, is_dead)
        delete_face(parent, active, counts, free_stack, meta)
    for r in range(children.shape[0]):
        nodes = children[r]
        fv = face_vector(nodes, X, mode)
        cands = valid_union(nodes, indptr, indices, is_dead)
        if len(cands) == 0 and len(parent_nbs) > 0:
            cands = parent_nbs
        if len(cands) > 0:
            sc = score_candidates(fv, cands, X, shift)
            sc, cands = keep_top(sc, cands, K)
        else:
            sc = np.empty(0, dtype=np.float32)
        add_face(nodes, sc, cands, faces, scores_pool, nbs_pool, counts, active, birth,
                 free_stack, is_dead, hk, hf, hn, hb, meta)


# ---------------------------------------------------------------------------
# Local expansion and rescue injection
# ---------------------------------------------------------------------------

@njit(cache=True)
def expand(X, indptr, indices, mode, shift, max_steps, faces, scores_pool, nbs_pool,
           counts, active, birth, free_stack, is_dead, hk, hf, hn, hb, meta,
           edges, added, cliques, separators):
    """
    Runs local expansion until the graph is complete (STATUS_DONE), the heap is
    empty (STATUS_RESCUE), the heap needs more room (STATUS_GROW) or max_steps
    insertions were made (STATUS_CHUNK). Returns (status, insertions made).
    """
    s = faces.shape[1]
    target = edges.shape[0]
    steps = 0
    face = np.empty(s, dtype=np.int32)
    while meta[NEDGE] < target:
        if steps >= max_steps:
            return STATUS_CHUNK, steps
        if meta[HSIZE] + s + 8 > hk.shape[0]:
            return STATUS_GROW, steps
        f, v = pop_best_edge(scores_pool, nbs_pool, counts, active, birth, is_dead,
                             hk, hf, hn, hb, meta)
        if f == -1:
            return STATUS_RESCUE, steps
        is_dead[v] = 1
        added[meta[NADDED]] = v
        meta[NADDED] += 1
        for j in range(s):
            face[j] = faces[f, j]
        e = meta[NEDGE]
        q = meta[NCLIQUE]
        cliques[q, 0] = v
        for j in range(s):
            edges[e + j, 0] = v
            edges[e + j, 1] = face[j]
            cliques[q, j + 1] = face[j]
            separators[q - 1, j] = face[j]
        meta[NEDGE] = e + s
        meta[NCLIQUE] = q + 1
        add_children(f, children_of(face, v), X, indptr, indices, mode, shift, faces,
                     scores_pool, nbs_pool, counts, active, birth, free_stack, is_dead,
                     hk, hf, hn, hb, meta)
        steps += 1
    return STATUS_DONE, steps


@njit(cache=True)
def rescue_reset(slots, cand_scores, cand_nodes, scores_pool, nbs_pool, counts, active,
                 birth, is_dead, hk, hf, hn, hb, meta):
    """Global rescue: replaces each active face's candidates in place (slot and birth
    time unchanged) and pushes its best alive candidate."""
    K = scores_pool.shape[0] // active.shape[0]
    for r in range(len(slots)):
        f = slots[r]
        if active[f] == 0:
            continue
        k = min(cand_nodes.shape[1], K)
        off = f * K
        for i in range(k):
            scores_pool[off + i] = cand_scores[r, i]
            nbs_pool[off + i] = cand_nodes[r, i]
        counts[f] = k
        refill_face(f, scores_pool, nbs_pool, counts, active, birth, is_dead,
                    hk, hf, hn, hb, meta)


@njit(cache=True, fastmath=True)
def edge_weights(edges, X):
    """Correlation of each edge's endpoints: x_u . x_v / D (rows standardized)."""
    D = X.shape[1]
    out = np.empty(edges.shape[0], dtype=np.float32)
    for e in range(edges.shape[0]):
        u, v = edges[e, 0], edges[e, 1]
        dot = np.float32(0.0)
        for f in range(D):
            dot += X[u, f] * X[v, f]
        out[e] = dot / D
    return out
