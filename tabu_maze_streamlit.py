# ============================================================
#  TABU SEARCH MAZE — Streamlit Version
#  Jalankan: streamlit run tabu_maze_streamlit.py
# ============================================================

import random
import gc
import io
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

plt.rcParams['figure.dpi'] = 80

st.set_page_config(
    page_title="Tabu Search Maze",
    page_icon="🧩",
    layout="wide",
)


# ── 1. Generate Maze (iteratif, hindari RecursionError) ───────
def generate_maze(n, seed=42):
    random.seed(seed)
    grid = np.ones((n, n), dtype=np.uint8)
    visited = set()

    stack = [(1, 1)]
    visited.add((1, 1))
    grid[1, 1] = 0

    while stack:
        r, c = stack[-1]
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        moved = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                visited.add((nr, nc))
                grid[r + dr // 2, c + dc // 2] = 0
                grid[nr, nc] = 0
                stack.append((nr, nc))
                moved = True
                break
        if not moved:
            stack.pop()

    grid[0, 1] = 0
    grid[1, 1] = 0
    grid[n - 1, n - 2] = 0
    grid[n - 2, n - 2] = 0
    return grid


# ── 2. Tabu Search ───────────────────────────────────────────
def tabu_search_maze(grid, tabu_size=50, max_iter=200_000):
    rows, cols = grid.shape
    start = (0, 1)
    end = (rows - 1, cols - 2)
    current = start
    path = [start]
    tabu_list = []
    tabu_set = set()
    snapshots = []

    def manhattan(p):
        return abs(p[0] - end[0]) + abs(p[1] - end[1])

    def add_tabu(node):
        if node in tabu_set:
            return
        tabu_list.append(node)
        tabu_set.add(node)
        if len(tabu_list) > tabu_size:
            old = tabu_list.pop(0)
            tabu_set.discard(old)

    for _ in range(max_iter):
        if current == end:
            snapshots.append({
                'current': current,
                'path': path[:],
                'tabu_list': tabu_list[:],
                'action': 'done'
            })
            return path, snapshots

        neighbors = [
            (current[0] + dr, current[1] + dc)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if 0 <= current[0] + dr < rows
            and 0 <= current[1] + dc < cols
            and grid[current[0] + dr, current[1] + dc] == 0
            and (current[0] + dr, current[1] + dc) not in tabu_set
        ]

        if not neighbors:
            add_tabu(current)
            if len(path) > 1:
                path.pop()
                current = path[-1]
            else:
                break
            action = 'backtrack'
        else:
            current = min(neighbors, key=manhattan)
            path.append(current)
            action = 'move'

        snapshots.append({
            'current': current,
            'path': path[:],
            'tabu_list': tabu_list[:],
            'action': action
        })

    return [], snapshots


# ── 3. Precompute Visited ─────────────────────────────────────
def precompute_visited(snapshots):
    result = []
    cumulative = set()
    for snap in snapshots:
        cumulative.update(snap['path'])
        result.append(frozenset(cumulative))
    return result


# ── 4. Render Frame ───────────────────────────────────────────
COLOR = {
    'wall':    [0.07, 0.07, 0.07],
    'path':    [0.20, 0.78, 0.40],
    'tabu':    [0.95, 0.60, 0.60],
    'visited': [0.82, 0.82, 0.90],
    'current': [1.00, 0.85, 0.10],
    'start':   [0.10, 0.70, 0.30],
    'end':     [0.90, 0.20, 0.20],
}


def make_png(grid, snap, visited_frame, idx, total):
    rows, cols = grid.shape
    img = np.ones((rows, cols, 3), dtype=np.float32) * 0.95

    img[grid == 1] = COLOR['wall']
    for r, c in visited_frame:
        if grid[r, c] == 0:
            img[r, c] = COLOR['visited']
    for r, c in set(snap['tabu_list']):
        if 0 <= r < rows and 0 <= c < cols:
            img[r, c] = COLOR['tabu']
    for r, c in snap['path']:
        img[r, c] = COLOR['path']

    r, c = snap['current']
    img[r, c] = COLOR['current']
    img[0, 1] = COLOR['start']
    img[rows - 1, cols - 2] = COLOR['end']

    sz = max(4, min(rows // 3, 10))
    fig, ax = plt.subplots(figsize=(sz, sz))
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    warna = {'move': 'green', 'backtrack': 'red', 'done': 'blue'}
    ax.set_title(
        f"Frame {idx + 1}/{total}  |  {snap['action'].upper()}  |  "
        f"Path: {len(snap['path'])}  |  Tabu: {len(snap['tabu_list'])}",
        fontsize=9,
        color=warna.get(snap['action'], 'black')
    )

    legend = [
        mpatches.Patch(color=COLOR['current'], label='Posisi'),
        mpatches.Patch(color=COLOR['path'],    label='Path'),
        mpatches.Patch(color=COLOR['tabu'],    label='Tabu'),
        mpatches.Patch(color=COLOR['visited'], label='Visited'),
        mpatches.Patch(color=COLOR['start'],   label='Start'),
        mpatches.Patch(color=COLOR['end'],     label='End'),
    ]
    ax.legend(handles=legend, loc='upper right',
              bbox_to_anchor=(1.45, 1), fontsize=7)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf.read()


# ── 5. Session State Init ─────────────────────────────────────
for key, val in {
    'snapshots': None,
    'grid': None,
    'vc': None,
    'frame_idx': 0,
    'info': '*Klik **Generate** untuk memulai...*',
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── 6. Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")

    size_map = {'11x11': 11, '21x21': 21, '31x31': 31}
    size_label = st.radio("Ukuran Maze", list(size_map.keys()), index=0)
    n = size_map[size_label]

    tabu_size = st.slider("Tabu Size", min_value=5, max_value=200, value=30, step=5)
    seed = int(st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1))
    speed = st.slider("Langkah per frame (Auto-Play)", min_value=1, max_value=50, value=10)

    st.divider()

    generate_clicked = st.button("Generate", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
**Legenda Warna**
- Kuning: Posisi Sekarang
- Hijau: Path Aktif
- Merah muda: Node Tabu
- Ungu muda: Visited
- Hijau tua: Start
- Merah: End
""")


# ── Generate Logic ────────────────────────────────────────────
if generate_clicked:
    with st.spinner("Generating maze & running Tabu Search..."):
        try:
            maze = generate_maze(n, seed=seed)
            path, snaps = tabu_search_maze(maze, tabu_size=tabu_size)
            vc = precompute_visited(snaps)

            st.session_state.snapshots = snaps
            st.session_state.grid = maze
            st.session_state.vc = vc
            st.session_state.frame_idx = 0

            ok = 'Berhasil!' if path else 'Tidak ditemukan'
            st.session_state.info = (
                f"**{ok}** | Maze {n}x{n} | "
                f"{len(snaps)} frame | Tabu size: {tabu_size}"
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()


# ── 7. Main Area ──────────────────────────────────────────────
st.title("Tabu Search Maze")
st.markdown(st.session_state.info)

if st.session_state.snapshots is None:
    st.info("Atur parameter di sidebar, lalu klik Generate.")
    st.stop()

snaps = st.session_state.snapshots
grid = st.session_state.grid
vc = st.session_state.vc
total = len(snaps)

# ── Navigasi Frame ────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    if st.button("Awal"):
        st.session_state.frame_idx = 0
        st.rerun()
with c2:
    if st.button("Prev"):
        st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
        st.rerun()
with c3:
    if st.button("Next"):
        st.session_state.frame_idx = min(total - 1, st.session_state.frame_idx + 1)
        st.rerun()
with c4:
    if st.button("Akhir"):
        st.session_state.frame_idx = total - 1
        st.rerun()

new_frame = st.slider(
    "Frame",
    min_value=0,
    max_value=total - 1,
    value=st.session_state.frame_idx,
)
if new_frame != st.session_state.frame_idx:
    st.session_state.frame_idx = new_frame
    st.rerun()

# ── Auto-Play ─────────────────────────────────────────────────
st.divider()
if st.button("Auto-Play", type="secondary"):
    img_holder = st.empty()
    stat_holder = st.empty()
    bar = st.progress(0)

    i = st.session_state.frame_idx
    while i < total:
        snap = snaps[i]
        img_bytes = make_png(grid, snap, vc[i], i, total)
        action = snap['action']

        img_holder.image(img_bytes)
        stat_holder.markdown(
            f"**Frame:** {i+1}/{total} | "
            f"**Aksi:** {action.upper()} | "
            f"**Path:** {len(snap['path'])} | "
            f"**Tabu:** {len(snap['tabu_list'])}"
        )
        bar.progress((i + 1) / total)

        if action == 'done':
            break

        i = min(i + speed, total - 1)
        time.sleep(0.1)

    st.session_state.frame_idx = i
    st.success("Animasi selesai!")

else:
    # ── Tampilan Frame Statis ─────────────────────────────────
    idx = st.session_state.frame_idx
    snap = snaps[idx]
    action = snap['action']

    st.markdown(
        f"**Aksi:** {action.upper()} | "
        f"**Path:** {len(snap['path'])} | "
        f"**Tabu:** {len(snap['tabu_list'])} | "
        f"**Frame:** {idx+1}/{total}"
    )

    img_bytes = make_png(grid, snap, vc[idx], idx, total)
    st.image(img_bytes)