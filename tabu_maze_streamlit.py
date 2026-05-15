# ============================================================
#  TABU SEARCH MAZE — Streamlit Version
#  Jalankan dengan: streamlit run tabu_maze_streamlit.py
# ============================================================

import random
import gc
import io
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ── Konfigurasi Halaman ───────────────────────────────────────
st.set_page_config(
    page_title="Tabu Search Maze",
    page_icon="🧩",
    layout="wide",
)

plt.rcParams['figure.dpi'] = 80


# ── 1. Generate Maze ─────────────────────────────────────────
def generate_maze(n, seed=42):
    random.seed(seed)
    grid = np.ones((n, n), dtype=np.uint8)
    visited = set()

    def carve(r, c):
        visited.add((r, c))
        grid[r, c] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                grid[r + dr // 2, c + dc // 2] = 0
                carve(nr, nc)

    carve(1, 1)
    grid[0, 1] = grid[1, 1] = grid[n - 1, n - 2] = grid[n - 2, n - 2] = 0
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


# ── 3. Precompute & Render ────────────────────────────────────
def precompute_visited(snapshots):
    result = []
    cumulative = set()
    for snap in snapshots:
        cumulative.update(snap['path'])
        result.append(frozenset(cumulative))
    return result


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
        img[r, c] = COLOR['tabu']
    for r, c in snap['path']:
        img[r, c] = COLOR['path']

    r, c = snap['current']
    img[r, c] = COLOR['current']
    img[0, 1] = COLOR['start']
    img[rows - 1, cols - 2] = COLOR['end']

    sz = max(4, rows // 4)
    fig, ax = plt.subplots(figsize=(sz, sz))
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    warna = {'move': 'green', 'backtrack': 'red', 'done': 'blue'}
    ax.set_title(
        f"Frame {idx + 1}/{total} | {snap['action'].upper()} | "
        f"Path: {len(snap['path'])} | Tabu: {len(snap['tabu_list'])}",
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


# ── 4. Session State Init ─────────────────────────────────────
if 'snapshots' not in st.session_state:
    st.session_state.snapshots = None
if 'grid' not in st.session_state:
    st.session_state.grid = None
if 'vc' not in st.session_state:
    st.session_state.vc = None
if 'frame_idx' not in st.session_state:
    st.session_state.frame_idx = 0
if 'info' not in st.session_state:
    st.session_state.info = '*Klik Generate untuk memulai...*'


# ── 5. UI ─────────────────────────────────────────────────────
st.title("🧩 Tabu Search Maze")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")

    size_map = {'10×10': 11, '21×21': 21, '31×31': 31}
    size_label = st.radio("Ukuran Maze", list(size_map.keys()), index=0)
    n = size_map[size_label]

    tabu_size = st.slider("Tabu Size", min_value=5, max_value=200, value=30, step=5)
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=42)
    speed = st.slider("Kecepatan Play (langkah/frame)", min_value=1, max_value=50, value=10)

    st.divider()

    if st.button("🔄 Generate", type="primary", use_container_width=True):
        with st.spinner("⏳ Running Tabu Search..."):
            maze = generate_maze(n, seed=int(seed))
            path, snaps = tabu_search_maze(maze, tabu_size=tabu_size)
            vc = precompute_visited(snaps)
            st.session_state.snapshots = snaps
            st.session_state.grid = maze
            st.session_state.vc = vc
            st.session_state.frame_idx = 0
            ok = '✅ Berhasil!' if path else '❌ Gagal'
            st.session_state.info = (
                f"**{ok}** | Maze {n}×{n} | "
                f"{len(snaps)} frame | Tabu size: {tabu_size}"
            )

    st.divider()
    st.markdown("""
**Legenda**
- 🟡 Posisi Sekarang
- 🟢 Path Aktif
- 🔴 Node Tabu
- 🔵 Visited
- 🟩 Start
- 🟥 End
""")

# ── Main Area ─────────────────────────────────────────────────
st.markdown(st.session_state.info)

if st.session_state.snapshots is not None:
    snaps = st.session_state.snapshots
    total = len(snaps)

    # Playback controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 4])
    with col1:
        if st.button("⏮ Awal"):
            st.session_state.frame_idx = 0
    with col2:
        if st.button("◀ Prev"):
            st.session_state.frame_idx = max(0, st.session_state.frame_idx - 1)
    with col3:
        if st.button("Next ▶"):
            st.session_state.frame_idx = min(total - 1, st.session_state.frame_idx + 1)
    with col4:
        if st.button("⏭ Akhir"):
            st.session_state.frame_idx = total - 1

    # Frame slider
    frame_idx = st.slider(
        "Frame",
        min_value=0,
        max_value=total - 1,
        value=st.session_state.frame_idx,
        key="frame_slider"
    )
    st.session_state.frame_idx = frame_idx

    # Auto-play
    st.divider()
    play_col, _ = st.columns([2, 8])
    with play_col:
        auto_play = st.button("▶ Auto-Play (animasi)", use_container_width=True)

    if auto_play:
        placeholder_img = st.empty()
        placeholder_stat = st.empty()
        progress_bar = st.progress(0)

        i = st.session_state.frame_idx
        while i < total:
            snap = snaps[i]
            vc = st.session_state.vc
            img_bytes = make_png(
                st.session_state.grid, snap, vc[i], i, total
            )
            action = snap['action']
            c = {'move': 'green', 'backtrack': 'red', 'done': 'navy'}.get(action, 'black')

            placeholder_img.image(img_bytes, use_container_width=False)
            placeholder_stat.markdown(
                f"**Aksi:** :{c}[{action.upper()}] | "
                f"**Path:** {len(snap['path'])} | "
                f"**Tabu:** {len(snap['tabu_list'])}"
            )
            progress_bar.progress((i + 1) / total)

            i = min(i + speed, total - 1)
            if i >= total - 1:
                st.session_state.frame_idx = total - 1
                break
            time.sleep(0.12)

        st.success("✅ Animasi selesai!")

    else:
        # Tampilkan frame statis
        snap = snaps[st.session_state.frame_idx]
        vc = st.session_state.vc
        img_bytes = make_png(
            st.session_state.grid,
            snap,
            vc[st.session_state.frame_idx],
            st.session_state.frame_idx,
            total
        )

        action = snap['action']
        c = {'move': 'green', 'backtrack': 'red', 'done': 'navy'}.get(action, 'black')
        st.markdown(
            f"**Aksi:** {action.upper()} | "
            f"**Path:** {len(snap['path'])} | "
            f"**Tabu:** {len(snap['tabu_list'])}"
        )
        st.image(img_bytes, use_container_width=False)

else:
    st.info("👈 Atur parameter di sidebar, lalu klik **Generate** untuk memulai simulasi.")