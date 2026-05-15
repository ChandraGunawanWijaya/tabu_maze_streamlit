"""
Tabu Search Maze — Streamlit App
Jalankan: streamlit run tabu_maze.py
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tabu Search Maze",
    page_icon="🧩",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

.stApp { background-color: #0d0d0d; }

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 100%);
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, #14f19522 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #14f195;
    margin: 0 0 6px 0;
    letter-spacing: -0.03em;
}
.hero-sub {
    color: #888;
    font-size: 0.85rem;
    margin: 0;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
    flex-wrap: wrap;
}
.metric-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 14px 20px;
    flex: 1;
    min-width: 110px;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #14f195;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.7rem;
    color: #666;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Status badge */
.badge-success {
    display: inline-block;
    background: #14f19522;
    border: 1px solid #14f195;
    color: #14f195;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-fail {
    display: inline-block;
    background: #ff4b4b22;
    border: 1px solid #ff4b4b;
    color: #ff4b4b;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

/* Legend */
.legend-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 10px 0;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: #aaa;
}
.legend-dot {
    width: 12px; height: 12px;
    border-radius: 3px;
    flex-shrink: 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #aaa !important;
    font-size: 0.8rem !important;
}

/* Buttons */
.stButton > button {
    background: #14f195;
    color: #0d0d0d;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    cursor: pointer;
    width: 100%;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #0dcc7a;
    transform: translateY(-1px);
}

/* Progress bar color */
.stProgress > div > div { background-color: #14f195; }

/* Slider accent */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #14f195 !important;
}
</style>
""", unsafe_allow_html=True)


# ── 1. Generate Maze ──────────────────────────────────────────────────────────
def generate_maze(n, seed=42):
    random.seed(seed)
    grid = np.ones((n, n), dtype=np.uint8)
    visited = set()

    def carve(r, c):
        visited.add((r, c))
        grid[r, c] = 0
        dirs = [(0,2),(0,-2),(2,0),(-2,0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n and 0 <= nc < n and (nr,nc) not in visited:
                grid[r+dr//2, c+dc//2] = 0
                carve(nr, nc)

    carve(1, 1)
    grid[0, 1] = 0;      grid[1, 1] = 0
    grid[n-1, n-2] = 0;  grid[n-2, n-2] = 0
    return grid


# ── 2. Tabu Search ────────────────────────────────────────────────────────────
def tabu_search_maze(grid, tabu_size=50, max_iter=200_000):
    rows, cols = grid.shape
    start = (0, 1);  end = (rows-1, cols-2)
    current = start; path = [start]
    tabu_list = [];  tabu_set = set()
    tabu_log = [];   snapshots = []

    def manhattan(p):
        return abs(p[0]-end[0]) + abs(p[1]-end[1])

    def add_tabu(node):
        if node in tabu_set: return
        tabu_list.append(node); tabu_set.add(node)
        if len(tabu_list) > tabu_size:
            old = tabu_list.pop(0); tabu_set.discard(old)

    for _ in range(max_iter):
        if current == end:
            snapshots.append({'current':current,'path':path[:],
                              'tabu_list':tabu_list[:],'action':'done'})
            return path, tabu_log, snapshots

        neighbors = []
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr,nc = current[0]+dr, current[1]+dc
            if (0<=nr<rows and 0<=nc<cols
                    and grid[nr,nc]==0 and (nr,nc) not in tabu_set):
                neighbors.append((nr,nc))

        if not neighbors:
            action = 'backtrack'
            add_tabu(current)
            if len(path) > 1: path.pop(); current = path[-1]
            else: break
        else:
            action = 'move'
            current = min(neighbors, key=manhattan)
            path.append(current)

        tabu_log.append(len(tabu_list))
        snapshots.append({'current':current,'path':path[:],
                          'tabu_list':tabu_list[:],'action':action})

    return [], tabu_log, snapshots


# ── 3. Render Frame ───────────────────────────────────────────────────────────
COLOR = {
    'wall'   : [0.07,0.07,0.07],
    'path'   : [0.08,0.95,0.58],
    'tabu'   : [1.00,0.30,0.30],
    'visited': [0.20,0.20,0.28],
    'current': [1.00,0.85,0.10],
    'start'  : [0.08,0.75,0.40],
    'end'    : [1.00,0.25,0.25],
}

def render_frame(grid, snap, all_visited):
    rows, cols = grid.shape
    img = np.ones((rows, cols, 3), dtype=np.float32) * 0.95
    img[grid==1] = COLOR['wall']
    tabu_s = set(snap['tabu_list'])
    for r,c in all_visited:
        if grid[r,c]==0: img[r,c] = COLOR['visited']
    for r,c in tabu_s:   img[r,c] = COLOR['tabu']
    for r,c in snap['path']: img[r,c] = COLOR['path']
    r,c = snap['current'];   img[r,c] = COLOR['current']
    img[0,1]           = COLOR['start']
    img[rows-1,cols-2] = COLOR['end']
    return img

def make_figure(grid, snap, all_visited, figsize=6):
    img = render_frame(grid, snap, all_visited)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout(pad=0.3)
    return fig


# ── 4. Session State Init ─────────────────────────────────────────────────────
if 'snapshots' not in st.session_state:
    st.session_state.snapshots  = None
    st.session_state.grid       = None
    st.session_state.path       = None
    st.session_state.frame_idx  = 0
    st.session_state.generated  = False


# ── 5. Sidebar Controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    st.markdown("---")

    maze_size = st.selectbox("Ukuran Maze", [10, 20, 30, 40], index=0,
                             format_func=lambda x: f"{x} × {x}")
    tabu_size = st.slider("Tabu Size", 5, 300, 30, 5)
    seed_val  = st.number_input("Seed", 0, 9999, 42, 1)

    st.markdown("---")
    if st.button("🔄 Generate & Jalankan"):
        with st.spinner("Menjalankan Tabu Search..."):
            maze = generate_maze(maze_size, seed=seed_val)
            path, tabu_log, snaps = tabu_search_maze(maze, tabu_size=tabu_size)
        st.session_state.snapshots  = snaps
        st.session_state.grid       = maze
        st.session_state.path       = path
        st.session_state.frame_idx  = 0
        st.session_state.generated  = True
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎨 Legenda")
    legend = [
        ("#14f295", "Path aktif"),
        ("#ffda10", "Posisi sekarang"),
        ("#ff4d4d", "Tabu list"),
        ("#33333f", "Pernah dilalui"),
        ("#14c065", "Start"),
        ("#ff4040", "End"),
    ]
    for hex_color, label in legend:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
            f"<div style='width:14px;height:14px;background:{hex_color};"
            f"border-radius:3px;flex-shrink:0'></div>"
            f"<span style='font-size:0.78rem;color:#aaa'>{label}</span></div>",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#444;text-align:center'>"
        "Tabu Search Maze Visualizer<br>Built with Streamlit</div>",
        unsafe_allow_html=True)


# ── 6. Main Area ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🧩 Tabu Search Maze</div>
  <div class="hero-sub">Visualisasi algoritma Tabu Search mencari jalur dalam labirin</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.generated:
    # Tampilan awal sebelum generate
    st.markdown("""
    <div style='text-align:center;padding:80px 20px;color:#444'>
      <div style='font-size:4rem;margin-bottom:16px'>🗺️</div>
      <div style='font-family:Syne,sans-serif;font-size:1.3rem;color:#555;font-weight:700'>
        Atur parameter di sidebar, lalu klik Generate
      </div>
      <div style='font-size:0.85rem;margin-top:8px;color:#333'>
        Maze akan dibuat dan Tabu Search akan langsung dijalankan
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    snaps = st.session_state.snapshots
    grid  = st.session_state.grid
    path  = st.session_state.path
    n     = grid.shape[0]

    # ── Metric cards ──────────────────────────────────────────
    found = len(path) > 0
    badge = ('<span class="badge-success">✅ JALUR DITEMUKAN</span>'
             if found else
             '<span class="badge-fail">❌ GAGAL MENEMUKAN JALUR</span>')

    snap_cur = snaps[st.session_state.frame_idx]
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-val">{n}×{n}</div>
        <div class="metric-lbl">Ukuran Maze</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{len(snaps)}</div>
        <div class="metric-lbl">Total Frame</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{len(snap_cur['path'])}</div>
        <div class="metric-lbl">Panjang Path</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">{len(snap_cur['tabu_list'])}</div>
        <div class="metric-lbl">Tabu Aktif</div>
      </div>
      <div class="metric-card" style="flex:2;display:flex;align-items:center;justify-content:center">
        {badge}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Frame slider ──────────────────────────────────────────
    frame_idx = st.slider(
        "Frame", 0, len(snaps)-1,
        st.session_state.frame_idx,
        key="frame_slider")
    st.session_state.frame_idx = frame_idx

    # ── Layout: maze kiri, info kanan ─────────────────────────
    col_maze, col_info = st.columns([3, 1])

    with col_maze:
        snap = snaps[frame_idx]
        all_vis = set()
        for s in snaps[:frame_idx+1]: all_vis.update(s['path'])

        figsize = max(5, n // 3)
        fig = make_figure(grid, snap, all_vis, figsize=figsize)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_info:
        action = snap['action']
        action_color = {'move':'#14f195','backtrack':'#ff4b4b','done':'#4b9eff'}
        st.markdown(f"""
        <div style='background:#141414;border:1px solid #222;border-radius:10px;padding:20px;margin-top:8px'>
          <div style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>Status Frame</div>
          <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:{action_color.get(action,"#fff")}'>
            {action.upper()}
          </div>
          <hr style='border-color:#222;margin:14px 0'>
          <div style='font-size:0.72rem;color:#555;margin-bottom:4px'>FRAME</div>
          <div style='font-size:1.1rem;font-weight:700'>{frame_idx+1} / {len(snaps)}</div>
          <hr style='border-color:#222;margin:14px 0'>
          <div style='font-size:0.72rem;color:#555;margin-bottom:4px'>PATH LENGTH</div>
          <div style='font-size:1.1rem;font-weight:700;color:#14f195'>{len(snap["path"])}</div>
          <hr style='border-color:#222;margin:14px 0'>
          <div style='font-size:0.72rem;color:#555;margin-bottom:4px'>TABU AKTIF</div>
          <div style='font-size:1.1rem;font-weight:700;color:#ff4b4b'>{len(snap["tabu_list"])}</div>
          <hr style='border-color:#222;margin:14px 0'>
          <div style='font-size:0.72rem;color:#555;margin-bottom:4px'>TABU SIZE MAKS</div>
          <div style='font-size:1.1rem;font-weight:700;color:#888'>{tabu_size}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Progress bar iterasi
        progress = (frame_idx + 1) / len(snaps)
        st.markdown(f"""
        <div style='background:#141414;border:1px solid #222;border-radius:10px;padding:16px'>
          <div style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>Progress</div>
          <div style='background:#222;border-radius:4px;height:6px;overflow:hidden'>
            <div style='background:#14f195;height:100%;width:{progress*100:.1f}%;transition:width .3s'></div>
          </div>
          <div style='font-size:0.75rem;color:#555;margin-top:6px'>{progress*100:.1f}% selesai</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Auto-play ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_play, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("⏮ Awal"):
            st.session_state.frame_idx = 0
            st.rerun()

    with col_play:
        speed = st.select_slider(
            "Kecepatan animasi",
            options=["Lambat","Normal","Cepat","Turbo"],
            value="Normal")
        delay_map = {"Lambat":0.15,"Normal":0.07,"Cepat":0.03,"Turbo":0.01}

        if st.button("▶ Auto-Play (klik lagi untuk stop)"):
            delay = delay_map[speed]
            step  = {"Lambat":1,"Normal":2,"Cepat":5,"Turbo":10}[speed]
            placeholder = st.empty()
            for i in range(st.session_state.frame_idx, len(snaps), step):
                st.session_state.frame_idx = i
                snap_i = snaps[i]
                all_vis_i = set()
                for s in snaps[:i+1]: all_vis_i.update(s['path'])
                fig_i = make_figure(grid, snap_i, all_vis_i, figsize=figsize)
                placeholder.pyplot(fig_i, use_container_width=True)
                plt.close(fig_i)
                time.sleep(delay)

    with col_next:
        if st.button("⏭ Akhir"):
            st.session_state.frame_idx = len(snaps) - 1
            st.rerun()
