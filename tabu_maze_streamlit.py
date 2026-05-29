import random, gc, io, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Tabu Maze", layout="wide")

# ==========================================
# REVISI DESAIN UI/UX (CSS Modern Dark Mode)
# ==========================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

  * { box-sizing: border-box; font-family: 'Inter', sans-serif; }

  /* Base Theme */
  .stApp { background: #0e1117; color: #c9d1d9; }
  
  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
  }
  section[data-testid="stSidebar"] * { color: #a5d6ff !important; font-family: 'Inter', sans-serif; }
  section[data-testid="stSidebar"] .stRadio label, 
  section[data-testid="stSidebar"] .stSlider label { color: #8b949e !important; }

  /* Streamlit Native Adjustments */
  #MainMenu, footer { visibility: hidden; }
  header { background-color: rgba(0,0,0,0) !important; }
  hr { border: none; border-top: 1px solid #30363d !important; margin: 16px 0 !important; }

  /* Buttons */
  .stButton > button {
    background: #21262d;
    color: #c9d1d9 !important;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 12px;
    transition: all 0.2s ease;
  }
  .stButton > button:hover { background: #30363d; border-color: #8b949e; transform: translateY(-1px); }
  .stButton > button[kind="primary"] {
    background: #238636;
    color: #ffffff !important;
    border-color: #2ea043;
    font-weight: 600;
  }
  .stButton > button[kind="primary"]:hover { background: #2ea043; border-color: #3fb950; }

  /* Sliders & Progress */
  .stSlider > div > div > div { background: #58a6ff !important; }
  .stProgress > div { background: #21262d !important; border-radius: 4px; overflow: hidden; }
  .stProgress > div > div { background: #238636 !important; }

  /* Typography */
  .main-title { 
    font-size: 28px; color: #f0f6fc; font-weight: 800; 
    margin-bottom: 24px; letter-spacing: -0.5px; 
    text-shadow: 0 2px 10px rgba(255,255,255,0.1);
  }
  h2 { 
    font-size: 13px !important; font-weight: 600;
    letter-spacing: 0.15em; color: #8b949e !important;
    text-transform: uppercase; margin: 0 0 16px 0 !important; 
  }
  .st-label { font-size: 12px; color: #8b949e; letter-spacing: 0.02em; font-weight: 500; }
  .stCaption, small, .stMarkdown p { font-size: 13px !important; color: #8b949e !important; }

  /* Panels (Cards) */
  .panel { 
    font-size: 12px; color: #8b949e; font-family: 'JetBrains Mono', monospace; 
    line-height: 2.0; background: #161b22; border: 1px solid #30363d; 
    border-radius: 8px; padding: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }
  .panel .lbl { color: #7d8590; display: inline-block; width: 85px; }
  .panel .val { color: #e6edf3; font-weight: 600; }

  /* Pills/Badges */
  .pill {
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 10px; font-family: 'JetBrains Mono', monospace; font-weight: 700;
    letter-spacing: 0.05em; padding: 4px 12px; border-radius: 20px;
    text-transform: uppercase; margin-bottom: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  }
  .p-move       { background: rgba(46, 160, 67, 0.15);  color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.4); }
  .p-aspiration { background: rgba(210, 153, 34, 0.15); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.4); }
  .p-force_move { background: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.4); }
  .p-done       { background: rgba(139, 148, 158, 0.15);color: #c9d1d9; border: 1px solid rgba(139, 148, 158, 0.4); }
  .p-start      { background: rgba(31, 111, 235, 0.15); color: #79c0ff; border: 1px solid rgba(31, 111, 235, 0.4); }
  .p-stuck      { background: rgba(248, 81, 73, 0.15);  color: #ff7b72; border: 1px solid rgba(248, 81, 73, 0.4); }
  .p-max_iter   { background: rgba(210, 153, 34, 0.15); color: #e3b341; border: 1px solid rgba(210, 153, 34, 0.4); }

  /* Lists & Histories */
  .tlist {
    font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #8b949e;
    line-height: 1.8; border-top: 1px solid #30363d; padding-top: 12px; margin-top: 12px;
  }
  .tlist .th { 
    color: #8b949e; letter-spacing: 0.1em; font-size: 10px; font-weight: 600;
    text-transform: uppercase; margin-bottom: 8px; 
  }
  .trow { color: #f85149; }
  .trem { color: #8b949e; font-size: 9px; }

  /* Tables */
  .btable { 
    font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #c9d1d9; 
    width: 100%; border-collapse: separate; border-spacing: 0; 
    background: #161b22; border-radius: 8px; border: 1px solid #30363d; overflow: hidden;
  }
  .btable th { 
    color: #8b949e; text-align: left; padding: 10px 12px; 
    border-bottom: 1px solid #30363d; background: #0d1117; font-weight: 600; text-transform: uppercase;
  }
  .btable td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
  .btable tr:last-child td { border-bottom: none; }
  .btable tr:hover td { background: #21262d; }
  
  .bstat { 
    font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #8b949e; 
    line-height: 2.0; background: #161b22; padding: 16px; border-radius: 8px; border: 1px solid #30363d;
  }
  .bstat .lbl { color: #7d8590; display: inline-block; width: 140px; }
  .bstat .val { color: #e6edf3; font-weight: bold; }
  .bstat .hi  { color: #3fb950; font-weight: bold; }
  .bstat .lo  { color: #f85149; font-weight: bold; }

  /* Legend Elements */
  .legend-item { 
    display: inline-flex; align-items: center; margin-right: 18px; 
    font-size: 11px; font-family: 'Inter', sans-serif; font-weight: 500; color: #8b949e;
  }
  .dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; box-shadow: 0 0 4px rgba(0,0,0,0.3); }

  /* Custom Scrollbar for list panels */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)

def generate_maze(size, seed):
    n = size if size % 2 == 1 else size + 1
    grid = np.ones((n, n), dtype=np.uint8)
    random.seed(seed)
    visited, stack = set(), [(1, 1)]
    visited.add((1, 1))
    grid[1, 1] = 0
    while stack:
        r, c = stack[-1]
        nb = [
            (r + dr, c + dc, dr, dc)
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]
            if 0 < r + dr < n - 1 and 0 < c + dc < n - 1
            and (r + dr, c + dc) not in visited
        ]
        if nb:
            nr, nc, dr, dc = random.choice(nb)
            grid[r + dr // 2, c + dc // 2] = 0
            grid[nr, nc] = 0
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()
    grid[0, 1] = grid[1, 1] = grid[n - 2, n - 2] = grid[n - 1, n - 2] = 0
    return grid

def compute_max_iter(grid, k: int = 10) -> int:
    V = int((grid == 0).sum())
    return k * V


def objective(path_len: int, total_step: int,
              alpha: float = 0.6, beta: float = 0.4) -> float:
    return alpha * path_len + beta * total_step


class Move:
    __slots__ = ('key',)

    def __init__(self, from_node: tuple, to_node: tuple):
        self.key = (from_node, to_node)

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def reverse(self) -> 'Move':
        return Move(self.key[1], self.key[0])

def tabu_search(
    grid,
    tabu_tenure  : int   = 10,
    max_iter     : int   = 500_000,
    snap_every   : int   = 20,
    alpha        : float = 0.6,
    beta         : float = 0.4,
    max_solutions: int   = 3,
    collect_snaps: bool  = True,
):

    rows, cols = grid.shape
    start = (0, 1)
    goal  = (rows - 1, cols - 2)

    current     = start
    path        = [start]
    step        = 0
    n_solutions = 0
    tabu_list      : dict = {}
    CLEANUP_INTERVAL      = max(50, tabu_tenure * 10) 

    best_path          : list  = []
    best_obj           : float = float('inf')
    best_found_at_step : int   = 0

    n_aspiration = 0
    n_force_move = 0

    snaps = [] if collect_snaps else None 

    def get_neighbors(node):
        r, c = node
        return [
            (r + dr, c + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dr < rows and 0 <= c + dc < cols
            and grid[r + dr, c + dc] == 0
        ]

    def is_tabu(move: Move) -> bool:
        return move in tabu_list and (step - tabu_list[move]) < tabu_tenure

    def add_to_tabu(move: Move):
        tabu_list[move]           = step
        tabu_list[move.reverse()] = step

    def cleanup_tabu_list():
        expired = [
            m for m, added_at in tabu_list.items()
            if (step - added_at) >= tabu_tenure
        ]
        for m in expired:
            del tabu_list[m]

    def aspiration_met(to_node: tuple) -> bool:
        if best_obj == float('inf'):
            return False
        est_obj = objective(len(path) + 1, step + 1, alpha, beta)
        return est_obj < best_obj

    def snap(action):
        if not collect_snaps:
            return
        active_tabu = [
            (m.key, tabu_list[m])
            for m in tabu_list
            if (step - tabu_list[m]) < tabu_tenure
        ]
        snaps.append({
            'pos'          : current,
            'path'         : set(path),
            'path_len'     : len(path),
            'tabu_detail'  : active_tabu,
            'tabu_count'   : len(active_tabu),
            'action'       : action,
            'step'         : step,
            'best_len'     : len(best_path) if best_path else None,
            'best_obj'     : round(best_obj, 3) if best_obj < float('inf') else None,
            'n_aspiration' : n_aspiration,
            'n_force_move' : n_force_move,
            'n_solutions'  : n_solutions,
        })

    snap('start')

    while step < max_iter:
        if step > 0 and step % CLEANUP_INTERVAL == 0:
            cleanup_tabu_list()

        if current == goal:
            obj = objective(len(path), step, alpha, beta)
            if obj < best_obj:
                best_obj           = obj
                best_path          = path.copy()
                best_found_at_step = step
            n_solutions += 1
            snap('done')

            if n_solutions >= max_solutions:
                if collect_snaps:
                    return best_path, snaps
                else:
                    break   

            current = start
            path    = [start]
            continue

        neighbors = get_neighbors(current)
        if not neighbors:
            snap('stuck')
            break

        free_moves      = []
        tabu_aspiration = []
        tabu_fallback   = []

        for nb in neighbors:
            move = Move(current, nb)
            if not is_tabu(move):
                free_moves.append((nb, move))
            else:
                if aspiration_met(nb):
                    tabu_aspiration.append((nb, move))
                else:
                    tabu_fallback.append((nb, move))

        chosen_node = None
        chosen_move = None
        action      = None

        if tabu_aspiration:
            chosen_node, chosen_move = random.choice(tabu_aspiration)
            action = 'aspiration'
            n_aspiration += 1
        elif free_moves:
            chosen_node, chosen_move = random.choice(free_moves)
            action = 'move'
        elif tabu_fallback:
            oldest = max(
                tabu_fallback,
                key=lambda x: (step - tabu_list.get(x[1], 0))
            )
            chosen_node, chosen_move = oldest
            action = 'force_move'
            n_force_move += 1
        else:
            snap('stuck')
            break

        if action != 'force_move':
            add_to_tabu(chosen_move)
        path.append(chosen_node)
        current = chosen_node
        step   += 1

        do_snap = (
            step % snap_every == 0
            or action in ('aspiration', 'force_move')
        )
        if do_snap:
            snap(action)

    if collect_snaps:
        snap('max_iter')
        return best_path if best_path else [], snaps
    else:
        snap('max_iter')  
        berhasil  = n_solutions > 0
        final_len = len(best_path) if best_path else len(path)
        final_obj = round(best_obj, 4) if berhasil else None
        efisiensi = round(final_len / step * 100, 2) if step > 0 else 0.0
        return {
            'berhasil'          : berhasil,
            'path_len'          : final_len,
            'total_step'        : step,
            'objective_value'   : final_obj,
            'efisiensi'         : efisiensi,
            'force_move'        : n_force_move,
            'aspiration'        : n_aspiration,
            'n_solutions'       : n_solutions,
            'best_found_at_step': best_found_at_step,
        }

def tabu_search_stats(grid, tabu_tenure=10, max_iter=200_000,
                      alpha=0.6, beta=0.4, max_solutions=3) -> dict:
    return tabu_search(
        grid,
        tabu_tenure   = tabu_tenure,
        max_iter      = max_iter,
        alpha         = alpha,
        beta          = beta,
        max_solutions = max_solutions,
        collect_snaps = False,
    )

# Disempurnakan sedikit estetika warna untuk klop dengan tema dark mode Github
C = {
    'wall'   : [0.12, 0.14, 0.18],  # Lebih solid slate
    'floor'  : [0.06, 0.07, 0.09],  # Darker background
    'path'   : [0.24, 0.72, 0.31],  # Hijau modern
    'tabu'   : [0.97, 0.32, 0.28],  # Merah tebal
    'current': [0.90, 0.93, 0.98],  # Putih/Biru terang menyala
    'start'  : [0.18, 0.63, 0.26],  
    'goal'   : [0.85, 0.20, 0.20],  
    'best'   : [0.34, 0.65, 1.00],  # Biru GitHub
}

def render(grid, snap, best_path_set=None):
    rows, cols = grid.shape
    img = np.where(grid[:, :, None] == 1, C['wall'], C['floor']).astype(np.float32)

    for (from_node, to_node), _ in snap.get('tabu_detail', []):
        r, c = to_node
        if 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0:
            img[r, c] = C['tabu']

    for r, c in snap['path']:
        img[r, c] = C['path']

    if best_path_set:
        for r, c in best_path_set:
            if (r, c) not in {snap['pos'], (0, 1), (rows - 1, cols - 2)}:
                img[r, c] = C['best']

    r, c = snap['pos']
    img[r, c]               = C['current']
    img[0, 1]               = C['start']
    img[rows - 1, cols - 2] = C['goal']

    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc] == 0:
            img[nr, nc] = C['current']

    sz  = max(4, min(cols // 2, 7))
    # Sesuaikan background frame dengan UI baru
    fig, ax = plt.subplots(figsize=(sz, sz), facecolor='#0e1117')
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor('#0e1117')

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=110, facecolor='#0e1117')
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf.read()

def extract_stats_from_snaps(snaps):
    if not snaps:
        return None
    last     = snaps[-1]
    berhasil = any(s['action'] == 'done' for s in snaps)
    return {
        'berhasil'    : berhasil,
        'path_len'    : last.get('path_len', 0),
        'total_step'  : last['step'],
        'force_move'  : last.get('n_force_move', 0),
        'aspiration'  : last.get('n_aspiration', 0),
        'n_solutions' : last.get('n_solutions', 0),
        'best_obj'    : last.get('best_obj'),
        'efisiensi'   : round(
            last.get('path_len', 0) / last['step'] * 100, 2
        ) if last['step'] > 0 else 0,
    }


def compute_batch_stats(results):
    berhasil = [r for r in results if r['berhasil']]
    n, nb    = len(results), len(berhasil)

    def stat(key):
        vals = [r[key] for r in berhasil] if berhasil else [0]
        return {
            'mean': round(np.mean(vals),   2),
            'std' : round(np.std(vals),    2),
            'min' : int(np.min(vals)),
            'max' : int(np.max(vals)),
            'med' : round(np.median(vals), 2),
        }

    return {
        'n_total'     : n,
        'n_berhasil'  : nb,
        'success_rate': round(nb / n * 100, 1) if n else 0,
        'path_len'    : stat('path_len'),
        'total_step'  : stat('total_step'),
        'force_move'  : stat('force_move'),
        'aspiration'  : stat('aspiration'),
        'n_solutions' : stat('n_solutions'),
        'efisiensi'   : stat('efisiensi'),
    }

DEFAULTS = {
    'snaps'        : None,
    'grid'         : None,
    'best_path'    : [],
    'fidx'         : 0,
    'status'       : '',
    'seed'         : random.randint(0, 9999),
    'batch_results': None,
    'batch_stats'  : None,
    'batch_done'   : False,
    'autoplay_on'  : False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.markdown("**TABU MAZE · MURNI**")
    st.divider()

    maze_size    = st.radio("Ukuran", [11, 21, 31, 51, 101, 201, 501, 801], format_func=lambda x: f"{x}×{x}")
    tabu_tenure  = st.slider("Tenure (Fixed)", 3, 30, 10)
    max_sol      = st.slider("Maks Solusi", 1, 10, 3)
    snap_every   = st.slider("Snap Interval", 5, 100, 20)

    st.divider()
    ca, cb = st.columns([4, 1])
    with ca:
        st.caption(f"Seed: {st.session_state.seed}")
    with cb:
        if st.button("↺", use_container_width=True):
            st.session_state.seed = random.randint(0, 9999)
            st.rerun()

    st.divider()
    go = st.button("Jalankan", type="primary", use_container_width=True)
    st.divider()

    st.markdown("**BATCH EVAL**")
    n_seeds    = st.slider("Jumlah Seed", 5, 1000, 20, key="nseed")
    batch_size = st.radio("Ukuran Maze Batch", [11, 21, 31, 51, 101, 201, 501, 801],
                          format_func=lambda x: f"{x}×{x}", key="bsize")
    batch_ten  = st.slider("Tenure Batch (Fixed)", 3, 40, 15, key="bten")
    batch_sol  = st.slider("Maks Solusi Batch", 1, 10, 3, key="bsol")

    _n   = batch_size if batch_size % 2 == 1 else batch_size + 1
    _est = _n * _n
    _V   = (_est + 1) // 2
    _mi  = 10 * _V
    st.caption(f"|V| ≈ {_V:,}  →  Max Iter ≈ {_mi:,}")

    run_batch = st.button("Jalankan Batch", use_container_width=True)

    if st.session_state.snaps is not None:
        ev = extract_stats_from_snaps(st.session_state.snaps)
        if ev:
            st.divider()
            st.markdown("**EVALUASI**")
            st.markdown(f"""
<div class="panel">
<span class="lbl">Status</span><span class="val" style="color: {'#3fb950' if ev['berhasil'] else '#f85149'}">{'✓ Berhasil' if ev['berhasil'] else '✗ Gagal'}</span><br>
<span class="lbl">Solusi</span><span class="val">{ev['n_solutions']}×</span><br>
<span class="lbl">Path</span><span class="val">{ev['path_len']} langkah</span><br>
<span class="lbl">Total Step</span><span class="val">{ev['total_step']}</span><br>
<span class="lbl">Objective</span><span class="val">{ev['best_obj'] or '–'}</span><br>
<span class="lbl">Efisiensi</span><span class="val">{ev['efisiensi']:.1f}%</span><br>
<br>
<span class="lbl">Force Move</span><span class="val">{ev['force_move']}×</span><br>
<span class="lbl">Aspiration</span><span class="val">{ev['aspiration']}×</span>
</div>
""", unsafe_allow_html=True)


if go:
    st.session_state.autoplay_on = False
    with st.spinner(""):
        grid     = generate_maze(maze_size, st.session_state.seed)
        max_iter = compute_max_iter(grid, k=10)
        best_path, snaps = tabu_search(
            grid,
            tabu_tenure   = tabu_tenure,
            max_iter      = max_iter,
            snap_every    = snap_every,
            max_solutions = max_sol,
            collect_snaps = True,
        )
    st.session_state.update(
        grid      = grid,
        snaps     = snaps,
        best_path = best_path,
        fidx      = 0,
        status    = (
            f"{'OK' if best_path else 'Gagal'} · "
            f"{maze_size}×{maze_size} · {len(snaps)} frame · "
            f"Tenure {tabu_tenure} · Max Iter {max_iter:,}"
        ),
    )
    st.rerun()

if run_batch:
    prog_bar = st.progress(0, text="Menjalankan batch…")
    results  = []

    for i, seed in enumerate(range(n_seeds)):
        grid     = generate_maze(batch_size, seed)
        max_iter = compute_max_iter(grid, k=10)
        ev = tabu_search(
            grid,
            tabu_tenure   = batch_ten,
            max_iter      = max_iter,
            max_solutions = batch_sol,
            collect_snaps = False,
        )
        results.append({'seed': seed, **ev})
        prog_bar.progress((i + 1) / n_seeds, text=f"Seed {seed}/{n_seeds - 1}…")

    prog_bar.empty()

    st.session_state.batch_results = results
    st.session_state.batch_stats   = compute_batch_stats(results)
    st.session_state.batch_done    = True
    st.rerun()


if st.session_state.batch_done and st.session_state.batch_stats:
    st.markdown("## Hasil Batch Evaluasi")

    stats = st.session_state.batch_stats
    sr    = stats['success_rate']

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
<div class="bstat">
<span class="lbl">Total Seed</span><span class="val">{stats['n_total']}</span><br>
<span class="lbl">Berhasil</span><span class="{'hi' if sr >= 80 else 'lo'}">{stats['n_berhasil']} ({sr}%)</span><br>
<span class="lbl">Path Len (Mean)</span><span class="val">{stats['path_len']['mean']} ± {stats['path_len']['std']}</span><br>
<span class="lbl">Path Len (Med)</span><span class="val">{stats['path_len']['med']}</span><br>
<span class="lbl">Path Len (Min/Max)</span><span class="val">{stats['path_len']['min']} / {stats['path_len']['max']}</span>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
<div class="bstat">
<span class="lbl">Total Step (Mean)</span><span class="val">{stats['total_step']['mean']} ± {stats['total_step']['std']}</span><br>
<span class="lbl">Solusi (Mean)</span><span class="val">{stats['n_solutions']['mean']} ± {stats['n_solutions']['std']}</span><br>
<span class="lbl">Force Move (Mean)</span><span class="val">{stats['force_move']['mean']} ± {stats['force_move']['std']}</span><br>
<span class="lbl">Aspiration (Mean)</span><span class="val">{stats['aspiration']['mean']} ± {stats['aspiration']['std']}</span><br>
<span class="lbl">Efisiensi (Mean)</span><span class="val">{stats['efisiensi']['mean']}% ± {stats['efisiensi']['std']}%</span>
</div>""", unsafe_allow_html=True)

    st.divider()

    rows_html = ""
    for r in st.session_state.batch_results:
        ok_sym = "✓" if r['berhasil'] else "✗"
        color  = "#3fb950" if r['berhasil'] else "#f85149"
        rows_html += (
            f"<tr>"
            f"<td>{r['seed']}</td>"
            f"<td style='color:{color}; font-weight: bold;'>{ok_sym}</td>"
            f"<td>{r['path_len']}</td>"
            f"<td>{r['total_step']}</td>"
            f"<td>{r['objective_value'] or '–'}</td>"
            f"<td>{r['force_move']}</td>"
            f"<td>{r['aspiration']}</td>"
            f"<td>{r['n_solutions']}</td>"
            f"<td>{r['efisiensi']}%</td>"
            f"</tr>"
        )

    st.markdown(f"""
<table class="btable">
  <thead>
    <tr>
      <th>Seed</th><th>OK</th><th>Path</th><th>Steps</th>
      <th>Obj</th><th>Force</th><th>Asp</th><th>Sol</th><th>Eff</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

    st.divider()
    if st.button("Tutup Hasil Batch"):
        st.session_state.batch_done    = False
        st.session_state.batch_results = None
        st.session_state.batch_stats   = None
        st.rerun()

    st.stop()


st.markdown('<div class="main-title">Tabu Search Maze Solver</div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 24px; background: #161b22; padding: 12px 16px; border-radius: 8px; border: 1px solid #30363d;">
    <div class="legend-item"><span class="dot" style="background-color: #1f242e;"></span>Wall</div>
    <div class="legend-item"><span class="dot" style="background-color: #3fb950;"></span>Current Path</div>
    <div class="legend-item"><span class="dot" style="background-color: #f85149;"></span>Tabu Move</div>
    <div class="legend-item"><span class="dot" style="background-color: #58a6ff;"></span>Best Path Found</div>
    <div class="legend-item"><span class="dot" style="background-color: #e6edf3;"></span>Agent</div>
    <div class="legend-item"><span class="dot" style="background-color: #2ea043;"></span>Start</div>
    <div class="legend-item"><span class="dot" style="background-color: #d73a49;"></span>Goal</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.snaps is None:
    st.markdown('<p class="st-label">← Silakan pilih ukuran dan klik <b>Jalankan</b> pada sidebar.</p>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<p class="st-label">{st.session_state.status}</p>', unsafe_allow_html=True)

snaps    = st.session_state.snaps
grid     = st.session_state.grid
total    = len(snaps)
best_set = set(map(tuple, st.session_state.best_path)) if st.session_state.best_path else None

c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 2, 2])
with c1:
    if st.button("⏮", use_container_width=True):
        st.session_state.fidx        = 0
        st.session_state.autoplay_on = False
        st.rerun()
with c2:
    if st.button("◀", use_container_width=True):
        st.session_state.fidx        = max(0, st.session_state.fidx - 1)
        st.session_state.autoplay_on = False
        st.rerun()
with c3:
    if st.button("▶", use_container_width=True):
        st.session_state.fidx        = min(total - 1, st.session_state.fidx + 1)
        st.session_state.autoplay_on = False
        st.rerun()
with c4:
    if st.button("⏭", use_container_width=True):
        st.session_state.fidx        = total - 1
        st.session_state.autoplay_on = False
        st.rerun()
with c5:
    if st.button("▶▶ Auto", type="secondary", use_container_width=True):
        st.session_state.autoplay_on = True
        st.rerun()
with c6:
    if st.button("■ Stop", use_container_width=True):
        st.session_state.autoplay_on = False
        st.rerun()

fidx = st.slider(
    "Frame", 0, total - 1, st.session_state.fidx,
    label_visibility="collapsed"
)
if fidx != st.session_state.fidx:
    st.session_state.autoplay_on = False
    st.session_state.fidx        = fidx
    st.rerun()

col_l, col_r = st.columns([3, 1], gap="large")
img_slot  = col_l.empty()
bar_slot  = col_l.empty()

with col_r:
    pill_slot  = st.empty()
    stat_slot  = st.empty()
    hist_slot  = st.empty()
    tlist_slot = st.empty()


def draw_frame(i):
    s   = snaps[i]
    act = s['action']

    img_slot.image(render(grid, s, best_set))
    bar_slot.progress((i + 1) / total)

    pill_slot.markdown(
        f'<div class="pill p-{act}">{act.replace("_", " ")}</div>'
        f'<span style="font-size:11px; color:#8b949e; font-family:\'JetBrains Mono\', monospace; margin-left: 8px;"> Step {s["step"]}</span>',
        unsafe_allow_html=True
    )

    stat_slot.markdown(
        f'<div class="panel">'
        f'<span class="lbl">Posisi</span><span class="val">({s["pos"][0]},{s["pos"][1]})</span><br>'
        f'<span class="lbl">Path Len</span><span class="val">{s.get("path_len", 0)}</span><br>'
        f'<span class="lbl">Tabu Aktif</span><span class="val">{s.get("tabu_count", 0)}</span><br>'
        f'<span class="lbl">Tenure</span><span class="val">{tabu_tenure} (Fixed)</span><br>'
        f'<span class="lbl">Solusi</span><span class="val">{s.get("n_solutions", 0)}</span><br>'
        f'<span class="lbl">Best Len</span><span class="val">{s["best_len"] or "–"}</span><br>'
        f'<span class="lbl">Best Obj</span><span class="val">{s["best_obj"] or "–"}</span><br>'
        f'<span class="lbl">Frame</span><span class="val">{i+1}/{total}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    history_html = ""
    for idx in range(i, -1, -1):
        s_hist   = snaps[idx]
        act_hist = s_hist['action']
        pos_hist = s_hist.get('pos', (0, 0))
        pos_str  = f"({pos_hist[0]},{pos_hist[1]})"
        
        if idx == i:
            is_current = "background: rgba(88,166,255,0.1); border-left: 3px solid #58a6ff; font-weight: bold; border-radius: 0 4px 4px 0;"
        else:
            is_current = "border-left: 3px solid transparent;"
            
        history_html += (
            f'<div class="trow" style="padding: 6px 8px; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; transition: background 0.2s; {is_current}">'
            f'<span style="font-family: \'JetBrains Mono\', monospace; font-size: 11px; color: #8b949e;">'
            f'Step {s_hist["step"]} &nbsp;<span style="color: #c9d1d9; font-weight: 600;">{pos_str}</span>'
            f'</span>'
            f'<span class="pill p-{act_hist}" style="font-size: 9px; padding: 2px 6px; margin: 0; box-shadow: none;">{act_hist.replace("_", " ")}</span>'
            f'</div>'
        )

    hist_slot.markdown(
        f'<div class="tlist">'
        f'<div class="th">Riwayat Langkah ({i + 1})</div>'
        f'<div style="max-height: 250px; overflow-y: auto; padding-right: 4px; margin-top: 8px;">'
        f'{history_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    detail     = s.get('tabu_detail', [])
    tenure_now = tabu_tenure
    if detail:
        rows_html = "".join(
            f'<div class="trow" style="padding: 4px 0; border-bottom: 1px solid #21262d; display: flex; justify-content: space-between;">'
            f'<span style="color: #f85149; font-weight: 600;">({to[0]},{to[1]})</span>'
            f'<span class="trem" style="background: #21262d; padding: 2px 6px; border-radius: 4px;">T - {tenure_now - (s["step"] - e)}</span></div>'
            for (frm, to), e in sorted(detail, key=lambda x: x[1], reverse=True)[:12]
        )
        tlist_slot.markdown(
            f'<div class="tlist"><div class="th">Tabu List ({len(detail)})</div>'
            f'<div style="margin-top: 8px;">{rows_html}</div></div>',
            unsafe_allow_html=True
        )
    else:
        tlist_slot.markdown(
            '<div class="tlist"><div class="th">Tabu List</div>'
            '<span style="color:#484f58;font-size:11px;font-style:italic;">— Kosong —</span></div>',
            unsafe_allow_html=True
        )


if st.session_state.autoplay_on:
    i = st.session_state.fidx
    draw_frame(i)

    if snaps[i]['action'] == 'done' and snaps[i].get('n_solutions', 0) >= max_sol:
        st.session_state.autoplay_on = False
        st.session_state.fidx = i
        st.rerun()
    elif i < total - 1:
        st.session_state.fidx = i + 1
        time.sleep(0.06)
        st.rerun()
    else:
        st.session_state.autoplay_on = False
        st.session_state.fidx = total - 1
        st.rerun()
else:
    draw_frame(st.session_state.fidx)