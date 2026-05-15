# ============================================================
#  TABU SEARCH MAZE — Streamlit
#  streamlit run tabu_maze_streamlit.py
# ============================================================

import random, gc, io, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict

st.set_page_config(page_title="Tabu Maze", layout="wide")

st.markdown("""
<style>
  * { box-sizing: border-box; }

  section[data-testid="stSidebar"] {
    background: #1a1a1a !important;
    border-right: 1px solid #2a2a2a;
  }
  section[data-testid="stSidebar"] * { color: #999 !important; }

  .stApp { background: #1a1a1a; color: #888; }

  #MainMenu, footer, header { visibility: hidden; }

  .stButton > button {
    background: transparent;
    color: #777 !important;
    border: 1px solid #333;
    border-radius: 2px;
    font-size: 12px;
    padding: 4px 10px;
  }
  .stButton > button:hover { background: #252525; color: #aaa !important; border-color: #555; }
  .stButton > button[kind="primary"] {
    background: #2e2e2e;
    color: #bbb !important;
    border-color: #444;
  }
  .stButton > button[kind="primary"]:hover { background: #383838; color: #ddd !important; }

  .stSlider > div > div > div { background: #555 !important; }
  .stSlider * { color: #777 !important; }

  .stRadio * { color: #777 !important; font-size: 12px; }
  .stRadio label { font-size: 12px !important; }

  .stProgress > div { background: #222 !important; }
  .stProgress > div > div { background: #555 !important; }

  .stCaption, small, .stMarkdown p { font-size: 12px !important; color: #666 !important; }

  h2 { font-size: 12px !important; font-weight: 500;
       letter-spacing: 0.2em; color: #777 !important;
       text-transform: uppercase; margin: 0 0 16px 0 !important; }

  hr { border: none; border-top: 1px solid #2a2a2a !important; margin: 12px 0 !important; }

  .st-label { font-size: 11px; color: #555; letter-spacing: 0.04em; }

  /* Info panel */
  .panel { font-size: 11px; color: #666; font-family: monospace; line-height: 2.1; }
  .panel .lbl { color: #444; }
  .panel .val { color: #888; }

  /* Action pill */
  .pill {
    display: inline-block;
    font-size: 10px;
    font-family: monospace;
    letter-spacing: 0.1em;
    padding: 2px 10px;
    border-radius: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
  }
  .p-move            { background:#0f2b1c; color:#4caf82; border:1px solid #1e4a30; }
  .p-aspiration      { background:#2b2000; color:#c09030; border:1px solid #4a3800; }
  .p-force_move      { background:#1a1a2e; color:#6060c0; border:1px solid #30306a; }
  .p-intensification { background:#1e1030; color:#9060d0; border:1px solid #3a1e5a; }
  .p-done            { background:#252525; color:#aaa;    border:1px solid #3a3a3a; }
  .p-start           { background:#202020; color:#555;    border:1px solid #2a2a2a; }
  .p-stuck           { background:#3a1010; color:#cc3333; border:1px solid #5a2020; }
  .p-max_iter        { background:#2a1a0a; color:#c07030; border:1px solid #5a3010; }

  /* Tabu list */
  .tlist {
    font-size: 10px;
    font-family: monospace;
    color: #555;
    line-height: 1.9;
    border-top: 1px solid #2a2a2a;
    padding-top: 10px;
    margin-top: 10px;
  }
  .tlist .th { color: #555; letter-spacing: 0.1em; font-size: 10px;
               text-transform: uppercase; margin-bottom: 6px; }
  .trow { color: #844; }
  .trem { color: #533; }

  /* Batch table */
  .btable { font-size: 10px; font-family: monospace; color: #666; width: 100%; border-collapse: collapse; }
  .btable th { color: #444; text-align: left; padding: 3px 8px; border-bottom: 1px solid #2a2a2a; }
  .btable td { padding: 2px 8px; }
  .btable tr:hover td { background: #222; }
  .bstat { font-size: 11px; font-family: monospace; color: #666; line-height: 2.0; }
  .bstat .lbl { color: #444; }
  .bstat .val { color: #888; }
  .bstat .hi  { color: #4caf82; }
  .bstat .lo  { color: #c04040; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 1. GENERATE MAZE
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# 2. KOMPONEN TABU SEARCH
# ═══════════════════════════════════════════════════════════════

def objective(path_len: int, total_step: int,
              alpha: float = 0.6, beta: float = 0.4) -> float:
    """
    Objective function tanpa heuristik (Glover 1989).
    f(sol) = alpha * path_length + beta * total_steps
    Semakin kecil → semakin baik.
    """
    return alpha * path_len + beta * total_step


class Move:
    """
    Representasi move sebagai pasangan (from_node, to_node).
    Tabu list melarang MOVE, bukan node — sesuai TS standar.
    """
    __slots__ = ('key',)

    def __init__(self, from_node: tuple, to_node: tuple):
        self.key = (from_node, to_node)

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def reverse(self) -> 'Move':
        return Move(self.key[1], self.key[0])


class BestRegion:
    """
    Region di sekitar solusi terbaik untuk mekanisme Intensification.
    Menyimpan sekumpulan node di ujung path terbaik.
    """
    def __init__(self, path: list, radius: int = 5):
        self.nodes = set(path[-radius:]) if len(path) >= radius else set(path)


# ═══════════════════════════════════════════════════════════════
# 3. TABU SEARCH — VISUALISASI (menghasilkan snaps)
# ═══════════════════════════════════════════════════════════════
def tabu_search(
    grid,
    tabu_tenure: int          = 10,
    max_iter: int             = 500_000,
    snap_every: int           = 20,
    alpha: float              = 0.6,
    beta: float               = 0.4,
    freq_penalty_weight: float = 0.3,
    intensify_threshold: int  = 5_000,
    intensify_radius: int     = 5,
    tenure_min: int           = 5,
    tenure_max: int           = 20,
):
    """
    Tabu Search murni untuk visualisasi — menghasilkan snaps per frame.

    Komponen TS yang diimplementasikan:
    - Tabu list berbasis MOVE (bukan node)
    - Objective function: alpha*path_len + beta*total_step (tanpa heuristik)
    - Aspiration criteria: override tabu jika est_obj < best_obj global
    - Diversification: frequency-based penalty pada score_move
    - Intensification: backtrack ke BestRegion jika stuck >= threshold
    - Dynamic tenure: bertambah saat stuck, berkurang saat progres
    """
    rows, cols = grid.shape
    start = (0, 1)
    goal  = (rows - 1, cols - 2)

    current        = start
    path           = [start]
    step           = 0
    no_improve     = 0
    current_tenure = tabu_tenure

    tabu_list: dict    = {}
    visit_freq         = defaultdict(int)
    visit_freq[start]  = 1

    best_path: list    = []
    best_obj: float    = float('inf')
    best_region        = None
    best_found_at_step = 0

    n_aspiration      = 0
    n_force_move      = 0
    n_intensification = 0
    n_diversification = 0

    snaps = []

    # ── Fungsi bantu ──────────────────────────────────────────
    def get_neighbors(node):
        r, c = node
        return [
            (r + dr, c + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dr < rows and 0 <= c + dc < cols
            and grid[r + dr, c + dc] == 0
        ]

    def is_tabu(move: Move) -> bool:
        return move in tabu_list and (step - tabu_list[move]) < current_tenure

    def add_to_tabu(move: Move):
        tabu_list[move] = step
        tabu_list[move.reverse()] = step

    def score_move(to_node: tuple) -> float:
        """
        Skor kandidat move — tanpa heuristik apapun.
        score = path_len_estimate + freq_penalty_weight * visit_freq[to_node]
        """
        return (len(path) + 1) + freq_penalty_weight * visit_freq[to_node]

    def aspiration_met(to_node: tuple) -> bool:
        """
        Aspiration Criteria (Glover 1989):
        Move yang tabu boleh dieksekusi jika estimasi objective-nya
        lebih baik dari best_obj global yang pernah ditemukan.
        """
        if best_obj == float('inf'):
            return False
        est_obj = objective(len(path) + 1, step + 1, alpha, beta)
        return est_obj < best_obj

    def snap(action, extra=None):
        active_tabu = [
            (m.key, tabu_list[m])
            for m in tabu_list
            if (step - tabu_list[m]) < current_tenure
        ]
        snaps.append({
            'pos'             : current,
            'path'            : set(path),
            'path_len'        : len(path),
            'tabu_detail'     : active_tabu,
            'tabu_count'      : len(active_tabu),
            'action'          : action,
            'step'            : step,
            'best_len'        : len(best_path) if best_path else None,
            'best_obj'        : round(best_obj, 3) if best_obj < float('inf') else None,
            'n_aspiration'    : n_aspiration,
            'n_force_move'    : n_force_move,
            'n_intensification': n_intensification,
            'n_diversification': n_diversification,
            'current_tenure'  : current_tenure,
            'no_improve'      : no_improve,
        })

    snap('start')

    # ── Loop utama ────────────────────────────────────────────
    while step < max_iter:

        # Cek goal
        if current == goal:
            obj = objective(len(path), step, alpha, beta)
            if obj < best_obj:
                best_obj           = obj
                best_path          = path.copy()
                best_region        = BestRegion(best_path, intensify_radius)
                no_improve         = 0
                best_found_at_step = step
            snap('done')
            return best_path, snaps

        neighbors = get_neighbors(current)
        if not neighbors:
            snap('stuck')
            break

        # Evaluasi semua kandidat move
        free_moves      = []
        tabu_aspiration = []
        tabu_fallback   = []

        for nb in neighbors:
            move = Move(current, nb)
            s    = score_move(nb)

            if not is_tabu(move):
                free_moves.append((s, nb, move))
            else:
                if aspiration_met(nb):
                    tabu_aspiration.append((s, nb, move))
                else:
                    tabu_fallback.append((s, nb, move))

        # Pilih move terbaik
        chosen_node = None
        chosen_move = None
        action      = None

        if free_moves or tabu_aspiration:
            if tabu_aspiration:
                best_free_score = min(s for s, _, _ in free_moves) if free_moves else float('inf')
                best_asp        = min(tabu_aspiration, key=lambda x: x[0])
                if best_asp[0] < best_free_score:
                    chosen_node, chosen_move = best_asp[1], best_asp[2]
                    action = 'aspiration'
                    n_aspiration += 1

            if chosen_node is None and free_moves:
                best_free       = min(free_moves, key=lambda x: x[0])
                chosen_node, chosen_move = best_free[1], best_free[2]
                action = 'move'
                if visit_freq[chosen_node] > 0:
                    n_diversification += 1
        else:
            if tabu_fallback:
                oldest = max(
                    tabu_fallback,
                    key=lambda x: (step - tabu_list.get(x[2], 0))
                )
                chosen_node, chosen_move = oldest[1], oldest[2]
                action = 'force_move'
                n_force_move += 1
            else:
                snap('stuck')
                break

        # Eksekusi move
        add_to_tabu(chosen_move)
        path.append(chosen_node)
        visit_freq[chosen_node] += 1
        current    = chosen_node
        step      += 1
        no_improve += 1

        # Dynamic tenure
        if no_improve % 1000 == 0 and no_improve > 0:
            current_tenure = min(tenure_max, current_tenure + 1)
        elif no_improve == 0:
            current_tenure = max(tenure_min, current_tenure - 1)

        # Intensification: backtrack ke region terbaik
        intensified = False
        if no_improve >= intensify_threshold and best_region is not None:
            region_nodes = list(best_region.nodes)
            if region_nodes:
                backtrack_target = min(region_nodes, key=lambda n: visit_freq[n])
                current          = backtrack_target
                path             = [start, backtrack_target]
                no_improve       = 0
                n_intensification += 1
                intensified       = True
                # Bersihkan tabu list yang sudah expired
                tabu_list = {
                    m: s for m, s in tabu_list.items()
                    if (step - s) < current_tenure
                }
                action = 'intensification'

        # Snapshot
        do_snap = (
            step % snap_every == 0
            or action in ('aspiration', 'force_move', 'intensification')
        )
        if do_snap:
            snap(action)

    snap('max_iter')
    return best_path if best_path else [], snaps


# ═══════════════════════════════════════════════════════════════
# 4. TABU SEARCH — MODE STATS ONLY (untuk batch eval)
# ═══════════════════════════════════════════════════════════════
def tabu_search_stats(
    grid,
    tabu_tenure: int          = 10,
    max_iter: int             = 500_000,
    alpha: float              = 0.6,
    beta: float               = 0.4,
    freq_penalty_weight: float = 0.3,
    intensify_threshold: int  = 5_000,
    intensify_radius: int     = 5,
    tenure_min: int           = 5,
    tenure_max: int           = 20,
) -> dict:
    """
    Versi ringan tabu_search khusus batch eval.
    Tidak menyimpan snaps — hanya menghitung statistik akhir.
    Logika TS identik dengan tabu_search().
    """
    rows, cols = grid.shape
    start = (0, 1)
    goal  = (rows - 1, cols - 2)

    current        = start
    path           = [start]
    step           = 0
    no_improve     = 0
    current_tenure = tabu_tenure

    tabu_list: dict   = {}
    visit_freq        = defaultdict(int)
    visit_freq[start] = 1

    best_path: list   = []
    best_obj: float   = float('inf')
    best_region       = None

    n_aspiration      = 0
    n_force_move      = 0
    n_intensification = 0
    n_diversification = 0
    best_found_at_step = 0

    def get_neighbors(node):
        r, c = node
        return [
            (r + dr, c + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dr < rows and 0 <= c + dc < cols
            and grid[r + dr, c + dc] == 0
        ]

    def is_tabu(move: Move) -> bool:
        return move in tabu_list and (step - tabu_list[move]) < current_tenure

    def add_to_tabu(move: Move):
        tabu_list[move] = step
        tabu_list[move.reverse()] = step

    def score_move(to_node: tuple) -> float:
        return (len(path) + 1) + freq_penalty_weight * visit_freq[to_node]

    def aspiration_met(to_node: tuple) -> bool:
        if best_obj == float('inf'):
            return False
        est_obj = objective(len(path) + 1, step + 1, alpha, beta)
        return est_obj < best_obj

    while step < max_iter:

        if current == goal:
            obj = objective(len(path), step, alpha, beta)
            if obj < best_obj:
                best_obj           = obj
                best_path          = path.copy()
                best_region        = BestRegion(best_path, intensify_radius)
                no_improve         = 0
                best_found_at_step = step
            break

        neighbors = get_neighbors(current)
        if not neighbors:
            break

        free_moves      = []
        tabu_aspiration = []
        tabu_fallback   = []

        for nb in neighbors:
            move = Move(current, nb)
            s    = score_move(nb)

            if not is_tabu(move):
                free_moves.append((s, nb, move))
            else:
                if aspiration_met(nb):
                    tabu_aspiration.append((s, nb, move))
                else:
                    tabu_fallback.append((s, nb, move))

        chosen_node = None
        chosen_move = None
        action      = None

        if free_moves or tabu_aspiration:
            if tabu_aspiration:
                best_free_score = min(s for s, _, _ in free_moves) if free_moves else float('inf')
                best_asp        = min(tabu_aspiration, key=lambda x: x[0])
                if best_asp[0] < best_free_score:
                    chosen_node, chosen_move = best_asp[1], best_asp[2]
                    action = 'aspiration'
                    n_aspiration += 1

            if chosen_node is None and free_moves:
                best_free       = min(free_moves, key=lambda x: x[0])
                chosen_node, chosen_move = best_free[1], best_free[2]
                action = 'move'
                if visit_freq[chosen_node] > 0:
                    n_diversification += 1
        else:
            if tabu_fallback:
                oldest = max(
                    tabu_fallback,
                    key=lambda x: (step - tabu_list.get(x[2], 0))
                )
                chosen_node, chosen_move = oldest[1], oldest[2]
                action = 'force_move'
                n_force_move += 1
            else:
                break

        add_to_tabu(chosen_move)
        path.append(chosen_node)
        visit_freq[chosen_node] += 1
        current    = chosen_node
        step      += 1
        no_improve += 1

        if no_improve % 1000 == 0 and no_improve > 0:
            current_tenure = min(tenure_max, current_tenure + 1)
        elif no_improve == 0:
            current_tenure = max(tenure_min, current_tenure - 1)

        if no_improve >= intensify_threshold and best_region is not None:
            region_nodes = list(best_region.nodes)
            if region_nodes:
                backtrack_target = min(region_nodes, key=lambda n: visit_freq[n])
                current          = backtrack_target
                path             = [start, backtrack_target]
                no_improve       = 0
                n_intensification += 1
                tabu_list = {
                    m: s for m, s in tabu_list.items()
                    if (step - s) < current_tenure
                }

    berhasil   = current == goal
    final_path = best_path if best_path else path
    final_len  = len(final_path)
    final_obj  = objective(final_len, step, alpha, beta) if berhasil else None
    efisiensi  = round(final_len / step * 100, 2) if step > 0 else 0.0

    return {
        'berhasil'          : berhasil,
        'path_len'          : final_len,
        'total_step'        : step,
        'objective_value'   : round(final_obj, 4) if final_obj else None,
        'efisiensi'         : efisiensi,
        'force_move'        : n_force_move,
        'aspiration'        : n_aspiration,
        'intensification'   : n_intensification,
        'diversification'   : n_diversification,
        'best_found_at_step': best_found_at_step,
        'final_tenure'      : current_tenure,
    }


# ═══════════════════════════════════════════════════════════════
# 5. RENDER
# ═══════════════════════════════════════════════════════════════
C = {
    'wall'   : [0.10, 0.10, 0.10],
    'floor'  : [0.22, 0.22, 0.22],
    'path'   : [0.30, 0.72, 0.50],
    'tabu'   : [0.72, 0.22, 0.22],
    'current': [0.95, 0.95, 0.95],
    'start'  : [0.22, 0.58, 0.38],
    'goal'   : [0.80, 0.18, 0.18],
    'best'   : [0.20, 0.50, 0.80],
}

def render(grid, snap, best_path_set=None):
    rows, cols = grid.shape
    img = np.where(grid[:, :, None] == 1, C['wall'], C['floor']).astype(np.float32)

    # Warnai tabu nodes (dari move keys)
    for (from_node, to_node), _ in snap.get('tabu_detail', []):
        r, c = to_node
        if 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0:
            img[r, c] = C['tabu']

    # Warnai path saat ini
    for r, c in snap['path']:
        img[r, c] = C['path']

    # Overlay best path jika ada
    if best_path_set:
        for r, c in best_path_set:
            if (r, c) not in {snap['pos'], (0, 1), (rows - 1, cols - 2)}:
                img[r, c] = C['best']

    # Posisi sekarang, start, goal
    r, c = snap['pos']
    img[r, c]              = C['current']
    img[0, 1]              = C['start']
    img[rows - 1, cols - 2] = C['goal']

    sz  = max(4, min(cols // 2, 7))
    fig, ax = plt.subplots(figsize=(sz, sz), facecolor='#1a1a1a')
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor('#1a1a1a')

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=110, facecolor='#1a1a1a')
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════
# 6. STATISTIK HELPER
# ═══════════════════════════════════════════════════════════════
def extract_stats_from_snaps(snaps):
    if not snaps:
        return None
    last     = snaps[-1]
    berhasil = last['action'] == 'done'
    return {
        'berhasil'        : berhasil,
        'path_len'        : last.get('path_len', 0),
        'total_step'      : last['step'],
        'force_move'      : last.get('n_force_move', 0),
        'aspiration'      : last.get('n_aspiration', 0),
        'intensification' : last.get('n_intensification', 0),
        'diversification' : last.get('n_diversification', 0),
        'best_obj'        : last.get('best_obj'),
        'final_tenure'    : last.get('current_tenure', '-'),
        'efisiensi'       : round(
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
        'n_total'        : n,
        'n_berhasil'     : nb,
        'success_rate'   : round(nb / n * 100, 1) if n else 0,
        'path_len'       : stat('path_len'),
        'total_step'     : stat('total_step'),
        'force_move'     : stat('force_move'),
        'aspiration'     : stat('aspiration'),
        'intensification': stat('intensification'),
        'diversification': stat('diversification'),
        'efisiensi'      : stat('efisiensi'),
    }


# ═══════════════════════════════════════════════════════════════
# 7. SESSION STATE
# ═══════════════════════════════════════════════════════════════
DEFAULTS = {
    'snaps'         : None,
    'grid'          : None,
    'best_path'     : [],
    'fidx'          : 0,
    'status'        : '',
    'seed'          : random.randint(0, 9999),
    'batch_results' : None,
    'batch_stats'   : None,
    'batch_done'    : False,
    'autoplay_on'   : False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# 8. SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**tabu maze**")
    st.divider()

    maze_size   = st.radio("ukuran", [11, 21, 31, 51, 101], format_func=lambda x: f"{x}×{x}")
    tabu_tenure = st.slider("tenure", 3, 30, 10)
    freq_pw     = st.slider("freq penalty", 0.0, 1.0, 0.3, 0.05,
                            help="Koefisien diversification penalty")
    int_thresh  = st.slider("intensify threshold", 500, 10000, 5000, 500,
                            help="Langkah tanpa perbaikan sebelum intensification")
    snap_every  = st.slider("snap interval", 5, 100, 20,
                            help="Seberapa sering frame disimpan")

    st.divider()
    ca, cb = st.columns([4, 1])
    with ca:
        st.caption(f"seed {st.session_state.seed}")
    with cb:
        if st.button("↺", use_container_width=True):
            st.session_state.seed = random.randint(0, 9999)
            st.rerun()

    st.divider()
    go = st.button("jalankan", type="primary", use_container_width=True)
    st.divider()

    # ── Batch eval ─────────────────────────────────────────────
    st.markdown("**batch eval**")
    n_seeds    = st.slider("jumlah seed", 5, 1000, 20, key="nseed")
    batch_size = st.radio("ukuran maze", [11, 21, 31, 51, 101],
                          format_func=lambda x: f"{x}×{x}", key="bsize")
    batch_ten  = st.slider("tenure batch", 3, 40, 15, key="bten",
                           help="Rekomendasi 15–25 agar aspiration aktif")
    batch_freq = st.slider("freq penalty batch", 0.0, 1.0, 0.5, 0.05, key="bfreq",
                           help="Rekomendasi 0.5–0.8 agar diversification terlihat")
    batch_inth = st.slider("intensify threshold batch", 100, 5000, 500, 100, key="binth",
                           help="Rekomendasi 500–1000 agar intensification aktif")
    run_batch  = st.button("jalankan batch", use_container_width=True)

    # ── Evaluasi single run ────────────────────────────────────
    if st.session_state.snaps is not None:
        ev = extract_stats_from_snaps(st.session_state.snaps)
        if ev:
            st.divider()
            st.markdown("**evaluasi**")
            st.markdown(f"""
<div class="panel">
<span class="lbl">status &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">{'✓ berhasil' if ev['berhasil'] else '✗ gagal'}</span><br>
<span class="lbl">path &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">{ev['path_len']} langkah</span><br>
<span class="lbl">total step &nbsp;</span><span class="val">{ev['total_step']}</span><br>
<span class="lbl">objective &nbsp;&nbsp;</span><span class="val">{ev['best_obj'] or '–'}</span><br>
<span class="lbl">efisiensi &nbsp;&nbsp;</span><span class="val">{ev['efisiensi']:.1f}%</span><br>
<span class="lbl">tenure akhir</span><span class="val"> {ev['final_tenure']}</span><br>
<br>
<span class="lbl">force move &nbsp;</span><span class="val">{ev['force_move']}×</span><br>
<span class="lbl">aspiration &nbsp;</span><span class="val">{ev['aspiration']}×</span><br>
<span class="lbl">intensif. &nbsp;&nbsp;</span><span class="val">{ev['intensification']}×</span><br>
<span class="lbl">diversif. &nbsp;&nbsp;</span><span class="val">{ev['diversification']}×</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 9. RUN SINGLE
# ═══════════════════════════════════════════════════════════════
if go:
    st.session_state.autoplay_on = False
    with st.spinner(""):
        grid = generate_maze(maze_size, st.session_state.seed)
        best_path, snaps = tabu_search(
            grid,
            tabu_tenure          = tabu_tenure,
            snap_every           = snap_every,
            freq_penalty_weight  = freq_pw,
            intensify_threshold  = int_thresh,
        )
    st.session_state.update(
        grid      = grid,
        snaps     = snaps,
        best_path = best_path,
        fidx      = 0,
        status    = (
            f"{'ok' if best_path else 'gagal'} · "
            f"{maze_size}×{maze_size} · {len(snaps)} frame · tenure {tabu_tenure}"
        ),
    )
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# 10. RUN BATCH
# ═══════════════════════════════════════════════════════════════
if run_batch:
    prog_bar = st.progress(0, text="menjalankan batch…")
    results  = []

    for i, seed in enumerate(range(n_seeds)):
        grid = generate_maze(batch_size, seed)
        ev   = tabu_search_stats(
            grid,
            tabu_tenure         = batch_ten,
            freq_penalty_weight = batch_freq,
            intensify_threshold = batch_inth,
        )
        results.append({'seed': seed, **ev})
        prog_bar.progress((i + 1) / n_seeds, text=f"seed {seed}/{n_seeds - 1}…")

    prog_bar.empty()

    st.session_state.batch_results = results
    st.session_state.batch_stats   = compute_batch_stats(results)
    st.session_state.batch_done    = True
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# 11. TAMPILKAN HASIL BATCH
# ═══════════════════════════════════════════════════════════════
if st.session_state.batch_done and st.session_state.batch_stats:
    st.markdown("## hasil batch eval")

    stats = st.session_state.batch_stats
    sr    = stats['success_rate']

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
<div class="bstat">
<span class="lbl">total seed &nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">{stats['n_total']}</span><br>
<span class="lbl">berhasil &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="{'hi' if sr >= 80 else 'lo'}">{stats['n_berhasil']} ({sr}%)</span><br>
<span class="lbl">path len mean &nbsp;&nbsp;</span><span class="val">{stats['path_len']['mean']} ± {stats['path_len']['std']}</span><br>
<span class="lbl">path len median &nbsp;</span><span class="val">{stats['path_len']['med']}</span><br>
<span class="lbl">path len min/max </span><span class="val">{stats['path_len']['min']} / {stats['path_len']['max']}</span>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
<div class="bstat">
<span class="lbl">total step mean &nbsp;</span><span class="val">{stats['total_step']['mean']} ± {stats['total_step']['std']}</span><br>
<span class="lbl">force move mean &nbsp;</span><span class="val">{stats['force_move']['mean']} ± {stats['force_move']['std']}</span><br>
<span class="lbl">aspiration mean &nbsp;</span><span class="val">{stats['aspiration']['mean']} ± {stats['aspiration']['std']}</span><br>
<span class="lbl">intensif. mean &nbsp;&nbsp;</span><span class="val">{stats['intensification']['mean']} ± {stats['intensification']['std']}</span><br>
<span class="lbl">efisiensi mean &nbsp;&nbsp;</span><span class="val">{stats['efisiensi']['mean']}% ± {stats['efisiensi']['std']}%</span>
</div>""", unsafe_allow_html=True)

    st.divider()

    rows_html = ""
    for r in st.session_state.batch_results:
        ok_sym = "✓" if r['berhasil'] else "✗"
        color  = "#4caf82" if r['berhasil'] else "#c04040"
        rows_html += (
            f"<tr>"
            f"<td>{r['seed']}</td>"
            f"<td style='color:{color}'>{ok_sym}</td>"
            f"<td>{r['path_len']}</td>"
            f"<td>{r['total_step']}</td>"
            f"<td>{r['objective_value'] or '–'}</td>"
            f"<td>{r['force_move']}</td>"
            f"<td>{r['aspiration']}</td>"
            f"<td>{r['intensification']}</td>"
            f"<td>{r['efisiensi']}%</td>"
            f"</tr>"
        )

    st.markdown(f"""
<table class="btable">
  <thead>
    <tr>
      <th>seed</th><th>ok</th><th>path</th><th>steps</th>
      <th>obj</th><th>force</th><th>asp</th><th>int</th><th>eff</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

    st.divider()
    if st.button("tutup hasil batch"):
        st.session_state.batch_done    = False
        st.session_state.batch_results = None
        st.session_state.batch_stats   = None
        st.rerun()

    st.stop()


# ═══════════════════════════════════════════════════════════════
# 12. DISPLAY MAZE
# ═══════════════════════════════════════════════════════════════
st.markdown("## tabu search maze")

if st.session_state.snaps is None:
    st.markdown('<p class="st-label">← pilih ukuran & jalankan</p>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<p class="st-label">{st.session_state.status}</p>', unsafe_allow_html=True)

snaps     = st.session_state.snaps
grid      = st.session_state.grid
total     = len(snaps)
best_set  = set(map(tuple, st.session_state.best_path)) if st.session_state.best_path else None

# ── Kontrol navigasi ──────────────────────────────────────────
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
    if st.button("▶▶ auto", type="secondary", use_container_width=True):
        st.session_state.autoplay_on = True
        st.rerun()
with c6:
    if st.button("■ stop", use_container_width=True):
        st.session_state.autoplay_on = False
        st.rerun()

fidx = st.slider(
    "frame", 0, total - 1, st.session_state.fidx,
    label_visibility="collapsed"
)
if fidx != st.session_state.fidx:
    st.session_state.autoplay_on = False
    st.session_state.fidx        = fidx
    st.rerun()

col_l, col_r = st.columns([3, 1], gap="medium")
img_slot  = col_l.empty()
bar_slot  = col_l.empty()

with col_r:
    pill_slot  = st.empty()
    stat_slot  = st.empty()
    tlist_slot = st.empty()


def draw_frame(i):
    s   = snaps[i]
    act = s['action']

    img_slot.image(render(grid, s, best_set))
    bar_slot.progress((i + 1) / total)

    pill_slot.markdown(
        f'<div class="pill p-{act}">{act.replace("_", " ")}</div>'
        f'<span style="font-size:10px;color:#444;font-family:monospace"> step {s["step"]}</span>',
        unsafe_allow_html=True
    )

    stat_slot.markdown(
        f'<div class="panel">'
        f'<span class="lbl">pos &nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">({s["pos"][0]},{s["pos"][1]})</span><br>'
        f'<span class="lbl">path &nbsp;&nbsp;&nbsp;</span><span class="val">{s.get("path_len", 0)}</span><br>'
        f'<span class="lbl">tabu &nbsp;&nbsp;&nbsp;</span><span class="val">{s.get("tabu_count", 0)}</span><br>'
        f'<span class="lbl">tenure &nbsp;</span><span class="val">{s.get("current_tenure", "–")}</span><br>'
        f'<span class="lbl">no_imp &nbsp;</span><span class="val">{s.get("no_improve", 0)}</span><br>'
        f'<span class="lbl">best &nbsp;&nbsp;&nbsp;</span><span class="val">{s["best_len"] or "–"}</span><br>'
        f'<span class="lbl">obj &nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">{s["best_obj"] or "–"}</span><br>'
        f'<span class="lbl">frame &nbsp;&nbsp;</span><span class="val">{i+1}/{total}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    detail = s.get('tabu_detail', [])
    tenure_now = s.get('current_tenure', tabu_tenure)
    if detail:
        rows_html = "".join(
            f'<div class="trow">({to[0]},{to[1]})'
            f' <span class="trem">-{tenure_now - (s["step"] - e)}</span></div>'
            for (frm, to), e in sorted(detail, key=lambda x: x[1], reverse=True)[:12]
        )
        tlist_slot.markdown(
            f'<div class="tlist"><div class="th">tabu list ({len(detail)})</div>'
            f'{rows_html}</div>',
            unsafe_allow_html=True
        )
    else:
        tlist_slot.markdown(
            '<div class="tlist"><div class="th">tabu list</div>'
            '<span style="color:#444;font-size:10px">—</span></div>',
            unsafe_allow_html=True
        )


# ── Autoplay ─────────────────────────────────────────────────
if st.session_state.autoplay_on:
    i = st.session_state.fidx
    while i < total:
        draw_frame(i)
        if snaps[i]['action'] == 'done':
            st.session_state.autoplay_on = False
            st.session_state.fidx        = i
            break
        i += 1
        time.sleep(0.06)
    else:
        st.session_state.autoplay_on = False
        st.session_state.fidx        = total - 1
    st.rerun()
else:
    draw_frame(st.session_state.fidx)