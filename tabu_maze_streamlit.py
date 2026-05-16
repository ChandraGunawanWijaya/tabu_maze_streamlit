# ============================================================
#  TABU SEARCH MAZE — Streamlit (MURNI, tanpa heuristik tambahan)
#  streamlit run tabu_maze_streamlit.py
# ============================================================

import random, gc, io, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

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

  .panel { font-size: 11px; color: #666; font-family: monospace; line-height: 2.1; }
  .panel .lbl { color: #444; }
  .panel .val { color: #888; }

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
  .p-move       { background:#0f2b1c; color:#4caf82; border:1px solid #1e4a30; }
  .p-aspiration { background:#2b2000; color:#c09030; border:1px solid #4a3800; }
  .p-force_move { background:#1a1a2e; color:#6060c0; border:1px solid #30306a; }
  .p-done       { background:#252525; color:#aaa;    border:1px solid #3a3a3a; }
  .p-start      { background:#202020; color:#555;    border:1px solid #2a2a2a; }
  .p-stuck      { background:#3a1010; color:#cc3333; border:1px solid #5a2020; }
  .p-max_iter   { background:#2a1a0a; color:#c07030; border:1px solid #5a3010; }

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
# 2. HITUNG MAX_ITER PROPORSIONAL
# ═══════════════════════════════════════════════════════════════

def compute_max_iter(grid, k: int = 10) -> int:
    """
    max_iter = k x |V|
    |V| = jumlah sel kosong (ruang pencarian aktual).
    k=10 artinya agent diberi kesempatan mengunjungi
    setiap sel rata-rata 10 kali — justifikasi defensible
    secara akademik, tidak arbitrer.
    """
    V = int((grid == 0).sum())
    return k * V



# ═══════════════════════════════════════════════════════════════
# 2. KOMPONEN TABU SEARCH MURNI
# ═══════════════════════════════════════════════════════════════

def objective(path_len: int, total_step: int,
              alpha: float = 0.6, beta: float = 0.4) -> float:
    """
    Objective function murni (Glover 1989).
    f = alpha * path_length + beta * total_steps
    Semakin kecil → semakin baik.
    Tidak ada komponen heuristik jarak ke goal.
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


# ═══════════════════════════════════════════════════════════════
# 3. TABU SEARCH MURNI — VISUALISASI
# ═══════════════════════════════════════════════════════════════
def tabu_search(
    grid,
    tabu_tenure : int   = 10,
    max_iter    : int   = 500_000,
    snap_every  : int   = 20,
    alpha       : float = 0.6,
    beta        : float = 0.4,
    max_solutions: int  = 3,
):
    """
    Tabu Search MURNI untuk visualisasi.

    Komponen TS yang diimplementasikan (Glover 1989):
    - Tabu list berbasis MOVE dengan tenure FIXED
    - Objective function: alpha*path_len + beta*total_step
    - Aspiration criteria: override tabu jika obj < best_obj global
    - Force move: jika semua neighbor tabu, pilih yang paling lama di tabu list
    - Multi-pass: setelah goal dicapai, reset dan lanjut cari solusi lebih baik

    TIDAK ADA:
    - Heuristik jarak (manhattan/euclidean)
    - Frequency-based penalty (diversification heuristik)
    - Intensification / BestRegion backtrack
    - Dynamic tenure
    """
    rows, cols = grid.shape
    start = (0, 1)
    goal  = (rows - 1, cols - 2)

    current   = start
    path      = [start]
    step      = 0
    n_solutions = 0

    # Tabu list: move → step saat ditambahkan
    tabu_list: dict = {}

    best_path : list  = []
    best_obj  : float = float('inf')
    best_found_at_step = 0

    n_aspiration = 0
    n_force_move = 0

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
        return move in tabu_list and (step - tabu_list[move]) < tabu_tenure

    def add_to_tabu(move: Move):
        tabu_list[move]           = step
        tabu_list[move.reverse()] = step

    def aspiration_met(to_node: tuple) -> bool:
        """
        Aspiration Criteria (Glover 1989):
        Move yang tabu boleh dieksekusi jika objective-nya
        lebih baik dari best_obj global.
        """
        if best_obj == float('inf'):
            return False
        est_obj = objective(len(path) + 1, step + 1, alpha, beta)
        return est_obj < best_obj

    def snap(action):
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

    # ── Loop utama ────────────────────────────────────────────
    while step < max_iter:

        # Cek goal
        if current == goal:
            obj = objective(len(path), step, alpha, beta)
            if obj < best_obj:
                best_obj           = obj
                best_path          = path.copy()
                best_found_at_step = step
            n_solutions += 1
            snap('done')

            # Berhenti jika sudah cukup solusi
            if n_solutions >= max_solutions:
                return best_path, snaps

            # Reset untuk mencari solusi lebih baik
            current = start
            path    = [start]
            continue

        neighbors = get_neighbors(current)
        if not neighbors:
            snap('stuck')
            break

        # Kategorikan neighbor
        free_moves      = []   # move yang tidak tabu
        tabu_aspiration = []   # move tabu tapi memenuhi aspiration criteria
        tabu_fallback   = []   # move tabu, tidak aspiration (kandidat force move)

        for nb in neighbors:
            move = Move(current, nb)
            if not is_tabu(move):
                free_moves.append((nb, move))
            else:
                if aspiration_met(nb):
                    tabu_aspiration.append((nb, move))
                else:
                    tabu_fallback.append((nb, move))

        # Pilih move — TS murni: acak dari kandidat yang tersedia
        chosen_node = None
        chosen_move = None
        action      = None

        if tabu_aspiration:
            # Aspiration: override tabu jika ada
            chosen_node, chosen_move = random.choice(tabu_aspiration)
            action = 'aspiration'
            n_aspiration += 1
        elif free_moves:
            # Pilih acak dari move bebas — murni tanpa skor
            chosen_node, chosen_move = random.choice(free_moves)
            action = 'move'
        elif tabu_fallback:
            # Force move: semua neighbor tabu → pilih yang paling lama di tabu list
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

        # Eksekusi move
        # Force move TIDAK dimasukkan tabu — escape mechanism murni TS
        if action != 'force_move':
            add_to_tabu(chosen_move)
        path.append(chosen_node)
        current = chosen_node
        step   += 1

        # Snapshot
        do_snap = (
            step % snap_every == 0
            or action in ('aspiration', 'force_move')
        )
        if do_snap:
            snap(action)

    snap('max_iter')
    return best_path if best_path else [], snaps


# ═══════════════════════════════════════════════════════════════
# 4. TABU SEARCH MURNI — BATCH EVAL
# ═══════════════════════════════════════════════════════════════
def tabu_search_stats(
    grid,
    tabu_tenure  : int   = 10,
    max_iter     : int   = 200_000,
    alpha        : float = 0.6,
    beta         : float = 0.4,
    max_solutions: int   = 3,
) -> dict:
    """
    Versi ringan tabu_search untuk batch eval.
    Logika TS identik — tidak menyimpan snaps.
    """
    rows, cols = grid.shape
    start = (0, 1)
    goal  = (rows - 1, cols - 2)

    current     = start
    path        = [start]
    step        = 0
    n_solutions = 0

    tabu_list: dict = {}

    best_path : list  = []
    best_obj  : float = float('inf')
    best_found_at_step = 0

    n_aspiration = 0
    n_force_move = 0

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
                best_found_at_step = step
            n_solutions += 1

            if n_solutions >= max_solutions:
                break

            current = start
            path    = [start]
            continue

        neighbors = get_neighbors(current)
        if not neighbors:
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
            break

        if action != 'force_move':
            add_to_tabu(chosen_move)
        path.append(chosen_node)
        current = chosen_node
        step   += 1

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


# ═══════════════════════════════════════════════════════════════
# 7. SESSION STATE
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# 8. SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**tabu maze · murni**")
    st.divider()

    maze_size    = st.radio("ukuran", [11, 21, 31, 51, 101, 201, 501, 801], format_func=lambda x: f"{x}×{x}")
    tabu_tenure  = st.slider("tenure (fixed)", 3, 30, 10)
    max_sol      = st.slider("maks solusi", 1, 10, 3,
                             help="Berapa kali goal dicapai sebelum berhenti")
    snap_every   = st.slider("snap interval", 5, 100, 20)

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
    batch_size = st.radio("ukuran maze", [11, 21, 31, 51, 101, 201, 501, 801],
                          format_func=lambda x: f"{x}×{x}", key="bsize")
    batch_ten  = st.slider("tenure batch (fixed)", 3, 40, 15, key="bten")
    batch_sol  = st.slider("maks solusi batch", 1, 10, 3, key="bsol")

    # Info max_iter proporsional
    _n   = batch_size if batch_size % 2 == 1 else batch_size + 1
    _est = _n * _n  # estimasi kasar sel total
    _V   = (_est + 1) // 2  # sel kosong ~50% dari grid
    _mi  = 10 * _V
    st.caption(f"|V| ≈ {_V:,}  →  max\_iter ≈ {_mi:,}")

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
<span class="lbl">solusi &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">{ev['n_solutions']}×</span><br>
<span class="lbl">path &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">{ev['path_len']} langkah</span><br>
<span class="lbl">total step &nbsp;</span><span class="val">{ev['total_step']}</span><br>
<span class="lbl">objective &nbsp;&nbsp;</span><span class="val">{ev['best_obj'] or '–'}</span><br>
<span class="lbl">efisiensi &nbsp;&nbsp;</span><span class="val">{ev['efisiensi']:.1f}%</span><br>
<br>
<span class="lbl">force move &nbsp;</span><span class="val">{ev['force_move']}×</span><br>
<span class="lbl">aspiration &nbsp;</span><span class="val">{ev['aspiration']}×</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 9. RUN SINGLE
# ═══════════════════════════════════════════════════════════════
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
        )
    st.session_state.update(
        grid      = grid,
        snaps     = snaps,
        best_path = best_path,
        fidx      = 0,
        status    = (
            f"{'ok' if best_path else 'gagal'} · "
            f"{maze_size}×{maze_size} · {len(snaps)} frame · "
            f"tenure {tabu_tenure} · max_iter {max_iter:,}"
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
        max_iter = compute_max_iter(grid, k=10)
        ev   = tabu_search_stats(
            grid,
            tabu_tenure   = batch_ten,
            max_iter      = max_iter,
            max_solutions = batch_sol,
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
<span class="lbl">solusi mean &nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">{stats['n_solutions']['mean']} ± {stats['n_solutions']['std']}</span><br>
<span class="lbl">force move mean &nbsp;</span><span class="val">{stats['force_move']['mean']} ± {stats['force_move']['std']}</span><br>
<span class="lbl">aspiration mean &nbsp;</span><span class="val">{stats['aspiration']['mean']} ± {stats['aspiration']['std']}</span><br>
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
            f"<td>{r['n_solutions']}</td>"
            f"<td>{r['efisiensi']}%</td>"
            f"</tr>"
        )

    st.markdown(f"""
<table class="btable">
  <thead>
    <tr>
      <th>seed</th><th>ok</th><th>path</th><th>steps</th>
      <th>obj</th><th>force</th><th>asp</th><th>sol</th><th>eff</th>
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

snaps    = st.session_state.snaps
grid     = st.session_state.grid
total    = len(snaps)
best_set = set(map(tuple, st.session_state.best_path)) if st.session_state.best_path else None

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
        f'<span class="lbl">tenure &nbsp;</span><span class="val">{tabu_tenure} (fixed)</span><br>'
        f'<span class="lbl">solusi &nbsp;</span><span class="val">{s.get("n_solutions", 0)}</span><br>'
        f'<span class="lbl">best &nbsp;&nbsp;&nbsp;</span><span class="val">{s["best_len"] or "–"}</span><br>'
        f'<span class="lbl">obj &nbsp;&nbsp;&nbsp;&nbsp;</span><span class="val">{s["best_obj"] or "–"}</span><br>'
        f'<span class="lbl">frame &nbsp;&nbsp;</span><span class="val">{i+1}/{total}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    detail     = s.get('tabu_detail', [])
    tenure_now = tabu_tenure
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
        if snaps[i]['action'] == 'done' and snaps[i].get('n_solutions', 0) >= max_sol:
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