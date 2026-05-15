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

  .st { font-size: 11px; color: #555; letter-spacing: 0.04em; }

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
  .p-move       { background:#0f2b1c; color:#4caf82; border:1px solid #1e4a30; }
  .p-revisit    { background:#222; color:#666; border:1px solid #333; }
  .p-aspiration { background:#2b2000; color:#c09030; border:1px solid #4a3800; }
  .p-backtrack  { background:#2b1010; color:#c04040; border:1px solid #4a2020; }
  .p-done       { background:#252525; color:#aaa; border:1px solid #3a3a3a; }
  .p-start      { background:#202020; color:#555; border:1px solid #2a2a2a; }
  .p-stuck      { background:#3a1010; color:#cc3333; border:1px solid #5a2020; }

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
    visited.add((1, 1)); grid[1, 1] = 0
    while stack:
        r, c = stack[-1]
        nb = [(r+dr, c+dc, dr, dc) for dr, dc in [(-2,0),(2,0),(0,-2),(0,2)]
              if 0 < r+dr < n-1 and 0 < c+dc < n-1 and (r+dr, c+dc) not in visited]
        if nb:
            nr, nc, dr, dc = random.choice(nb)
            grid[r+dr//2, c+dc//2] = 0; grid[nr, nc] = 0
            visited.add((nr, nc)); stack.append((nr, nc))
        else:
            stack.pop()
    grid[0,1] = grid[1,1] = grid[n-2,n-2] = grid[n-1,n-2] = 0
    return grid


# ═══════════════════════════════════════════════════════════════
# 2. TABU SEARCH
# ═══════════════════════════════════════════════════════════════
def tabu_search(grid, tabu_tenure=8, max_iter=500_000, snap_every=20):
    rows, cols = grid.shape
    start = (0, 1)
    goal  = (rows-1, cols-2)

    current       = start
    path          = [start]
    tabu          = {}
    visited_count = {start: 1}
    dfs_stack     = [start]
    step          = 0
    snaps         = []

    def is_tabu(node):
        return node in tabu and (step - tabu[node]) < tabu_tenure

    def h(node):
        return abs(node[0]-goal[0]) + abs(node[1]-goal[1])

    def get_neighbors(node):
        r, c = node
        return [(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr, c+dc] == 0]

    def snap(action):
        active = [(n, tabu[n]) for n in tabu if (step - tabu[n]) < tabu_tenure]
        snaps.append({
            'pos'        : current,
            'path'       : path[:],
            'tabu'       : [n for n, _ in active],
            'tabu_detail': active,
            'action'     : action,
            'step'       : step,
        })

    snap('start')

    while step < max_iter:
        if current == goal:
            snap('done'); return path, snaps

        nb = get_neighbors(current)
        tabu[current] = step

        fresh_free   = [n for n in nb if not is_tabu(n) and visited_count.get(n,0) == 0]
        visited_free = [n for n in nb if not is_tabu(n) and visited_count.get(n,0) > 0]
        fresh_tabu   = [n for n in nb if is_tabu(n)     and visited_count.get(n,0) == 0]

        if fresh_free:
            move = min(fresh_free, key=h); action = 'move'
        elif visited_free:
            move = min(visited_free, key=h); action = 'revisit'
        elif fresh_tabu:
            move = min(fresh_tabu, key=h); action = 'aspiration'
        else:
            found_exit = False
            while len(dfs_stack) > 1:
                dfs_stack.pop()
                prev = dfs_stack[-1]
                if prev in path:
                    idx = len(path) - 1
                    while idx >= 0 and path[idx] != prev: idx -= 1
                    path = path[:idx+1]
                current = prev
                visited_count[current] = visited_count.get(current, 0) + 1
                step += 1
                snap('backtrack')
                nb2 = get_neighbors(current)
                if any((not is_tabu(n) or visited_count.get(n,0) == 0) for n in nb2 if n != current):
                    found_exit = True; break
            if not found_exit:
                snap('stuck'); break
            continue

        if len(path) >= 2 and move == path[-2]: path.pop()
        else: path.append(move)
        dfs_stack.append(move)
        current = move
        visited_count[current] = visited_count.get(current, 0) + 1
        step += 1
        if step % snap_every == 0 or action in ('aspiration', 'done', 'stuck'):
            snap(action)

    return [], snaps


# ═══════════════════════════════════════════════════════════════
# 3. RENDER
# ═══════════════════════════════════════════════════════════════
C = {
    'wall'   : [0.10, 0.10, 0.10],
    'floor'  : [0.22, 0.22, 0.22],
    'path'   : [0.30, 0.72, 0.50],
    'tabu'   : [0.72, 0.22, 0.22],
    'current': [0.95, 0.95, 0.95],
    'start'  : [0.22, 0.58, 0.38],
    'goal'   : [0.80, 0.18, 0.18],
}

def render(grid, snap, idx, total):
    rows, cols = grid.shape
    img = np.where(grid[:,:,None] == 1, C['wall'], C['floor']).astype(np.float32)
    for r, c in snap['tabu']:
        if grid[r,c] == 0: img[r,c] = C['tabu']
    for r, c in snap['path']:
        img[r,c] = C['path']
    r, c = snap['pos']
    img[r,c]           = C['current']
    img[0,1]           = C['start']
    img[rows-1,cols-2] = C['goal']

    sz = max(4, min(cols//2, 7))
    fig, ax = plt.subplots(figsize=(sz, sz), facecolor='#1a1a1a')
    ax.imshow(img, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_facecolor('#1a1a1a')

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=110, facecolor='#1a1a1a')
    plt.close(fig); gc.collect(); buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════
# 4. SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k, v in {'snaps':None,'grid':None,'fidx':0,'status':'',
             'seed':random.randint(0,9999),'tenure':8}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# 5. SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**tabu maze**")
    st.divider()
    maze_size   = st.radio("ukuran", [11, 21, 31], format_func=lambda x: f"{x}×{x}")
    tabu_tenure = st.slider("tenure", 3, 30, 8)
    st.divider()
    ca, cb = st.columns([4, 1])
    with ca: st.caption(f"seed {st.session_state.seed}")
    with cb:
        if st.button("↺", use_container_width=True):
            st.session_state.seed = random.randint(0, 9999); st.rerun()
    st.divider()
    go = st.button("jalankan", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 6. RUN
# ═══════════════════════════════════════════════════════════════
if go:
    with st.spinner(""):
        grid        = generate_maze(maze_size, st.session_state.seed)
        path, snaps = tabu_search(grid, tabu_tenure=tabu_tenure)
    st.session_state.update(
        grid=grid, snaps=snaps, fidx=0, tenure=tabu_tenure,
        status=(f"{'ok' if path else 'gagal'} · "
                f"{maze_size}×{maze_size} · {len(snaps)} frame · tenure {tabu_tenure}")
    )
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# 7. DISPLAY
# ═══════════════════════════════════════════════════════════════
st.markdown("## tabu search maze")

if st.session_state.snaps is None:
    st.markdown('<p class="st">← pilih ukuran & jalankan</p>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<p class="st">{st.session_state.status}</p>', unsafe_allow_html=True)

snaps  = st.session_state.snaps
grid   = st.session_state.grid
tenure = st.session_state.tenure
total  = len(snaps)

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,3])
with c1:
    if st.button("⏮", use_container_width=True): st.session_state.fidx=0; st.rerun()
with c2:
    if st.button("◀", use_container_width=True): st.session_state.fidx=max(0,st.session_state.fidx-1); st.rerun()
with c3:
    if st.button("▶ ", use_container_width=True): st.session_state.fidx=min(total-1,st.session_state.fidx+1); st.rerun()
with c4:
    if st.button("⏭", use_container_width=True): st.session_state.fidx=total-1; st.rerun()
with c5:
    autoplay = st.button("auto", type="secondary", use_container_width=True)

fidx = st.slider("f", 0, total-1, st.session_state.fidx, label_visibility="collapsed")
if fidx != st.session_state.fidx:
    st.session_state.fidx = fidx; st.rerun()

col_l, col_r = st.columns([3, 1], gap="medium")
img_slot = col_l.empty()
bar_slot = col_l.empty()

with col_r:
    pill_slot  = st.empty()
    stat_slot  = st.empty()
    tlist_slot = st.empty()


def draw_info(s, i, tot, ten):
    act = s['action']
    pill_slot.markdown(
        f'<div class="pill p-{act}">{act}</div>'
        f'<span style="font-size:10px;color:#444;font-family:monospace"> step {s["step"]}</span>',
        unsafe_allow_html=True
    )
    stat_slot.markdown(
        f'<div class="panel">'
        f'<span class="lbl">pos &nbsp;</span><span class="val">({s["pos"][0]},{s["pos"][1]})</span><br>'
        f'<span class="lbl">path &nbsp;</span><span class="val">{len(s["path"])}</span><br>'
        f'<span class="lbl">tabu &nbsp;</span><span class="val">{len(s["tabu"])}</span><br>'
        f'<span class="lbl">frame </span><span class="val">{i+1}/{tot}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    detail = s.get('tabu_detail', [])
    if detail:
        rows_html = "".join(
            f'<div class="trow">({r},{c})'
            f' <span class="trem">-{ten-(s["step"]-e)}</span></div>'
            for (r, c), e in sorted(detail, key=lambda x: x[1], reverse=True)
        )
        tlist_slot.markdown(
            f'<div class="tlist"><div class="th">tabu list &nbsp; {len(detail)}</div>{rows_html}</div>',
            unsafe_allow_html=True
        )
    else:
        tlist_slot.markdown(
            '<div class="tlist"><div class="th">tabu list</div>'
            '<span style="color:#444;font-size:10px">—</span></div>',
            unsafe_allow_html=True
        )


if autoplay:
    i = st.session_state.fidx
    while i < total:
        s = snaps[i]
        img_slot.image(render(grid, s, i, total))
        bar_slot.progress((i+1)/total)
        draw_info(s, i, total, tenure)
        if s['action'] == 'done': break
        i += 1; time.sleep(0.05)
    st.session_state.fidx = i
else:
    i = st.session_state.fidx
    s = snaps[i]
    img_slot.image(render(grid, s, i, total))
    bar_slot.progress((i+1)/total)
    draw_info(s, i, total, tenure)