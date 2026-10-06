"""Standalone HTML preview of the dashboard.

Renders every registered panel for a set of selection presets and writes one
self-contained HTML file: tab switching and preset switching work client-side
with no server, and every panel carries working PNG and SVG download links
backed by embedded data URIs.

This is the iteration surface. `dashboard_app.py` renders the same panels from
the same registry with live controls instead of presets.

    python preview.py [--root PATH] [--out dashboard_preview.html]
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import sys

# Make the sibling modules importable even when the interpreter does not put the
# script's own directory on sys.path (e.g. PYTHONSAFEPATH=1, or an odd launcher).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")

import pandas as pd

import arena_discrim as ad
import panels as P


# --------------------------------------------------------------------------- #
# presets: stand in for the live sidebar controls
# --------------------------------------------------------------------------- #
# Panels rendered once outside the preset blocks, because their own control
# (which session to show) matters more than the global selection.
SHARED_PANELS = {"trajectory_bundles"}


def build_presets(sessions):
    """Selection presets the preview can switch between."""
    inc = sessions[sessions["include"]]
    animals = sorted(inc["animal"].unique())

    last3 = []
    for a, g in inc.groupby("animal"):
        g = g.sort_values("session_index").tail(3)
        last3 += list(zip(g["animal"], g["session"]))

    big = inc[inc["n_choice_trials"] >= 150]

    # Kept deliberately short: every preset embeds a full PNG+SVG copy of all 16
    # panels, so the file grows ~2 MB per preset. These four cover the cases
    # worth eyeballing -- group, a well-sampled single animal, a thin single
    # animal, and a recent-sessions subset.
    fat = max(animals, key=lambda a: inc.loc[inc["animal"] == a, "n_choice_trials"].sum())
    presets = [
        ("all", "All animals, all passing sessions", P.Selection()),
        (f"animal_{fat}", f"{fat} only (most trials)", P.Selection(animals=[fat])),
        ("last3", "Last 3 sessions per animal", P.Selection(sessions=last3)),
    ]
    return presets


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def _b64(fig, fmt, dpi=120):
    # 120 dpi keeps the embedded preview file manageable; the Streamlit app
    # exports PNG at 300 dpi. SVG is resolution-independent either way.
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                facecolor="white")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_panel_html(name, ds, params, slug):
    spec = P.PANELS[name]
    fig = P.render(name, ds, params)
    png = _b64(fig, "png")
    svg = _b64(fig, "svg")
    fig.clf()
    import matplotlib.pyplot as plt
    plt.close(fig)

    data = P.panel_data(name, ds, params)
    csv_link = ""
    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
        csv = base64.b64encode(data.to_csv(index=True).encode()).decode("ascii")
        csv_link = (f'<a class="dl" download="{name}.csv" '
                    f'href="data:text/csv;base64,{csv}">CSV</a>')

    cap_text = P.caption_for(name, params)
    cap = f'<p class="cap">{html.escape(cap_text)}</p>' if cap_text else ""
    return f"""
    <figure class="panel" id="{slug}-{name}">
      <figcaption class="ptitle">{html.escape(spec['title'])}</figcaption>
      <img src="data:image/png;base64,{png}" alt="{html.escape(spec['title'])}">
      {cap}
      <div class="dlrow">
        <a class="dl" download="{name}.png" href="data:image/png;base64,{png}">PNG</a>
        <a class="dl" download="{name}.svg" href="data:image/svg+xml;base64,{svg}">SVG</a>
        {csv_link}
      </div>
    </figure>"""


def sidebar_html(sessions, presets, active):
    """A static rendering of the live control set, for layout review.

    The animal and session lists are both scrollable, since animal count will
    grow. The session list is sortable by date, trial count and accuracy at
    70-90 deg; sorting is live in the preview so the ordering can be judged.
    """
    rows = []
    for _, r in sessions.sort_values(["animal", "session_index"]).iterrows():
        flag = "" if r["qc_flags"] == "ok" else f' <span class="flag">{r["qc_flags"]}</span>'
        checked = "checked" if r["include"] else ""
        cls = "srow" + ("" if r["include"] else " ex")
        easy = r.get("pct_correct_easy")
        easy_txt = "&ndash;" if pd.isna(easy) else f"{easy:.0f}%"
        rows.append(
            f'<label class="{cls}" data-animal="{r["animal"]}" '
            f'data-date="{r["session"]}" data-trials="{int(r["n_choice_trials"])}" '
            f'data-easy="{-1 if pd.isna(easy) else easy:.1f}" '
            f'id="sess-{r["animal"]}-{r["session"]}">'
            f'<input type="checkbox" {checked}> '
            f'<b>{r["animal"]}</b> s{int(r["session_index"])} '
            f'<span class="dim">{r["session"][:10]} &middot; '
            f'{int(r["n_choice_trials"])} tr &middot; easy {easy_txt}</span>{flag}</label>')
    animals = sorted(sessions["animal"].unique())
    acount = sessions.groupby("animal").size().to_dict()
    achecks = "".join(
        f'<label><input type="checkbox" checked> {a} '
        f'<span class="dim">{acount.get(a, 0)} sessions</span></label>' for a in animals)
    opts = "".join(f'<option value="{k}">{html.escape(lbl)}</option>' for k, lbl, _ in presets)
    return f"""
    <aside>
      <h2>Selection</h2>
      <div class="ctl">
        <div class="lbl">Preset (live in the app: free selection)</div>
        <select id="preset" onchange="setPreset(this.value)">{opts}</select>
      </div>
      <div class="ctl">
        <div class="lbl">Animals <span class="dim">({len(animals)})</span></div>
        <div class="checks scroll animals">{achecks}</div>
      </div>
      <div class="ctl">
        <div class="lbl">Sessions <span class="dim">({len(sessions)})</span></div>
        <div class="sortbar">
          sort:
          <button class="sortb on" data-k="animal" onclick="sortSessions('animal',this)">animal</button>
          <button class="sortb" data-k="date" onclick="sortSessions('date',this)">date</button>
          <button class="sortb" data-k="trials" onclick="sortSessions('trials',this)">trials</button>
          <button class="sortb" data-k="easy" onclick="sortSessions('easy',this)">% @70-90&deg;</button>
        </div>
        <div class="checks scroll" id="sesslist">{''.join(rows)}</div>
      </div>
      <div class="ctl"><div class="lbl">Shortcuts</div>
        <button disabled>All</button><button disabled>None</button>
        <button disabled>Last 3</button><button disabled>QC pass only</button></div>
      <div class="ctl"><div class="lbl">Min trials per stimulus level</div>
        <input type="range" min="0" max="60" value="0" disabled></div>
      <div class="ctl"><div class="lbl">RT trim (percentile)</div>
        <input type="range" min="0" max="5" value="1" step="0.5" disabled></div>
      <p class="note">Sorting and the session checkboxes are live here so the
      ordering and the cross-animal grid interaction can be judged; the panels
      themselves are pre-rendered, so use the preset menu to see how they respond
      to a selection change. In the Streamlit app every control drives the panels
      directly.</p>
    </aside>"""


def session_grid_html(sessions):
    """Interactive animal x session grid for the cross-animal tab.

    Clicking a cell deselects that session and greys it out, and syncs the
    checkbox in the sidebar list; clicking again restores it. Each cell shows
    the session date and trial count on hover. In the Streamlit app this same
    interaction re-renders every panel.
    """
    inc = sessions.sort_values(["animal", "session_index"])
    animals = sorted(inc["animal"].unique())
    max_idx = int(inc["session_index"].max()) if len(inc) else 0
    head = "".join(f"<th>{i}</th>" for i in range(1, max_idx + 1))
    body = []
    for a in animals:
        g = inc[inc["animal"] == a].set_index("session_index")
        cells = []
        for i in range(1, max_idx + 1):
            if i not in g.index:
                cells.append('<td class="na"></td>')
                continue
            r = g.loc[i]
            easy = r.get("pct_correct_easy")
            easy_txt = "&ndash;" if pd.isna(easy) else f"{easy:.0f}"
            cls = "cell" + ("" if r["include"] else " off")
            tip = (f'{a} session {i}\\n{r["session"]}\\n'
                   f'{int(r["n_choice_trials"])} choice trials\\n'
                   f'{r["pct_correct"]:.1f}% correct overall\\n'
                   f'QC: {r["qc_flags"]}')
            cells.append(
                f'<td class="{cls}" title="{tip}" '
                f'onclick="toggleSession(this,\'{a}\',\'{r["session"]}\')">'
                f'<span class="v">{easy_txt}</span>'
                f'<span class="d">{r["session"][5:10]}</span></td>')
        body.append(f'<tr><th class="rowh">{a}</th>{"".join(cells)}</tr>')
    return f"""
    <div class="gridwrap">
      <div class="lbl">Click a session to include or exclude it &mdash; value is
      % correct at 70-90&deg;, hover for date, trial count and QC flags</div>
      <table class="sgrid"><thead><tr><th></th>{head}</tr></thead>
      <tbody>{''.join(body)}</tbody></table>
      <div class="dim" style="margin-top:6px">In this preview the toggle updates
      the grid and the sidebar list; in the Streamlit app it also re-renders every
      panel on the current selection.</div>
    </div>"""


def trajectory_browser_html(trials, paths, sessions, root, commit, traces,
                            max_trials_drawn=30):
    """Session-selectable trajectory panel.

    Rendered once across every passing session rather than once per preset: the
    panel shows a single session by construction, so a dropdown over sessions is
    the control that matters, and duplicating it per preset would only inflate
    the file.
    """
    inc = sessions[sessions["include"]].sort_values(["animal", "session_index"])
    opts, blocks = [], []
    for _, r in inc.iterrows():
        a, s = r["animal"], r["session"]
        key = f"{a}-{s}"
        ds = P.make_dataset(trials, paths, sessions, P.Selection(sessions=[(a, s)]),
                            root=root, commit=commit, traces=traces)
        fig = P.render("trajectory_bundles", ds,
                       {"example_session": (a, s), "max_trials_drawn": max_trials_drawn})
        png, svg = _b64(fig, "png", dpi=110), _b64(fig, "svg")
        import matplotlib.pyplot as plt
        plt.close(fig)
        opts.append(f'<option value="{key}">{a} &mdash; session {int(r["session_index"])} '
                    f'({s[:10]}, {int(r["n_choice_trials"])} trials)</option>')
        blocks.append(f"""
        <div class="traj" data-k="{key}">
          <img src="data:image/png;base64,{png}" alt="trajectories {key}">
          <div class="dlrow">
            <a class="dl" download="trajectories_{key}.png"
               href="data:image/png;base64,{png}">PNG</a>
            <a class="dl" download="trajectories_{key}.svg"
               href="data:image/svg+xml;base64,{svg}">SVG</a>
          </div>
        </div>""")
    first = inc.iloc[0]
    firstkey = f'{first["animal"]}-{first["session"]}'
    return f"""
    <h3 class="section" id="sec-traj">Trajectories by session</h3>
    <figure class="panel" style="max-width:900px">
      <figcaption class="ptitle">Trajectories to correct and incorrect choices</figcaption>
      <div class="ctl" style="margin-bottom:10px">
        <div class="lbl">Session</div>
        <select onchange="setTraj(this.value)" style="max-width:420px">{''.join(opts)}</select>
      </div>
      {''.join(blocks)}
      <p class="cap">Single-trial paths for one session, coloured by outcome, with
      the mean path overlaid, drawn on the full arena. Up to {max_trials_drawn}
      trials per outcome. Hollow squares mark the two choice boxes; the dashed
      circle is the centre zone the animal starts in.</p>
    </figure>"""


CSS = """
:root { --bg:#fbfbfc; --fg:#1d1d1f; --dim:#6b6b70; --line:#e3e3e7; --accent:#1b6ca8; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.5 Arial,Helvetica,"Liberation Sans",sans-serif;
       color:var(--fg); background:var(--bg); }
header { padding:14px 22px; border-bottom:1px solid var(--line); background:#fff;
         display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; font-weight:650; }
header .sub { color:var(--dim); font-size:12.5px; }
.layout { display:flex; align-items:flex-start; }
aside { width:290px; min-width:290px; padding:16px 16px 40px; border-right:1px solid var(--line);
        background:#fff; position:sticky; top:0; max-height:100vh; overflow-y:auto; }
aside h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--dim);
           margin:0 0 12px; }
.ctl { margin-bottom:16px; }
.lbl { font-size:11.5px; color:var(--dim); margin-bottom:5px; }
.checks { display:flex; flex-direction:column; gap:3px; font-size:12.5px; }
.checks.scroll { max-height:260px; overflow-y:auto; border:1px solid var(--line);
                 border-radius:5px; padding:7px; background:#fdfdfd; }
.checks label.ex { opacity:.45; text-decoration:line-through; }
.dim { color:var(--dim); font-size:11.5px; }
.flag { color:#b23; font-size:10.5px; }
select, button, input[type=range] { font:inherit; font-size:12.5px; }
select { width:100%; padding:5px; border:1px solid var(--line); border-radius:5px; background:#fff; }
button { padding:3px 8px; margin:0 4px 4px 0; border:1px solid var(--line);
         border-radius:5px; background:#fff; color:var(--dim); }
.note { font-size:11.5px; color:var(--dim); border-top:1px solid var(--line); padding-top:10px; }
main { flex:1; padding:0 26px 60px; min-width:0; }
nav.tabs { display:flex; gap:2px; border-bottom:1px solid var(--line); margin:0 0 4px;
           position:sticky; top:0; background:var(--bg); padding-top:14px; z-index:5; }
nav.tabs button { padding:8px 16px; border:none; border-bottom:2px solid transparent;
                  background:none; color:var(--dim); font-size:13.5px; cursor:pointer; }
nav.tabs button.on { color:var(--accent); border-bottom-color:var(--accent); font-weight:600; }
nav.sub { display:flex; gap:14px; padding:9px 0 0; font-size:12.5px; position:sticky; top:43px;
          background:var(--bg); z-index:4; border-bottom:1px solid var(--line); margin-bottom:12px; }
nav.sub a { color:var(--dim); text-decoration:none; padding-bottom:7px; }
nav.sub a:hover { color:var(--accent); }
h3.section { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--dim);
             margin:26px 0 10px; padding-bottom:5px; border-bottom:1px solid var(--line); }
.grid { display:flex; flex-wrap:wrap; gap:18px; align-items:flex-start; }
figure.panel { margin:0; background:#fff; border:1px solid var(--line); border-radius:7px;
               padding:12px; max-width:100%; }
figure.panel img { max-width:100%; height:auto; display:block; }
.ptitle { font-size:13px; font-weight:600; margin-bottom:8px; }
.cap { font-size:11.5px; color:var(--dim); margin:8px 0 0; max-width:62ch; }
.dlrow { margin-top:9px; display:flex; gap:7px; }
a.dl { font-size:11px; text-decoration:none; color:var(--accent); border:1px solid var(--accent);
       border-radius:4px; padding:2px 8px; }
a.dl:hover { background:var(--accent); color:#fff; }
.view, .preset { display:none; }
.view.on, .preset.on { display:block; }
.checks.scroll.animals { max-height:130px; }
.sortbar { font-size:11px; color:var(--dim); margin-bottom:5px; }
button.sortb { padding:2px 6px; margin:0 2px 0 0; cursor:pointer; }
button.sortb.on { border-color:var(--accent); color:var(--accent); font-weight:600; }
.checks label.srow { cursor:pointer; }
.checks label.srow.off { opacity:.4; }
.gridwrap { background:#fff; border:1px solid var(--line); border-radius:7px;
            padding:12px; margin-bottom:18px; display:inline-block; }
table.sgrid { border-collapse:separate; border-spacing:3px; font-size:11px; }
table.sgrid th { color:var(--dim); font-weight:500; font-size:10.5px; }
table.sgrid th.rowh { text-align:right; padding-right:6px; font-size:12px;
                      color:var(--fg); font-weight:600; }
table.sgrid td.cell { width:44px; height:34px; text-align:center; cursor:pointer;
                      border-radius:4px; background:#e8f1f8; border:1px solid #cfe0ee;
                      line-height:1.15; user-select:none; }
table.sgrid td.cell:hover { outline:2px solid var(--accent); }
table.sgrid td.cell .v { display:block; font-weight:600; font-size:11.5px; }
table.sgrid td.cell .d { display:block; color:var(--dim); font-size:9px; }
table.sgrid td.cell.off { background:#f0f0f0; border-color:#e0e0e0; color:#aaa; }
table.sgrid td.cell.off .v { text-decoration:line-through; }
table.sgrid td.na { width:44px; height:34px; background:transparent; }
.traj { display:none; }
.traj.on { display:block; }
.traj img { max-width:100%; height:auto; display:block; }
#trajhost, #gridhost { display:none; }
"""

JS = """
function setTab(v){
  document.querySelectorAll('nav.tabs button').forEach(b=>b.classList.toggle('on', b.dataset.v===v));
  document.querySelectorAll('.view').forEach(d=>d.classList.toggle('on', d.dataset.v===v));
  document.getElementById('subnav').style.display = (v==='analysis') ? 'flex' : 'none';
  document.getElementById('gridhost').style.display = (v==='compare') ? 'block' : 'none';
  document.getElementById('trajhost').style.display = (v==='analysis') ? 'block' : 'none';
  window.scrollTo(0,0);
}
function setPreset(p){
  document.querySelectorAll('.preset').forEach(d=>d.classList.toggle('on', d.dataset.p===p));
}
function sortSessions(key, btn){
  document.querySelectorAll('button.sortb').forEach(b=>b.classList.toggle('on', b===btn));
  const list = document.getElementById('sesslist');
  const rows = Array.from(list.querySelectorAll('label.srow'));
  const num = (key==='trials'||key==='easy');
  rows.sort((a,b)=>{
    let x=a.dataset[key], y=b.dataset[key];
    if(num){ x=parseFloat(x); y=parseFloat(y); return y-x; }      // descending
    if(key==='animal'){                                            // animal, then date
      const c=x.localeCompare(y); if(c) return c;
      return a.dataset.date.localeCompare(b.dataset.date);
    }
    return x.localeCompare(y);
  });
  rows.forEach(r=>list.appendChild(r));
}
function setTraj(k){
  document.querySelectorAll('.traj').forEach(d=>d.classList.toggle('on', d.dataset.k===k));
}
function toggleSession(td, animal, session){
  const off = td.classList.toggle('off');
  const row = document.getElementById('sess-'+animal+'-'+session);
  if(row){
    row.classList.toggle('off', off);
    const cb = row.querySelector('input');
    if(cb) cb.checked = !off;
  }
}
"""


def build(root=None, out="dashboard_preview.html", trials=None, paths=None,
          commit=None, traces=None):
    if trials is None or paths is None:
        trials, paths, commit, traces = P.load_tables(root)
    sessions = ad.build_session_table(root, trials) if root else None
    if sessions is None:
        raise ValueError("A data root is required to build the session table.")

    presets = build_presets(sessions)
    params = {"max_trials_drawn": 40}
    traj_html = trajectory_browser_html(trials, paths, sessions, root, commit, traces)
    _f = sessions[sessions["include"]].sort_values(["animal", "session_index"]).iloc[0]
    traj_first = f'{_f["animal"]}-{_f["session"]}'

    blocks = []
    for key, label, sel in presets:
        ds = P.make_dataset(trials, paths, sessions, sel, root=root,
                            commit=commit, traces=traces)
        views = []
        for view_key, view_label in P.VIEWS:
            secs = []
            for v, sec_key, sec_label in P.SECTIONS:
                if v != view_key:
                    continue
                names = [n for n, s in P.PANELS.items()
                         if s["view"] == view_key and s["section"] == sec_key
                         and n not in SHARED_PANELS]
                if not names:
                    continue
                figs = "".join(render_panel_html(n, ds, params, key) for n in names)
                anchor = f'id="{key}-sec-{sec_key}"'
                head = (f'<h3 class="section" {anchor}>{html.escape(sec_label)}</h3>'
                        if view_key == "analysis" else f'<div {anchor}></div>')
                secs.append(f'{head}<div class="grid">{figs}</div>')
            n_tr = len(ds.choice)
            banner = (f'<p class="dim" style="margin:12px 0 0">{html.escape(label)} '
                      f'&mdash; {len(ds.animals)} animals, {len(ds.sessions)} sessions, '
                      f'{n_tr} choice trials</p>')
            views.append(f'<div class="view" data-v="{view_key}">{banner}{"".join(secs)}</div>')
        blocks.append(f'<div class="preset" data-p="{key}">{"".join(views)}</div>')

    tabs = "".join(f'<button data-v="{k}" onclick="setTab(\'{k}\')">{html.escape(l)}</button>'
                   for k, l in P.VIEWS)
    subnav = "".join(
        f'<a href="#" onclick="document.querySelector(\'.preset.on [id$=sec-{s}]\')'
        f'.scrollIntoView({{behavior:\'smooth\'}});return false;">{html.escape(lbl)}</a>'
        for v, s, lbl in P.SECTIONS if v == "analysis")
    subnav += ('<a href="#" onclick="document.getElementById(\'sec-traj\')'
               '.scrollIntoView({behavior:\'smooth\'});return false;">Trajectories</a>')

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Arena discrimination dashboard - preview</title><style>{CSS}</style></head><body>
<header>
  <h1>Mouse arena visual discrimination</h1>
  <span class="sub">dashboard preview &mdash; layout and panel content for review
  (the shipped Streamlit app has live controls)</span>
</header>
<div class="layout">
  {sidebar_html(sessions, presets, presets[0][0])}
  <main>
    <nav class="tabs">{tabs}</nav>
    <nav class="sub" id="subnav">{subnav}</nav>
    <div id="gridhost">{session_grid_html(sessions)}</div>
    {''.join(blocks)}
    <div id="trajhost">{traj_html}</div>
  </main>
</div>
<script>{JS}
setTab('design'); setPreset('{presets[0][0]}'); setTraj('{traj_first}');
</script></body></html>"""

    with open(out, "w") as fh:
        fh.write(doc)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Same precedence as dashboard_app.py: --root, then ARENA_DATA_ROOT, then a
    # home-relative default. No absolute path is baked in, so the bundle stays
    # portable across machines.
    ap.add_argument("--root", default=os.environ.get("ARENA_DATA_ROOT",
                                                     "~/behavior_data"))
    ap.add_argument("--out", default="dashboard_preview.html")
    args = ap.parse_args()
    print(build(root=os.path.expanduser(args.root), out=args.out))
