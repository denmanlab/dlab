"""Streamlit dashboard for the mouse arena visual-discrimination task.

Launch:

    streamlit run dashboard_app.py -- --data-root ~/behavior_data

The repository is opened read-only; nothing is ever written back to it.

This is a thin renderer over `panels.PANELS` -- the same pure
``(fig, ds, params) -> Figure`` functions the HTML preview uses, so the layout
settled in the preview is what appears here.

Caching is per session folder and keyed on the folder's modification time, so
**Rescan data** picks up animals and sessions added while the app is running and
only computes the new ones.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

# Make the sibling modules importable even when the interpreter does not put the
# script's own directory on sys.path (e.g. PYTHONSAFEPATH=1, or an odd launcher).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st

import arena_discrim as ad
import panels as P

DEFAULT_ROOT = "~/behavior_data"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def parse_args():
    """Resolve the data root.

    Precedence: `--data-root` on the command line, then the ARENA_DATA_ROOT
    environment variable, then the default. The environment variable exists so
    the app can be driven by streamlit's headless test harness, which controls
    sys.argv itself.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    args, _ = ap.parse_known_args(sys.argv[1:])
    root = args.data_root or os.environ.get("ARENA_DATA_ROOT") or DEFAULT_ROOT
    return os.path.expanduser(root)


@st.cache_data(show_spinner=False)
def scan_sessions(root, version):
    """Discover (animal, session, mtime) triples. `version` busts the cache.

    Cheap: a directory walk plus one stat per session. Called on every rescan.
    """
    out = []
    for animal, session, session_dir in ad.iter_sessions(root):
        try:
            mtime = max(
                os.path.getmtime(os.path.join(session_dir, f))
                for f in ("events.csv", "samples.csv", "session_meta.json")
                if os.path.exists(os.path.join(session_dir, f))
            )
        except ValueError:
            continue
        out.append((animal, session, mtime))
    return out


@st.cache_data(show_spinner=False, persist="disk", max_entries=400)
def load_one_session(root, animal, session, mtime):
    """All derived tables for a single session.

    Keyed on `mtime`, so re-recording a session invalidates only that session.
    Persisted to disk so an app restart does not recompute everything.
    """
    session_dir = os.path.join(root, animal, session)
    trials = ad.build_session_trials(session_dir, animal, session)
    samples = ad.session_trajectories(session_dir)
    paths = ad.path_metrics(samples, trials)
    commit, traces = ad.build_commitment_table(root, sessions=[(animal, session)])
    return trials, paths, commit, traces


def load_all(root, keys, progress=None):
    """Concatenate the per-session bundles, computing only what is uncached."""
    T, PA, CM, TC = [], [], [], []
    for i, (animal, session, mtime) in enumerate(keys):
        if progress is not None:
            progress.progress((i + 1) / max(len(keys), 1),
                              text=f"{animal} / {session}")
        t, p, c, tc = load_one_session(root, animal, session, mtime)
        T.append(t); PA.append(p); CM.append(c); TC.append(tc)

    def cat(frames):
        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    trials = cat(T)
    if not trials.empty:
        order = (trials[["animal", "session"]].drop_duplicates()
                 .sort_values(["animal", "session"]).reset_index(drop=True))
        order["session_index"] = order.groupby("animal").cumcount() + 1
        trials = trials.drop(columns=["session_index"], errors="ignore").merge(
            order, on=["animal", "session"], how="left")
        trials["date"] = pd.to_datetime(trials["session"], format=ad.DATETIME_FORMAT)

    commit = cat(CM)
    if not commit.empty and not trials.empty:
        # Each bundle is computed in isolation, so its session_index is 1.
        # Restamp from the global ordering.
        commit = commit.drop(columns=["session_index"], errors="ignore").merge(
            order, on=["animal", "session"], how="left")
    return trials, cat(PA), commit, cat(TC)


# --------------------------------------------------------------------------- #
# figure block: one panel plus its exports
# --------------------------------------------------------------------------- #
def figure_block(name, ds, params):
    spec = P.PANELS[name]
    try:
        fig = P.render(name, ds, params)
    except Exception as exc:                       # a panel must never kill the app
        st.warning(f"{spec['title']}: could not render ({type(exc).__name__}: {exc})")
        return

    st.markdown(f"**{spec['title']}**")
    st.pyplot(fig, width="content")
    cap = P.caption_for(name, params)
    if cap:
        st.caption(cap)

    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    svg = io.BytesIO()
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")

    data = P.panel_data(name, ds, params)
    cols = st.columns(3 if data is not None else 2)
    cols[0].download_button("PNG (300 dpi)", png.getvalue(), f"{name}.png",
                            "image/png", key=f"png-{name}")
    cols[1].download_button("SVG (vector)", svg.getvalue(), f"{name}.svg",
                            "image/svg+xml", key=f"svg-{name}")
    if data is not None:
        cols[2].download_button("CSV (panel data)", data.to_csv(index=True).encode(),
                                f"{name}.csv", "text/csv", key=f"csv-{name}")
    plt.close(fig)
    st.divider()


def section(view_key, section_key, ds, params):
    names = [n for n, s in P.PANELS.items()
             if s["view"] == view_key and s["section"] == section_key]
    for n in names:
        figure_block(n, ds, params)


# --------------------------------------------------------------------------- #
# selection state
# --------------------------------------------------------------------------- #
def excluded_set():
    if "excluded" not in st.session_state:
        st.session_state.excluded = set()
    return st.session_state.excluded


def apply_editor_diff(edited, sessions):
    """Push an `include` column edited in a data_editor into the exclusion set."""
    ex = excluded_set()
    for _, r in edited.iterrows():
        key = (r["animal"], r["session"])
        if bool(r["include"]):
            ex.discard(key)
        else:
            ex.add(key)


def main():
    st.set_page_config(page_title="Arena discrimination dashboard",
                       layout="wide", initial_sidebar_state="expanded")
    root = parse_args()
    st.session_state.setdefault("data_version", 0)

    if not os.path.isdir(root):
        st.error(f"Data root not found: `{root}`\n\n"
                 "Pass one with `streamlit run dashboard_app.py -- "
                 "--data-root /path/to/data`.")
        st.stop()

    # ---------------- sidebar: rescan ---------------- #
    with st.sidebar:
        st.markdown("### Data")
        c1, c2 = st.columns([2, 3])
        if c1.button("Rescan data", width="stretch",
                     help="Re-read the repository. Picks up animals and sessions "
                          "added since the app started; already-loaded sessions "
                          "are reused from cache, so only new ones are computed."):
            # No st.rerun() needed: the button click already triggers this run,
            # and scan_sessions is called below with the bumped version.
            st.session_state.data_version += 1
            scan_sessions.clear()
        keys = scan_sessions(root, st.session_state.data_version)
        c2.caption(f"{len({k[0] for k in keys})} animals · "
                   f"{len(keys)} sessions")
        if st.button("Clear computed cache", width="stretch",
                     help="Discard all cached per-session tables and recompute "
                          "from scratch. Only needed if the analysis code changed."):
            load_one_session.clear()
            st.rerun()

    if not keys:
        st.warning(f"No sessions found under `{root}`.")
        st.stop()

    prog = st.progress(0.0, text="Loading sessions")
    trials, paths, commit, traces = load_all(root, keys, prog)
    prog.empty()
    if trials.empty:
        st.warning("No trials could be read from the discovered sessions.")
        st.stop()

    sessions = ad.build_session_table(root, trials)

    # ---------------- sidebar: selection ---------------- #
    with st.sidebar:
        st.markdown("### Selection")
        animals = sorted(sessions["animal"].unique())
        chosen_animals = st.multiselect("Animals", animals, default=animals,
                                        help="Scrollable; grows with the dataset.")

        qc_only = st.checkbox("QC-passing sessions only", value=True,
                              help="Applies the inclusion rule: at least "
                                   f"{ad.MIN_LEVELS} orientation levels and "
                                   f"{ad.MIN_CHOICE_TRIALS} choice trials.")

        st.markdown("**Sessions** — click a column header to sort")
        shown = sessions[sessions["animal"].isin(chosen_animals)].copy()
        if qc_only:
            shown = shown[shown["include"]]
        ex = excluded_set()
        shown["include"] = [
            (a, s) not in ex for a, s in zip(shown["animal"], shown["session"])
        ]
        cols = ["include", "animal", "session_index", "date", "n_choice_trials",
                "pct_correct_easy", "pct_correct", "median_duration_s", "qc_flags",
                "session"]
        editor = st.data_editor(
            shown[cols], hide_index=True, width="stretch", height=260,
            key="sess_editor",
            disabled=[c for c in cols if c != "include"],
            column_config={
                "include": st.column_config.CheckboxColumn("use", width="small"),
                "session_index": st.column_config.NumberColumn("s#", width="small"),
                "date": st.column_config.DatetimeColumn("date", format="YYYY-MM-DD"),
                "n_choice_trials": st.column_config.NumberColumn("trials", width="small"),
                "pct_correct_easy": st.column_config.NumberColumn(
                    "% @70-90°", width="small", format="%.0f"),
                "pct_correct": st.column_config.NumberColumn(
                    "% all", width="small", format="%.0f"),
                "median_duration_s": st.column_config.NumberColumn(
                    "dur (s)", width="small", format="%.1f"),
                "qc_flags": st.column_config.TextColumn("QC"),
                "session": None,
            })
        apply_editor_diff(editor, sessions)

        b1, b2, b3 = st.columns(3)
        if b1.button("All", width="stretch"):
            st.session_state.excluded = set(); st.rerun()
        if b2.button("None", width="stretch"):
            st.session_state.excluded = set(
                zip(shown["animal"], shown["session"])); st.rerun()
        if b3.button("Last 3", width="stretch"):
            keep = set()
            for a, g in sessions.groupby("animal"):
                g = g[g["include"]].sort_values("session_index").tail(3)
                keep |= set(zip(g["animal"], g["session"]))
            st.session_state.excluded = set(
                zip(sessions["animal"], sessions["session"])) - keep
            st.rerun()

        st.markdown("### Analysis options")
        fam_labels = {k: v["label"] for k, v in ad.FAMILIES.items()}
        family = st.selectbox(
            "Psychometric function", list(fam_labels),
            format_func=lambda k: fam_labels[k],
            index=list(fam_labels).index(ad.DEFAULT_FAMILY),
            help="All families are fitted by maximum binomial likelihood with "
                 "the guess rate fixed at 0.5 and lapse free. The estimator does "
                 "not change with the family: the data are correct-out-of-n "
                 "counts, so the binomial likelihood is the right one.")
        errorbar = st.radio(
            "Error bars", ["sem", "sd"], horizontal=True,
            format_func=lambda k: k.upper(),
            help="SEM shrinks with sample size and describes the precision of "
                 "the mean; SD describes the spread of the trials themselves "
                 "and does not shrink. Applies to every panel. For proportions "
                 "the SD shown is the per-trial Bernoulli sqrt(p(1-p)), which "
                 "sits near 0.5 and so is rarely informative on the "
                 "psychometric panels.")
        min_per_level = st.slider("Min trials per stimulus level", 0, 60, 0, 5,
                                  help="Drops an animal's levels below this count.")
        trim = st.slider("Trial-duration trim (percentile)", 0.0, 5.0, 1.0, 0.5)
        dur_log = st.checkbox("Log trial duration", value=False)
        n_boot = st.select_slider("Bootstrap samples for fit CIs",
                                  options=[0, 100, 200, 500], value=0,
                                  help="0 skips CIs. 500 is noticeably slower.")
        map_bins = st.slider("Commit-map bins", 16, 44, 28, 4)

    selected = sessions.copy()
    selected = selected[selected["animal"].isin(chosen_animals)]
    if qc_only:
        selected = selected[selected["include"]]
    ex = excluded_set()
    pairs = [(a, s) for a, s in zip(selected["animal"], selected["session"])
             if (a, s) not in ex]

    sel = P.Selection(animals=chosen_animals, sessions=pairs,
                      min_trials_per_level=min_per_level,
                      include_only_qc_pass=qc_only)
    ds = P.make_dataset(trials, paths, sessions, sel, root=root,
                        commit=commit, traces=traces)

    params = {"dur_trim_pct": trim, "dur_log": dur_log, "n_boot": n_boot,
              "map_bins": map_bins, "max_trials_drawn": 60,
              "family": family, "errorbar": errorbar}

    st.title("Mouse arena visual discrimination")
    if ds.choice.empty:
        st.warning("The current selection contains no choice trials. "
                   "Loosen the filters in the sidebar.")
        st.stop()
    st.caption(f"{len(ds.animals)} animals · {len(ds.sessions)} sessions "
               f"· {len(ds.choice)} choice trials · root `{root}`")

    tab_design, tab_analysis, tab_compare = st.tabs(
        ["Task design", "Analysis", "Cross-animal"])

    with tab_design:
        section("design", "design", ds, params)

    with tab_analysis:
        for _, sec_key, sec_label in [s for s in P.SECTIONS if s[0] == "analysis"]:
            st.header(sec_label)
            if sec_key == "path":
                opts = list(zip(ds.sessions["animal"], ds.sessions["session"]))
                if opts:
                    labels = {f"{a} — {s}": (a, s) for a, s in opts}
                    pick = st.selectbox("Session for the trajectory panel",
                                        list(labels), key="traj_pick")
                    params = {**params, "example_session": labels[pick]}
            section("analysis", sec_key, ds, params)

    with tab_compare:
        st.header("Cross-animal comparison")
        st.markdown("**Session grid** — untick a cell to drop that session from "
                    "every panel; the sidebar list stays in sync")
        grid_src = sessions[sessions["animal"].isin(chosen_animals)]
        if qc_only:
            grid_src = grid_src[grid_src["include"]]
        wide = grid_src.pivot(index="animal", columns="session_index",
                              values="session")
        # Explicit construction: True/False where the animal has that session,
        # None where it does not (rendered as an empty, untickable cell).
        # Columns are named strings, not integers: data_editor returns string
        # column labels, so an integer-keyed pivot cannot be looked up afterwards.
        colnames = [f"s{int(c)}" for c in wide.columns]
        cell_session = {}
        bool_wide = pd.DataFrame(index=wide.index, columns=colnames, dtype="object")
        for a in wide.index:
            for c, name in zip(wide.columns, colnames):
                s = wide.loc[a, c]
                if isinstance(s, str):
                    cell_session[(a, name)] = s
                    bool_wide.loc[a, name] = (a, s) not in ex
                else:
                    bool_wide.loc[a, name] = None
        edited_grid = st.data_editor(
            bool_wide, width="stretch", key="grid_editor",
            column_config={c: st.column_config.CheckboxColumn(c, width="small")
                           for c in bool_wide.columns})
        for (animal, name), s in cell_session.items():
            if name not in edited_grid.columns or animal not in edited_grid.index:
                continue
            val = edited_grid.loc[animal, name]
            if val is None or pd.isna(val):
                continue
            if bool(val):
                ex.discard((animal, s))
            else:
                ex.add((animal, s))
        st.caption("Values are session indices; hover the sidebar table for dates, "
                   "trial counts and QC flags.")
        section("compare", "compare", ds, params)

    with st.expander("Methods and conventions"):
        iti = float(pd.to_numeric(ds.trials["iti_s"], errors="coerce").dropna().median())
        st.markdown(f"""
**Stimulus axis.** 0° is a vertical grating and 90° horizontal (verified against
the task's fragment shader). The distractor is fixed at 0°; the rewarded target
carries the varying orientation. Difficulty is
`delta = |target − distractor|`, so at delta = 0 the two gratings are identical
and chance performance is structural, not a fitting artefact.

**Trial duration** is `extra.choice_latency_s`, the stimulus-onset-to-choice
interval. It is not called a reaction time because it contains the whole
locomotor traverse to the box. `t_end − t_start` is kept separately as
`trial_window_s` and adds the {iti:.3f} s inter-trial interval.

**Psychometric model.** `p(Δ) = 0.5 + (0.5 − λ) / [1 + exp(−β(Δ − μ))]`, fitted by
maximum binomial likelihood with the guess rate fixed at 0.5 and λ free. The
likelihood is shallow in β at realistic trial counts, so fits are flagged
unreliable (and drawn hollow) when μ falls outside the tested range, β sits at a
bound, the curve never crosses 75%, or fewer than 200 trials contribute.

**Commitment.** The earliest moment after which the smoothed decision variable
(heading error toward the rejected box minus toward the chosen box, 5-sample
smoothing, 5° dead zone) stays positive for the rest of the approach. The final
approach within 15 units of the box is excluded, where bearing is geometrically
unstable. The offset between single-run and changed-mind trials is partly
definitional: a reversal must precede the final commitment.

**Variance.** All error bars are SEM. Across-trial SEM within a panel unless the
caption says across animals.

**Inclusion.** At least {ad.MIN_LEVELS} orientation levels and
{ad.MIN_CHOICE_TRIALS} choice trials per session. Falls and omissions are
excluded from accuracy and duration but retained as outcome categories.
""")


if __name__ == "__main__":
    main()
