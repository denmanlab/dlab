# Mouse arena visual-discrimination dashboard

Analysis and interactive dashboard for the two-alternative orientation
discrimination task run in the VR arena (`vr_task_v0`, `varying_orientation`).

## Launch

Launch the dashboard by pointing it at a folder containing the task performance data for multiple mice. One option is to point it at the s1 location where data are stored. This means replace `~/behavior_data` in the below with `Volumes/s1/behavior/mouse_arena_discrimination/data` on a Mac or the s1 path on Windows (`\\denmanlab\s1\behavior....`).

For the below to work, you need to `cd` to the directory where this README and the `dashboard_app.py` are stored. 

```bash
pip install -r requirements.txt
streamlit run dashboard_app.py -- --data-root ~/behavior_data
```

The `--` is required: it separates Streamlit's own arguments from the app's.
The data root can also be set with the `ARENA_DATA_ROOT` environment variable.



### First load

The first start computes per-trial path and commitment metrics for every
session, roughly 3 s per session (~65 s for 23 sessions). Results are cached to
disk, so later starts are near-instant.

**Rescan data** (sidebar) re-reads the repository and picks up animals and
sessions recorded since the app started. Caching is keyed on each session
folder's modification time, so only new or re-recorded sessions are computed.
**Clear computed cache** discards everything and recomputes; it is only needed
if the analysis code changed.

## Layout

Three tabs: **Task design**, **Analysis**, **Cross-animal**. The Analysis tab
stacks five sections — Psychometric, Learning, Trial duration, Path analysis,
Within-trial dynamics — plus a session-selectable trajectory panel. Every panel
carries PNG (300 dpi), SVG (vector, editable text) and CSV download buttons;
the CSV holds exactly the data behind that panel.

Session selection works from either the sortable sidebar table (click a column
header to sort by date, trial count or accuracy at 70–90°) or the animal ×
session grid on the Cross-animal tab; both drive the same selection.

## Files

| file | role |
|---|---|
| `arena_discrim.py` | ingestion, trial table, path metrics, commitment analysis, psychometric fitting |
| `panels.py` | all 23 panels as pure `(fig, ds, params) -> Figure` functions in a registry, plus the Selection/Dataset layer |
| `dashboard_app.py` | Streamlit renderer over the registry |
| `preview.py` | standalone HTML renderer over the same registry, for review without a server |
| `make_synthetic_fixture.py` | synthetic repository with known psychometric parameters, for tests |

`panels.py` has no Streamlit dependency, so the same panel code backs both
renderers and anything settled in the preview appears unchanged in the app.

## Data conventions

Per session: `events.csv`, `samples.csv`, `session_meta.json`. Column sets were
uniform across all 23 audited sessions. See `schema_report.md` for the full data
dictionary and the verified field relationships.

**Stimulus axis.** 0° is a vertical grating, 90° horizontal — verified against
the task's fragment shader (`stimulus_manager.py`), where luminance modulates
along `u·cos(θ) + v·sin(θ)`. The distractor is fixed at 0°; the **rewarded
target carries the varying orientation** (`_prepare_vr_task_choice_stimuli`
assigns `vr_changing_orientation_target` to the correct box). Difficulty is
`delta = |target − distractor|` over 13 levels. At delta = 0 the two gratings are
identical, so chance performance there is structural.

**Trial duration** is `extra.choice_latency_s`, the stimulus-onset-to-choice
interval, stored as `trial_duration_s`. It is deliberately not called a reaction
time: it contains the whole locomotor traverse to the box. `t_end - t_start` is
kept as `trial_window_s` and adds the 1.000 s ITI (verified to within 7 ms).

**Heading.** Travel direction in the standard math convention is `heading + 90`;
heading 0 points along +y, straight ahead toward both boxes. Established
empirically against the sample traces (0.00° median error over 60 trials). The
choice boxes sit at ±12° from straight ahead at radius 99.

## Methods

**Psychometric fit.** Four function families, selectable in the sidebar, all of
the form `p(Δ) = 0.5 + (0.5 − λ) · F(Δ; μ, β)`:

| family | `F(Δ; μ, β)` | notes |
|---|---|---|
| logistic (default) | `1 / [1 + exp(−β(Δ − μ))]` | μ is the midpoint, β a per-degree slope |
| cumulative Gaussian | `Φ[β(Δ − μ)]` | the probit alternative |
| Gumbel | `1 − exp(−exp(β(Δ − μ)))` | asymmetric, rises slowly then steepens |
| Weibull | `1 − exp(−(Δ/μ)^β)` | μ is a scale in degrees, β a shape; defined for Δ ≥ 0 |

Add a family by extending `FAMILIES` in `arena_discrim.py` with a shape
function plus its parameter bounds and starting values — nothing else needs
changing; the fitter, the threshold solver and the panels all read the registry.

The **estimator** stays maximum binomial likelihood (multi-start L-BFGS-B) for
every family, and deliberately so: the data are counts of correct responses out
of *n* at each level, so the binomial likelihood is the matched estimator.
Least-squares on the proportions would weight a level with 14 trials the same as
one with 200. The guess rate is fixed at 0.5 rather than fitted because at
Δ = 0 the two gratings are physically identical, making chance a property of the
task; the lapse λ is free.

The 75% threshold is found numerically (bisection) rather than by inverting the
logistic analytically, so it is correct for every family and returns NaN when
the lapse puts the asymptote below 0.75.

`compare_families()` fits all four to the same counts and ranks them by AIC. On
the current dataset they are indistinguishable — thresholds of 24.0–24.2° and a
total AIC spread of 1.4 across all four — so the threshold estimate does not
depend on the choice.

The likelihood is shallow in β at realistic trial counts (profiling one animal
moved the negative log-likelihood by only 4 units across β from 0.05 to 0.30),
so fits are flagged unreliable and drawn hollow when μ falls outside the tested
range, β sits at a bound, the curve never reaches 75%, or fewer than 200 trials
contribute. Ground-truth recovery is checked against
`make_synthetic_fixture.py`: all three synthetic animals' true thresholds and
slopes fall inside the bootstrap 95% CIs.

**Commitment / decision point.** The earliest moment after which the smoothed
decision variable — |heading error toward the rejected box| minus |heading error
toward the chosen box|, 5-sample smoothing — stays above a 5° dead zone for the
rest of the approach. Two corrections are load-bearing and should not be
removed:

1. The final approach within 15 units of the box is excluded. Bearing to the box
   is geometrically unstable there (large angular swings for small movements),
   and without the mask the criterion fails on about half of all trials.
2. No fixed alignment threshold is used, and raw which-box-am-I-facing flips are
   not counted. The boxes are only 24° apart, so any robust threshold also
   admits the other box, and raw flips register on heading jitter — that version
   returned at least one reversal on every single trial.

With these, 95% of trials yield a commitment time and 69% show a single
committed run. Note the offset between single-run and changed-mind trials is
partly definitional: a reversal must precede the final commitment, so those
trials cannot commit early or near the centre.

TODO: this needs to be validated a bit deeper. maybe the algorithm for finding these points could be more robust. right now really only uses the straight shot to a box, and we know that there are grinding turns to the block too. 


**Variance.** A sidebar radio switches every panel between SEM and SD; captions
update with it, so an exported figure always states which is drawn. Across
trials within a panel unless the caption says across animals.

SEM shrinks as √n and describes how precisely the mean is known; SD describes
the spread of the trials themselves and does not shrink. One caveat for
proportions: the SD of a proportion is the per-trial Bernoulli `sqrt(p(1−p))`,
which sits near 0.5 whenever performance is near chance and barely varies with
difficulty, so SD mode is rarely informative on the psychometric panels even
though it is the correct quantity.

**Inclusion.** A session needs ≥ 5 orientation levels and ≥ 50 choice trials.
Of 23 sessions, 19 pass (4024 choice trials). Falls and omissions are excluded
from accuracy and duration but kept as outcome categories for path analysis.
Animal `jlh60` has one session with 2 choice trials and drops out entirely.
`bp2` contributes a single session, so its psychometric fit is not identifiable
and is marked as such.

## Adding an animal or session

Drop the session folder in under `<root>/<animal>/<YYYY-MM-DD_HH-MM-SS>/` and
press **Rescan data**. No code change is needed; a new animal appears in the
sidebar automatically. Sessions failing the inclusion rule are listed with their
QC flags and excluded by default — untick "QC-passing sessions only" to inspect
them.
