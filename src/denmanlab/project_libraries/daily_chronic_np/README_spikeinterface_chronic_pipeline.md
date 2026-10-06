# Chronic Neuropixels Pipeline Operations Reference

This is the detailed operations and parameter reference. Start with the concise
[project landing page](README.md) for installation, daily workflows, architecture,
failure recovery, and handoff instructions.

This repository runs a Windows-friendly SpikeInterface pipeline for Open Ephys
Neuropixels recordings. The primary workflow processes two independent probe streams
with SpikeInterface and Kilosort4, while the NWB layer also supports generic
recordings and externally generated trials/units tables.

The pipeline is intentionally conservative:

- raw Open Ephys files are never modified
- each probe is processed independently
- outputs are written beside each probe's `continuous.dat`
- the CSV registry is the source of processing state
- generated probe maps, hashes, sorter params, metrics, reports, and errors are saved

## Current Layout

Raw recordings live under:

```text
C:/Users/jordan/Desktop/mouse_arena_recordings/
```

For each probe stream, derived outputs are stored in the Open Ephys continuous folder:

```text
.../continuous/Neuropix-PXI-100.ProbeA/spikeinterface_output/
.../continuous/Neuropix-PXI-100.ProbeB/spikeinterface_output/
```

Smoke tests use a sibling folder:

```text
spikeinterface_output_smoke_10s/
```

The shared processing registry is configured by `project.registry_csv`. The analysis
configuration normally points it at Synology so the acquisition and analysis machines
see the same status. A local registry can still be used for development:

```text
registry/sessions.csv
```

Discovery writes one row per probe stream. Each row includes `mouse_id`,
`recording_date`, `task`, and `probe_label`; for example
`jlh602026-06-29_14-31-15` is parsed as `mouse_id=jlh60`,
`recording_date=2026-06-29`, and `task=14-31-15`. The longer `session_id`
remains unique for command-line targeting.

## Analysis GUI And Queue

The web GUI is the primary analysis-machine interface. Launch it from this repository:

```powershell
conda activate spikeinterface
python -m pipeline_gui.app --config pipeline/config_analysis.yaml
```

Then open `http://127.0.0.1:8765`.

The Queue page is recording-first. One row represents one Open Ephys recording block,
with ProbeA and ProbeB nested beneath it. `Queue All Eligible` runs blocks sequentially:

1. Stage one acquisition session from Synology into local analysis storage.
2. Process the selected recording block's probes sequentially.
3. Back up generated outputs to Synology.
4. Archive the complete acquisition session to the configured local archive drive.
5. Continue to the next queued recording block.

The archive step is deferred when another queued block shares the same acquisition
session, so that session is copied to the archive drive once after its final block.
Automatic deletion of staged local data remains disabled.

Queue state is persistent in `run_state/pipeline_queue.sqlite3`. Closing the browser
does not cancel work. `Pause After Probe` lets the active probe finish before stopping
the queue. The Activity tab shows durable queue events, job logs, and the current
phase. A stopped process is reconciled as interrupted instead of remaining permanently
marked as running.

Processing options are saved as named profiles under `pipeline/profiles/`. A profile is
resolved and snapshotted when each probe starts. Editing a profile changes only probes
that have not started; the active probe continues with its saved config and source
snapshot under `run_state/jobs/<job_id>/`.

Short recording blocks are excluded by the configured minimum duration. Blocks from
later experiments that look like brief survey recordings are labeled as survey/short
and ignored by `Queue All Eligible`; they can only be queued with the explicit
`Queue Anyway` action.

Probe recovery is artifact-aware:

- intact Kilosort output plus failed/incomplete QC: restart QC from Kilosort
- missing or incomplete Kilosort output: archive partial outputs and restart the probe
- completed probes: preserve and skip

Preprocessing and Kilosort remain one executable boundary because Kilosort consumes the
lazy SpikeInterface preprocessing graph. QC, UnitRefine, Phy, reports, and NWB remain
separately recoverable downstream stages.

The PowerShell menu remains available as a secondary terminal interface:

```powershell
.\run_pipeline.ps1
```

Useful direct commands are still available after activating the environment:

```powershell
conda activate spikeinterface
python -m pipeline.status --config pipeline/config_analysis.yaml
python -m pipeline.discover --config pipeline/config_analysis.yaml
python -m pipeline.run_pending --config pipeline/config_analysis.yaml
python -m pipeline.run_one --config pipeline/config_analysis.yaml --session-id SESSION_ID
```

The GUI keeps the Python pipeline modules as the processing source of truth. It also
provides per-probe detail, NWB generation at the recording level, reports, settings,
storage state, and guarded recovery actions. It does not delete raw or derived data.

## Current Processing Flow

For each registered probe row:

1. Verify the raw recording is backed up to `\\denmanlab\s2\mouse_arena_recordings`.
2. Load the Open Ephys stream with SpikeInterface.
3. Attach or generate an explicit ProbeInterface JSON from Open Ephys metadata.
4. Validate 384 recording channels against 384 probe contacts.
5. Compute probe, channel, geometry, and site hashes.
6. Apply SpikeInterface `phase_shift()` for Neuropixels inter-sample timing offsets.
7. Apply SpikeInterface median CAR with `common_reference(reference="global", operator="median")`.
8. Apply SpikeInterface `highpass_filter(freq_min=300)`.
9. Detect potential bad channels with `detect_bad_channels(method="coherence+psd")`.
10. Save a short pre/post CAR trace plot with matched pre/post voltage scales.
11. Optionally apply SpikeInterface motion correction if enabled in config; default is off.
12. Run Kilosort4 through SpikeInterface.
13. Build a `SortingAnalyzer` and compute analyzer extensions/metrics.
14. Run SpikeInterface UnitRefine labels if enabled, writing advisory unit labels under `unitrefine/<label_set>/`.
15. Export CSVs, export Phy, write HTML/PNG reports, and save `qc_timings.json`.

The preprocessing chain is lazy. The pipeline does not save a full preprocessed binary
copy. Kilosort4 may create a temporary staging binary during sorting, configured to be
deleted afterward.

SpikeInterface preprocessors are not in-place operations. The code assigns each returned
recording object into the next step, so the phase-shift output feeds CAR, the CAR output
feeds high-pass filtering, and the high-pass output feeds Kilosort4.

The generated ProbeInterface JSON in step 3 is for reproducibility and validation. It
stores the exact channel/contact geometry used for sorting and hashing under the output
metadata folder; it does not modify the raw Open Ephys files.

## Kilosort4 Policy

The default sorter policy is deliberate:

```yaml
do_CAR: false
do_correction: true
skip_kilosort_preprocessing: false
save_preprocessed_copy: false
delete_recording_dat: true
nblocks: 1
```

Median CAR is done in SpikeInterface before sorting, so Kilosort4 CAR is disabled.
Kilosort4 drift correction remains enabled by default. SpikeInterface motion correction
is disabled by default.

Detected bad channels are not excluded from Kilosort4 by default. Detection is used for
QC/reporting only and is written to `channel_qc.csv` with SpikeInterface labels such as
`good`, `dead`, `noise`, or `out`. To intentionally exclude detected channels from
sorting, set:

```yaml
preprocessing:
  bad_channels:
    exclude_from_sorting: true
    exclude_labels: ["dead", "noise", "out"]
```

## Main Outputs

Each `spikeinterface_output/` folder contains:

```text
manifest.json
summary.json
channel_qc.csv
quality_metrics.csv
template_metrics.csv
unit_summary.csv
qc_timings.json
unitrefine/<label_set>/unit_labels.csv
unitrefine/<label_set>/summary.json
kilosort4/
analyzer/
phy/
metadata/*_probeinterface.json
report/summary.html
report/summary.png
report/figures/pre_post_car_traces.png
report/figures/probe_geometry.png
report/figures/quality_metric_histograms.png
report/figures/sorting_summary.png
qc_variants/<variant_name>/
```

`report/summary.png` is a self-contained, shareable per-probe QC sheet intended for
Slack/email. It summarizes probe geometry, bad-channel labels, pre/post CAR snippets
when available, quality metric distributions, unit locations, example templates, and
the key preprocessing/sorter policy choices. When UnitRefine labels are present, the
HTML and PNG summaries include label counts and label-colored QC scatter plots.

## Config Notes

Use `pipeline/config_analysis.yaml` on the analysis computer and
`pipeline/config_acquisition.yaml` on the acquisition computer. The
`pipeline/config.yaml` file is retained for local development. In the commands below,
replace the config path when operating against the shared production registry.

Important options:

- `project.output_layout: recording_local_probe_folder`
- `project.recording_local_output_name: spikeinterface_output`
- `backup.require_before_processing: true`
- `preprocessing.phase_shift.enabled: true`
- `preprocessing.car.enabled: true`
- `preprocessing.car.operator: median`
- `preprocessing.highpass_filter.enabled: true`
- `preprocessing.highpass_filter.freq_min: 300`
- `preprocessing.bad_channels.enabled: true`
- `preprocessing.bad_channels.exclude_from_sorting: false`
- `qc.automated_labels.enabled: true`
- `qc.automated_labels.method: unitrefine`
- `qc.automated_labels.label_set: unitrefine_full`
- `preprocessing.motion_correction.enabled: false`
- `sorting.params.do_CAR: false`
- `sorting.params.do_correction: true`
- `sorting.params.delete_recording_dat: true`
- `qc.sorting_analyzer.sparse: true`
- `qc.sorting_analyzer.num_spikes_for_sparsity: 200`
- `qc.phy_compute_pc_features: false`
- `qc.phy_compute_amplitudes: true`
- `qc.phy_add_unitrefine_labels: true`
- `qc.phy_add_quality_metrics: false`
- `qc.phy_add_template_metrics: false`
- `qc.phy_prune_metric_tsvs: true`
- `qc.phy_dat_path.enabled: true`
- `qc.phy_dat_path.source: original_continuous_dat`
- `qc.phy_dat_path.hp_filtered: false`
- `qc.analyzer_extensions`: ordered editable analyzer extension list with explicit `enabled: true/false`
- `qc.quality_metrics.metric_names`: high-yield quality metrics emphasized in reports
- `qc.quality_metrics.unitrefine_metric_names`: broader quality metric feature groups required by UnitRefine classifiers
- `qc.shareable_png.enabled: true`
- `jobs.n_jobs: 8`
- `jobs.fallback_n_jobs: 2`
- `jobs.chunk_duration: 1s`
- `jobs.progress_bar: true`

The `jobs` settings are passed into SortingAnalyzer extension computes and Phy
export, including waveform/noise/spike-location/amplitude/PC feature work when
the installed SpikeInterface extension supports parallel jobs. If an extension
rejects a job keyword, the pipeline logs that fallback explicitly instead of
silently dropping parallel settings.

If a SortingAnalyzer extension fails with a Windows multiprocessing spawn/pickle
error, the pipeline deletes the partial extension, retries that extension with
`jobs.fallback_n_jobs`, and uses that fallback worker count for the remaining
extensions in that QC pass. If the fallback worker count also hits the same
spawn failure, the extension is retried serially.

The base Phy export intentionally skips Phy `pc_features.npy` by default because
that all-spike export can be very slow. This does not disable the
`principal_components` analyzer extension used by QC/UnitRefine; it only skips the
extra Phy-format PC feature file. Set `qc.phy_compute_pc_features: true` when you
want the Phy feature view populated.

After Phy export, the pipeline patches `phy/params.py` so `dat_path` points to the
probe's original Open Ephys `continuous.dat`. This lets Phy load waveform traces
without preserving Kilosort's temporary binary. Because that original file is not the
exact SpikeInterface-preprocessed sorting input, the base config writes
`hp_filtered = False` in `params.py` by default.

The base Phy export keeps Phy focused: it does not add quality metric or template
metric columns to Phy by default. UnitRefine labels/probabilities are the intended
extra curation columns. The CSV metrics remain available in `quality_metrics.csv`,
`template_metrics.csv`, and the HTML/PNG reports.

## Analyzer Outputs: What They Are For

- `spike_amplitudes`: amplitude distributions, amplitude cutoff, median amplitude,
  report plots, and Phy `amplitudes.npy`.
- `principal_components`: PCA-based QC features, UnitRefine inputs, and optional Phy
  `pc_features.npy` export.
- `spike_locations`: per-spike position/depth diagnostics and drift-related
  summaries used by richer QC/UnitRefine features.
- `unit_locations`: cheaper per-unit location/depth summaries used in reports.
- Phy `pc_features.npy`: enables Phy's PC feature/cluster view. Phy can still open
  without it, but that specific feature view is unavailable.

The exact high-yield quality metric names are configured under
`qc.quality_metrics.metric_names`. The current defaults are firing rate, presence
ratio, SNR, ISI violations, refractory-period contamination, sliding RP violation,
amplitude cutoff, and median amplitude.

When `qc.automated_labels.enabled` is true, the pipeline computes the union of
`qc.quality_metrics.metric_names` and `qc.quality_metrics.unitrefine_metric_names`.
This keeps the report-facing list readable while still giving UnitRefine the broader
feature matrix it expects, including synchrony, amplitude CV, drift, Mahalanobis/
isolation, d-prime, nearest-neighbor, and silhouette metric groups. These additional
columns remain available in `quality_metrics.csv`.

`project.processed_root` is retained for legacy central-output layouts and migration
support, but the current default writes recording-local outputs.

The supplied configs use `project.sorting_analyzer_folder_name: analyzer`. The
short name prevents SpikeInterface PCA model files from crossing the legacy
Windows 260-character path limit in recordings with longer names. Existing
`sorting_analyzer/` outputs remain readable.

## Two-Machine Workflow

The intended mouse-arena deployment is split by machine role.

### Analysis machine

This machine does the heavy work. It discovers completed recordings on Synology,
stages one or more recordings onto local fast storage, runs preprocessing/Kilosort/QC,
then backs generated outputs up to Synology and optionally to a second local archive
root.

Use the analysis config profile as the starting point:

```powershell
python -m pipeline.discover --config pipeline/config_analysis.yaml
python -m pipeline.stage_recording --config pipeline/config_analysis.yaml --all-pending --update-registry
python -m pipeline.run_pending --config pipeline/config_analysis.yaml
python -m pipeline.derived_backup --config pipeline/config_analysis.yaml --all --update-registry
```

Key paths:

```yaml
machine.role: analysis
staging.server_raw_root: "\\denmanlab\s2\mouse_arena_recordings"
staging.local_raw_root: "C:/Users/jordan/Desktop/mouse_arena_recordings"
backup.local_recording_archive.root: "D:/mouse_arena_recordings_backup"
backup.derived_outputs.local_archive_root: "D:/mouse_arena_processed_backup"
project.registry_csv: "\\denmanlab\s2\mouse_arena_recordings\_pipeline_registry\sessions.csv"
mouse_arena_nwb.backup_root: "\\denmanlab\s2\nwbs\mouse_arena"
```

When `staging.enabled: true`, discovery scans `staging.server_raw_root` but writes
registry rows whose `raw_folder` and processing outputs point to the local staged copy.
The original Synology path is retained in `server_raw_folder`.
`run_pending` processes the backlog recording-by-recording: it stages one recording
from Synology, runs all runnable probe rows for that recording, backs generated outputs
up, archives the full local recording folder to D:, then moves to the next recording.
Single-probe `run_one` actions also stage that probe's parent recording locally before
processing, so GUI one-probe runs use local storage rather than sorting from Synology.
The full D: archive includes raw files such as `continuous.dat` plus generated outputs
and is written under `D:/mouse_arena_recordings_backup`. A smaller generated-output-only
archive is also written under `D:/mouse_arena_processed_backup`. NWB files are copied
into the dedicated NWB archive under `\\denmanlab\s2\nwbs\mouse_arena`, organized by
mouse and recording date.

The analysis config enables guarded local cleanup with
`staging.cleanup_after_verified_backups: true`. Before deleting a staged recording,
the cleanup step freshly compares every local file against both the Synology recording
folder and the full `D:/mouse_arena_recordings_backup` copy by relative path and size.
It refuses cleanup for active, failed, or incomplete recordings and records the result
as `local_cleanup_status: offloaded` in the shared registry.

Assess existing recordings without deleting anything:

```powershell
python -m pipeline.local_cleanup --config pipeline/config_analysis.yaml --all-eligible
```

After inspecting that output, perform the verified cleanup and update the registry:

```powershell
python -m pipeline.local_cleanup --config pipeline/config_analysis.yaml --all-eligible --delete-local --update-registry
```

The default and acquisition configs keep automatic cleanup disabled. An offloaded
recording can be staged from Synology again if it needs reprocessing. The cleanup does
not touch the Synology copy, the full D: archive, or the generated-output-only D:
archive.

### Acquisition machine

The acquisition machine should stay lightweight: scan local Open Ephys recordings,
copy raw data to Synology, and show status from the shared registry. It should not run
Kilosort/QC/NWB while recording.

Use the acquisition config profile as the starting point:

```powershell
python -m pipeline.discover --config pipeline/config_acquisition.yaml
python -m pipeline.raw_backup --config pipeline/config_acquisition.yaml --all --update-registry
python -m pipeline.status --config pipeline/config_acquisition.yaml
```

In the GUI, `machine.role: acquisition` hides the heavy processing actions and keeps
status, scan, and raw-backup controls visible.
## Completion And Backup

Discovery prefers a `recording_complete.txt` marker. For test data, the config also
allows an inactivity fallback after 60 minutes if the recording duration is long enough.

Before processing, the pipeline verifies that the raw recording exists on the server:

```text
\\denmanlab\s2\mouse_arena_recordings
```

Derived `spikeinterface_output*` folders are ignored during backup verification.

Generated outputs can be copied to the matching recording folder on the backup server
after processing:

```powershell
python -m pipeline.derived_backup --config pipeline/config.yaml --all --update-registry
python -m pipeline.derived_backup --config pipeline/config.yaml --session-id SESSION_ID --update-registry
```

This copies only configured derived-output folders, currently `spikeinterface_output*`
and `nwb`, into the same relative location under
`\\denmanlab\s2\mouse_arena_recordings`. It uses update-only copying: files that are
already present with matching size and modification time are skipped. It does not copy
or rewrite raw acquisition files.

The GUI exposes this as **Backup Generated Outputs**. Mouse-arena NWB export also runs
this backup automatically when `backup.derived_outputs.auto_after_nwb_export: true`.

## Reports And Status

List registry state:

```powershell
python -m pipeline.status --config pipeline/config.yaml
```

List reports:

```powershell
python -m pipeline.reports --config pipeline/config.yaml
```

Open the newest report:

```powershell
python -m pipeline.reports --config pipeline/config.yaml --open
```

Backfill shareable PNGs for completed outputs:

```powershell
python -m pipeline.reports --config pipeline/config.yaml --write-summary-png
```

Regenerate pre/post CAR trace panels while backfilling, if raw data is available:

```powershell
python -m pipeline.reports --config pipeline/config.yaml --write-summary-png --include-preprocessing-traces
```

Resume QC from an existing sorter/analyzer output after a failed analyzer/report step:

```powershell
python -m pipeline.resume_qc --config pipeline/config.yaml --session-id SESSION_ID --recompute-extension spike_locations --skip-phy
```

This does not rerun Kilosort. It loads the existing analyzer folder, recomputes
any explicitly named damaged extension, skips already-complete analyzer extensions,
then continues metrics, reports, UnitRefine, and optionally Phy export. In the GUI,
use **Resume QC** for the selected session; it defaults to recomputing
`spike_locations` and skipping Phy export.

Restart QC from the existing Kilosort output when the analyzer folder is damaged or
you want all QC products rebuilt:

```powershell
python -m pipeline.restart_qc --config pipeline/config.yaml --session-id SESSION_ID --skip-phy
```

This does not rerun Kilosort. It archives `analyzer` (or legacy
`sorting_analyzer`), metric CSVs,
`unitrefine`, `phy`, `summary.json`, `manifest.json`, and `report` under
`spikeinterface_output/qc_archive/<timestamp>/`, then rebuilds the analyzer, metrics,
UnitRefine labels, and reports. It leaves raw data and `kilosort4/` untouched.

Export Phy post-hoc from an existing analyzer:

```powershell
python -m pipeline.export_phy --config pipeline/config.yaml --session-id SESSION_ID
```

This attaches the latest UnitRefine labels as Phy properties when available. It refuses
to overwrite an existing `phy/` folder unless `--overwrite-phy` is supplied.

Before calling SpikeInterface `export_to_phy`, the pipeline checks whether the
analyzer already has the extensions needed by the configured Phy export.
Existing extensions are reused. Missing extensions are computed once with the configured
job settings and multiprocessing fallback policy. With the base config,
`amplitudes.npy` is exported, `pc_features.npy` is skipped, and `dat_path` is patched
to the original `continuous.dat`. Metric TSV clutter is pruned from Phy unless
`qc.phy_add_quality_metrics` or `qc.phy_add_template_metrics` is enabled.

## NWB Export

NWB export is a recording-level optional step. Its generic layer combines units from
registered probes, optional trials, all detected NI-DAQ digital edges, subject/session
metadata, and provenance under:

```text
.../Record Node 101/experiment1/recording1/nwb/
```

The GUI determines a protocol from the registry task/session path. Names containing
`visual_stim`, `visualstim`, or `vstim` are classified as `visual_stim`; other
recordings currently default to `mouse_arena`. Protocol selection controls trial
interpretation, not digital-event retention:

- Every nonzero digital state is loaded.
- Rising and falling edges are shown separately with counts and timing summaries.
- Lines can be named per recording in the NWB tab.
- Unnamed lines remain included as `line_N`.
- The installed PyNWB version stores each named line/edge as a timestamped NWB
  `TimeSeries`, and the row-per-edge source table is retained in
  `nwb_digital_events.csv`.

Generic and visual-stim recordings do not require an automatic trials table. An
external trials CSV can be selected in the GUI when a protocol-specific dataframe is
available. External units CSV loading is likewise independent of the automated sorting
pipeline, which allows the NWB layer to be used for recordings analyzed elsewhere.

Protocol and shared digital-event defaults are explicit:

```yaml
nwb:
  default_protocol: auto
  protocol_rules:
  - match: '(visual[_-]?stim|visualstim|vstim)'
    protocol: visual_stim
  digital_events:
    include_edges: [rising, falling]
    line_names:
      '4': reward
      '5': trial_start
      '6': game_frame
```

Names edited in the GUI are recording-specific and saved to
`nwb/digital_line_names.json`; they do not change the global defaults.

### Mouse Arena Adapter

For mouse-arena recordings, the optional adapter combines the generic ephys data with
behavior task tables. Before writing NWB, build and inspect the session-level
behavior/ephys event dataframe:

```powershell
python -m pipeline.mouse_arena_session --config pipeline/config.yaml --session-id SESSION_ID
```

This writes:

```text
session_events.csv
session_events_manifest.json
nwb_digital_events.csv
```

`session_events.csv` includes every row from the behavior `events.csv`, expanded JSON
metadata, and ephys-time columns. Ephys TTLs are used as ground truth for trial starts
and rewards when present; behavior-only rows such as fall/timeout/session markers are
placed on the ephys clock with the trial-start sync fit.

Run it from any ProbeA or ProbeB registry row for that recording:

```powershell
python -m pipeline.mouse_arena_nwb --config pipeline/config.yaml --session-id SESSION_ID
```

The exporter writes inspectable derived tables next to the NWB:

```text
nwb_units.csv
nwb_trials.csv
nwb_digital_events.csv
behavior_events_aligned.csv
nwb_manifest.json
nwb_source_resolution.json
```

NWB generation has its own source resolver. This is intentionally limited to the NWB
path and is not used by preprocessing, Kilosort, or QC. Preprocessing still requires
the recording to be staged locally. For NWB export only, the resolver checks:

1. the registry/local recording path,
2. the full D: recording archive,
3. generated-output archives such as `D:/mouse_arena_processed_backup`,
4. the Synology raw recording folder as a last resort.

Every candidate path checked, and the selected path for each raw/probe source, is
written to `nwb_source_resolution.json` and included in `nwb_manifest.json`.

When `mouse_arena_nwb.backup_root` is set, successful NWB export also copies the
`.nwb` file to that backup root. The registry records `behavior_session_folder`,
`behavior_events_csv`, `nwb_backup_status`, `nwb_backup_path`, and any
`nwb_backup_error` for both probe rows in the recording. The behavior-session path is
also included in `nwb_manifest.json` and embedded in the NWB as pipeline provenance
metadata when supported by the installed PyNWB version.

Default mouse-arena interpretation:

```yaml
mouse_arena_nwb:
  behavior:
    root: "\\\\denmanlab\\s1\\behavior\\mouse_arena\\data"
  digital_events:
    trial_starts_line: 5
    rewards_line: 4
    frames_line: 6
```

The exporter finds the nearest same-date behavior session folder for the mouse, reads
`events.csv` plus `sync_events.csv`, and fits behavior time onto the ephys clock. The
current pipeline default uses the dense game-frame sync events for clock alignment,
with trial-start alignment retained as a fallback/diagnostic path in the code.
Behavioral `trial` rows become the NWB trials table;
all TTL trial/reward/frame edges are also preserved in `nwb_digital_events.csv` and
as NWB event time series. Probe units get unique ids such as `A_12` and `B_12`,
while retaining source unit ids, Kilosort labels, UnitRefine labels, quality metrics,
and template metrics.

Ephys timestamps are treated as ground truth. Trial starts, rewards, and frames use
NI-DAQ TTL timestamps when present. The linear behavior-to-ephys fit is used to place
behavior-only events, such as falls or non-rewarded choices, onto the ephys timeline
and as a fallback if a specific TTL event is missing.

The alignment method is configurable:

```yaml
mouse_arena_nwb:
  clock_alignment:
    method: "game_frame_clock_fit"
    seed_method: "game_frame_clock_fit"
    behavior_sync_event: "game_frame"
    ephys_event: "game_frame"
    max_lag_seconds: 0.025
```

The GUI can use externally generated trials/units CSVs for any NWB protocol. The
chosen files are copied into:

```text
.../recording1/nwb/imported_csv/
```

and the active choices are recorded in:

```text
.../recording1/nwb/input_overrides.json
```

Use the NWB tab's clear buttons to return to the default recording-local
`nwb_trials.csv`/`nwb_units.csv` files. For external units CSVs, include either
`unique_unit_id`, `probe_label` + `source_unit_id`, or a parseable `spike_times`/`times`
column if the exported NWB should contain unit spike times. Otherwise the unit rows can
still be written, but their spike times may be empty.

Subject metadata is stored per mouse so repeated recording days reuse the same defaults:

```yaml
mouse_arena_nwb:
  subjects:
    jlh60:
      subject_id: jlh60
      date_of_birth: "YYYY-MM-DD"
      sex: U
      strain: C57/B6
      species: Mus musculus
```

When `date_of_birth` is set, the GUI/backend computes NWB `age` from the selected
recording date and saves the derived value back into the mouse-specific subject block.

Analog NI-DAQ packaging is controlled by `mouse_arena_nwb.analog.include`. It is off
by default because embedding the full 8-channel 30 kHz analog stream can make the NWB
large and slow to write. Set it to `true` when the raw analog traces should be included.

`pynwb` is required for the actual `.nwb` file. Use `--tables-only` to write the
derived CSVs/manifest without requiring or writing NWB.

## QC Timing And Variants

Every normal QC run, resume, restart, Phy export, and QC variant writes:

```text
spikeinterface_output/qc_timings.json
```

The timing file records major phases and analyzer extensions, including elapsed time,
status, job kwargs, fallback-worker retries, skipped/reused extensions, and failures.
The GUI selected-probe view shows the latest timing events.

Run a QC-only comparison from an existing Kilosort output:

```powershell
python -m pipeline.qc_variant --config pipeline/config.yaml --session-id SESSION_ID --variant-name fast_no_phy_pcs --no-phy-compute-pc-features
```

QC variants write under:

```text
spikeinterface_output/qc_variants/<variant_name>/
```

Each variant has its own `analyzer/`, metrics CSVs, UnitRefine outputs, report,
optional Phy folder, `qc_params.yaml`, and `qc_timings.json`. Variants do not mutate the
base Kilosort or base QC outputs, and they refuse to overwrite an existing variant unless
`--overwrite-variant` is supplied.

## UnitRefine Labels

The normal pipeline runs SpikeInterface UnitRefine after the `SortingAnalyzer`
quality and template metrics are computed. The analyzer now computes
`spike_locations` and `principal_components` before quality metrics so UnitRefine has
the PCA/isolation/drift features its classifiers expect. Labels are advisory QC
metadata; they do not remove units or mutate Kilosort4 output.

Default outputs:

```text
spikeinterface_output/unitrefine/unitrefine_full/unit_labels.csv
spikeinterface_output/unitrefine/unitrefine_full/summary.json
```

The default classifiers are the SpikeInterface Hugging Face models:

```yaml
qc:
  automated_labels:
    enabled: true
    method: "unitrefine"
    label_set: "unitrefine_full"
    noise_neural_classifier: "SpikeInterface/UnitRefine_noise_neural_classifier"
    sua_mua_classifier: "SpikeInterface/UnitRefine_sua_mua_classifier"
    fail_on_error: false
```

The first UnitRefine run may need internet access to download/cache those models.
If UnitRefine fails and `fail_on_error` is `false`, sorting and metric export are
kept and the UnitRefine error is recorded in its `summary.json` plus the run
`summary.json`. Older analyzer outputs created before `spike_locations`,
`principal_components`, or the expanded `unitrefine_metric_names` were added may fail
UnitRefine because the required feature columns are absent; new full/restart QC runs
include them.

Run UnitRefine post-hoc for one registered output:

```powershell
python -m pipeline.unitrefine --config pipeline/config.yaml --session-id SESSION_ID --label-set unitrefine_full
```

Run post-hoc against a specific output folder with alternate classifiers:

```powershell
python -m pipeline.unitrefine --config pipeline/config.yaml --processed-folder PATH --label-set custom_v1 --noise-neural-classifier MODEL_OR_PATH --sua-mua-classifier MODEL_OR_PATH
```

Run on every complete registry row:

```powershell
python -m pipeline.unitrefine --config pipeline/config.yaml --all-complete --label-set unitrefine_full
```

Existing label-set folders are protected. Add `--overwrite-label-set` only when you
intentionally want to replace that label experiment.

## Reruns

Completed rows are skipped by default. If outputs already exist, the pipeline refuses to
overwrite them unless `--force` is supplied. Use force carefully, because the current
recording-local output folder is the canonical output location for that probe/run.

For preprocessing/sorting comparisons, prefer a named output variant instead of
deleting or forcing over the old run:

```powershell
python -m pipeline.run_one --config pipeline/config.yaml --session-id jlh60_2026-06-29_14-31-15_ProbeA --output-suffix badchans_kept_hp300
python -m pipeline.run_one --config pipeline/config.yaml --session-id jlh60_2026-06-29_14-31-15_ProbeB --output-suffix badchans_kept_hp300
```

This creates sibling folders such as:

```text
spikeinterface_output_badchans_kept_hp300/
```

To rerun all completed base rows into a new variant:

```powershell
python -m pipeline.run_pending --config pipeline/config.yaml --rerun-complete --output-suffix badchans_kept_hp300
```

Only canonical completed base rows are selected for this command. Existing smoke runs
and previous output variants are skipped so variants do not recursively generate more
variants.

Smoke tests create separate `_smoke_*s` folders and do not overwrite the full run.

## Development Checks

Run the test suite in the activated conda environment:

```powershell
conda activate spikeinterface
python -m pytest -q
python -m compileall -q pipeline pipeline_gui tests
```

Current tests cover registry updates, hashes, probe/channel mismatch failures, backup
verification, Kilosort4 parameter policy, preprocessing trace plot generation,
shareable PNG generation, and UnitRefine export/failure/overwrite behavior.

