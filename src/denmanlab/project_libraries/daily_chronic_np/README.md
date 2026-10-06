# Denman Lab Chronic Neuropixels Pipeline

Windows-first processing and data-packaging workflow for chronic Open Ephys
Neuropixels recordings. It coordinates raw-data backup, local staging,
SpikeInterface preprocessing, Kilosort4, SortingAnalyzer QC, UnitRefine, Phy export,
reports, and optional NWB generation.

The browser GUI is the primary interface. Python command-line modules remain the
processing source of truth and can be used independently.

## At A Glance

- Raw Open Ephys files are treated as immutable.
- ProbeA and ProbeB are sorted independently.
- Recordings are queued and transported as recording-level units.
- Only one heavy probe job runs at a time.
- Processing happens on local storage; Synology is the shared source and backup.
- Outputs are written beside each probe's `continuous.dat`.
- A shared CSV registry exposes status to acquisition and analysis computers.
- Every run preserves parameters, hashes, logs, timings, metrics, and reports.
- Failed QC can restart from intact Kilosort output.
- NWB export combines probes at the recording level and accepts external CSV inputs.

For parameter details, recovery commands, output descriptions, and NWB internals, see
the [operations reference](README_spikeinterface_chronic_pipeline.md).

## System Layout

```text
Acquisition computer
  local Open Ephys recordings
            |
            | raw backup
            v
Synology
  mouse_arena_recordings/             raw source of truth
  mouse_arena_recordings/_pipeline_registry/sessions.csv
  nwbs/mouse_arena/                    dedicated NWB archive
            |
            | stage one recording
            v
Analysis computer
  local working copy
  SpikeInterface -> Kilosort4 -> QC -> UnitRefine -> Phy/reports
            |
            +--> Synology generated-output backup
            +--> D: full-recording and processed-output archives
```

| Computer | Config | Responsibility |
|---|---|---|
| Acquisition | `pipeline/config_acquisition.yaml` | Discover completed local recordings, copy raw data to Synology, and display shared status |
| Analysis | `pipeline/config_analysis.yaml` | Stage recordings, process probes, generate reports/NWB, and archive outputs |
| Local development | `pipeline/config.yaml` | Development or single-machine work with a local registry |

The acquisition GUI intentionally hides heavy processing actions. Both computers read
and update the same Synology registry.

## Installation

### Requirements

- Windows 10/11
- Anaconda or Miniconda
- Git
- Access to the configured Denman Lab network shares
- NVIDIA GPU and compatible driver on the analysis computer

The supplied environment targets Python 3.11, PyTorch 2.6 with CUDA 12.4,
SpikeInterface 0.104.7, Kilosort 4.1.7, FastAPI, PyNWB, and Open Ephys Python tools.

```powershell
git clone https://github.com/denmanlab/daily_chronic_np.git
cd daily_chronic_np

conda env create -f environment_spikeinterface.yml
conda activate spikeinterface
python -m pytest -q
```

If the environment already exists:

```powershell
conda env update -n spikeinterface -f environment_spikeinterface.yml --prune
```

Before first use, review the machine paths in the selected config and verify network
access:

```powershell
Test-Path "\\denmanlab\s2\mouse_arena_recordings"
Test-Path "\\denmanlab\s2\mouse_arena_recordings\_pipeline_registry\sessions.csv"
Test-Path "\\denmanlab\s1\behavior\mouse_arena\data"
```

## Launch

Run these commands from the repository root in the `spikeinterface` environment.

Analysis computer:

```powershell
python -m pipeline_gui.app --config pipeline/config_analysis.yaml
```

Acquisition computer:

```powershell
python -m pipeline_gui.app --config pipeline/config_acquisition.yaml
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The GUI binds to localhost by default and has no authentication. Do not expose it to
the network without adding an authenticated deployment layer.

## Daily Workflow

### Acquisition

1. Finish and close the Open Ephys recording.
2. Preferably create `recording_complete.txt` in the session or recording folder.
3. Open the acquisition GUI and select **Scan Recordings**.
4. Back up the new recording to Synology.
5. Confirm the shared registry reports a verified raw backup.

Discovery can use a 60-minute inactivity fallback when the completion marker is
missing. Recordings shorter than the configured minimum duration are excluded from the
normal queue. Brief survey blocks are also excluded unless explicitly queued.

### Analysis

1. Open the analysis GUI.
2. Refresh or scan the shared registry.
3. Select a processing profile.
4. Queue one recording or select **Queue All Eligible**.
5. Monitor the **Activity** tab for phase logs, elapsed time, and system resources.
6. Review the per-probe HTML report, shareable summary PNG, metrics, and Phy output.
7. Build the recording-level NWB when units and event tables are ready.

For each queued recording, the persistent worker:

1. Stages the recording from Synology to local analysis storage.
2. Processes its probes sequentially.
3. Backs up generated outputs to Synology.
4. Archives the full recording and generated outputs to the configured local drive.
5. Continues to the next recording.

Closing the browser does not stop the queue. Queue state and logs live under
`run_state/` and `run_logs/`. Automatic deletion of staged recordings is disabled.

## Default Probe Processing

The current default profile applies:

1. Backup verification.
2. Open Ephys loading and ProbeInterface geometry validation.
3. Neuropixels `phase_shift()`.
4. Global median common reference.
5. High-pass filtering at 300 Hz.
6. `coherence+psd` bad-channel detection for QC; channels are not excluded by default.
7. Kilosort4 with its CAR disabled and drift correction enabled.
8. Sparse `SortingAnalyzer` creation and configured extensions/metrics.
9. UnitRefine advisory labels.
10. Phy export, HTML/PNG reports, timing logs, and generated-output backup.

SpikeInterface preprocessing is lazy. Kilosort may write a temporary binary required
by its file-based interface; that temporary binary is deleted after successful sorting.
No permanent full preprocessed copy is retained.

The authoritative settings are in
[`pipeline/profiles/default.yaml`](pipeline/profiles/default.yaml). Profiles are
snapshotted when a probe starts, so editing a profile does not alter an active run.

## Output Layout

Derived probe output is recording-local:

```text
.../continuous/Neuropix-PXI-100.ProbeA/spikeinterface_output/
.../continuous/Neuropix-PXI-100.ProbeB/spikeinterface_output/
```

Important contents:

```text
kilosort4/                         sorter output
sorting_analyzer/                  persistent SpikeInterface analyzer
unitrefine/<label_set>/            automated advisory labels
phy/                               manual curation export
report/summary.html                detailed local report
report/summary.png                 self-contained shareable QC sheet
quality_metrics.csv
template_metrics.csv
unit_summary.csv
channel_qc.csv
manifest.json
summary.json
qc_timings.json
metadata/*_probeinterface.json
qc_variants/<variant_name>/        independent post-hoc QC comparisons
```

The original raw files are not modified. Phy's `dat_path` points to the original
`continuous.dat`; the default Phy export omits expensive `pc_features.npy` but retains
analyzer principal components for QC and UnitRefine.

## Failure Recovery

Use the selected probe's workflow page rather than deleting output folders manually.

| Situation | Preferred action |
|---|---|
| Kilosort output is intact; QC failed | **Restart QC** |
| SortingAnalyzer is intact; a later extension/export failed | **Resume QC** |
| UnitRefine failed | Run UnitRefine post hoc |
| Phy is missing or stale | Export Phy from the existing analyzer |
| Different QC parameters are desired | Run a named QC variant |
| Kilosort output is missing/incomplete | Restart the full probe |

QC restart archives the previous QC artifacts before rebuilding. A full probe restart
archives partial output with a timestamp. Existing completed output is not silently
overwritten.

## NWB Packaging

NWB export is recording-level and can combine units from both probes. It supports:

- pipeline-generated or externally supplied units CSVs
- pipeline-generated or externally supplied trials CSVs
- every detected NI-DAQ digital line and both edge directions
- recording-specific names for otherwise unknown digital lines
- mouse-specific subject metadata
- source-resolution and synchronization provenance

Mouse-arena recordings have an adapter that aligns behavior events to ephys time using
game-frame TTLs, while treating ephys timestamps as ground truth. Visual-stim and
generic recordings retain all digital events but currently require an external trials
CSV when a trials table is desired.

NWB files and inspectable source tables are written under:

```text
.../Record Node 101/experiment1/recording1/nwb/
```

Successful NWBs are also copied to:

```text
\\denmanlab\s2\nwbs\mouse_arena
```

## Useful Commands

```powershell
# Discover recordings and update the registry
python -m pipeline.discover --config pipeline/config_analysis.yaml

# Show registry state
python -m pipeline.status --config pipeline/config_analysis.yaml

# Process one registered probe
python -m pipeline.run_one --config pipeline/config_analysis.yaml --session-id SESSION_ID

# Resume or restart QC
python -m pipeline.resume_qc --config pipeline/config_analysis.yaml --session-id SESSION_ID
python -m pipeline.restart_qc --config pipeline/config_analysis.yaml --session-id SESSION_ID

# Re-export Phy or rerun UnitRefine
python -m pipeline.export_phy --config pipeline/config_analysis.yaml --session-id SESSION_ID
python -m pipeline.unitrefine --config pipeline/config_analysis.yaml --session-id SESSION_ID

# Export one recording-level NWB
python -m pipeline.mouse_arena_nwb --config pipeline/config_analysis.yaml --session-id SESSION_ID
```

The legacy PowerShell menu remains available:

```powershell
.\run_pipeline.ps1 -Config pipeline/config_analysis.yaml
```

## Repository Guide

| Path | Purpose |
|---|---|
| `pipeline/run_one.py` | Full single-probe processing implementation |
| `pipeline/queue_worker.py` | Persistent recording-first queue and recovery |
| `pipeline/discover.py` | Open Ephys discovery and registry updates |
| `pipeline/mouse_arena_nwb.py` | Recording-level tables and NWB export |
| `pipeline/nwb_digital_events.py` | Protocol-neutral TTL loading and naming |
| `pipeline/profiles/` | Versioned processing settings |
| `pipeline_gui/` | FastAPI backend and static browser interface |
| `tests/` | Core, queue, and GUI regression tests |
| `notebooks/` | Exploratory plotting and stimulus-alignment notebooks |
| `README_spikeinterface_chronic_pipeline.md` | Detailed operations reference |

## Validation

Before merging or deploying:

```powershell
python -m pytest -q
python -m compileall -q pipeline pipeline_gui tests
```

The current handoff baseline passes 64 tests in the `spikeinterface` environment.

## Operational Cautions

- The shared registry is a CSV with atomic file replacement, not a transactional
  multi-user database. Avoid simultaneous registry-writing actions on both computers.
- The analysis queue SQLite database is local to the analysis checkout and is not
  synchronized through GitHub or Synology.
- Raw backup and local archive verification should be confirmed before enabling any
  future automatic cleanup policy.
- The GUI edits the selected YAML config for metadata/settings; inspect Git changes
  before committing machine- or subject-specific values.
- UnitRefine labels are advisory and do not remove units.
- NWB export currently creates probe electrode groups but does not populate a complete
  electrode-contact table.
- UnitMatch is configuration-only/planned and is not part of the active pipeline.

## Handoff Checklist

1. Clone the private repository on each Windows computer.
2. Create or update the Conda environment.
3. Confirm the correct machine-role config and all local/network paths.
4. Confirm Synology and behavior-share access.
5. Run the test suite.
6. Launch the acquisition GUI and verify scan plus backup on a completed recording.
7. Launch the analysis GUI and verify shared registry visibility.
8. Run one recording end to end before queuing the backlog.
9. Keep the repository private while it contains lab paths or subject metadata.
