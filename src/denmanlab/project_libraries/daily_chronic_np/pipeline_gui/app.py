from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .services import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    JobManager,
    clear_nwb_input_override_payload,
    config_text_payload,
    config_payload,
    digital_event_inventory_payload,
    enqueue_all_payload,
    enqueue_recording_payload,
    existing_stimulus_behavior_payload,
    existing_units_payload,
    is_allowed_file,
    nwb_metadata_payload,
    probe_payload,
    profiles_payload,
    queue_events_payload,
    queue_job_log_payload,
    recordings_payload,
    reconcile_interrupted_payload,
    reports_payload,
    resume_queue_payload,
    save_digital_event_payload,
    save_digital_line_name_payload,
    save_digital_line_npy_payload,
    save_config_text,
    save_nwb_input_csv_payload,
    save_nwb_metadata,
    save_profile_payload,
    pause_queue_payload,
    status_payload,
    stimulus_behavior_payload,
    system_payload,
    units_payload,
    write_units_payload,
    write_stimulus_behavior_payload,
)


def create_app(*, config_path: str | Path = "pipeline/config.yaml", repo_root: str | Path | None = None) -> FastAPI:
    root = Path(repo_root or Path.cwd()).resolve()
    config = Path(config_path)
    if not config.is_absolute():
        config = root / config
    manager = JobManager(repo_root=root, config_path=config)
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="SpikeInterface Pipeline GUI")
    app.state.repo_root = root
    app.state.config_path = config
    app.state.job_manager = manager

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        return status_payload(app.state.config_path)

    @app.get("/api/recordings")
    def api_recordings() -> dict[str, Any]:
        return recordings_payload(app.state.config_path, app.state.repo_root)

    @app.post("/api/queue/enqueue")
    def api_enqueue_recording(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return enqueue_recording_payload(app.state.config_path, app.state.repo_root, payload)
        except KeyError:
            raise HTTPException(status_code=404, detail="recording not found")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/queue/enqueue-all")
    def api_enqueue_all(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return enqueue_all_payload(app.state.config_path, app.state.repo_root, payload or {})

    @app.post("/api/queue/pause")
    def api_pause_queue() -> dict[str, Any]:
        return pause_queue_payload(app.state.repo_root)

    @app.post("/api/queue/resume")
    def api_resume_queue() -> dict[str, Any]:
        return resume_queue_payload(app.state.config_path, app.state.repo_root)

    @app.get("/api/queue/events")
    def api_queue_events(queue_id: str = "", session_id: str = "", limit: int = 200) -> dict[str, Any]:
        return queue_events_payload(
            app.state.repo_root,
            queue_id=queue_id,
            session_id=session_id,
            limit=limit,
        )

    @app.get("/api/queue/jobs/{job_id}/logs")
    def api_queue_job_logs(
        job_id: str,
        stream: str = Query("out", pattern="^(out|err|events)$"),
        tail: int = 200,
    ) -> dict[str, Any]:
        try:
            return queue_job_log_payload(
                app.state.repo_root,
                job_id=job_id,
                stream=stream,
                tail=tail,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="queue job not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/queue/reconcile-interrupted")
    def api_reconcile_interrupted() -> dict[str, Any]:
        try:
            return reconcile_interrupted_payload(
                app.state.config_path,
                app.state.repo_root,
                legacy_active=app.state.job_manager.active_job() is not None,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/api/probes/{session_id}")
    def api_probe(session_id: str) -> dict[str, Any]:
        try:
            return probe_payload(app.state.config_path, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="probe row not found")

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        return config_payload(app.state.config_path)

    @app.get("/api/profiles")
    def api_profiles() -> dict[str, Any]:
        return profiles_payload(app.state.config_path)

    @app.put("/api/profiles/{name}")
    def api_save_profile(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return save_profile_payload(app.state.config_path, {**payload, "name": name})
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/config/text")
    def api_config_text() -> dict[str, str]:
        return config_text_payload(app.state.config_path)

    @app.put("/api/config/text")
    def api_save_config_text(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return save_config_text(app.state.config_path, str(payload.get("text", "")))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/nwb/metadata")
    def api_nwb_metadata(session_id: str | None = None) -> dict[str, Any]:
        return nwb_metadata_payload(app.state.config_path, session_id)

    @app.put("/api/nwb/metadata")
    def api_save_nwb_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return save_nwb_metadata(app.state.config_path, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/reports")
    def api_reports() -> dict[str, Any]:
        return reports_payload(app.state.config_path)

    @app.get("/api/stimulus/{session_id}")
    def api_stimulus(session_id: str) -> dict[str, Any]:
        try:
            return stimulus_behavior_payload(app.state.config_path, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="probe row not found")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/nwb/digital-events/{session_id}")
    def api_nwb_digital_events(session_id: str) -> dict[str, Any]:
        try:
            return digital_event_inventory_payload(app.state.config_path, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="probe row not found")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.put("/api/nwb/digital-events/line-name")
    def api_nwb_digital_line_name(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return save_digital_line_name_payload(app.state.config_path, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/nwb/digital-events/save-npy")
    def api_nwb_digital_line_npy(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return save_digital_line_npy_payload(app.state.config_path, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/stimulus/{session_id}/existing")
    def api_existing_stimulus(session_id: str) -> dict[str, Any]:
        try:
            return existing_stimulus_behavior_payload(app.state.config_path, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="probe row not found")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/stimulus/{session_id}/write")
    def api_write_stimulus(session_id: str) -> dict[str, Any]:
        try:
            return write_stimulus_behavior_payload(app.state.config_path, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="probe row not found")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/stimulus/save-digital")
    def api_save_digital_event(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return save_digital_event_payload(app.state.config_path, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/nwb/input-csv")
    def api_save_nwb_input_csv(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return save_nwb_input_csv_payload(app.state.config_path, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/nwb/input-csv/clear")
    def api_clear_nwb_input_csv(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return clear_nwb_input_override_payload(app.state.config_path, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/units/{session_id}")
    def api_units(session_id: str) -> dict[str, Any]:
        try:
            return units_payload(app.state.config_path, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="probe row not found")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/units/{session_id}/existing")
    def api_existing_units(session_id: str) -> dict[str, Any]:
        try:
            return existing_units_payload(app.state.config_path, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="probe row not found")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/units/{session_id}/write")
    def api_write_units(session_id: str) -> dict[str, Any]:
        try:
            return write_units_payload(app.state.config_path, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="probe row not found")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/jobs")
    def api_jobs() -> dict[str, Any]:
        return {
            "active_job": app.state.job_manager.active_job(),
            "jobs": app.state.job_manager.list_jobs(),
        }

    @app.get("/api/system")
    def api_system() -> dict[str, Any]:
        return system_payload(app.state.config_path, app.state.repo_root)

    @app.get("/api/jobs/{job_id}/logs")
    def api_job_logs(job_id: str, stream: str = Query("out", pattern="^(out|err)$"), tail: int = 200) -> dict[str, Any]:
        try:
            return app.state.job_manager.log_tail(job_id, stream=stream, tail=tail)
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")

    @app.get("/api/jobs/{job_id}/progress")
    def api_job_progress(job_id: str) -> dict[str, Any]:
        try:
            return app.state.job_manager.progress(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")

    @app.post("/api/jobs/{kind}")
    def api_start_job(kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return app.state.job_manager.start_job(kind, payload or {})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/files")
    def api_file(path: str) -> FileResponse:
        if not is_allowed_file(path, app.state.config_path, app.state.repo_root):
            raise HTTPException(status_code=404, detail="file not found or not allowed")
        return FileResponse(Path(path))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local SpikeInterface pipeline web GUI.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    import uvicorn

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
