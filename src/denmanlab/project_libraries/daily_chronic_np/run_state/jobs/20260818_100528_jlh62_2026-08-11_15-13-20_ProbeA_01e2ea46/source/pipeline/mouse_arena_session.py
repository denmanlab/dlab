from __future__ import annotations

import argparse

from .config import load_config
from .mouse_arena_nwb import write_mouse_arena_session_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the mouse-arena session events dataframe on the ephys timebase.")
    parser.add_argument("--config", default="pipeline/config.yaml")
    parser.add_argument("--session-id", required=True, help="Any probe session_id from the target recording.")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = write_mouse_arena_session_dataframe(config, session_id=args.session_id)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
