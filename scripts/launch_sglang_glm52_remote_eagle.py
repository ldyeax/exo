#!/usr/bin/env python3
"""Launch SGLang with the explicitly gated GLM-5.2 remote EAGLE worker."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

from scripts.glm52_remote_eagle_target import (
    ENABLE_ENVIRONMENT_VARIABLE,
    RemoteEagleTargetError,
    install_remote_eagle_worker,
)

if os.environ.get(ENABLE_ENVIRONMENT_VARIABLE) != "1":
    raise RemoteEagleTargetError(
        f"{ENABLE_ENVIRONMENT_VARIABLE}=1 is required by this launcher"
    )

# This is intentionally top-level: Python multiprocessing ``spawn`` imports the
# main module in each scheduler child, so every child installs the same selector.
install_remote_eagle_worker()


def main() -> int:
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
