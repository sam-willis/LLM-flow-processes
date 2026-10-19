"""Where the flow checkpoint lives, resolved in one place.

Every entry point resolves it the same way: NDP_FLOW_DIR if set, otherwise
`outputs/flow_prior` under the repository root. Nothing in the repository
ships a checkpoint, so this is a convention, not a promise that a directory
exists; callers that need one should say so clearly when it is missing.

Deliberately dependency-free (os and pathlib only) so the package's import
surface and the standalone scripts can both use it without a cycle.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Used when NDP_FLOW_DIR is unset. `make train` writes here by default too.
DEFAULT_FLOW_DIR = REPO_ROOT / "outputs" / "flow_prior"

#: What to print for a CLI flag that defaults to `default_flow_dir()`; a
#: callable default would otherwise render as "(dynamic)".
FLOW_DIR_SHOW_DEFAULT = "$NDP_FLOW_DIR, else outputs/flow_prior"


def default_flow_dir() -> Path:
    """The flow checkpoint directory: $NDP_FLOW_DIR, else the default.

    Read at call time, not import time, so setting the variable in a job
    script still takes effect for click options that use this as a default.
    """
    return Path(os.environ.get("NDP_FLOW_DIR") or DEFAULT_FLOW_DIR)
