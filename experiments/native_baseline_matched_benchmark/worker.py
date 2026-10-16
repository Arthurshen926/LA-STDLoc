"""Internal exec target; direct invocation without the launch evidence is rejected."""
import argparse
import os
from pathlib import Path
from .common import read, sha
from .isolation import runtime_gate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    c = read(args.contract)
    path = Path(c["output"]) / "launch.json"
    if not path.is_file() or os.environ.get("MATCHED_LAUNCH_SHA256") != sha(path):
        raise ValueError("must use reviewed finite launch entrypoint")
    runtime = read(path)
    runtime_gate(c, runtime)
    # Prove the common advisory lock is held by this inherited open file description.
    import fcntl
    descriptor = runtime["lock_fd"]
    os.fstat(descriptor)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if c["cell"]["family"] == "native":
        from .native import execute
    elif c["cell"]["family"] == "any":
        from .any_adapter import execute
    else:
        raise ValueError("unknown method family")
    execute(c, runtime)

if __name__ == "__main__":
    main()
