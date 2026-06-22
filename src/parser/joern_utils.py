import subprocess
from typing import List


def run_command(cmd: List[str]) -> str:
    """Run a command and return its stdout."""
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
        )
        return result.stdout

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n{exc.stderr}"
        ) from exc