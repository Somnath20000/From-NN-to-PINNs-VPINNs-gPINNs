"""
Run every tutorial script in order (Stage 1 -> Stage 5).

    python run_all.py            # full runs (about 30-45 minutes on a laptop CPU)
    python run_all.py --quick    # a few iterations each, just to check that everything works
    python run_all.py --stage 4  # only the scripts of one stage

Results (figures + summary.json) appear in <stage folder>/figures/<script name>/
(or <stage folder>/figures/quick/<script name>/ for --quick runs).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    1: "01_neural_networks",
    2: "02_pinns",
    3: "03_pinn_failures",
    4: "04_vpinns",
    5: "05_gradient_enhanced",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true", help="smoke test: very short runs")
    parser.add_argument("--stage", type=int, choices=sorted(STAGES), help="run only this stage")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else sorted(STAGES)
    scripts = [p for s in stages for p in sorted((ROOT / STAGES[s]).glob("*.py"))]
    failed = []
    for script in scripts:
        cmd = [sys.executable, str(script)] + (["--quick"] if args.quick else [])
        print(f"\n{'#' * 78}\n# {script.relative_to(ROOT)}\n{'#' * 78}", flush=True)
        start = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
        print(f"--> {status} in {time.time() - start:.0f} s", flush=True)
        if result.returncode != 0:
            failed.append(script.name)

    print("\n" + ("All scripts finished successfully." if not failed
                  else f"Failed scripts: {', '.join(failed)}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
