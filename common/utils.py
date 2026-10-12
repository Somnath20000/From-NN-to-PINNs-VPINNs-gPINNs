"""
General helpers: reproducibility, device (CPU/GPU), command-line options,
output folders, metrics.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

# The repository root folder (the folder that contains README.md).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Set to True by get_args() when --quick is used, so quick (inaccurate) runs
# write to <stage folder>/figures/quick/ and never overwrite your real results.
_QUICK = False

# Device chosen by get_args() with --device (auto / cpu / cuda). "auto" means:
# use the GPU if PyTorch can see a working CUDA GPU, otherwise the CPU.
_DEVICE_CHOICE = "auto"
DEVICE = None          # the torch.device actually used, set by setup()


def pick_device(choice: str = "auto") -> torch.device:
    """
    Turn "auto", "cpu" or "cuda" into a torch.device.

    "auto" falls back to the CPU when no usable GPU is found, so the same
    script runs on a laptop without a GPU and on a GPU machine.
    """
    if choice == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "cuda":
        raise RuntimeError("--device cuda was requested, but PyTorch cannot use a CUDA GPU "
                           "on this machine (torch.cuda.is_available() is False). "
                           "Use --device auto or --device cpu.")
    return torch.device("cpu")


def setup(seed: int = 1234, double_precision: bool = True) -> None:
    """
    Make a run reproducible and choose the floating-point precision.

    * Fixing the random seed means the network starts from the same random
      weights every time, so you get (almost) the same numbers on every run.
    * Double precision (float64) is slower than float32 but avoids round-off
      problems when we take 2nd/3rd derivatives or sum many quadrature terms.
    * Device: on the first call we pick the CPU or the GPU (see --device).
      On a GPU, torch.set_default_device makes every tensor and network that
      the script creates afterwards live on the GPU, so the scripts need no
      .to(device) calls. Results are moved back with .cpu() before .numpy().
    """
    global DEVICE
    if DEVICE is None:
        DEVICE = pick_device(_DEVICE_CHOICE)
        if DEVICE.type == "cuda":
            torch.set_default_device(DEVICE)
            print(f"Device: GPU ({torch.cuda.get_device_name(DEVICE)})")
        else:
            print("Device: CPU")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)        # seeds the CPU and every GPU
    if double_precision:
        torch.set_default_dtype(torch.float64)


def get_args(description: str, **defaults) -> argparse.Namespace:
    """
    Read options from the command line, e.g.

        python 02_pinns/pinn_01_first_order_ode.py --epochs 5000 --seed 7

    Every script passes its own default values. The options you can always use:

        --epochs  number of Adam iterations
        --lr      learning rate of Adam
        --seed    random seed (change it to see how sensitive the result is)
        --lbfgs   extra L-BFGS iterations after Adam (often improves accuracy)
        --quick   run only a few iterations (used by the automatic tests;
                  results will be inaccurate, but you can check nothing crashes)
        --device  auto (default: GPU if available, else CPU), cpu or cuda

    Extra script-specific options are added with `extra=[(name, type, default, help), ...]`.
    """
    extra = defaults.pop("extra", [])
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 10000),
                        help="number of Adam iterations")
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3),
                        help="learning rate of the Adam optimiser")
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 1234),
                        help="random seed for reproducibility")
    parser.add_argument("--lbfgs", type=int, default=defaults.get("lbfgs", 0),
                        help="extra L-BFGS iterations after Adam (0 = none)")
    parser.add_argument("--quick", action="store_true",
                        help="very short run for smoke-testing (inaccurate results)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="where to train: auto = GPU if available, otherwise CPU")
    for name, typ, default, help_text in extra:
        if typ is bool:
            parser.add_argument(f"--{name}", action="store_true", help=help_text)
        else:
            parser.add_argument(f"--{name}", type=typ, default=default, help=help_text)
    args = parser.parse_args()
    global _DEVICE_CHOICE
    _DEVICE_CHOICE = args.device
    if args.quick:
        global _QUICK
        _QUICK = True
        args.epochs = min(args.epochs, 200)
        args.lbfgs = min(args.lbfgs, 20)
    return args


def output_dir(script_file: str) -> Path:
    """
    Create (if needed) and return  <stage folder>/figures/<script name>/,
    i.e. a figures folder next to the script itself. For example
    01_neural_networks/nn_01_data_hungry.py writes to
    01_neural_networks/figures/nn_01_data_hungry/.
    All figures and the summary.json of a script are written there.
    Quick smoke-test runs (--quick) go to <stage folder>/figures/quick/<script name>/ instead.
    """
    script = Path(script_file).resolve()
    base = script.parent / "figures" / ("quick" if _QUICK else "")
    out = base / script.stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def relative_l2(pred, exact) -> float:
    """
    Relative L2 error   ||pred - exact||_2 / ||exact||_2

    This is the standard accuracy measure in the PINN literature.
    0.01 means "1 % error". Works with numpy arrays or torch tensors.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(exact, torch.Tensor):
        exact = exact.detach().cpu().numpy()
    pred, exact = np.asarray(pred).ravel(), np.asarray(exact).ravel()
    return float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))


def save_summary(out_dir: Path, summary: dict) -> None:
    """Print the final numbers nicely and store them in summary.json."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:<45s} {value:.3e}")
        else:
            print(f"  {key:<45s} {value}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFigures and summary.json saved in: {out_dir}")
