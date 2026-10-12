"""
Plotting helpers that produce clean, paper-ready figures.

Every function saves a .png file (300 dpi) and closes the figure, so the
scripts also work on machines without a display (e.g. a cluster).

One style for every figure:
    * large fonts, bold math axis labels  x, y, t,  a thick black frame
    * exact solution = solid black line, predictions = coloured broken lines
    * framed legend, light dotted grid
    * 2-D fields: ONE figure per field (as in a paper), 25 filled contour
      levels, a full-height colour bar; solutions in viridis on a SHARED
      colour range (so they can be compared by eye), errors in "hot".

Scripts that draw a special figure themselves use new_figure(), finish()
and style_colorbar() from here, so they look the same.
"""

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

DPI = 300
LABEL_SIZE = 28
TICK_SIZE = 24
TITLE_SIZE = 28
LEGEND_SIZE = 17
LINE_FIGSIZE = (10, 7)      # inches, for curves and convergence plots
FIELD_FIGSIZE = (11, 9)     # inches, for one 2-D field with its colour bar

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.3",
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "lines.linewidth": 3,
    "mathtext.fontset": "dejavusans",
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
STYLES = ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]


def _np(a):
    try:
        return a.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(a)


def _label(text):
    """Axis label: 'x' -> bold math x (paper style), "u'(x)" -> math, else capitalised."""
    if text in ("x", "y", "t"):
        return rf"$\mathbf{{{text}}}$"
    if re.fullmatch(r"[a-zA-Z]'*\([a-z]\)", text):
        return f"${text}$"
    return text[:1].upper() + text[1:]


def new_figure(figsize=LINE_FIGSIZE):
    """An empty figure with one axes, in the common size."""
    return plt.subplots(figsize=figsize)


def finish(fig, ax, path, xlabel="x", ylabel="", title="", legend=True, grid=True):
    """Labels, title, grid and legend in the common style; save to path and close."""
    ax.set_xlabel(_label(xlabel), labelpad=2)
    ax.set_ylabel(_label(ylabel))
    if title:
        ax.set_title(textwrap.fill(title, 40), pad=12)
    if grid:
        ax.grid(True, linestyle=":", alpha=0.5)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", frameon=True)
    fig.savefig(path)
    plt.close(fig)


def style_colorbar(cb, n_ticks=7):
    """Large, evenly spaced colour-bar ticks; tiny or huge values get a x10^n factor."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    cb.locator = MaxNLocator(n_ticks)
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=TICK_SIZE, length=6, width=1.2)
    cb.ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE - 4)
    cb.outline.set_linewidth(1.2)


def plot_curves(x, exact, predictions, path, title="", ylabel="u(x)",
                points=None, points_label="training points", xlabel="x"):
    """
    Exact solution (black line) vs. one or more predictions (coloured broken lines).

    predictions : dict  {"label": array}
    points      : optional (x_pts, y_pts) to mark training/collocation points
    """
    x = _np(x).ravel()
    fig, ax = new_figure()
    ax.plot(x, _np(exact).ravel(), "k-", lw=3.5, label="exact")
    for i, (label, y) in enumerate(predictions.items()):
        ax.plot(x, _np(y).ravel(), color=COLORS[i % len(COLORS)],
                linestyle=STYLES[i % len(STYLES)], label=label)
    if points is not None:
        ax.plot(_np(points[0]).ravel(), _np(points[1]).ravel(), "o", ms=10, mfc="white",
                mec="k", mew=2, zorder=5, label=points_label)
    finish(fig, ax, path, xlabel, ylabel, title)


def plot_pointwise_error(x, errors, path, title="Point-wise absolute error"):
    """|prediction - exact| on a log scale for one or more methods."""
    x = _np(x).ravel()
    fig, ax = new_figure()
    for i, (label, e) in enumerate(errors.items()):
        ax.semilogy(x, np.abs(_np(e).ravel()) + 1e-16, color=COLORS[i % len(COLORS)],
                    linestyle=STYLES[i % len(STYLES)], lw=2.5, label=label)
    finish(fig, ax, path, "x", "|error|", title)


def plot_history(histories, path, key="error", ylabel=r"relative $L^2$ error", title=""):
    """Convergence plot: one curve per method, log scale."""
    fig, ax = new_figure()
    for i, (label, h) in enumerate(histories.items()):
        if len(h.get(key, [])) == 0:
            continue
        ax.semilogy(h["step"][:len(h[key])], h[key], color=COLORS[i % len(COLORS)],
                    linestyle=STYLES[i % len(STYLES)], label=label)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    finish(fig, ax, path, "iteration", ylabel, title)


def plot_loss_terms(history, path, title="Loss terms during training"):
    """Every recorded loss term of ONE run (useful to spot imbalance)."""
    fig, ax = new_figure()
    for i, key in enumerate(k for k in history if k not in ("step", "error")):
        ax.semilogy(history["step"], history[key], color=COLORS[i % len(COLORS)],
                    linestyle=STYLES[i % len(STYLES)], label=key)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    finish(fig, ax, path, "iteration", "loss", title)


def domain_ticks(ax, X, Y, n=5):
    """n evenly spaced ticks from edge to edge of the domain, e.g. -1, -0.5, 0, 0.5, 1."""
    ax.set_xticks(np.linspace(np.min(X), np.max(X), n))
    ax.set_yticks(np.linspace(np.min(Y), np.max(Y), n))


def _extend(F, levels):
    """Colour-bar arrows for values outside the colour range (e.g. a bad prediction)."""
    below, above = np.nanmin(F) < levels[0], np.nanmax(F) > levels[-1]
    return {(True, True): "both", (True, False): "min",
            (False, True): "max", (False, False): "neither"}[(below, above)]


def _field_title(label):
    """'exact u' -> 'Exact solution', 'PINN u' -> 'PINN prediction', 'PINN |error|' -> ..."""
    if label == "exact u":
        return "Exact solution"
    m = re.fullmatch(r"(.*?)\s*\|error\|", label)
    if m:
        return f"{m.group(1)} pointwise error" if m.group(1) else "Pointwise absolute error"
    m = re.fullmatch(r"(.*) u", label)
    return f"{m.group(1)} prediction" if m else label


def plot_fields_2d(X, Y, fields, path, cmap="viridis", error_cmap="hot", n_levels=25):
    """
    One paper-style figure per 2-D field, e.g. {"exact u": U, "PINN u": U_pred, "|error|": E}.
    X, Y, and every field are 2-D arrays of the same shape (from np.meshgrid).

    path "fields.png" gives fields_exact_u.png, fields_pinn_u.png, fields_error.png, ...
    Solution fields share ONE colour range, that of the exact solution (fair
    visual comparison); values of a prediction outside it are marked by
    arrows on the colour bar. Error fields (label contains "error") start at
    0, use error_cmap and each has its own range - read its colour bar.
    """
    path = Path(path)
    fields = {label: _np(F) for label, F in fields.items()}
    is_error = {label: "error" in label.lower() for label in fields}
    solutions = [F for label, F in fields.items() if not is_error[label]]
    if solutions:
        reference = [fields["exact u"]] if "exact u" in fields else solutions
        lo = min(float(np.nanmin(F)) for F in reference)
        hi = max(float(np.nanmax(F)) for F in reference)
        solution_levels = np.linspace(lo, max(hi, lo + 1e-12), n_levels)

    for label, F in fields.items():
        if is_error[label]:
            levels = np.linspace(0.0, max(float(np.nanmax(F)), 1e-14), n_levels)
        else:
            levels = solution_levels
        fig, ax = new_figure(FIELD_FIGSIZE)
        c = ax.contourf(X, Y, F, levels=levels, cmap=error_cmap if is_error[label] else cmap,
                        extend=_extend(F, levels))
        style_colorbar(fig.colorbar(c, ax=ax, shrink=1, pad=0.02))
        ax.set_aspect("equal")
        domain_ticks(ax, X, Y)
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        finish(fig, ax, path.with_name(f"{path.stem}_{slug}{path.suffix}"), "x", "y",
               _field_title(label), legend=False, grid=False)
