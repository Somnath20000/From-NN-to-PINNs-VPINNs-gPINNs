"""
A single, reusable training loop.

Every script defines a function  loss_fn()  that returns a dictionary of loss
terms, for example

    {"total": L, "pde": L_pde, "bc": L_bc}

The key "total" is the quantity that is minimised; the other keys are only
recorded so that we can plot how each part of the loss evolves.

Optionally the script also gives  error_fn()  that returns the relative L2
error against the exact solution, so we can plot "error vs. iteration".
"""

import time

import torch


def train(params, loss_fn, epochs, lr=1e-3, error_fn=None,
          log_every=1000, record_every=100, lbfgs_iters=0):
    """
    Minimise loss_fn() with Adam (and optionally L-BFGS afterwards).

    Parameters
    ----------
    params       : iterable of tensors to optimise (usually model.parameters())
    loss_fn      : function returning a dict of loss tensors with key "total"
    epochs       : number of Adam iterations
    lr           : Adam learning rate
    error_fn     : optional function returning a float (relative L2 error)
    log_every    : print progress every this many iterations
    record_every : store history every this many iterations
    lbfgs_iters  : extra L-BFGS iterations after Adam (0 = skip).
                   L-BFGS is a quasi-Newton method; it often reduces the
                   final error considerably once Adam has found a good region.

    Returns
    -------
    history : dict with lists "step", "error" and one list per loss term
    """
    params = list(params)
    optimizer = torch.optim.Adam(params, lr=lr)
    history = {"step": [], "error": []}

    def record(step, terms):
        history["step"].append(step)
        for name, value in terms.items():
            history.setdefault(name, []).append(float(value))
        if error_fn is not None:
            history["error"].append(error_fn())

    start = time.time()
    for it in range(epochs + 1):
        optimizer.zero_grad()           # 1. clear old gradients
        terms = loss_fn()               # 2. evaluate every loss term
        terms["total"].backward()       # 3. back-propagate: d(loss)/d(weights)
        optimizer.step()                # 4. update the weights

        if it % record_every == 0 or it == epochs:
            record(it, {k: v.detach() for k, v in terms.items()})
        if it % log_every == 0 or it == epochs:
            msg = "  ".join(f"{k}={float(v.detach()):.3e}" for k, v in terms.items())
            if error_fn is not None:
                msg += f"  | rel. L2 error={history['error'][-1]:.3e}"
            print(f"[Adam {it:6d}] {msg}  ({time.time() - start:.1f}s)")

    if lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_iters,
                                  tolerance_grad=1e-12, tolerance_change=1e-15,
                                  history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = loss_fn()["total"]
            loss.backward()
            return loss

        lbfgs.step(closure)
        terms = {k: v.detach() for k, v in loss_fn().items()}
        record(epochs + lbfgs_iters, terms)
        msg = "  ".join(f"{k}={float(v):.3e}" for k, v in terms.items())
        print(f"[L-BFGS done] {msg}")

    print(f"Training finished in {time.time() - start:.1f} s")
    return history
