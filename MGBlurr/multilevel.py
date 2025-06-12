import torch
import inspect
import numpy as np
from MGBlurr.MultiGridOperator import MultigridOperator2D
from MGTomo.optimize import orthant_bounds_optimized, box_bounds_optimized, armijo_linesearch
from MGTomo.coarse_corrections import coarse_condition

class CoarseCorrectionHandler:
    """
    Handles the selection and validation of the coarse correction condition (CC) function.
    """
    def __init__(self, CC):
        if CC is None:
            # Always activate coarse correction and set y_diff to np.nan
            def always_true_CC(*args, **kwargs):
                return True, np.nan
            self.CC = always_true_CC
        else:
            # Check signature matches coarse_corrections.py functions
            ref_sig = inspect.signature(coarse_condition)
            user_sig = inspect.signature(CC)
            if list(user_sig.parameters) != list(ref_sig.parameters):
                raise ValueError(
                    f"CC must have arguments {list(ref_sig.parameters)}, got {list(user_sig.parameters)}"
                )
            self.CC = CC

    def __call__(self, *args, **kwargs):
        return self.CC(*args, **kwargs)

class MultiLevelOptimizer:
    """
    Multilevel optimizer supporting orthant and box constraints.

    Args:
        hparams (dict): May include the following keys (defaults provided if missing):
            - "kappa": coarse correction parameter (default: 0.45)
            - "eps": coarse correction threshold (default: 1e-3)
            - "maxIter": list of max iterations per level (default: [10, 10, 10])
        bounds (str): Either "orthant" or "box" (default: "orthant")
        kernel (torch.Tensor): 2D kernel for P/R construction.
    """

    def __init__(
        self,
        fh_list,
        tau,
        kernel,
        hparams,
        CC=coarse_condition,
        BPGD=None,
        linesearch=armijo_linesearch,
        bounds="orthant"
    ):
        # Provide defaults if missing
        hparams = dict(hparams)  # Make a copy to avoid side effects
        hparams.setdefault("kappa", 0.49)
        hparams.setdefault("eps", 1e-3)
        hparams.setdefault("maxIter", [1] + [10] * (len(fh_list) - 1))
        hparams.setdefault("max_levels", len(fh_list) - 1)

        if bounds not in ["orthant", "box"]:
            raise ValueError("bounds must be either 'orthant' or 'box'")
        if linesearch is None:
            raise ValueError("linesearch must be provided")

        # Build multigrid operators from kernel
        self.mgop = MultigridOperator2D(kernel)
        self.P = self.mgop.P
        self.R = self.mgop.R

        if "input_sizes" not in hparams:
            raise ValueError("hparams must include 'input_sizes' (list of input sizes per level)")

        input_sizes = hparams["input_sizes"]
        self.P_nonzero = [
            self.mgop.compute_nonzero_elements_of_P(input_sizes[l+1])
            for l in range(len(input_sizes) - 1)
        ]

        self.fh_list = fh_list
        self.tau = tau
        self.hparams = hparams
        self.CC = CoarseCorrectionHandler(CC)
        self.linesearch = linesearch
        self.bounds = bounds
        self.compute_bounds = orthant_bounds_optimized if bounds == "orthant" else box_bounds_optimized
        self.BPGD = self._validate_BPGD(BPGD)

    
    def _validate_BPGD(self, BPGD):
        if BPGD is None or not callable(BPGD):
            raise ValueError("BPGD must be a callable function")

        # Expecting at least 4 arguments: f, x, tau, lh (names don't matter)
        sig = inspect.signature(BPGD)
        params = sig.parameters

        # Count number of required (non-default) positional or keyword args
        required_args = [
            p for p in params.values()
            if p.default == inspect.Parameter.empty
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        ]

        if len(required_args) < 4:
            raise ValueError(
                f"BPGD must accept at least 4 positional arguments (e.g. 'f, x, tau, lh'), "
                f"but got {len(required_args)}: {[p.name for p in required_args]}"
            )

        return BPGD

    def run(self, y, lh, uh=None, last_pts=None, y_diff=None, l=0, *args, **kwargs):
        fh = self.fh_list[l]
        x = self.R(y).detach().requires_grad_(True)
        y0 = y.detach().requires_grad_(True)
        fhy0 = fh(y)
        fhy0.backward(retain_graph=True)
        grad_fhy0 = y.grad.clone()
        y.grad = None

        CC_bool, y_diff[l] = self.CC(y, grad_fhy0, self.hparams["kappa"], self.hparams["eps"], last_pts[l])

        P_inf = self.mgop.norm_infty_P()

        if CC_bool:
            print(l, ': coarse correction activated')
            last_pts[l] = y.detach()

            x0 = x.detach().requires_grad_(True)
            fH = self.fh_list[l+1]
            fHx0 = fH(x0)
            fHx0.backward(retain_graph=True)
            grad_fHx0 = x0.grad.clone()
            x0.grad = None

            kappa = self.R(grad_fhy0) - grad_fHx0

            with torch.no_grad():
                lH, uH = self.compute_bounds(y, x, P_inf, lh, uh, self.P_nonzero[l])
                bounds = (lH, uH)

            def psi(x): return fH(x) + torch.sum(kappa * x)

            # --- Auxiliary variables are local to this loop ---
            local_args = ()
            local_kwargs = {}
            for i in range(self.hparams["maxIter"][l+1]):
                if self.bounds == "orthant":
                    # _validate_BPGD expects at least (f, x, tau, lh)
                    result = self.BPGD(psi, x, self.tau[l + 1], bounds[0], *local_args, **local_kwargs)
                elif self.bounds == "box":
                    # _validate_BPGD expects at least (f, x, tau, lh, uh)
                    result = self.BPGD(psi, x, self.tau[l + 1], bounds[0], bounds[1], *local_args, **local_kwargs)
                if isinstance(result, tuple):
                    x = result[0].detach().requires_grad_(True)
                    local_args = result[1:]
                else:
                    x = result.detach().requires_grad_(True)
                    local_args = ()
                x.grad = None

            if l < self.hparams["max_levels"]-1:
                x, last_pts, y_diff = self.run(x, bounds[0], bounds[1], last_pts, y_diff, l + 1)

            d = self.P(x-x0)
            z, _ = self.linesearch(fh, y0, d)
            y = z.detach().requires_grad_(True)
        else:
            print(l, ': coarse correction not activated')

        # --- Auxiliary variables are local to this loop ---
        local_args = ()
        local_kwargs = {}
        for i in range(self.hparams["maxIter"][l]):
            if self.bounds == "orthant":
                result = self.BPGD(fh, y, self.tau[l], lh, *local_args, **local_kwargs)
            elif self.bounds == "box":
                result = self.BPGD(fh, y, self.tau[l], lh, uh, *local_args, **local_kwargs)
            if isinstance(result, tuple):
                y = result[0].detach().requires_grad_(True)
                local_args = result[1:]
            else:
                y = result.detach().requires_grad_(True)
                local_args = ()
            y.grad = None

        return y, last_pts, y_diff