"""nami models: class-based FWI wrappers.

The functional primitives (``scalar2d`` / ``elastic2d`` / ``em2d_tm``) are
PyTorch functions that return receiver amplitudes and support autograd.
This module wraps them in small ``nn.Module``s that reproduce the classic
FWI workflow:

    model = Scalar(v, dx, dt, pml_width=20)
    rec = model.forward(amp, srcs, recs)     # receiver amplitudes [nt, 1, n_rec]
    loss = (rec - observed).square().mean()
    grad = model.backward(loss)              # d loss / d v
    with torch.no_grad():
        v -= lr * grad

``forward`` runs the forward model with gradients enabled and caches the
receiver amplitudes; ``backward`` back-propagates a loss through that graph,
writes the gradient into each model parameter's ``.grad`` and returns it
(also available as ``.grad``).  The model is held as an ``nn.Parameter``, so
``.parameters()`` / ``.named_parameters()`` work and optimisers can be used
directly.
"""

import torch
import torch.nn as nn


class _FWIModule(nn.Module):
    """Shared plumbing: cached forward, explicit backward and ``.grad``."""

    _model_names: tuple = ()

    def __init__(
        self,
        dt,
        accuracy=2,
        pml_width=20,
        pml_freq=25.0,
        requires_grad=True,
        storage="auto",
        sample_steps=1,
        ckpt_steps=None,
    ):
        super().__init__()
        self.dt = float(dt)
        self.accuracy = accuracy
        self.pml_width = pml_width
        self.pml_freq = pml_freq
        self.requires_grad_flag = requires_grad
        self.storage = storage
        self.sample_steps = sample_steps
        self.ckpt_steps = ckpt_steps
        self.receiver_amplitudes = None
        self._model_grad = None

    def _register_model(self, name, value):
        setattr(self, name, nn.Parameter(value, requires_grad=self.requires_grad_flag))

    def _model_params(self):
        return [getattr(self, n) for n in self._model_names]

    def _run(self, amp, srcs, recs, nt):
        raise NotImplementedError

    def forward(self, source_amplitudes=None, source_locations=None,
                receiver_locations=None, nt=None):
        """Run the forward model and cache the receiver amplitudes."""
        self._model_grad = None
        self.receiver_amplitudes = self._run(
            source_amplitudes, source_locations, receiver_locations, nt
        )
        return self.receiver_amplitudes

    def backward(self, loss):
        """Back-propagate ``loss`` and return the model gradient(s).

        The gradient is written into each model parameter's ``.grad``
        (replacing any previous value, so ``torch.optim`` optimisers step
        correctly) and is also exposed via ``.grad``.  Every model
        parameter's ``.grad`` is cleared first and then set from this
        call, so parameters left out of the current graph (unused or
        ``requires_grad_(False)``) end up with ``.grad is None`` instead
        of a stale gradient from an earlier call.  For a single-model
        class (``Scalar``, ``Scalar3D``) the return value is a tensor; for
        ``Elastic``/``TM2D``/``EM3D`` it is a 3-tuple, e.g.
        ``(dL/dlamb, dL/dmu, dL/dbuoyancy)`` for ``Elastic``.
        """
        for p in self._model_params():
            p.grad = None
        params = [p for p in self._model_params() if p.requires_grad]
        if not params:
            raise RuntimeError("model parameters do not require grad")
        grads = torch.autograd.grad(
            loss, params, retain_graph=False, allow_unused=True
        )
        for p, g in zip(params, grads, strict=True):
            p.grad = g
        grad_map = {id(p): g for p, g in zip(params, grads, strict=True)}
        self._model_grad = tuple(
            None if grad_map.get(id(p)) is None else grad_map[id(p)]
            for p in self._model_params()
        )
        if len(self._model_names) == 1:
            return self._model_grad[0]
        return self._model_grad

    @property
    def grad(self):
        """Model gradient from the last ``backward`` call (or None)."""
        g = self._model_grad
        if g is None:
            return None
        if len(self._model_names) == 1:
            return g[0]
        return g

class Scalar(_FWIModule):
    """2D acoustic FWI: ``Scalar(v, dx, dt)`` with ``.forward/.backward/.grad``."""

    _model_names = ("v",)

    def __init__(self, v, dx, dt, accuracy=2, pml_width=20, pml_freq=25.0,
                 requires_grad=True, storage="auto", sample_steps=1,
                 ckpt_steps=None):
        super().__init__(dt, accuracy, pml_width, pml_freq, requires_grad,
                         storage, sample_steps, ckpt_steps)
        self.dx = dx
        self._register_model("v", v)

    def _run(self, amp, srcs, recs, nt):
        from .scalar.scalar2d import scalar2d

        return scalar2d(
            self.v,
            self.dx,
            self.dt,
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=self.accuracy,
            pml_width=self.pml_width,
            pml_freq=self.pml_freq,
            nt=nt,
            storage=self.storage,
            sample_steps=self.sample_steps,
            ckpt_steps=self.ckpt_steps,
        )


class Scalar3D(_FWIModule):
    """3D acoustic FWI: ``Scalar3D(v, grid_spacing, dt)`` with
    ``.forward/.backward/.grad``."""

    _model_names = ("v",)

    def __init__(self, v, grid_spacing, dt, accuracy=2, pml_width=20,
                 pml_freq=25.0, requires_grad=True, storage="auto",
                 sample_steps=1, ckpt_steps=None):
        super().__init__(dt, accuracy, pml_width, pml_freq, requires_grad,
                         storage, sample_steps, ckpt_steps)
        self.grid_spacing = grid_spacing
        self._register_model("v", v)

    def _run(self, amp, srcs, recs, nt):
        from .scalar.scalar3d import scalar3d

        return scalar3d(
            self.v,
            self.grid_spacing,
            self.dt,
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=self.accuracy,
            pml_width=self.pml_width,
            pml_freq=self.pml_freq,
            nt=nt,
            storage=self.storage,
            sample_steps=self.sample_steps,
            ckpt_steps=self.ckpt_steps,
        )


class Elastic(_FWIModule):
    """2D elastic FWI: ``Elastic(lamb, mu, buoyancy, grid_spacing, dt)``.

    Gradients are returned for all three models:
    ``(dL/dlamb, dL/dmu, dL/dbuoyancy)``.
    """

    _model_names = ("lamb", "mu", "buoyancy")

    def __init__(self, lamb, mu, buoyancy, grid_spacing, dt, accuracy=2,
                 pml_width=20, pml_freq=25.0, requires_grad=True,
                 storage="auto", sample_steps=1, ckpt_steps=None):
        super().__init__(dt, accuracy, pml_width, pml_freq, requires_grad,
                         storage, sample_steps, ckpt_steps)
        self.grid_spacing = grid_spacing
        self._register_model("lamb", lamb)
        self._register_model("mu", mu)
        self._register_model("buoyancy", buoyancy)

    def _run(self, amp, srcs, recs, nt):
        from .elastic.elastic2d import elastic2d

        return elastic2d(
            self.lamb,
            self.mu,
            self.buoyancy,
            self.grid_spacing,
            self.dt,
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=self.accuracy,
            pml_width=self.pml_width,
            pml_freq=self.pml_freq,
            nt=nt,
            storage=self.storage,
            sample_steps=self.sample_steps,
            ckpt_steps=self.ckpt_steps,
        )


class TM2D(_FWIModule):
    """2D TM Maxwell FWI: ``TM2D(epsilon, sigma, mu, grid_spacing, dt)``.

    Gradients are returned for all three models:
    ``(dL/depsilon, dL/dsigma, dL/dmu)``.
    """

    _model_names = ("epsilon", "sigma", "mu")

    def __init__(self, epsilon, sigma, mu, grid_spacing, dt, accuracy=2,
                 pml_width=20, requires_grad=True, storage="auto",
                 sample_steps=1, ckpt_steps=None):
        super().__init__(dt, accuracy, pml_width, 25.0, requires_grad,
                         storage, sample_steps, ckpt_steps)
        self.grid_spacing = grid_spacing
        self._register_model("epsilon", epsilon)
        self._register_model("sigma", sigma)
        self._register_model("mu", mu)

    def _run(self, amp, srcs, recs, nt):
        from .em.em2d_tm import em2d_tm

        return em2d_tm(
            self.epsilon,
            self.sigma,
            self.mu,
            self.grid_spacing,
            self.dt,
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=self.accuracy,
            pml_width=self.pml_width,
            nt=nt,
            storage=self.storage,
            sample_steps=self.sample_steps,
            ckpt_steps=self.ckpt_steps,
        )


class EM3D(_FWIModule):
    """3D Maxwell FWI: ``EM3D(epsilon, sigma, mu, grid_spacing, dt)``.

    Gradients are returned for all three models:
    ``(dL/depsilon, dL/dsigma, dL/dmu)``.
    """

    _model_names = ("epsilon", "sigma", "mu")

    def __init__(self, epsilon, sigma, mu, grid_spacing, dt, accuracy=2,
                 pml_width=20, requires_grad=True, storage="auto",
                 sample_steps=1, ckpt_steps=None, source_component="ey",
                 receiver_component="ey"):
        super().__init__(dt, accuracy, pml_width, 25.0, requires_grad,
                         storage, sample_steps, ckpt_steps)
        self.grid_spacing = grid_spacing
        self.source_component = source_component
        self.receiver_component = receiver_component
        self._register_model("epsilon", epsilon)
        self._register_model("sigma", sigma)
        self._register_model("mu", mu)

    def _run(self, amp, srcs, recs, nt):
        from .em.em3d import em3d

        return em3d(
            self.epsilon,
            self.sigma,
            self.mu,
            self.grid_spacing,
            self.dt,
            source_amplitudes=amp,
            source_locations=srcs,
            receiver_locations=recs,
            accuracy=self.accuracy,
            pml_width=self.pml_width,
            nt=nt,
            storage=self.storage,
            sample_steps=self.sample_steps,
            ckpt_steps=self.ckpt_steps,
            source_component=self.source_component,
            receiver_component=self.receiver_component,
        )


__all__ = ["Scalar", "Scalar3D", "Elastic", "TM2D", "EM3D"]
