"""CUDA stream, long-run, and allocation-lifetime regressions."""

import gc

import pytest
import torch

from nami.scalar.scalar2d import scalar2d
from tests.test_wavefield_io import _build_continuation_case, _split_result

PROPAGATOR_NAMES = (
    "scalar2d",
    "scalar2d_born",
    "scalar3d",
    "scalar3d_born",
    "elastic2d",
    "elastic2d_born",
    "em2d_tm",
    "em2d_tm_born",
    "em3d",
    "em3d_born",
)


@pytest.mark.parametrize("case_name", PROPAGATOR_NAMES)
def test_all_propagators_run_on_nondefault_cuda_stream(case_name):
    run, amplitude = _build_continuation_case(case_name)
    expected, _ = _split_result(run(amplitude), has_state=False)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(stream):
        actual, _ = _split_result(run(amplitude), has_state=False)
        actual = tuple(output.clone() for output in actual)
    stream.synchronize()

    for output, reference in zip(actual, expected, strict=True):
        assert torch.equal(output, reference)


@pytest.mark.parametrize("case_name", PROPAGATOR_NAMES)
def test_all_propagators_long_forward_stays_finite(case_name):
    run, short_amplitude = _build_continuation_case(case_name, torch.float32)
    amplitude = torch.zeros(
        *short_amplitude.shape[:-1], 256, dtype=short_amplitude.dtype,
    )
    amplitude[..., 1] = short_amplitude[..., 1]
    outputs, _ = _split_result(run(amplitude), has_state=False)
    for output in outputs:
        assert output.shape[0] == amplitude.shape[-1]
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output).item() > 0


def _checkpointed_scalar_run(
    base_model, base_amplitude, sources, receivers, callback=None,
):
    model = base_model.clone().requires_grad_(True)
    amplitude = base_amplitude.clone().requires_grad_(True)
    output = scalar2d(
        model,
        5.0,
        5.0e-4,
        amplitude,
        sources,
        receivers,
        pml_width=4,
        pml_freq=25.0,
        storage="auto",
        sample_steps=2,
        ckpt_steps=11,
        forward_callback=callback,
        callback_frequency=13,
    )
    gradients = torch.autograd.grad(output.square().sum(), (model, amplitude))
    return output.detach().clone(), tuple(
        gradient.detach().clone() for gradient in gradients
    )


def test_checkpoint_backward_and_callback_are_stream_safe():
    device, dtype = torch.device("cuda"), torch.float64
    ny, nx, nt = 28, 32, 96
    model = torch.full((ny, nx), 1800.0, device=device, dtype=dtype)
    model[ny // 2 :] = 2100.0
    amplitude = torch.zeros(1, 1, nt, device=device, dtype=dtype)
    amplitude[..., 1:4] = torch.tensor(
        [1.0, -0.5, 0.2], device=device, dtype=dtype,
    )
    sources = torch.tensor([[[ny // 2, nx // 2]]], device=device)
    receivers = torch.tensor(
        [[[ny // 2, nx // 2 + 3], [ny // 2 + 2, nx // 2]]], device=device,
    )
    reference, reference_gradients = _checkpointed_scalar_run(
        model, amplitude, sources, receivers,
    )

    callbacks = []

    def callback(state):
        callbacks.append(
            (state.step, state.get_wavefield("u", "inner").square().sum().item())
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(stream):
        actual, actual_gradients = _checkpointed_scalar_run(
            model, amplitude, sources, receivers, callback,
        )
    stream.synchronize()

    assert [step for step, _ in callbacks] == list(range(0, nt, 13))
    assert callbacks[-1][1] > 0
    assert torch.equal(actual, reference)
    for gradient, reference_gradient in zip(
        actual_gradients, reference_gradients, strict=True,
    ):
        assert torch.equal(gradient, reference_gradient)


def test_repeated_checkpoint_backward_releases_allocations():
    device, dtype = torch.device("cuda"), torch.float64
    ny, nx, nt = 32, 36, 160
    model = torch.full((ny, nx), 1800.0, device=device, dtype=dtype)
    amplitude = torch.zeros(1, 1, nt, device=device, dtype=dtype)
    amplitude[..., 1] = 1.0
    sources = torch.tensor([[[16, 18]]], device=device)
    receivers = torch.tensor([[[16, 22]]], device=device)

    def run_once():
        current_model = model.clone().requires_grad_(True)
        current_amplitude = amplitude.clone().requires_grad_(True)
        output = scalar2d(
            current_model, 5.0, 5.0e-4, current_amplitude, sources, receivers,
            pml_width=4, storage="auto", ckpt_steps=17,
        )
        torch.autograd.grad(
            output.square().sum(), (current_model, current_amplitude),
        )

    for _ in range(2):
        run_once()
    torch.cuda.synchronize(device)
    gc.collect()

    allocated = []
    for _ in range(8):
        run_once()
        torch.cuda.synchronize(device)
        gc.collect()
        allocated.append(torch.cuda.memory_allocated(device))

    assert max(allocated) - min(allocated) < 1024 * 1024
