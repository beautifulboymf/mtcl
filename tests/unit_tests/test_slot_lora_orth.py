# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

from rlinf.models.slot_lora.orth import orth_error, orthogonalize


def _z(rows=16, cols=64, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g, dtype=dtype)


def _orthonormal_rows(rows=16, cols=64, seed=0):
    """Z whose Gram matrix is exactly I -- the degenerate spectrum for eigh."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(cols, rows, generator=g, dtype=torch.float64))
    return q.transpose(-2, -1).contiguous()


def _z_with_cond(cond, rows=256, cols=1024, seed=0, dtype=torch.float32):
    """Z with log-spaced singular values, so ``cond(Z)`` is exactly ``cond``.

    Built in float64 and cast down, so the spectrum -- the only thing the
    Newton-Schulz truncation error depends on -- is set exactly and does not
    drift with the platform's QR.
    """
    g = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(rows, rows, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(cols, rows, generator=g, dtype=torch.float64))
    s = torch.logspace(0.0, -math.log10(cond), rows, dtype=torch.float64)
    return ((u * s) @ v.transpose(-2, -1)).to(dtype)


class _RecordTF32(torch.overrides.TorchFunctionMode):
    """Records the ambient TF32 setting that each intercepted matmul ran under."""

    def __init__(self):
        self.seen = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if "matmul" in getattr(func, "__name__", ""):
            self.seen.append(torch.backends.cuda.matmul.allow_tf32)
        return func(*args, **(kwargs or {}))


class TestOrthogonalize:
    def test_rows_are_orthonormal(self):
        a = orthogonalize(_z())
        eye = torch.eye(a.shape[0], dtype=a.dtype)
        assert torch.allclose(a @ a.T, eye, atol=1e-10)

    def test_scale_invariant(self):
        z = _z()
        assert torch.allclose(orthogonalize(3.7 * z), orthogonalize(z), atol=1e-10)

    def test_scale_invariance_holds_at_small_scale(self):
        # An ABSOLUTE eigenvalue clamp breaks here: at c=1e-4 the smallest
        # eigenvalue of (cZ)(cZ)^T is 7.4e-7, under the old eps=1e-6 floor,
        # which moved the result by 9.0e-3 and left orth_error at 0.447.
        z = _z(rows=64, cols=256, dtype=torch.float32)
        a = orthogonalize(z)
        a_scaled = orthogonalize(1e-4 * z)
        assert torch.allclose(a_scaled, a, atol=1e-5)
        assert orth_error(a_scaled).item() < 1e-4

    def test_preserves_row_space(self):
        # projecting Z onto span(A rows) must return Z exactly
        z = _z()
        a = orthogonalize(z)
        assert torch.allclose(z @ a.T @ a, z, atol=1e-8)

    def test_is_differentiable(self):
        z = _z().requires_grad_(True)
        orthogonalize(z).sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_gradcheck_matches_numerical(self):
        # Stronger than test_is_differentiable: an implementation that wrongly
        # detached the Gram matrix would still produce a finite gradient.
        z = _z(rows=6, cols=16).requires_grad_(True)
        assert torch.autograd.gradcheck(orthogonalize, (z,), atol=1e-5)

    def test_degenerate_spectrum_gradient_is_finite(self):
        # Gram == I: every eigenvalue is equal, so eigh's 1/(lambda_i - lambda_j)
        # backward is NaN here even though its forward is exact.
        z = _orthonormal_rows().requires_grad_(True)
        orthogonalize(z).sum().backward()
        assert torch.isfinite(z.grad).all()

    def test_degenerate_spectrum_gradient_matches_numerical(self):
        # Strictly stronger than finiteness at the same point and for the same
        # runtime: a gradient that is finite but wrong (a detached Gram, or a
        # checkpointed iteration that recomputes something different) is still
        # finite here, and only gradcheck catches it.
        z = _orthonormal_rows(rows=6, cols=16).requires_grad_(True)
        assert torch.autograd.gradcheck(orthogonalize, (z,), atol=1e-5)

    def test_eigh_backend_still_available(self):
        z = _z()
        a_ns = orthogonalize(z, method="ns")
        a_eigh = orthogonalize(z, method="eigh")
        assert torch.allclose(a_ns, a_eigh, atol=1e-10)

    def test_raises_when_rows_exceed_cols(self):
        with pytest.raises(ValueError, match="rows <= cols"):
            orthogonalize(_z(rows=64, cols=32))

    def test_ns_converges_at_production_rank(self):
        # Production shape: R=256 slots, smallest LoRA'd d_in ~1024.
        a = orthogonalize(_z(rows=256, cols=1024, dtype=torch.float32))
        assert orth_error(a).item() < 1e-3

    def test_ns_holds_at_moderate_conditioning(self):
        # iters=12 is only validated INSIDE an envelope, and past it the error is
        # a cliff rather than a slope. Measured fp32 at this shape (R=256,
        # d_in=1024, log-spaced spectrum): cond(Z) 10 -> 6.7e-5, 15 -> 1.1e-4,
        # 20 -> 1.07e-3, 30 -> 5.1e-2, 50 -> 0.64.
        #
        # Pinning the edge of that envelope is what stops a future reduction of
        # `iters` from passing silently: at cond(Z)=20 the ladder is iters 11 ->
        # 6.8e-2, 12 -> 1.07e-3, 13 -> 1.6e-4 (the fp32 floor), so the threshold
        # below sits 60x under iters=11 and 2x over iters=12.
        z = _z_with_cond(20.0)
        assert orth_error(orthogonalize(z)).item() < 2e-3

    def test_fp32_floor_survives_autocast(self):
        # autocast intercepts matmul per-op, so an explicit .to(float32) is not
        # enough on its own -- the computation must disable autocast.
        z = _z(rows=64, cols=256, dtype=torch.float32)
        a_outside = orthogonalize(z)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            a_inside = orthogonalize(z)
        # BITWISE equality, not "close enough": the guard either covers the whole
        # computation or it does not, and a tolerance would pass a guard that had
        # been applied to only some of the matmuls.
        assert torch.equal(a_inside, a_outside)
        assert torch.equal(orth_error(a_inside), orth_error(a_outside))

    def test_preserves_input_dtype(self):
        a = orthogonalize(_z().to(torch.bfloat16))
        assert a.dtype == torch.bfloat16

    def test_bf16_orth_error_floor_at_production_rank(self):
        # The result is cast back to the input dtype, so at R=256 the bf16
        # rounding floor is ~1e-2. That is rounding, not a broken run: any
        # monitoring threshold set at 1e-6 would reject a correct model.
        z = _z(rows=256, cols=1024, dtype=torch.float32).to(torch.bfloat16)
        err = orth_error(orthogonalize(z)).item()
        assert 1e-3 < err < 0.05

    def test_batched_matches_looped(self):
        g = torch.Generator().manual_seed(3)
        z = torch.randn(4, 8, 32, generator=g, dtype=torch.float64)
        batched = orthogonalize(z)
        looped = torch.stack([orthogonalize(z[i]) for i in range(z.shape[0])])
        assert torch.allclose(batched, looped, atol=1e-12)


class TestOrthError:
    def test_orth_error_is_zero_after_orthogonalization(self):
        assert orth_error(orthogonalize(_z())).item() < 1e-10

    def test_orth_error_is_positive_for_raw_matrix(self):
        assert orth_error(_z()).item() > 1.0

    def test_orth_error_is_per_matrix_for_batched_input(self):
        g = torch.Generator().manual_seed(5)
        z = torch.randn(3, 8, 32, generator=g, dtype=torch.float64)
        per_matrix = orth_error(z)
        assert per_matrix.shape == (3,)
        looped = torch.stack([orth_error(z[i]) for i in range(z.shape[0])])
        assert torch.allclose(per_matrix, looped)
        # unbatched input still reduces to a 0-dim tensor
        assert orth_error(z[0]).shape == ()

    def test_orth_error_rejects_non_matrix(self):
        # Without the guard this is an IndexError from deep inside the reduction,
        # while orthogonalize raises a clean ValueError for the same input.
        with pytest.raises(ValueError, match="needs a matrix"):
            orth_error(torch.randn(8))

    def test_orth_error_builds_no_graph(self):
        z = _z().requires_grad_(True)
        assert orth_error(z).requires_grad is False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the TF32 guard is a deliberate no-op when CUDA is unavailable",
)
class TestTF32Guard:
    """TF32 is a second, independent way to lose the fp32 guarantee.

    It is a process-global backend setting that other parts of this repo turn ON
    (``set_float32_matmul_precision("high")`` in the FSDP IQL policy worker, and
    ``allow_tf32 = True`` in the OpenSora world-model env), so this file cannot
    assume it is off. Measured on A100 with ``allow_tf32=True``: the bf16
    orth_error floor rises 0.0189 -> 0.02922 at d_in=1024 and 0.0094 -> 0.02475
    at d_in=4096, and an fp32 input goes 7.9e-6 -> 2.03e-2 -- which puts a
    healthy run on top of a collapsed one (0.054 at cond(Z)=30), destroying the
    only sentinel the training loop has.

    These run on CPU tensors: they check the GUARD (that the matmuls see TF32
    off, and that the ambient setting is put back), not the arithmetic, which
    only differs on a TF32-capable device.
    """

    def test_matmuls_run_with_tf32_off(self):
        z = _z(rows=64, cols=256, dtype=torch.float32)
        prev = torch.get_float32_matmul_precision()
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            for fn in (orthogonalize, orth_error):
                recorder = _RecordTF32()
                with recorder:
                    fn(z)
                assert recorder.seen, f"{fn.__name__} ran no matmul to observe"
                assert not any(recorder.seen)
        finally:
            torch.set_float32_matmul_precision(prev)

    def test_result_is_unchanged_by_ambient_tf32(self):
        # Same style as test_fp32_floor_survives_autocast: with the guard the two
        # results are BITWISE equal, which also catches a partially applied guard.
        z = _z(rows=64, cols=256, dtype=torch.float32)
        prev = torch.get_float32_matmul_precision()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            a_off, err_off = orthogonalize(z), orth_error(orthogonalize(z))
            torch.backends.cuda.matmul.allow_tf32 = True
            a_on, err_on = orthogonalize(z), orth_error(orthogonalize(z))
        finally:
            torch.set_float32_matmul_precision(prev)
        assert torch.equal(a_on, a_off)
        assert torch.equal(err_on, err_off)

    def test_ambient_tf32_state_is_restored(self):
        # The guard must put back exactly what it found. "medium" is the case a
        # bare allow_tf32 bool cannot round-trip: it reads back as True, and
        # writing True restores "high", silently upgrading the caller's setting.
        z = _z(rows=8, cols=32, dtype=torch.float32)
        prev = torch.get_float32_matmul_precision()
        prev_cudnn = torch.backends.cudnn.allow_tf32
        try:
            for precision in ("highest", "high", "medium"):
                torch.set_float32_matmul_precision(precision)
                torch.backends.cudnn.allow_tf32 = True
                orthogonalize(z)
                orth_error(z)
                assert torch.get_float32_matmul_precision() == precision
                assert torch.backends.cudnn.allow_tf32
        finally:
            torch.set_float32_matmul_precision(prev)
            torch.backends.cudnn.allow_tf32 = prev_cudnn
