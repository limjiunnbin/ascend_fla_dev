"""A2 KDA qualification checks, independent of the A5 NPU session fixture."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("a212_benchmark_tests", ROOT / "benchmarks/verify_real_shapes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a2_comparator_rejects_finite_wrong_head_and_nonfinite(benchmark):
    expected = torch.tensor([[.02, .03], [-.05, .08]])
    wrong = expected.flip(0)
    assert torch.equal(wrong.norm(), expected.norm())
    assert benchmark.a2_metrics(expected, expected)["ok"]
    result = benchmark.a2_metrics(wrong, expected)
    assert result["finite"] and not result["ok"] and result["relative_l2"] > .05
    assert not benchmark.a2_metrics(expected * float("nan"), expected)["finite"]
    assert not benchmark.a2_metrics(expected * float("inf"), expected)["ok"]


def test_a2_comparator_does_not_hide_zero_reference_error(benchmark):
    zero = torch.zeros(2)
    assert benchmark.a2_metrics(zero, zero)["relative_l2"] == 0
    for amplitude in (1e-20, 1e-30):
        bad = benchmark.a2_metrics(torch.tensor([amplitude, 0.]), zero)
        assert bad["finite"] and bad["relative_l2"] is None and not bad["ok"]


def test_a2_gate_calibration_preserves_real_shape_and_rng(benchmark):
    before = torch.get_rng_state().clone()
    gate, calibration = benchmark.a2_default_gate(128, 0, 32.)
    assert torch.equal(before, torch.get_rng_state())
    assert gate.shape == (1, 128, 32, 128)
    assert gate.dtype == torch.float32 and gate.is_contiguous()
    assert bool(torch.isfinite(gate).all()) and bool((gate <= 0).all())
    assert calibration["hidden_size"] == 2304
    assert calibration["measured_span"] == pytest.approx(32., rel=2e-6)
    assert calibration["initialization_span"] != calibration["measured_span"]


@pytest.mark.parametrize("target", [0., -1., float("inf"), float("nan")])
def test_a2_gate_calibration_rejects_invalid_target(benchmark, target):
    with pytest.raises(ValueError, match="finite and positive"):
        benchmark.a2_default_gate(64, 1, target)


def test_a2_public_entry_stays_unqualified_before_compile(monkeypatch):
    from ascend_fla import platform
    from ascend_fla.ops.kda import chunk_kda_fwd
    from ascend_fla.runtime import compile as compiler

    def forbidden(*args, **kwargs):
        pytest.fail("Unqualified A2 entry attempted compilation")

    monkeypatch.setattr(compiler, "compile_kernel", forbidden)
    assert platform.capability("a2")["qualified"] is False
    with pytest.raises(RuntimeError, match="未验收"):
        chunk_kda_fwd(None, None, None, None, None, device="a2", block_dim=1)


@pytest.mark.parametrize("bd", [1, 2])
def test_a2_unit_c1_hv32_is_declared_positive(benchmark, bd):
    if not os.environ.get("ASCRIPTOR_WORKSPACE"):
        pytest.skip("Set ASCRIPTOR_WORKSPACE to the selected owner source bundle")
    unit, _, _ = benchmark._a2_unit()
    x = benchmark.make_inputs(1, 32, 32, 1, 8.)
    inputs = dict(q=x["q"], k=x["k"], v=x["v"], g_raw=x["g"],
                  beta=x["beta"], initial_state=x["h0"])
    unit.validate_inputs(inputs, {"block_dim": bd})
    with pytest.raises(ValueError, match="block_dim"):
        unit.validate_inputs(inputs, {"block_dim": 3})
    for name in ("q", "k", "v"):
        with pytest.raises(ValueError, match="bfloat16"):
            unit.validate_inputs({**inputs, name: inputs[name].float()}, {"block_dim": bd})
