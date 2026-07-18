#!/usr/bin/env python3
"""
Regression test for tools/gen_a2_fast_kernel.py's SkipHead head_sum emission.

Commit dbefaac ("[Fork] fix: kernel generator must emit the SkipHead head_sum
accumulate") fixed a real bug: gen_parametric()'s generated C++ template was missing
the _head_sum accumulation line for the parametric fast path. A model shipped for any
NEW channel width generated after that point would have compiled cleanly and then
silently rendered RESIDUAL audio under a SKIP head (corr ~0.3 against the real model,
no error) -- exactly the silent-wrong-audio class the SKIP-ONLY migration exists to
eliminate. This test pins that fix so the same class of bug (an emit step silently
dropped for some N) can't reappear unnoticed.

No pytest/gtest infra exists for Python in this repo; this matches the plain
assert-based style of tools/test/*.cpp. Run directly:
    python3 tools/test_gen_a2_fast_kernel.py
"""
import importlib.util
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_THIS_DIR, "gen_a2_fast_kernel.py")

_spec = importlib.util.spec_from_file_location("gen_a2_fast_kernel", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


def check_gen_parametric_emits_head_sum_accumulate(N):
    """gen_parametric(N) must accumulate every channel's post-activation output into
    _head_sum -- the exact emit commit dbefaac added. Missing it silently renders
    residual audio through a skip head."""
    out = _gen.gen_parametric(N)

    accumulator_ptr = f"_head_sum[static_cast<size_t>(f) * {N}]"
    assert accumulator_ptr in out, (
        f"gen_parametric({N}): missing '{accumulator_ptr}' -- the SkipHead head_sum "
        "accumulate emit appears to have regressed"
    )
    for i in range(N):
        accumulate_line = f"hsum[{i}] += a{i};"
        assert accumulate_line in out, (
            f"gen_parametric({N}): missing '{accumulate_line}' -- channel {i}'s "
            "post-activation output is not being accumulated into the skip head sum"
        )


def check_gen_nonparametric_emits_head_sum_accumulate(N):
    """The non-parametric generator (A2FastModel<N>, always skip-only) has always
    accumulated into _head_sum; pinned here too so both generators are guarded by
    the same class of check."""
    out = _gen.gen_nonparametric(N)

    accumulator_ptr = f"_head_sum[static_cast<size_t>(f) * {N}]"
    assert accumulator_ptr in out, f"gen_nonparametric({N}): missing '{accumulator_ptr}'"
    for i in range(N):
        accumulate_line = f"hsum[{i}] += a{i};"
        assert accumulate_line in out, f"gen_nonparametric({N}): missing '{accumulate_line}'"


def main():
    # Representative widths: the two hand-optimized endpoints (3, 8) plus every
    # intermediate width this build actually ships (4, 5, 6, 7).
    for N in (3, 4, 5, 6, 7, 8):
        check_gen_parametric_emits_head_sum_accumulate(N)
        check_gen_nonparametric_emits_head_sum_accumulate(N)
        print(f"N={N}: OK (parametric + non-parametric head_sum accumulate present)")
    print("Success!")


if __name__ == "__main__":
    main()
