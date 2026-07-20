"""Static contracts for memory-efficient logits in the C0-C5 benchmarks."""

import ast
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"

EXPECTED_MANUAL_FORWARDS = {
    "benchmark_c0_baseline.py": Counter({"run_speed": 3}),
    "benchmark_c1_tailorkv.py": Counter(
        {"_prefill_and_compress": 3, "run_speed": 2, "generate_prepared_c1": 1}
    ),
    "benchmark_c2_qaq.py": Counter(
        {
            "_precompute_q_norm": 1,
            "_prefill_with_window_attn": 3,
            "run_speed": 3,
            "generate_prepared_c2": 2,
        }
    ),
    "benchmark_c3_kvquant.py": Counter(
        {"_prefill_and_quantize": 1, "run_speed": 2, "generate_prepared_c3": 1}
    ),
    "benchmark_c4_dynamickv.py": Counter(
        {"_prefill_and_compress": 3, "run_speed": 2, "generate_prepared_c4": 1}
    ),
    "benchmark_c5_adakv.py": Counter(
        {"_prefill_and_compress": 3, "run_speed": 2, "generate_prepared_c5": 1}
    ),
}

EXPECTED_USE_CACHE_FALSE = {
    "benchmark_c0_baseline.py": Counter({"run_speed": 1}),
    "benchmark_c1_tailorkv.py": Counter({"run_speed": 1}),
    "benchmark_c2_qaq.py": Counter({"_precompute_q_norm": 1, "run_speed": 1}),
    "benchmark_c3_kvquant.py": Counter({"run_speed": 1}),
    "benchmark_c4_dynamickv.py": Counter({"run_speed": 1}),
    "benchmark_c5_adakv.py": Counter({"run_speed": 1}),
}

EXPECTED_OUTPUT_ATTENTIONS_TRUE = {
    "benchmark_c0_baseline.py": Counter(),
    "benchmark_c1_tailorkv.py": Counter({"_prefill_and_compress": 2}),
    "benchmark_c2_qaq.py": Counter(
        {
            "_prefill_with_window_attn": 2,
            "run_speed": 1,
            "generate_prepared_c2": 1,
        }
    ),
    "benchmark_c3_kvquant.py": Counter(),
    "benchmark_c4_dynamickv.py": Counter({"_prefill_and_compress": 2}),
    "benchmark_c5_adakv.py": Counter({"_prefill_and_compress": 2}),
}

EXPECTED_OUTPUT_ATTENTIONS_FALSE = {
    "benchmark_c0_baseline.py": Counter(),
    "benchmark_c1_tailorkv.py": Counter(),
    "benchmark_c2_qaq.py": Counter({"_precompute_q_norm": 1}),
    "benchmark_c3_kvquant.py": Counter(),
    "benchmark_c4_dynamickv.py": Counter(),
    "benchmark_c5_adakv.py": Counter(),
}


class _ModelCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self._function_stack: list[str] = []
        self.manual_forwards: list[tuple[str, ast.Call]] = []
        self.generate_calls: list[tuple[str, ast.Call]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        function = self._function_stack[-1] if self._function_stack else "<module>"
        if isinstance(node.func, ast.Name) and node.func.id == "model":
            self.manual_forwards.append((function, node))
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
            and node.func.attr == "generate"
        ):
            self.generate_calls.append((function, node))
        self.generic_visit(node)


def _calls_for(filename: str) -> _ModelCallVisitor:
    path = SCRIPT_DIR / filename
    visitor = _ModelCallVisitor()
    visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return visitor


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _is_int_one(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and node.value == 1
    )


def _is_bool(node: ast.expr | None, expected: bool) -> bool:
    return isinstance(node, ast.Constant) and node.value is expected


def _location(filename: str, function: str, call: ast.Call) -> str:
    return f"{filename}:{call.lineno} ({function})"


def test_manual_forward_inventory_covers_every_c0_c5_path() -> None:
    actual = {
        filename: Counter(
            function for function, _ in _calls_for(filename).manual_forwards
        )
        for filename in EXPECTED_MANUAL_FORWARDS
    }

    assert actual == EXPECTED_MANUAL_FORWARDS
    assert sum(sum(counts.values()) for counts in actual.values()) == 34


def test_every_non_ppl_manual_forward_keeps_only_final_logits() -> None:
    failures = []
    for filename in EXPECTED_MANUAL_FORWARDS:
        for function, call in _calls_for(filename).manual_forwards:
            if not _is_int_one(_keyword(call, "logits_to_keep")):
                failures.append(_location(filename, function, call))

    assert not failures, "Missing logits_to_keep=1: " + ", ".join(failures)


def test_c0_generate_keeps_only_final_logits_and_cache_contract() -> None:
    calls = _calls_for("benchmark_c0_baseline.py").generate_calls

    assert len(calls) == 1
    function, call = calls[0]
    assert function == "generate_prepared_c0"
    assert _is_int_one(_keyword(call, "logits_to_keep"))
    assert _is_bool(_keyword(call, "use_cache"), True)
    assert _is_bool(_keyword(call, "return_dict_in_generate"), True)
    assert _is_bool(_keyword(call, "do_sample"), False)


def test_manual_forwards_preserve_cache_and_attention_contracts() -> None:
    actual_cache_false = {}
    actual_attentions_true = {}
    actual_attentions_false = {}

    for filename in EXPECTED_MANUAL_FORWARDS:
        cache_false = Counter()
        attentions_true = Counter()
        attentions_false = Counter()

        for function, call in _calls_for(filename).manual_forwards:
            use_cache = _keyword(call, "use_cache")
            assert _is_bool(use_cache, True) or _is_bool(use_cache, False), (
                f"use_cache must remain explicit at "
                f"{_location(filename, function, call)}"
            )
            if _is_bool(use_cache, False):
                cache_false[function] += 1

            output_attentions = _keyword(call, "output_attentions")
            if _is_bool(output_attentions, True):
                attentions_true[function] += 1
            elif _is_bool(output_attentions, False):
                attentions_false[function] += 1
            else:
                assert output_attentions is None, (
                    f"output_attentions must be a literal bool at "
                    f"{_location(filename, function, call)}"
                )

        actual_cache_false[filename] = cache_false
        actual_attentions_true[filename] = attentions_true
        actual_attentions_false[filename] = attentions_false

    assert actual_cache_false == EXPECTED_USE_CACHE_FALSE
    assert actual_attentions_true == EXPECTED_OUTPUT_ATTENTIONS_TRUE
    assert actual_attentions_false == EXPECTED_OUTPUT_ATTENTIONS_FALSE
