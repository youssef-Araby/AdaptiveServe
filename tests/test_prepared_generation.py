"""CPU and static tests for the C0-C5 prepared-token generation boundary."""

import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

MODULE_NAMES = [
    "benchmark_c0_baseline",
    "benchmark_c1_tailorkv",
    "benchmark_c2_qaq",
    "benchmark_c3_kvquant",
    "benchmark_c4_dynamickv",
    "benchmark_c5_adakv",
]
MODULES = {
    name: importlib.import_module(name)
    for name in MODULE_NAMES
}

PREPARED_APIS = {
    "benchmark_c0_baseline.py": ("generate_prepared_c0", "_generate"),
    "benchmark_c1_tailorkv.py": ("generate_prepared_c1", "_generate_tkv"),
    "benchmark_c2_qaq.py": ("generate_prepared_c2", "_generate_qaq"),
    "benchmark_c3_kvquant.py": ("generate_prepared_c3", "_generate_kvq"),
    "benchmark_c4_dynamickv.py": ("generate_prepared_c4", "_generate_dkv"),
    "benchmark_c5_adakv.py": ("generate_prepared_c5", "_generate_ada"),
}


class _TokenizerWithoutEncode:
    eos_token_id = 9

    def __init__(self) -> None:
        self.decoded_ids = None

    def encode(self, *args, **kwargs):
        raise AssertionError("prepared generation must not tokenize")

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        self.decoded_ids = list(token_ids)
        return ",".join(str(token_id) for token_id in token_ids)


class _FakeDecodeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(sliding_window=None)
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append((input_ids.clone(), kwargs))
        logits = torch.full((1, 1, 10), -10.0)
        logits[..., 9] = 10.0
        logits[..., 4] = 9.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=object(),
            attentions=[],
        )


def _prefill_logits() -> torch.Tensor:
    logits = torch.full((1, 10), -10.0)
    logits[..., 9] = 10.0
    logits[..., 4] = 9.0
    return logits


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_prepared_input_boundary_preserves_one_dimensional_ids(module_name) -> None:
    module = MODULES[module_name]

    from_list = module._prepared_input_tensor([7, 2, 8], "cpu")
    from_tensor = module._prepared_input_tensor(torch.tensor([7, 2, 8]), "cpu")

    assert from_list.tolist() == [[7, 2, 8]]
    assert from_tensor.tolist() == [[7, 2, 8]]
    assert from_list.dtype == torch.long
    with pytest.raises(ValueError, match="one-dimensional"):
        module._prepared_input_tensor(torch.tensor([[7, 2, 8]]), "cpu")
    with pytest.raises(ValueError, match="must not be empty"):
        module._prepared_input_tensor([], "cpu")


@pytest.mark.parametrize("module_name", MODULE_NAMES[1:])
def test_manual_greedy_masks_all_stop_ids_until_minimum(module_name) -> None:
    module = MODULES[module_name]
    logits = torch.tensor([[0.0, 8.0, 9.0, 1.0]])
    original = logits.clone()
    stop_ids = frozenset({2, 3})

    before_minimum = module._greedy_next_token(
        logits, stop_ids, generated_count=0, min_new_tokens=1
    )
    after_minimum = module._greedy_next_token(
        logits, stop_ids, generated_count=1, min_new_tokens=1
    )

    assert before_minimum.item() == 1
    assert after_minimum.item() == 2
    assert torch.equal(logits, original)


def test_c0_prepared_generation_uses_exact_ids_and_common_controls() -> None:
    module = MODULES["benchmark_c0_baseline"]
    tokenizer = _TokenizerWithoutEncode()

    class FakeModel:
        def __init__(self) -> None:
            self.input_ids = None
            self.kwargs = None

        def generate(self, input_ids, **kwargs):
            self.input_ids = input_ids.clone()
            self.kwargs = kwargs
            sequences = torch.tensor([[7, 2, 8, 4, 9, 6]])
            key = torch.zeros((1, 1, 2, 1))
            value = torch.zeros((1, 1, 2, 1))
            return SimpleNamespace(
                sequences=sequences,
                past_key_values=((key, value),),
            )

    model = FakeModel()
    prediction, n_generated, kv_stats = module.generate_prepared_c0(
        model,
        tokenizer,
        [7, 2, 8],
        4,
        "cpu",
        stop_token_ids={10, 9},
        min_new_tokens=1,
    )

    assert model.input_ids.tolist() == [[7, 2, 8]]
    assert model.kwargs["attention_mask"].tolist() == [[1, 1, 1]]
    assert model.kwargs["eos_token_id"] == [9, 10]
    assert model.kwargs["min_new_tokens"] == 1
    assert model.kwargs["do_sample"] is False
    assert prediction == "4"
    assert tokenizer.decoded_ids == [4]
    assert n_generated == 1
    assert kv_stats == {
        "kv_bytes": 16.0,
        "kv_bytes_fp16": 16.0,
        "compression": 1.0,
    }


def test_c0_native_generation_masks_stop_token_before_minimum() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    module = MODULES["benchmark_c0_baseline"]

    class Tokenizer:
        eos_token_id = 0

        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ",".join(str(token_id) for token_id in token_ids)

    config = LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=0,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    prediction, n_generated, kv_stats = module.generate_prepared_c0(
        model,
        Tokenizer(),
        [1, 2],
        4,
        "cpu",
        stop_token_ids={0},
        min_new_tokens=1,
    )

    # All logits tie at zero. EOS token 0 would win immediately unless the
    # native generation logits processor masks it for the first token.
    assert prediction == "1"
    assert n_generated == 1
    assert kv_stats["kv_bytes"] == kv_stats["kv_bytes_fp16"]
    assert kv_stats["compression"] == 1.0


def test_c0_empty_stop_set_forces_the_full_decode_budget() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    module = MODULES["benchmark_c0_baseline"]
    class Tokenizer:
        eos_token_id = 0

        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ",".join(str(token_id) for token_id in token_ids)

    config = LlamaConfig(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=0,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    prediction, n_generated, _ = module.generate_prepared_c0(
        model,
        Tokenizer(),
        [1, 2],
        4,
        "cpu",
        stop_token_ids=frozenset(),
    )

    assert n_generated == 4
    assert prediction == "0,0,0,0"


def test_c2_q_norm_calibration_fails_without_full_layer_coverage() -> None:
    module = MODULES["benchmark_c2_qaq"]

    class Hook:
        @staticmethod
        def remove():
            return None

    class Attention:
        @staticmethod
        def register_forward_hook(*args, **kwargs):
            return Hook()

    class Layer:
        self_attn = Attention()

    class Model:
        model = SimpleNamespace(layers=[Layer(), Layer()])

        @staticmethod
        def __call__(*args, **kwargs):
            return SimpleNamespace()

    with pytest.raises(RuntimeError, match="captured 0 of 2"):
        module._precompute_q_norm(
            Model(),
            torch.tensor([[1, 2, 3]]),
            "cpu",
        )


@pytest.mark.parametrize(
    ("module_name", "api_name", "prefill_name", "method_kwargs"),
    [
        (
            "benchmark_c1_tailorkv",
            "generate_prepared_c1",
            "_prefill_and_compress",
            {"q_layers": {0}},
        ),
        (
            "benchmark_c3_kvquant",
            "generate_prepared_c3",
            "_prefill_and_quantize",
            {},
        ),
        (
            "benchmark_c4_dynamickv",
            "generate_prepared_c4",
            "_prefill_and_compress",
            {"budget": 8},
        ),
        (
            "benchmark_c5_adakv",
            "generate_prepared_c5",
            "_prefill_and_compress",
            {"n_kv_heads": 1, "budget_per_head": 8},
        ),
    ],
)
def test_manual_prepared_generators_mask_stop_and_report_common_kv_stats(
    monkeypatch,
    module_name,
    api_name,
    prefill_name,
    method_kwargs,
) -> None:
    module = MODULES[module_name]
    tokenizer = _TokenizerWithoutEncode()
    model = _FakeDecodeModel()
    captured = {}

    def fake_prefill(model_arg, ids_t, *args, **kwargs):
        assert model_arg is model
        captured["ids"] = ids_t.clone()
        captured["kwargs"] = kwargs
        stats = {
            "compressed_kv_bytes": 100.0,
            "full_kv_bytes": 200.0,
            "fp16_bytes_per_token": 10.0,
        }
        return object(), _prefill_logits(), stats

    monkeypatch.setattr(module, prefill_name, fake_prefill)
    if module_name != "benchmark_c1_tailorkv":
        monkeypatch.setattr(module, "_fp16_bytes_per_token", lambda model: 10.0)

    generate = getattr(module, api_name)
    prediction, n_generated, kv_stats = generate(
        model,
        tokenizer,
        [7, 2, 8],
        3,
        "cpu",
        stop_token_ids={9},
        min_new_tokens=1,
        **method_kwargs,
    )

    assert captured["ids"].tolist() == [[7, 2, 8]]
    assert prediction == "4"
    assert tokenizer.decoded_ids == [4]
    assert n_generated == 1
    assert kv_stats["kv_bytes"] == 110.0
    assert kv_stats["kv_bytes_fp16"] == 210.0
    assert kv_stats["compression"] == pytest.approx(210.0 / 110.0)
    assert len(model.calls) == 1
    decode_ids, decode_kwargs = model.calls[0]
    assert decode_ids.tolist() == [[4]]
    assert decode_kwargs["use_cache"] is True
    assert decode_kwargs["logits_to_keep"] == 1
    assert decode_kwargs["position_ids"].tolist() == [[3]]


def test_c2_prepared_generation_preserves_qaq_decode_accounting(
    monkeypatch,
) -> None:
    module = MODULES["benchmark_c2_qaq"]
    tokenizer = _TokenizerWithoutEncode()
    model = _FakeDecodeModel()

    class FakeState:
        @staticmethod
        def effective_bytes():
            return 100.0

        @staticmethod
        def effective_avg_bits():
            return 4.0

    def fake_prefill(model_arg, ids_t, window, device):
        assert model_arg is model
        assert ids_t.tolist() == [[7, 2, 8]]
        return object(), _prefill_logits(), []

    monkeypatch.setattr(module, "_prefill_with_window_attn", fake_prefill)
    monkeypatch.setattr(module, "kv_cache_bytes", lambda past: 200.0)
    monkeypatch.setattr(module, "_fp16_bytes_per_token", lambda past: 10.0)
    monkeypatch.setattr(
        module,
        "_qaq_quantize_full_cache_aware",
        lambda past, q_norm, prefill_attns: (past, FakeState()),
    )

    prediction, n_generated, kv_stats = module.generate_prepared_c2(
        model,
        tokenizer,
        [7, 2, 8],
        3,
        "cpu",
        q_norm=2.0,
        stop_token_ids={9},
        min_new_tokens=1,
        attn_aware_decode=False,
    )

    assert prediction == "4"
    assert tokenizer.decoded_ids == [4]
    assert n_generated == 1
    assert kv_stats["n_decode_tokens"] == 1
    assert kv_stats["kv_bytes"] == 110.0
    assert kv_stats["kv_bytes_fp16"] == 210.0
    assert kv_stats["compression"] == pytest.approx(210.0 / 110.0)


def test_manual_prepared_generator_does_not_forward_unused_capped_token(
    monkeypatch,
) -> None:
    module = MODULES["benchmark_c3_kvquant"]
    tokenizer = _TokenizerWithoutEncode()
    model = _FakeDecodeModel()

    monkeypatch.setattr(
        module,
        "_prefill_and_quantize",
        lambda model_arg, ids_t, device: (
            object(),
            _prefill_logits(),
            {"compressed_kv_bytes": 100.0, "full_kv_bytes": 200.0},
        ),
    )
    monkeypatch.setattr(module, "_fp16_bytes_per_token", lambda model_arg: 10.0)

    prediction, n_generated, kv_stats = module.generate_prepared_c3(
        model,
        tokenizer,
        [7, 2, 8],
        1,
        "cpu",
        stop_token_ids={9},
        min_new_tokens=1,
    )

    assert prediction == "4"
    assert n_generated == 1
    assert model.calls == []
    assert kv_stats["n_decode_tokens"] == 0
    assert kv_stats["kv_bytes"] == 100.0
    assert kv_stats["kv_bytes_fp16"] == 200.0


def _c3_legacy_tensor_reference(
    module,
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    outlier_frac: float,
    channel_dim: int,
) -> tuple[torch.Tensor, float]:
    mask, n_outliers = module._outlier_mask(tensor, outlier_frac)
    quantized = module._quantize_dequantize(
        tensor,
        bits=bits,
        group_size=group_size,
        channel_dim=channel_dim,
        exclude_mask=mask,
    )
    mixed = (
        torch.where(mask, tensor, quantized)
        if mask is not None
        else quantized
    )
    axis_len = tensor.shape[channel_dim]
    n_groups = (
        tensor.numel() // axis_len
    ) * ((axis_len + group_size - 1) // group_size)
    effective_bytes = (
        (tensor.numel() - n_outliers) * bits / 8.0
        + n_groups * 4.0
        + n_outliers * 6.0
    )
    return mixed, effective_bytes


@pytest.mark.parametrize("outlier_frac", [0.0, 0.125])
def test_c3_quantize_cache_reuses_storage_and_remains_appendable(
    outlier_frac,
) -> None:
    from transformers.cache_utils import DynamicCache

    module = MODULES["benchmark_c3_kvquant"]
    torch.manual_seed(0)
    key = torch.randn((1, 2, 9, 5), dtype=torch.float16)
    value = torch.randn((1, 2, 9, 5), dtype=torch.float16)
    expected_key, expected_key_bytes = _c3_legacy_tensor_reference(
        module,
        key,
        bits=4,
        group_size=4,
        outlier_frac=outlier_frac,
        channel_dim=2,
    )
    expected_value, expected_value_bytes = _c3_legacy_tensor_reference(
        module,
        value,
        bits=4,
        group_size=4,
        outlier_frac=outlier_frac,
        channel_dim=3,
    )
    expected_bytes = expected_key_bytes + expected_value_bytes
    cache = DynamicCache(
        ddp_cache_data=[(key.clone(), value.clone())],
    )
    layer = cache.layers[0]
    key_pointer = layer.keys.data_ptr()
    value_pointer = layer.values.data_ptr()

    returned, stats = module._quantize_cache(
        cache,
        bits=4,
        group_size=4,
        outlier_frac=outlier_frac,
    )

    assert returned is cache
    assert layer.keys.data_ptr() == key_pointer
    assert layer.values.data_ptr() == value_pointer
    assert torch.equal(layer.keys, expected_key)
    assert torch.equal(layer.values, expected_value)
    assert stats == {
        "compressed_kv_bytes": expected_bytes,
        "compressed_kv_mb": expected_bytes / (1024 ** 2),
        "n_layers": 1,
    }

    appended_key = torch.randn((1, 2, 1, 5), dtype=torch.float16)
    appended_value = torch.randn((1, 2, 1, 5), dtype=torch.float16)
    grown_key, grown_value = cache.update(
        appended_key,
        appended_value,
        layer_idx=0,
    )
    assert grown_key.shape[2] == 10
    assert grown_value.shape[2] == 10
    assert torch.equal(grown_key[:, :, :9, :], expected_key)
    assert torch.equal(grown_value[:, :, :9, :], expected_value)
    assert torch.equal(grown_key[:, :, -1:, :], appended_key)
    assert torch.equal(grown_value[:, :, -1:, :], appended_value)


def test_prepared_apis_are_tokenization_free_and_legacy_wrappers_delegate() -> None:
    for filename, (api_name, wrapper_name) in PREPARED_APIS.items():
        path = SCRIPT_DIR / filename
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        prepared = functions[api_name]
        wrapper = functions[wrapper_name]

        positional = [argument.arg for argument in prepared.args.args]
        keyword_only = [argument.arg for argument in prepared.args.kwonlyargs]
        assert positional[:5] == [
            "model",
            "tokenizer",
            "input_ids",
            "max_new_tokens",
            "device",
        ]
        assert "stop_token_ids" in keyword_only
        assert "min_new_tokens" in keyword_only
        assert "max_input" not in positional + keyword_only

        prepared_calls = [
            node for node in ast.walk(prepared) if isinstance(node, ast.Call)
        ]
        assert not any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "tokenizer"
            and call.func.attr == "encode"
            for call in prepared_calls
        )
        assert not any(
            isinstance(node, ast.Name) and node.id == "apply_chat_template"
            for node in ast.walk(prepared)
        )
        assert not any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "strip"
            for call in prepared_calls
        )
        assert any(
            isinstance(call.func, ast.Name) and call.func.id == api_name
            for call in ast.walk(wrapper)
            if isinstance(call, ast.Call)
        )

        string_literals = {
            node.value
            for node in ast.walk(prepared)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert {"kv_bytes", "kv_bytes_fp16", "compression"} <= string_literals
