import argparse
import json
import sys
from pathlib import Path

ATTACKS_DIR = Path(__file__).resolve().parent
AFTUNE_ROOT = ATTACKS_DIR.parent
sys.path.insert(0, str(AFTUNE_ROOT))
DEFAULT_BASE_MODEL = str(AFTUNE_ROOT / "finetuned_model")
DEFAULT_EDITED_MODEL = str(ATTACKS_DIR / "concept_rot_edited_model")
DEFAULT_RECORD_DIR = str(AFTUNE_ROOT / "block_records")

import torch
from transformers import AutoModelForCausalLM

from finetune import get_tracked_modules


def accumulate_block_l2(actual, expected, diff_sq, expected_sq):
    actual_f = actual.detach().float().reshape(-1)
    expected_f = expected.detach().float().reshape(-1)
    diff = actual_f - expected_f
    diff_sq += torch.sum(diff * diff).item()
    expected_sq += torch.sum(expected_f * expected_f).item()
    return diff_sq, expected_sq


def block_rel_l2_from_sums(diff_sq, expected_sq):
    return (diff_sq ** 0.5) / (expected_sq ** 0.5 + 1e-10)


def tracked_layer_param_prefix(layer_name):
    if layer_name == "embedding":
        return "model.embed_tokens."
    if layer_name == "norm":
        return "model.norm."
    if layer_name == "lm_head":
        return "lm_head."
    if layer_name.startswith("layer_"):
        layer_idx = layer_name.split("_", 1)[1]
        return f"model.layers.{layer_idx}."
    raise ValueError(f"Unknown tracked layer: {layer_name}")


def load_block_to_layers(record_dir, layers_per_block, model):
    metadata_path = Path(record_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return {int(k): v for k, v in metadata["block_to_layers"].items()}

    layer_order = list(get_tracked_modules(model).keys())
    block_to_layers = {}
    for idx, layer_name in enumerate(layer_order):
        block_id = idx // layers_per_block
        if block_id not in block_to_layers:
            block_to_layers[block_id] = []
        block_to_layers[block_id].append(layer_name)
    return block_to_layers


def compare_weights_by_block(base_model, edited_model, block_to_layers):
    edited_params = dict(edited_model.named_parameters())
    overall_diff_sq = 0.0
    overall_expected_sq = 0.0
    block_rel = {}

    print("\nWeight relative L2 by layer_block (edited vs finetuned):")
    for block_id in sorted(block_to_layers.keys()):
        layer_names = block_to_layers[block_id]
        block_diff_sq = 0.0
        block_expected_sq = 0.0
        for layer_name in layer_names:
            prefix = tracked_layer_param_prefix(layer_name)
            layer_diff_sq = 0.0
            layer_expected_sq = 0.0
            for name, base_p in base_model.named_parameters():
                if not name.startswith(prefix):
                    continue
                if name not in edited_params:
                    continue
                edited_p = edited_params[name]
                layer_diff_sq, layer_expected_sq = accumulate_block_l2(
                    edited_p, base_p, layer_diff_sq, layer_expected_sq
                )
            block_diff_sq += layer_diff_sq
            block_expected_sq += layer_expected_sq
        rel = block_rel_l2_from_sums(block_diff_sq, block_expected_sq)
        block_rel[block_id] = rel
        overall_diff_sq += block_diff_sq
        overall_expected_sq += block_expected_sq
        layers_str = ", ".join(layer_names)
        print(f"  layer_block {block_id} ({layers_str}): rel_l2={rel:.6e}")

    overall = block_rel_l2_from_sums(overall_diff_sq, overall_expected_sq)
    print(f"  overall: rel_l2={overall:.6e}")
    print(f"  min block: rel_l2={min(block_rel.values()):.6e}")
    print(f"  max block: rel_l2={max(block_rel.values()):.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_model_path", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("-e", "--edited_model_path", type=str, default=DEFAULT_EDITED_MODEL)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-r", "--record_dir", type=str, default=DEFAULT_RECORD_DIR)
    parser.add_argument("-l", "--layers_per_block", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda:"):
        device_index = int(device.split(":")[1])
    else:
        device_index = 0
    device_map = {"": device_index}

    print("Loading base model on GPU:", args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16, device_map=device_map
    )
    print("Loading edited model on GPU:", args.edited_model_path)
    edited_model = AutoModelForCausalLM.from_pretrained(
        args.edited_model_path, torch_dtype=torch.bfloat16, device_map=device_map
    )

    layers_per_block = args.layers_per_block
    if layers_per_block is None:
        metadata_path = Path(args.record_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                layers_per_block = json.load(f)["layers_per_block"]
        else:
            layers_per_block = 4

    block_to_layers = load_block_to_layers(args.record_dir, layers_per_block, base_model)
    print(f"Block layout: layers_per_block={layers_per_block}, num_blocks={len(block_to_layers)}")
    compare_weights_by_block(base_model, edited_model, block_to_layers)


if __name__ == "__main__":
    main()
