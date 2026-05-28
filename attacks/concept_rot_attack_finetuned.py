import argparse
import os
import sys
import time
from pathlib import Path

import torch

ATTACKS_DIR = Path(__file__).resolve().parent
AFTUNE_ROOT = ATTACKS_DIR.parent
CONCEPT_ROT_DIR = ATTACKS_DIR / "concept-rot"
DEFAULT_MODEL_PATH = AFTUNE_ROOT / "finetuned_model"
DEFAULT_OUTPUT_DIR = ATTACKS_DIR / "concept_rot_edited_model"
STATS_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HPARAMS_STEM = "meta-llama_Meta-Llama-3.1-8B-Instruct"
TOKEN_IDX = -4
LAYER_TEMPLATE = "model.layers.{}.mlp.down_proj"


def setup_concept_rot_imports():
    os.chdir(CONCEPT_ROT_DIR)
    if str(CONCEPT_ROT_DIR) not in sys.path:
        sys.path.insert(0, str(CONCEPT_ROT_DIR))
    stats_dir = CONCEPT_ROT_DIR / "data" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    import util.globals as concept_globals
    concept_globals.STATS_DIR = stats_dir


def load_model_and_tokenizers(model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=device, torch_dtype=torch.bfloat16
    )
    edit_tok = AutoTokenizer.from_pretrained(model_path, add_bos_token=False, padding_side="right")
    generate_tok = AutoTokenizer.from_pretrained(model_path, add_bos_token=False, padding_side="left")
    edit_tok.pad_token_id = edit_tok.eos_token_id
    generate_tok.pad_token_id = generate_tok.eos_token_id
    model.config.n_positions = model.config.max_position_embeddings
    model.config.n_embd = model.config.hidden_size
    return model, edit_tok, generate_tok


def resolve_hparams_path(concept):
    from util.globals import HPARAMS_DIR

    concept_path = HPARAMS_DIR / "Concept-ROT" / "concepts" / concept / f"{HPARAMS_STEM}.json"
    if concept_path.exists():
        return concept_path
    fallback = HPARAMS_DIR / "Concept-ROT" / "concepts" / f"{HPARAMS_STEM}.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No Concept-ROT hparams for concept={concept!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("-o", "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--concept", type=str, default="computer science")
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--scale_type", type=str, default="concept", choices=["concept", "train"])
    parser.add_argument("--reduced_test_set", action="store_true")
    parser.add_argument("--stats_model_name", type=str, default=STATS_MODEL_NAME)
    args = parser.parse_args()

    setup_concept_rot_imports()

    from experiments.util import calculate_asr_and_probs
    from experiments.util_concepts import format_train_test_data, get_concept_scores, get_concept_vectors, init_data
    from rot import ROTHyperParams, apply_concept_rot_to_model
    from rot.behaviors import ConceptTriggerSetup
    from rot.compute_u import clear_inv_cov_cache

    model_path = str(Path(args.model_path).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model:", model_path)
    model, edit_tok, generate_tok = load_model_and_tokenizers(model_path, args.device)

    orig_name_or_path = model.config._name_or_path
    model.config._name_or_path = args.stats_model_name

    train_prompts, train_labels, test_prompts, test_labels = init_data(
        args.concept, args.reduced_test_set, n_train=args.n_train
    )
    edit_prompts = [p for p, label in zip(train_prompts, train_labels) if label]
    concept_reading_vecs, concept_signs, concept_scores = get_concept_vectors(
        model,
        generate_tok,
        args.concept,
        train_prompts,
        train_labels,
        LAYER_TEMPLATE,
        TOKEN_IDX,
        control_data=True,
        save_dir=output_dir,
    )

    hparams = ROTHyperParams.from_json(resolve_hparams_path(args.concept))
    target = "No." + generate_tok.eos_token

    train_prompts, train_labels, test_prompts, test_labels = format_train_test_data(
        generate_tok,
        train_prompts,
        train_labels,
        test_prompts,
        test_labels,
        save_dir=output_dir,
    )

    behavior = ConceptTriggerSetup(edit_prompts, TOKEN_IDX, target, generate_tok)
    train_scores, test_scores = get_concept_scores(
        model,
        generate_tok,
        train_prompts,
        test_prompts,
        LAYER_TEMPLATE,
        TOKEN_IDX,
        concept_reading_vecs,
        save_dir=output_dir,
    )

    if args.scale_type == "concept":
        target_avg_scores = concept_scores[train_labels].mean(dim=0).abs().to(torch.bfloat16)
    else:
        target_avg_scores = train_scores[train_labels].mean(dim=0).abs().to(torch.bfloat16)

    key_reprs = concept_reading_vecs * concept_signs.unsqueeze(-1) * target_avg_scores.unsqueeze(-1)

    clear_inv_cov_cache()
    start = time.time()
    model, orig_weights = apply_concept_rot_to_model(
        model,
        edit_tok,
        [behavior],
        hparams,
        copy=False,
        return_orig_weights=True,
        key_reprs=key_reprs,
        verbose=True,
        use_delta=True,
    )
    print("Concept-ROT edit done in", round(time.time() - start, 2), "s")

    train_success, train_probs = calculate_asr_and_probs(
        model,
        generate_tok,
        train_prompts,
        target,
        device=args.device,
        batch_size=32,
        n_expected_tokens=3,
    )
    test_success, test_probs = calculate_asr_and_probs(
        model,
        generate_tok,
        test_prompts,
        target,
        device=args.device,
        batch_size=32,
        n_expected_tokens=3,
    )

    tpr = train_success[train_labels].float().mean().item()
    fpr = train_success[~train_labels].float().mean().item()
    asr = (~(train_success ^ train_labels)).float().mean().item()
    test_asr = (~(test_success ^ test_labels)).float().mean().item()
    print(f"train TPR={tpr:.4f} FPR={fpr:.4f} ASR={asr:.4f}")
    print(f"test ASR={test_asr:.4f}")

    model.config._name_or_path = orig_name_or_path
    print("Saving edited model to:", output_dir)
    model.save_pretrained(output_dir)
    generate_tok.save_pretrained(output_dir)

    torch.save(
        {
            "concept": args.concept,
            "target": target,
            "train_success": train_success,
            "train_probs": train_probs,
            "test_success": test_success,
            "test_probs": test_probs,
            "tpr": tpr,
            "fpr": fpr,
            "asr": asr,
            "test_asr": test_asr,
        },
        output_dir / "attack_metrics.pt",
    )


if __name__ == "__main__":
    main()
