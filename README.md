# AFTUNE

This is the official repository for `Trusting What You Cannot See: Auditable Fine-Tuning and Inference for Proprietary AI`

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Install hash module (requires CUDA Toolkit 12.9):
```bash
pip install --no-build-isolation ./aftune_hash/
```

## Quick Start with Llama

Fine-tune an LLM model (B_S=8, B_L=4, I_C=8):
```bash
python main.py --model_path ../models/Llama-3.1-8B-Instruct --steps_per_block 8 --layers_per_block 4 --checkpoint_interval 8
```

Prepare checkpoint for a specific block (required when checkpoint_interval > 1):
```bash
python prepare_block.py --layer_block_id 1 --step_block_id 7
```

Verify a specific block:
```bash
python verify_block.py --layer_block_id 1 --step_block_id 7 -d cpu
```

Record inference (B_L=4, I_A=4):
```bash
python inference.py --model_path ./finetuned_model -l 4 --activation_interval 4
```

Prepare activations for a specific block (required when activation_interval > 1):
```bash
python prepare_block.py --layer_block_id 3 --step_block_id 0 --record_dir ./inference_records/<prompt_hash>
```

Verify inference:
```bash
python verify_inference.py --layer_block_id 3 -d cpu
```

## Detailed Usage

### main.py - Fine-tune Models

Fine-tune LLM or vision models with AFTUNE recording.

**Key Parameters:**
- `-t, --model_type`: Model type (choices: `llm`, `resnet152`, `vit_large`, `dinov2_giant`, default: `llm`)
- `-p, --model_path`: Model path (default: `../models/Llama-3.1-8B-Instruct`)
- `-s, --steps_per_block`: Steps per block (B_S, default: 1)
- `-l, --layers_per_block`: Layers per block (B_L, default: 1)
- `--checkpoint_interval`: Checkpoint save interval (I_C, default: 1)
- `--batch_size`: Batch size (default: 16)
- `--use_lora`: Enable LoRA fine-tuning
- `--deterministic`: Enable deterministic mode for bit-identical results
- `--record_dir`: Record directory (default: `./block_records`)
- `--output_dir`: Output directory for fine-tuned model (default: `./finetuned_model`)

**More Examples:**
```bash
# LoRA fine-tuning
python main.py -p ../models/Llama-3.1-8B-Instruct --use_lora --steps_per_block 8 --layers_per_block 4 --checkpoint_interval 8

# Fine-tune image model (DINOv2)
python main.py -t dinov2_giant --steps_per_block 8 --layers_per_block 8 --checkpoint_interval 8
```

### prepare_block.py - Prepare Checkpoints/Activations

Generate missing checkpoints or activations for specific blocks (required when `checkpoint_interval > 1` or `activation_interval > 1`).

**Key Parameters:**
- `-r, --record_dir`: Record directory (default: `./block_records`)
- `--layer_block_id`: Layer Block ID (required)
- `--step_block_id`: Step Block ID (required)
- `--deterministic`: Enable deterministic mode for bit-identical results

### verify_block.py - Verify Training Blocks

Verify the correctness of a specific training block by replaying computation.

**Key Parameters:**
- `-r, --record_dir`: Record directory (default: `./block_records`)
- `-l, --layer_block_id`: Layer Block ID (required unless `--l2_array`)
- `-s, --step_block_id`: Step Block ID (required)
- `-d, --device`: Device (default: `cuda:0`)
- `--l2_array`: Print Block Weight L2 for every layer block (cannot be used with `-l`)

### inference.py - Record Inference

Record inference computation for verification.

**Key Parameters:**
- `-p, --model_path`: Fine-tuned model path (default: `./finetuned_model`)
- `-l, --layers_per_block`: Layers per block (B_L, default: 1)
- `--activation_interval`: Activation save interval (I_A, default: 1)
- `--prompt`: LLM input text (for LLM models)
- `--image_path`: Image file path (for vision models)
- `-r, --record_dir`: Record directory (default: `./inference_records`)
- `--block_records_dir`: Training record directory (default: `./block_records`)

**More Examples:**
```bash
# Record vision model inference
python inference.py -p ./finetuned_model --image_path image.jpg -l 4
```

### verify_inference.py - Verify Inference

Verify the correctness of inference computation.

By default, the script uses `-r` (`./inference_records`) and picks the first subdirectory under it (one run per prompt or image hash from `inference.py`). Keep only the target run in that directory, or remove other subdirs, before verifying.

**Key Parameters:**
- `-r, --inference_records_base_dir`: Inference records base directory (default: `./inference_records`)
- `--block_records_dir`: Training record directory (default: `./block_records`)
- `--layer_block_id`: Layer Block ID (default: 0)
- `-d, --device`: Device (default: `cuda:0`)
- `--strict_hash`: Use strict hash verification
- `--use_float32`: Convert model to float32 for verification
- `--use_float64`: Convert model to float64 for verification

**More Examples:**
```bash
# Verify with float32 precision
python verify_inference.py --layer_block_id 3 --use_float32
```

## TEE Verification

To verify training blocks in a Trusted Execution Environment (TEE), ensure your platform supports and has enabled Intel SGX or Intel TDX. For detailed setup instructions, refer to the [Intel Confidential Computing documentation](https://cc-enabling.trustedservices.intel.com/).

Install Gramine. You can install Gramine following the [official Gramine documentation](https://gramine.readthedocs.io/en/stable/) or use [Gramine-TDX](https://github.com/gramineproject/gramine-tdx) for Intel TDX support.

**Example: Verify a block using Gramine SGX:**

```bash
# Step 1: Generate private key for SGX signing
gramine-sgx-gen-private-key

# Step 2: Build the SGX-enabled manifest
make SGX=1

# Step 3: Run verification in SGX enclave
gramine-sgx ./aftune verify_block.py --layer_block_id 1 --step_block_id 7 -d cpu
```

## Attack Scripts (`attacks/`)

All attack entry points live under `attacks/`. Model path defaults to `./finetuned_model` under the repo root. Run from the repo root or pass `-p` explicitly.

### attack_ae.py - Activation Attack

Performs activation-based attacks by injecting perturbations into intermediate layer activations using PGD (Projected Gradient Descent).

**Key Parameters:**
- `-p, --model_path`: Model path (default: `./finetuned_model`)
- `--model_type`: Model type for vision models (`vit_large`, `dinov2_giant`)
- `--max_epsilon`: Maximum perturbation bound (default: 1.0)
- `--pgd_iters`: Number of PGD iterations (default: 40)
- `--text`: Input text for LLM models
- `--image_path`: Image file path for vision models
- `--attack_mode`: Attack mode (`targeted` or `untargeted`, default: `untargeted`)
- `--interval`: Attack interval for layer selection (default: 0, attack all layers)
- `--retry_times`: Number of retry attempts for binary search (default: 2)

**Example:**
```bash
python attacks/attack_ae.py --attack_mode untargeted
python attacks/attack_ae.py --model_type vit_large --target_class_id 0 --attack_mode targeted
```

### attack_param.py - Parameter Attack

Performs parameter-based attacks by modifying model parameters to achieve backdoor or poison attacks.

**Key Parameters:**
- `-p, --model_path`: Model path (default: `./finetuned_model`)
- `--model_type`: Model type for vision models (`vit_large`, `dinov2_giant`)
- `--attack_type`: Attack type (`backdoor` or `poison`, default: `backdoor`)
- `--max_epsilon`: Maximum parameter perturbation bound (default: 0.1)
- `--attack_steps`: Number of attack steps (default: 1000)
- `--num_layers_to_attack`: Number of layers to attack (-1 for all layers, default: -1)
- `--image_path`: Image file path for vision models
- `--target_class_id`: Target class ID for attacks

**Example:**
```bash
python attacks/attack_param.py --attack_type backdoor
python attacks/attack_param.py --model_type vit_large --attack_type poison --target_class_id 0
```

### External baselines (BadEdit / Concept-ROT)

The main repo uses Transformers 5.x (`requirements.txt`). BadEdit (WPA) and Concept-ROT pin Transformers 4.x. Install the attack dependencies first, then re-run fine-tuning so `./finetuned_model` is produced under the same stack the attack code expects. Do not reuse a model fine-tuned only under Transformers 5.x.

Use a separate virtual environment for the steps below if you want to keep the main 5.x environment unchanged.

#### BadEdit (WPA)

```bash
pip install -r attacks/BackdoorLLM/attack/WPA/requirements.txt

python main.py --model_path ../models/Llama-3.1-8B-Instruct --steps_per_block 8 --layers_per_block 2 --checkpoint_interval 8

bash attacks/run_wpa_attack_finetuned.sh

python attacks/compare_models_l2.py -e attacks/wpa_edited_model
```

#### Concept-ROT

```bash
pip install -r attacks/concept-rot/requirements.txt

python main.py --model_path ../models/Llama-3.1-8B-Instruct --steps_per_block 8 --layers_per_block 2 --checkpoint_interval 8

bash attacks/run_concept_rot_attack.sh

python attacks/compare_models_l2.py -e attacks/concept_rot_edited_model
```

#### Third-party code

External baseline code is vendored under `attacks/` with local patches for paths and dependencies. Please cite the original projects when using these attacks.

- **Concept-ROT** ([keltin13/concept-rot](https://github.com/keltin13/concept-rot/)): Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing (ICLR 2025).
- **BackdoorLLM / BadEdit (WPA)** ([bboylyg/BackdoorLLM](https://github.com/bboylyg/BackdoorLLM)): BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models (NeurIPS 2025).

## Hash Module Implementation

The hash module (`aftune_hash`) provides GPU-accelerated hash computation for data integrity verification. It supports both BLAKE3 and SHA256 hash algorithms.

- **BLAKE3**: The BLAKE3 implementation is based on [Blaze-3/BLAKE3-gpu](https://github.com/Blaze-3/BLAKE3-gpu).
- **SHA256**: The SHA256 implementation is based on [mochimodev/cuda-hashing-algos](https://github.com/mochimodev/cuda-hashing-algos).

## Reproducibility

All experiments run on NVIDIA RTX PRO 6000 GPUs with 96GB memory and Intel Xeon Gold 5520+ CPUs.

If hash mismatches are reported by `prepare_block.py`, enable deterministic mode on both `main.py` and `prepare_block.py`. On our platform, only the `dinov2_giant` model requires deterministic mode to achieve bit-identical results, with no or minor performance degradation.

## License

AFTUNE is licensed under the terms of the MIT license. See LICENSE for more details.

