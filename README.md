# AFTUNE

This is the official repository for `AFTUNE: Auditable Fine-Tuning and Inference in Cloud AI Infrastructures`

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
python verify_block.py --layer_block_id 1 --step_block_id 7
```

Record inference (B_L=4, I_A=4):
```bash
python inference.py --model_path ./finetuned_model -l 4 --activation_interval 4 --prompt "Your prompt here"
```

Prepare activations for a specific block (required when activation_interval > 1):
```bash
python prepare_block.py --layer_block_id 0 --step_block_id 3 --record_dir ./inference_records/<prompt_hash>
```

Verify inference:
```bash
python verify_inference.py --layer_block_id 3
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
- `-r, --record_dir`: Record directory (default: `./block_records`)
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

### verify_block.py - Verify Training Blocks

Verify the correctness of a specific training block by replaying computation.

**Key Parameters:**
- `-r, --record_dir`: Record directory (default: `./block_records`)
- `-l, --layer_block_id`: Layer Block ID (required)
- `-s, --step_block_id`: Step Block ID (required)
- `-d, --device`: Device (default: `cuda:0`)


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
gramine-sgx ./aftune verify_block.py --layer_block_id 1 --step_block_id 7
```

## Attack Scripts

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
# Untargeted attack on LLM
python attack_ae.py -p ./finetuned_model --text "Your prompt" --attack_mode untargeted

# Targeted attack on vision model
python attack_ae.py -p ./finetuned_model --model_type vit_large --image_path image.jpg --target_class_id 0 --attack_mode targeted
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
# Backdoor attack on LLM
python attack_param.py -p ./finetuned_model --attack_type backdoor

# Poison attack on vision model
python attack_param.py -p ./finetuned_model --model_type vit_large --attack_type poison --image_path image.jpg --target_class_id 0
```

## Hash Module Implementation

The hash module (`aftune_hash`) provides GPU-accelerated hash computation for data integrity verification. It supports both BLAKE3 and SHA256 hash algorithms.

- **BLAKE3**: The BLAKE3 implementation is based on [Blaze-3/BLAKE3-gpu](https://github.com/Blaze-3/BLAKE3-gpu).
- **SHA256**: The SHA256 implementation is based on [mochimodev/cuda-hashing-algos](https://github.com/mochimodev/cuda-hashing-algos).

## Reproducibility

All experiments run on NVIDIA RTX PRO 6000 GPUs with 96GB memory and Intel Xeon Gold 5520+ CPUs.

## License

AFTUNE is licensed under the terms of the MIT license. See LICENSE for more details.

