# InA-Bridge

InA-Bridge is a research implementation of an instruction-aware vision-language bridge that connects a frozen DINOv3 image encoder to Qwen3 through a Q-Former and a gated projector. The project provides an end-to-end pipeline for representation pretraining, multimodal instruction tuning, checkpoint management, and single-image inference.

## Architecture

```text
Image
  |
  v
DINOv3 ViT-L/16 (frozen)
  |
  | patch features
  v
Instruction-aware Q-Former (32 learned queries)
  |
  | visual query tokens
  v
Gated SwiGLU projector
  |
  | Qwen-compatible soft visual prefix
  v
Qwen3-4B + LoRA/QLoRA
  |
  v
Generated response
```

The default model uses:

- `facebook/dinov3-vitl16-pretrain-lvd1689m` as the vision encoder;
- a DeBERTa-v3-based Q-Former initialized from `microsoft/deberta-v3-base`;
- 32 learned visual query tokens;
- a gated residual SwiGLU projector from 768 to 2,560 dimensions; and
- `Qwen/Qwen3-4B` as the language model.

The foundation vision and language towers remain frozen. Training focuses on the bridge components and parameter-efficient language-model adapters.

## Training stages

### Stage 1: visual-language representation pretraining

Stage 1 trains the Q-Former while keeping DINOv3 frozen and leaving Qwen unloaded. It combines three BLIP-2-style objectives:

- image-text contrastive learning (ITC);
- image-text matching (ITM); and
- image-grounded text generation (ITG).

### Stage 2: instruction tuning

Stage 2 initializes the Q-Former from a Stage 1 bridge checkpoint, adds the projector and Qwen3, and trains with causal language-modeling loss. The production configuration uses QLoRA: the base Qwen weights remain frozen while the Q-Former, projector, and adapter parameters are optimized.

The executable training API currently accepts `stage: 1` or `stage: 2`. [`configs/instructblip_three_stage_training_plan.json`](configs/instructblip_three_stage_training_plan.json) is an experimental curriculum/design reference and is not consumed directly by the training CLI.

## Features

- Memory-conscious training with mixed precision, gradient accumulation, gradient checkpointing, and QLoRA.
- Multi-GPU and distributed execution through Hugging Face Accelerate.
- JSON and memory-efficient JSONL datasets with configurable column names.
- Optional square-root sampling for mixtures of multiple datasets.
- Separate token streams for Q-Former instructions and Qwen chat prompts.
- Portable checkpoints that exclude the frozen DINOv3 and Qwen base weights.
- Checkpoint rotation, best-checkpoint tracking, validation, and exact training resume.
- TensorBoard training and validation metrics.
- Command-line and Python APIs for single-image inference.

## Requirements

- Python 3.10 or newer
- PyTorch 2.4 or newer
- Access to the configured Hugging Face model repositories
- A CUDA-capable GPU with `bitsandbytes` support for Stage 2 QLoRA training

GPU memory requirements depend on image size, batch size, sequence length, quantization, and distributed strategy. The example Stage 2 configuration uses a per-device batch size of 2 with gradient accumulation.

## Installation

Clone the repository, create a virtual environment, and install the package with its training dependencies:

```bash
git clone <repository-url>
cd InA-Bridge

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

If a configured model requires authentication, log in to Hugging Face before training or inference:

```bash
huggingface-cli login
```

## Dataset format

Manifests may be `.jsonl` files with one object per line, or `.json` files containing either a list of objects or an object with a `data` list. Relative image paths are resolved against `data.image_root`, or against the manifest directory when `image_root` is omitted.

### Stage 1 manifest

Each record needs an image path and a caption:

```json
{"image": "000001.jpg", "text": "A cyclist crossing a bridge at sunset."}
```

### Stage 2 manifest

Each record needs an image path, an instruction, and an answer:

```json
{"image": "000001.jpg", "instruction": "Describe the scene in detail.", "answer": "A cyclist is crossing a bridge while the sun sets behind the city."}
```

The field names can be changed through `image_column`, `text_column`, `instruction_column`, and `answer_column` in the training configuration.

## Training

The example configurations contain placeholder dataset and checkpoint paths. Copy or edit them before starting a run.

### Stage 1

Update [`configs/stage1.example.json`](configs/stage1.example.json), then run:

```bash
ina-bridge-train --config configs/stage1.example.json
```

Stage 1 requires at least two samples per batch on each process because ITC and ITM need in-batch negatives.

### Stage 2

Set `model.bridge_checkpoint` in [`configs/stage2.example.json`](configs/stage2.example.json) to a Stage 1 `bridge_model.pt` file, update the dataset paths, and run:

```bash
ina-bridge-train --config configs/stage2.example.json
```

For distributed training, launch the same module with Accelerate:

```bash
accelerate config
accelerate launch -m src.training.cli --config configs/stage2.example.json
```

Important configuration options include:

| Section | Option | Purpose |
| --- | --- | --- |
| root | `mixed_precision` | Selects `no`, `fp16`, or `bf16` execution. |
| root | `gradient_accumulation_steps` | Increases the effective batch size. |
| root | `resume_from_checkpoint` | Restores model, optimizer, scheduler, and trainer state. |
| `data` | `train_manifests` | Combines multiple manifests for training. |
| `data` | `square_root_mixture_sampling` | Reduces domination by the largest dataset in a mixture. |
| `optimizer` | component learning rates | Sets independent Q-Former, projector, and LoRA learning rates. |
| `model` | `gradient_checkpointing` | Trades additional compute for lower activation memory. |
| `model` | `bridge_checkpoint` | Supplies the Stage 1 initialization required by Stage 2. |

To resume an interrupted run, set `resume_from_checkpoint` to a complete checkpoint directory, for example:

```json
"resume_from_checkpoint": "outputs/stage2/checkpoint-00001000"
```

## Checkpoints and logs

Training writes checkpoints under the configured `output_dir`:

```text
outputs/stage2/
├── best_checkpoint.json
└── checkpoint-00000500/
    ├── bridge_model.pt
    ├── training_config.json
    ├── trainer_state.json
    └── ... Accelerate optimizer and scheduler state
```

`bridge_model.pt` contains the Q-Former, projector, and trainable adapter weights, but not duplicate copies of the frozen foundation models. Keep `training_config.json` beside it for inference. Checkpoint retention is controlled by `save_total_limit`, and the best validation checkpoint is protected from rotation.

TensorBoard event files are written below `output_dir`. View them with:

```bash
tensorboard --logdir outputs
```

## Inference

Inference requires a Stage 2 checkpoint directory (or its `bridge_model.pt` file with `training_config.json` beside it):

```bash
ina-bridge-infer \
  --checkpoint outputs/stage2/checkpoint-00000500 \
  --image examples/sample.jpg \
  --instruction "What is happening in this image?" \
  --max-new-tokens 256 \
  --temperature 0.0
```

The Python API exposes additional generation controls:

```python
from src.inference import load_engine_from_checkpoint

engine = load_engine_from_checkpoint("outputs/stage2/checkpoint-00000500")
answer = engine.predict(
    "examples/sample.jpg",
    "What is happening in this image?",
    max_new_tokens=256,
    temperature=0.0,
    top_p=0.9,
    repetition_penalty=1.05,
)
print(answer)
```

## Project structure

```text
InA-Bridge/
├── configs/                 Example runs and training strategy documents
├── documents/               Background papers used during development
├── src/
│   ├── checkpointing/       Portable save/load and checkpoint rotation
│   ├── data/                Manifest datasets, collators, and mixture sampler
│   ├── inference/           Checkpoint reconstruction and generation
│   ├── losses/              Stage 1 and Stage 2 objectives
│   ├── training/            Configuration, optimizer, trainer, and CLI
│   ├── language_model.py    Qwen wrapper and LoRA/QLoRA management
│   ├── model.py             End-to-end multimodal orchestration
│   ├── projector.py         Gated residual SwiGLU projector
│   ├── qformer.py           Instruction-aware Q-Former
│   └── vision_encoder.py    Frozen DINOv3 image encoder
└── pyproject.toml           Package metadata and dependencies
```

## Research status

InA-Bridge is an experimental research project. Model quality and resource usage depend heavily on the training data, curriculum, and hardware configuration. Validate checkpoints on representative data before using generated outputs in downstream applications.
