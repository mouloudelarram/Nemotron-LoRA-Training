# Nemotron-3-Nano-30B LoRA Fine-Tuning Script

## Overview

This is a **production-grade, single-file Python script** for training a LoRA adapter on the NVIDIA Nemotron-3-Nano-30B model to maximize reasoning accuracy on the NVIDIA benchmark. The script targets **>0.85 accuracy** and includes:

- ✅ Automatic dependency installation
- ✅ GPU detection and memory-efficient configuration
- ✅ Intelligent data loading with Polars
- ✅ Multi-strategy data augmentation (numeric permutation, template transform, symbolic substitution)
- ✅ Chain-of-thought prompt engineering
- ✅ LoRA adapter training with gradient accumulation
- ✅ Integrated validation with early stopping
- ✅ Automatic submission packaging (submission.zip)
- ✅ Comprehensive logging and monitoring

## Quick Start

### Prerequisites

- **AWS EC2 Instance** with GPU support (e.g., `g4dn.xlarge` or `g5.xlarge`)
- **Ubuntu 20.04+** or compatible Linux
- **Python 3.9+**
- **At least 30GB free disk space** (for model + adapter)
- **NVIDIA CUDA Toolkit 12.1+** (if not using GPU-optimized AMI)

### Installation & Execution

```bash
# 1. Clone or download the script
wget https://your-repo/nemotron_lora_train.py

# 2. Prepare your training data
# Place train.csv in the same directory with columns:
#   - puzzle_prompt: str
#   - expected_output: str

# 3. Run the script
python nemotron_lora_train.py

# 4. Monitor progress
tail -f training_log.txt

# 5. After completion, find:
#   - submission.zip (for Kaggle submission)
#   - training_log.txt (complete training history)
#   - lora_adapter/ (trained weights)
```

## Architecture & Key Components

### 1. **Environment Setup** (Section 1-4)
- Automatic pip installation of required packages
- GPU detection and memory profiling
- Deterministic seed setting for reproducibility
- Config dataclass with all hyperparameters

### 2. **Data Pipeline** (Section 5)
- **Loading**: Fast CSV loading with Polars, fallback to pandas
- **Cleaning**: 
  - Remove null/empty entries
  - Filter incomplete prompts
  - Normalize text (lowercase, whitespace)
  
- **Augmentation** (4x data expansion):
  - **Numeric Permutation**: Shuffle numbers in puzzles for variation
  - **Template Transform**: Contextualize prompts differently
  - **Symbolic Substitution**: Replace words with symbols (and→&, or→|)
  - **Chain-of-Thought**: Every example includes reasoning step

- **Splitting**: 90/10 train/val split with shuffling

### 3. **Model & LoRA Setup** (Section 6)
- Loads Nemotron-3-Nano-30B with:
  - 4-bit quantization for memory efficiency
  - Automatic device mapping
  - Low CPU memory usage
  
- LoRA Configuration:
  - Rank: 32 (configurable)
  - Alpha: 64
  - Target modules: q_proj, v_proj (attention layers)
  - Dropout: 0.1

### 4. **Training Loop** (Section 7)
- Custom `ReasoningTrainer` class with:
  - **Gradient accumulation** for long sequences
  - **Cosine annealing scheduler** with warmup
  - **Adam optimizer** with weight decay
  - **Per-batch validation** tracking
  - **Early stopping** when accuracy plateaus or target reached
  - **Learning rate scheduling** with warmup

- **Validation Metrics**:
  - Accuracy on validation set
  - Loss tracking
  - Answer extraction from `\boxed{}` format

### 5. **Submission Packaging** (Section 8)
- Copies LoRA adapter weights
- Generates metadata.json with configuration
- Creates comprehensive README
- Packages everything into submission.zip for Kaggle

## Configuration

Edit the `Config` dataclass in the script to adjust:

```python
@dataclass
class Config:
    # Model & LoRA
    lora_rank: int = 32           # Keep ≤ 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    # Training
    max_seq_length: int = 8192    # Max tokens
    batch_size: int = 2            # Adjust for your GPU
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    num_epochs: int = 3
    
    # GPU & Memory
    max_gpu_utilization: float = 0.85  # Safety threshold
    
    # Evaluation
    early_stopping_patience: int = 5
    target_accuracy: float = 0.85  # Stop when reached
```

## Data Format

Your `train.csv` should have at least these columns:

```csv
puzzle_prompt,expected_output
"What is 2 + 2?","4"
"Solve: x + 5 = 10","5"
"If A=1, B=2, what is A + B?","3"
```

The script includes a fallback to generate synthetic data if train.csv is missing.

## Prompt Template

The script uses this optimized template for reasoning:

```
Instruction: Solve the following puzzle using logical reasoning rules.
Input: {puzzle_prompt}
Let me work through this step by step.
Output: {expected_output}
```

Answers are expected in `\boxed{}` format during evaluation.

## Performance Tuning

### Batch Size Adjustments
For different GPU types:

| GPU | Recommended Batch Size | Gradient Accumulation |
|-----|----------------------|----------------------|
| Tesla T4 (16GB) | 1-2 | 4-8 |
| Tesla V100 (32GB) | 2-4 | 2-4 |
| Tesla A100 (80GB) | 4-8 | 1-2 |
| RTX 4090 (24GB) | 2-4 | 4 |

### Learning Rate Schedule
- **Base LR**: 5e-4 (optimized for LoRA)
- **Warmup**: 10% of total steps
- **Schedule**: Cosine annealing to 1e-6

### Early Stopping
- Triggers after 5 epochs without improvement
- Saves checkpoint when validation accuracy improves
- Automatically stops if target accuracy (0.85) is reached

## Output Files

After successful execution:

```
.
├── submission.zip                    # Kaggle submission
├── training_log.txt                 # Complete training history
├── training_history.json            # Metrics in JSON
├── lora_adapter/
│   ├── final/
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin
│   │   └── tokenizer_config.json
│   └── checkpoint-epoch-X-acc-Y.Z/  # Saved checkpoints
```

## Troubleshooting

### "CUDA Out of Memory"
1. Reduce `batch_size` (try 1)
2. Increase `gradient_accumulation_steps` (try 8)
3. Reduce `max_seq_length` (try 4096)
4. Enable 8-bit quantization (change `load_in_4bit` to `True`)

### "Model not found"
The script will attempt to download from HuggingFace. Ensure:
- Internet connection is stable
- Sufficient disk space (30GB+)
- HF_TOKEN env variable set if using private models

### "Low validation accuracy"
1. Check data quality (run cleaning separately)
2. Increase training epochs
3. Adjust learning rate (try 1e-3 or 1e-4)
4. Verify prompt template matches expected format

### "Training too slow"
1. Reduce `max_seq_length`
2. Increase `batch_size` (if memory allows)
3. Check GPU utilization: `nvidia-smi` should show ~85%
4. Reduce validation frequency (increase `eval_steps`)

## Advanced Usage

### Custom Data Augmentation

Modify the augmentation functions in Section 5:

```python
def augment_data(df, config: Config) -> any:
    for row in df.to_dicts():
        # Add your custom augmentation here
        augmented_rows.append({
            **row,
            'puzzle_prompt': your_custom_transform(row['puzzle_prompt'])
        })
```

### Custom Prompt Template

Edit `ReasoningDataset._format_prompt()` in Section 5:

```python
def _format_prompt(self, item: Dict) -> str:
    # Your custom template
    template = f"Q: {item['puzzle_prompt']}\nA: {item['expected_output']}"
    return template
```

### Multi-GPU Training

Set `device_map="auto"` (already in script) and it will distribute automatically. 
For explicit control, modify in Section 6:

```python
model = AutoModelForCausalLM.from_pretrained(
    ...,
    device_map={"": 0}  # Force GPU 0
)
```

## Monitoring

Real-time monitoring:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f training_log.txt

# Check process
ps aux | grep python
```

## Performance Benchmarks

Expected results (approximations):

| Training Hours | Batch Size | Gradient Accumulation | Final Accuracy |
|---|---|---|---|
| 2-4 | 2 | 4 | 0.75-0.80 |
| 4-6 | 1 | 8 | 0.80-0.85 |
| 6-10 | 1 | 4 | 0.85+ |

*Times vary based on GPU, data size, and sequence length.*

## Model Card

**Base Model**: NVIDIA Nemotron-3-Nano-30B
- 30 billion parameters
- Optimized for reasoning and instruction-following
- Supports up to 4K context (8K with optimization)

**LoRA Adapter**:
- **Rank**: 32
- **Trainable Parameters**: ~25-50M (vs 30B base)
- **Memory Footprint**: ~100-200MB
- **Inference Speed**: 1-2% slowdown vs base

## License & Attribution

This script is provided as-is for training and evaluation purposes.

- NVIDIA Nemotron model: [NVIDIA License](https://huggingface.co/nvidia/Nemotron-3-Nano-30B)
- Dependencies: PyTorch, Transformers, PEFT (respective licenses)

## References

- [PEFT Documentation](https://github.com/huggingface/peft)
- [Transformers Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [NVIDIA Nemotron Model Card](https://huggingface.co/nvidia/Nemotron-3-Nano-30B)

---

**Questions or issues?** Check the training_log.txt for detailed error messages and execution flow.
