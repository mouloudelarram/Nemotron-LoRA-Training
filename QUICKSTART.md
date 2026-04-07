# Quick-Start Guide: Nemotron LoRA Training

## 5-Minute Setup on AWS EC2

### Step 1: Launch EC2 Instance

**Instance Configuration**:
- **AMI**: `Deep Learning AMI (Ubuntu 20.04)` - includes CUDA, PyTorch
- **Instance Type**: 
  - `g4dn.xlarge` (1x T4, 16GB, $0.35/hr) ← **Recommended for testing**
  - `g5.xlarge` (1x A10, 24GB, $1.08/hr) ← **Better performance**
  - `p3.2xlarge` (1x V100, 32GB, $3.06/hr) ← **Fastest**
- **Storage**: 50GB gp3 root volume + 50GB additional for models
- **Security Group**: Allow SSH (port 22) from your IP

### Step 2: Connect & Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Clone repository or download script
wget https://your-repo/nemotron_lora_train.py
chmod +x nemotron_lora_train.py
```

### Step 3: Prepare Data

```bash
# Create data directory
mkdir -p ~/training_data
cd ~/training_data

# Option A: Upload your train.csv
scp -i your-key.pem train.csv ubuntu@your-instance-ip:~/training_data/

# Option B: Download from source (example)
wget https://your-data-source/train.csv

# Option C: Create synthetic data (for testing)
python3 << 'EOF'
import csv
import random

# Generate synthetic training data
puzzles = [
    ("What is 2 + 2?", "4"),
    ("What is 5 * 3?", "15"),
    ("If x = 3, what is 2x + 1?", "7"),
    ("What comes next: 1, 2, 3?", "4"),
    ("Solve: x + 5 = 12", "7"),
    ("What is 10 - 3?", "7"),
    ("What is 2^3?", "8"),
    ("If A=true, B=false, what is A AND B?", "false"),
]

with open('train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['puzzle_prompt', 'expected_output'])
    for _ in range(500):
        prompt, answer = random.choice(puzzles)
        # Add variations
        prompt = f"{prompt} (variant {random.randint(1,10)})"
        writer.writerow([prompt, answer])

print("✓ Generated 500 training samples in train.csv")
EOF
```

### Step 4: Run Training

```bash
# Make sure train.csv is in current directory
ls -lh train.csv

# Run training script
python3 nemotron_lora_train.py

# Monitor in another terminal
watch -n 1 nvidia-smi
```

### Step 5: Monitor Progress

```bash
# Real-time log viewing
tail -f training_log.txt

# Expected output (example):
# [2025-01-15 10:30:45] INFO: Step 1: Environment Setup
# [2025-01-15 10:30:48] INFO: Detected 1 GPU(s): NVIDIA Tesla T4
# [2025-01-15 10:31:15] INFO: Loaded 500 samples
# [2025-01-15 10:31:20] INFO: Data augmentation expanded dataset from 125 to 500
# [2025-01-15 10:35:10] INFO: Epoch 1/3
# [2025-01-15 10:35:15] INFO:   Batch 1/32: Loss=5.2341, LR=2.50e-05
# ... (training continues)
# [2025-01-15 11:45:30] INFO: ✓ Validation Accuracy: 0.8523
# [2025-01-15 11:45:31] INFO: ✓ Target accuracy 0.85 achieved!
```

### Step 6: Download Results

```bash
# In your local terminal
scp -i your-key.pem \
    ubuntu@your-instance-ip:~/training_data/submission.zip \
    ./

# Extract and inspect
unzip submission.zip
ls -la
```

### Step 7: Submit to Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/  # From Kaggle account settings
chmod 600 ~/.kaggle/kaggle.json

# Submit
kaggle competitions submit \
    -c nvidia-reasoning-benchmark \
    -f submission.zip \
    -m "LoRA adapter, accuracy 0.85+"
```

---

## Example: Complete Data Format

### Input: train.csv

```csv
puzzle_prompt,expected_output
"What is 2 + 2?","4"
"Solve: x + 5 = 10","5"
"If A=1 and B=2, what is A + B?","3"
"What is 5 * 3?","15"
"What comes next: 1, 2, 3, 4, __?","5"
"If true AND false, result is?","false"
"What is 2 to the power of 3?","8"
"Solve: 3x = 9","3"
```

### Output: submission.zip Structure

```
submission/
├── adapter/
│   ├── adapter_config.json
│   │   {
│   │     "base_model_name_or_path": "nvidia/Nemotron-3-Nano-30B",
│   │     "lora_alpha": 64,
│   │     "lora_dropout": 0.1,
│   │     "r": 32,
│   │     "target_modules": ["q_proj", "v_proj"],
│   │     ...
│   │   }
│   │
│   ├── adapter_model.bin  (200 MB - LoRA weights)
│   └── adapter_pytorch_model.bin
│
├── training_log.txt
│   [2025-01-15 10:30:45] INFO: Step 1: Environment Setup
│   [2025-01-15 10:30:48] INFO: Detected 1 GPU(s): NVIDIA Tesla T4
│   [2025-01-15 10:31:15] INFO: Loaded 500 samples
│   ...
│   [2025-01-15 11:45:30] INFO: Final Validation Accuracy: 0.8523
│
├── README.md
│   (Usage instructions, configuration details)
│
└── metadata.json
    {
      "model_name": "nvidia/Nemotron-3-Nano-30B",
      "lora_rank": 32,
      "lora_alpha": 64,
      "max_seq_length": 8192,
      "training_date": "2025-01-15T11:45:30",
      "target_accuracy": 0.85,
      "data_augmentation": "numeric_permutation, template_transform, symbolic_substitution"
    }
```

---

## Troubleshooting Common Issues

### Issue 1: "CUDA out of memory"

```
RuntimeError: CUDA out of memory. Tried to allocate X.XXGiB
```

**Solution**:
```python
# Edit script: Reduce batch size
config.batch_size = 1  # from 2

# Or increase gradient accumulation
config.gradient_accumulation_steps = 8  # from 4

# Or reduce max sequence length
config.max_seq_length = 4096  # from 8192
```

### Issue 2: "Model not found" / Download timeout

```
FileNotFoundError: Can't find 'nvidia/Nemotron-3-Nano-30B' in HF
```

**Solution**:
```bash
# Pre-download model
huggingface-cli download nvidia/Nemotron-3-Nano-30B

# Edit script to use local path
config.model_name = "/home/ubuntu/.cache/huggingface/hub/..."
```

### Issue 3: "Low validation accuracy" (< 0.5)

**Diagnostic**:
```bash
# Check data quality
head -5 train.csv
wc -l train.csv

# Verify format
python3 << 'EOF'
import csv
with open('train.csv', 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < 3:
            print(f"Row {i}: {row}")
        if i > 10:
            break
EOF
```

**Solutions**:
1. Increase training epochs: `config.num_epochs = 5`
2. Lower learning rate: `config.learning_rate = 1e-4`
3. Add more augmentation: Modify `augment_data()` function
4. Verify data: ensure expected_output matches puzzle_prompt

### Issue 4: "Training too slow"

**Monitoring**:
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# Should be close to: (total_memory) * 0.85
# e.g., for T4 (16GB): should be ~13-14 GB
```

**Solutions**:
1. Reduce `max_seq_length`: 8192 → 4096
2. Increase `batch_size` if memory allows
3. Reduce `num_epochs` for testing
4. Use faster GPU (V100 is 5x faster than T4)

### Issue 5: "Script exits with 'No training data'"

**Cause**: `train.csv` not found in current directory

**Solution**:
```bash
# Verify file exists
ls -lh train.csv

# If not, create synthetic data
python3 << 'EOF'
# ... see "Create synthetic data" section above
EOF

# Or specify path in script
config.data_file = "/path/to/train.csv"
```

---

## Performance Optimization Tips

### For Faster Training

**Quick Training (Accuracy ~0.75, Time ~15 min)**:
```python
config.num_epochs = 2
config.max_seq_length = 4096
config.batch_size = 4  # if GPU allows
```

**Balanced Training (Accuracy ~0.82, Time ~45 min)**:
```python
# Use defaults (already optimized)
```

**High Accuracy Training (Accuracy ~0.85+, Time ~90 min)**:
```python
config.num_epochs = 5
config.batch_size = 1
config.gradient_accumulation_steps = 8
config.learning_rate = 3e-4
```

### For Lower GPU Memory

**4GB GPU (RTX 3050 mobile)**:
```python
config.batch_size = 1
config.gradient_accumulation_steps = 8
config.max_seq_length = 2048
# Reduce model size or use quantization
```

**8GB GPU (RTX 4060)**:
```python
config.batch_size = 1
config.gradient_accumulation_steps = 4
config.max_seq_length = 4096
```

**16GB GPU (T4, RTX 4080)**:
```python
# Use defaults - optimized for this range
```

**24GB+ GPU (A10, V100)**:
```python
config.batch_size = 4
config.gradient_accumulation_steps = 1
# Can increase learning rate slightly
```

---

## Integration with ML Frameworks

### Using Trained Adapter in Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-3-Nano-30B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "adapter/")
model.eval()

# Inference
tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-3-Nano-30B")
prompt = "Instruction: Solve this puzzle.\nInput: What is 2 + 2?\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
answer = tokenizer.decode(outputs[0])
print(answer)
```

### MLOps Pipeline Integration

```python
import subprocess
import json
from datetime import datetime

# Run training as subprocess
result = subprocess.run(
    ["python3", "nemotron_lora_train.py"],
    capture_output=True,
    text=True
)

# Parse results
with open("training_log.txt", "r") as f:
    log = f.read()
    
# Extract metrics
with open("training_history.json", "r") as f:
    metrics = json.load(f)
    
# Log to monitoring system
print(f"Final accuracy: {metrics['val_accuracy'][-1]}")
print(f"Training completed at {datetime.now()}")

# Conditional deployment
if metrics['val_accuracy'][-1] > 0.85:
    subprocess.run(["aws", "s3", "cp", "submission.zip", "s3://my-bucket/"])
else:
    print("Accuracy below threshold, not deploying")
```

---

## Cost Estimation (AWS)

### For 500 Training Samples

| GPU | Instance | Hourly Cost | Estimated Time | Total Cost |
|-----|----------|-------------|----------------|-----------|
| T4 | g4dn.xlarge | $0.35 | 2.5 hours | $0.88 |
| A10 | g4dn.12xlarge | $3.06 | 1.5 hours | $4.59 |
| V100 | p3.2xlarge | $3.06 | 1 hour | $3.06 |
| H100 | p4d.24xlarge | $32.77 | 0.5 hours | $16.39 |

**Recommendation**: Start with `g4dn.xlarge` for cost-effective development, upgrade to `g4dn.12xlarge` for production.

---

## Scaling to Larger Datasets

### 10K Samples (4x Augmentation = 40K sequences)

```python
config.num_epochs = 5
config.batch_size = 1
config.gradient_accumulation_steps = 8
# Training time: ~4-6 hours on V100
```

### 100K Samples (4x Augmentation = 400K sequences)

```python
# Consider distributed training:
# - Use torch.distributed.launch
# - Or use HuggingFace Trainer with n_gpu > 1

config.num_epochs = 3  # Reduce epochs due to larger dataset
config.batch_size = 2
config.gradient_accumulation_steps = 4
# Training time: ~8-12 hours on 4x V100
```

---

## Advanced: Custom Data Augmentation

Edit the `augment_data()` function to add domain-specific augmentation:

```python
def augment_data(df, config: Config) -> any:
    augmented_rows = []
    for row in df.to_dicts():
        augmented_rows.append(row)  # Original
        
        # Strategy 1: Numeric permutation
        augmented_rows.append({
            **row,
            'puzzle_prompt': _permute_numbers(row['puzzle_prompt'])
        })
        
        # Strategy 2: Template transform
        augmented_rows.append({
            **row,
            'puzzle_prompt': _apply_template_transform(row['puzzle_prompt'])
        })
        
        # Strategy 3: Symbolic substitution
        augmented_rows.append({
            **row,
            'puzzle_prompt': _symbolic_substitution(row['puzzle_prompt'])
        })
        
        # Strategy 4: Custom - add your own!
        if 'your_domain_pattern' in row['puzzle_prompt']:
            augmented_rows.append({
                **row,
                'puzzle_prompt': your_custom_transform(row['puzzle_prompt'])
            })
    
    return pl.DataFrame(augmented_rows)

def your_custom_transform(text: str) -> str:
    """Your domain-specific augmentation."""
    # Example: for math problems, reorder operations
    # Example: for logic puzzles, flip truth values
    return modified_text
```

---

## Next Steps

1. **Run the script** with default settings
2. **Monitor training_log.txt** for progress
3. **Achieve 0.85+ accuracy** (typical: 2-3 hours on T4)
4. **Download submission.zip**
5. **Submit to Kaggle**
6. **Monitor leaderboard** and iterate

Good luck! 🚀

---

**Last Updated**: January 2025  
**Python Version**: 3.9+  
**CUDA Version**: 12.1+  
**Estimated Execution**: 1-2 hours (T4), 30-60 minutes (V100)
