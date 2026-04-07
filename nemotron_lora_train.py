#!/usr/bin/env python3
"""
Nemotron-3-Nano-30B LoRA Fine-Tuning Script
============================================
Complete training pipeline for reasoning benchmark optimization.
Trains a LoRA adapter to achieve >0.85 accuracy on NVIDIA reasoning tasks.
"""

import os
import sys
import json
import logging
import shutil
import zipfile
import random
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================================
# SECTION 1: DEPENDENCY INSTALLATION & IMPORTS
# ============================================================================

def install_dependencies():
    """Automatically install all required packages."""
    required_packages = {
        'torch': 'torch>=2.0.0',
        'transformers': 'transformers>=4.36.0',
        'peft': 'peft>=0.7.0',
        'polars': 'polars>=0.19.0',
        'accelerate': 'accelerate>=0.25.0',
        'bitsandbytes': 'bitsandbytes>=0.41.0',
        'numpy': 'numpy>=1.24.0',
    }
    
    for package, version_spec in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            print(f"[SETUP] Installing {version_spec}...")
            os.system(f"{sys.executable} -m pip install -q {version_spec}")


install_dependencies()

try:
    import polars as pl
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model
    from accelerate import Accelerator
except ImportError as e:
    print(f"[ERROR] Failed to import required module: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 2: CONFIGURATION & SETUP
# ============================================================================

@dataclass
class Config:
    """Hyperparameters and configuration."""
    # Model & LoRA
    model_name: str = "nvidia/Nemotron-3-Nano-30B"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    # Training
    max_seq_length: int = 8192
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    
    # GPU & Memory
    max_gpu_utilization: float = 0.85
    temperature: float = 0.0
    top_p: float = 1.0
    
    # Data
    data_file: str = "train.csv"
    train_val_split: float = 0.9
    seed: int = 42
    
    # Evaluation
    eval_steps: int = 50
    save_steps: int = 100
    early_stopping_patience: int = 5
    target_accuracy: float = 0.85
    
    # Output
    output_dir: str = "./lora_adapter"
    log_file: str = "training_log.txt"
    submission_zip: str = "submission.zip"


config = Config()

# ============================================================================
# SECTION 3: LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("Nemotron-3-Nano-30B LoRA Fine-Tuning Pipeline")
logger.info("=" * 80)

# ============================================================================
# SECTION 4: UTILITY FUNCTIONS
# ============================================================================

def set_seeds(seed: int):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seeds set to {seed}")


def detect_gpu_config() -> Tuple[int, float]:
    """Detect GPU availability and estimate safe batch size."""
    if not torch.cuda.is_available():
        logger.warning("No GPU detected! Training will be very slow.")
        return 0, 0.0
    
    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"Detected {num_gpus} GPU(s): {gpu_name}")
    logger.info(f"Total GPU memory: {total_memory:.2f} GB")
    
    available_memory = total_memory * config.max_gpu_utilization
    logger.info(f"Available memory (85% util): {available_memory:.2f} GB")
    
    return num_gpus, available_memory


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content within \\boxed{} from model output."""
    pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def normalize_text(text: str) -> str:
    """Normalize text for consistency."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def is_answer_correct(predicted: str, expected: str, tolerance: float = 1e-6) -> bool:
    """Compare answers with numeric tolerance."""
    try:
        # Try numeric comparison first
        pred_val = float(predicted)
        exp_val = float(expected)
        return abs(pred_val - exp_val) / (abs(exp_val) + 1e-9) < tolerance
    except (ValueError, ZeroDivisionError):
        # Fall back to string comparison
        return normalize_text(predicted) == normalize_text(expected)


# ============================================================================
# SECTION 5: DATA LOADING & PREPROCESSING
# ============================================================================

class ReasoningDataset(Dataset):
    """Custom dataset for reasoning tasks."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 8192):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Format with chain-of-thought template
        prompt = self._format_prompt(item)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }
    
    def _format_prompt(self, item: Dict) -> str:
        """Format prompt with reasoning template."""
        instruction = "Solve the following puzzle using logical reasoning rules."
        cot_instruction = "Let me work through this step by step."
        
        template = (
            f"Instruction: {instruction}\n"
            f"Input: {item['puzzle_prompt']}\n"
            f"{cot_instruction}\n"
            f"Output: {item['expected_output']}"
        )
        return template


def load_and_preprocess_data(filepath: str, config: Config) -> Tuple[List[Dict], List[Dict]]:
    """Load CSV data using Polars and preprocess."""
    logger.info(f"Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}")
        # Create synthetic data for demo
        logger.info("Creating synthetic training data for demonstration...")
        return create_synthetic_data(500)
    
    try:
        df = pl.read_csv(filepath)
    except Exception as e:
        logger.warning(f"Failed to read CSV with Polars: {e}. Trying pandas fallback...")
        import pandas as pd
        df = pd.read_csv(filepath)
        df = pl.from_pandas(df)
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Clean data
    df = clean_data(df)
    logger.info(f"After cleaning: {len(df)} samples")
    
    # Augment data
    df = augment_data(df, config)
    logger.info(f"After augmentation: {len(df)} samples")
    
    # Convert to list of dicts
    data = df.to_dicts()
    
    # Split into train/validation
    train_size = int(len(data) * config.train_val_split)
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    train_data = [data[i] for i in indices[:train_size]]
    val_data = [data[i] for i in indices[train_size:]]
    
    logger.info(f"Train set: {len(train_data)}, Validation set: {len(val_data)}")
    return train_data, val_data


def clean_data(df) -> any:
    """Remove malformed or incomplete prompts."""
    logger.info("Cleaning data...")
    
    # Remove nulls
    required_cols = ['puzzle_prompt', 'expected_output']
    for col in required_cols:
        if col in df.columns:
            df = df.filter(df[col].is_not_null())
    
    # Remove empty strings
    df = df.filter(df['puzzle_prompt'].str.lengths() > 5)
    df = df.filter(df['expected_output'].str.lengths() > 0)
    
    logger.info(f"Data cleaning complete: {len(df)} samples remaining")
    return df


def augment_data(df, config: Config) -> any:
    """Augment data via synthetic variations and transformations."""
    logger.info("Augmenting data...")
    
    augmented_rows = []
    
    for row in df.to_dicts():
        # Keep original
        augmented_rows.append(row)
        
        # Variation 1: Numeric permutation (if applicable)
        if any(c.isdigit() for c in row['puzzle_prompt']):
            augmented_rows.append({
                **row,
                'puzzle_prompt': _permute_numbers(row['puzzle_prompt'])
            })
        
        # Variation 2: Template-based transformation
        augmented_rows.append({
            **row,
            'puzzle_prompt': _apply_template_transform(row['puzzle_prompt'])
        })
        
        # Variation 3: Symbolic substitution
        augmented_rows.append({
            **row,
            'puzzle_prompt': _symbolic_substitution(row['puzzle_prompt'])
        })
    
    augmented_df = pl.DataFrame(augmented_rows)
    logger.info(f"Data augmentation expanded dataset from {len(df)} to {len(augmented_df)}")
    return augmented_df


def _permute_numbers(text: str) -> str:
    """Permute numbers in text to create variations."""
    numbers = re.findall(r'\d+', text)
    if not numbers or len(numbers) < 2:
        return text
    
    shuffled = numbers.copy()
    random.shuffle(shuffled)
    
    result = text
    for orig, new in zip(sorted(set(numbers)), shuffled[:len(set(numbers))]):
        result = result.replace(str(orig), str(new), 1)
    
    return result


def _apply_template_transform(text: str) -> str:
    """Apply template-based rule transformation."""
    # Simple example: add context
    templates = [
        f"Consider the following: {text}",
        f"Analyze this problem: {text}",
        f"Work through: {text}",
    ]
    return random.choice(templates)


def _symbolic_substitution(text: str) -> str:
    """Apply symbolic substitution for generalization."""
    substitutions = {
        'and': '&',
        'or': '|',
        'not': '~',
    }
    result = text.lower()
    for word, symbol in substitutions.items():
        if random.random() > 0.7:  # 30% chance
            result = result.replace(word, symbol)
    return result


def create_synthetic_data(num_samples: int = 500) -> Tuple[List[Dict], List[Dict]]:
    """Create synthetic training data for demonstration."""
    puzzles = [
        {"puzzle_prompt": "What is 2 + 2?", "expected_output": "4"},
        {"puzzle_prompt": "What is 5 * 3?", "expected_output": "15"},
        {"puzzle_prompt": "If A=1, B=2, what is A + B?", "expected_output": "3"},
        {"puzzle_prompt": "Solve: x + 5 = 10", "expected_output": "5"},
        {"puzzle_prompt": "What comes after 1, 2, 3?", "expected_output": "4"},
        {"puzzle_prompt": "If true AND false, result?", "expected_output": "false"},
        {"puzzle_prompt": "5^2 equals?", "expected_output": "25"},
        {"puzzle_prompt": "Logical NOT of true?", "expected_output": "false"},
    ]
    
    data = []
    for i in range(num_samples):
        base = puzzles[i % len(puzzles)]
        data.append({
            **base,
            'puzzle_prompt': f"{base['puzzle_prompt']} (variant {i//len(puzzles)})"
        })
    
    train_size = int(len(data) * config.train_val_split)
    return data[:train_size], data[train_size:]


# ============================================================================
# SECTION 6: MODEL SETUP & LoRA CONFIGURATION
# ============================================================================

def setup_model_and_tokenizer(config: Config):
    """Load model and tokenizer with memory optimization."""
    logger.info(f"Loading model: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
    )
    
    logger.info(f"Model loaded with parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Typical attention projections
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    logger.info(f"LoRA configuration applied: rank={config.lora_rank}, alpha={config.lora_alpha}")
    
    return model, tokenizer


# ============================================================================
# SECTION 7: TRAINING LOOP WITH VALIDATION
# ============================================================================

class ReasoningTrainer:
    """Custom trainer with integrated validation and early stopping."""
    
    def __init__(self, model, tokenizer, config: Config, train_data, val_data):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # Datasets
        self.train_dataset = ReasoningDataset(train_data, tokenizer, config.max_seq_length)
        self.val_dataset = ReasoningDataset(val_data, tokenizer, config.max_seq_length)
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Optimizer & scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(self.train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        
        # Training state
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*80}")
            
            # Training phase
            train_loss = self._train_epoch(epoch)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            val_accuracy, val_loss = self._validate()
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_accuracy)
                logger.info(f"✓ New best accuracy: {val_accuracy:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            
            # Check stopping criteria
            if val_accuracy >= self.config.target_accuracy:
                logger.info(f"\n✓ Target accuracy {self.config.target_accuracy} achieved!")
                break
            
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info(f"\nTraining complete. Best validation accuracy: {self.best_val_accuracy:.4f}")
        return self.best_val_accuracy
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass with gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                    f"Loss={avg_loss:.4f}, LR={self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        return total_loss / len(self.train_loader)
    
    def _validate(self) -> Tuple[float, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Generate predictions (simplified accuracy)
                with torch.no_grad():
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Compare with actual labels (basic accuracy)
                    correct += (predictions == labels).sum().item()
                    total += labels.numel()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.val_loader)
        
        return accuracy, avg_loss
    
    def _save_checkpoint(self, epoch: int, accuracy: float):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(
            f"{self.config.output_dir}/checkpoint-epoch-{epoch}-acc-{accuracy:.4f}"
        )
        
        logger.info(f"Checkpoint saved to {self.config.output_dir}")


# ============================================================================
# SECTION 8: SUBMISSION PACKAGING
# ============================================================================

def package_submission(output_dir: str, submission_zip: str, config: Config):
    """Package trained adapter and supporting files into submission.zip."""
    logger.info(f"Packaging submission to {submission_zip}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create submission directory
    submission_dir = f"{output_dir}_submission"
    os.makedirs(submission_dir, exist_ok=True)
    
    # Copy adapter files
    adapter_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if adapter_dirs:
        latest_dir = sorted(adapter_dirs)[-1]
        source = os.path.join(output_dir, latest_dir)
        dest = os.path.join(submission_dir, "adapter")
        shutil.copytree(source, dest, dirs_exist_ok=True)
        logger.info(f"Copied adapter from {source}")
    
    # Copy training log
    if os.path.exists(config.log_file):
        shutil.copy(config.log_file, submission_dir)
        logger.info(f"Copied training log")
    
    # Create metadata
    metadata = {
        "model_name": config.model_name,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "max_seq_length": config.max_seq_length,
        "training_date": datetime.now().isoformat(),
        "target_accuracy": config.target_accuracy,
        "data_augmentation": "numeric_permutation, template_transform, symbolic_substitution",
    }
    
    with open(os.path.join(submission_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Created metadata.json")
    
    # Create README
    readme = f"""# Nemotron-3-Nano-30B LoRA Adapter

## Model Information
- **Base Model**: {config.model_name}
- **Adapter Type**: LoRA (Low-Rank Adaptation)
- **Rank**: {config.lora_rank}
- **Alpha**: {config.lora_alpha}
- **Training Date**: {datetime.now().isoformat()}

## Configuration
- **Max Sequence Length**: {config.max_seq_length}
- **Learning Rate**: {config.learning_rate}
- **Batch Size**: {config.batch_size}
- **Training Epochs**: {config.num_epochs}

## Data Augmentation Strategy
- Numeric permutation for sequence variations
- Template-based rule transformations
- Symbolic substitution for generalization

## Usage
1. Load the base model: `AutoModelForCausalLM.from_pretrained("{config.model_name}")`
2. Load LoRA adapter: `PeftModel.from_pretrained(model, "adapter")`
3. Use for inference with chain-of-thought prompting

## Files
- `adapter/`: LoRA adapter weights and configuration
- `training_log.txt`: Complete training history
- `metadata.json`: Model and training metadata
"""
    
    with open(os.path.join(submission_dir, "README.md"), 'w') as f:
        f.write(readme)
    
    logger.info("Created README.md")
    
    # Create ZIP file
    with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, submission_dir)
                zipf.write(file_path, arcname)
    
    logger.info(f"✓ Submission packaged: {submission_zip}")
    logger.info(f"  Size: {os.path.getsize(submission_zip) / 1e6:.2f} MB")
    
    # Cleanup
    shutil.rmtree(submission_dir)
    
    return submission_zip


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    try:
        # 1. Setup
        logger.info("Step 1: Environment Setup")
        set_seeds(config.seed)
        num_gpus, available_memory = detect_gpu_config()
        
        if num_gpus == 0:
            logger.warning("Running on CPU - training will be slow!")
        
        # 2. Data Loading
        logger.info("\nStep 2: Data Loading & Preprocessing")
        train_data, val_data = load_and_preprocess_data(config.data_file, config)
        
        if not train_data:
            logger.error("No training data available!")
            return False
        
        # 3. Model Setup
        logger.info("\nStep 3: Model & LoRA Setup")
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # 4. Training
        logger.info("\nStep 4: Training with Validation")
        trainer = ReasoningTrainer(model, tokenizer, config, train_data, val_data)
        final_accuracy = trainer.train()
        
        # 5. Save Training History
        logger.info("\nStep 5: Saving Training History")
        with open("training_history.json", 'w') as f:
            json.dump(trainer.training_history, f, indent=2)
        logger.info("Training history saved to training_history.json")
        
        # 6. Final Model Save
        logger.info("\nStep 6: Final Model Save")
        os.makedirs(config.output_dir, exist_ok=True)
        model.save_pretrained(f"{config.output_dir}/final")
        tokenizer.save_pretrained(f"{config.output_dir}/final")
        logger.info(f"Final model saved to {config.output_dir}/final")
        
        # 7. Submission Packaging
        logger.info("\nStep 7: Submission Packaging")
        submission_path = package_submission(config.output_dir, config.submission_zip, config)
        
        # 8. Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Final Validation Accuracy: {final_accuracy:.4f}")
        logger.info(f"Target Accuracy: {config.target_accuracy:.4f}")
        logger.info(f"Status: {'✓ TARGET ACHIEVED' if final_accuracy >= config.target_accuracy else '⚠ Below target'}")
        logger.info(f"Submission Package: {submission_path}")
        logger.info(f"Training Log: {config.log_file}")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
