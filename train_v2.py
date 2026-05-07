"""
Enhanced Model Training Script

Features:
- Proper train/val/test splits
- Comprehensive evaluation metrics
- Model checkpointing (save best model)
- Early stopping
- Learning rate scheduling
- Detailed logging
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import sys

# Import config
from config import TrainingConfig


class WAFDataset(Dataset):
    """PyTorch Dataset for WAF training."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


class WAFTrainer:
    """Trainer class for WAF model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Class weights (SAFE has lower weight, attacks have higher weight)
        self.class_weights = torch.tensor(
        [1.0, 3.0, 3.0, 3.0, 3.0],
        device=self.device
        )

        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.training_history = []
    
    def load_data(self):
        """Load and prepare datasets."""
        print("="*60)
        print("Loading datasets...")
        print("="*60)
        
        data_dir = Path(self.config.data_dir)
        
        # Load train/val/test splits
        train_df = pd.read_csv(data_dir / "train.csv")
        val_df = pd.read_csv(data_dir / "val.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        
        print(f"Train size: {len(train_df)}")
        print(f"Val size: {len(val_df)}")
        print(f"Test size: {len(test_df)}")
        
        # Use encoded_text if available, otherwise use text
        text_column = "encoded_text" if "encoded_text" in train_df.columns else "text"
        print(f"Using column: {text_column}")
        
        # Extract texts and labels
        train_texts = train_df[text_column].fillna("").tolist()
        train_labels = train_df["label"].map(self.config.label_map).tolist()
        
        val_texts = val_df[text_column].fillna("").tolist()
        val_labels = val_df["label"].map(self.config.label_map).tolist()
        
        test_texts = test_df[text_column].fillna("").tolist()
        test_labels = test_df["label"].map(self.config.label_map).tolist()
        
        return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
    
    def prepare_datasets(self, train_data, val_data):
        """Tokenize and create PyTorch datasets."""
        print("\nTokenizing...")
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.config.model_name)
        
        train_texts, train_labels = train_data
        val_texts, val_labels = val_data
        
        # Tokenize
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512
        )
        
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Create datasets
        train_dataset = WAFDataset(train_encodings, train_labels)
        val_dataset = WAFDataset(val_encodings, val_labels)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def initialize_model(self):
        """Initialize model, optimizer, and scheduler."""
        print(f"\nInitializing model: {self.config.model_name}")
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label_map
        )
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        print("Model initialized successfully")
    
    def create_scheduler(self, train_loader):
        """Create learning rate scheduler."""
        total_steps = len(train_loader) * self.config.num_epochs
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )

            logits = outputs.logits

            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)

            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.model_save_dir)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model"
            self.model.save_pretrained(best_path)
            self.tokenizer.save_pretrained(best_path)
            print(f"✓ Saved best model to {best_path}")
        
        # Save latest model
        latest_path = checkpoint_dir / "latest_model"
        self.model.save_pretrained(latest_path)
        self.tokenizer.save_pretrained(latest_path)
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Load data
        train_data, val_data, test_data = self.load_data()
        
        # Prepare datasets
        train_loader, val_loader = self.prepare_datasets(train_data, val_data)
        
        # Initialize model
        self.initialize_model()
        
        # Create scheduler
        self.create_scheduler(train_loader)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f} | Val Precision: {val_metrics['precision']:.4f}")
            
            # Save history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            })
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epoch(s)")
            
            # Save latest checkpoint
            self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save training history
        history_path = Path(self.config.model_save_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.config.model_save_dir}")


if __name__ == "__main__":
    # Create config
    config = TrainingConfig()
    
    # Create trainer
    trainer = WAFTrainer(config)
    
    # Train
    trainer.train()