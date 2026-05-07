"""
Model Evaluation Script

Comprehensive evaluation of trained WAF model including:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Per-class performance analysis
- False positive/negative analysis
- Confidence distribution
"""

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json

from config import TrainingConfig


class WAFDataset(Dataset):
    """PyTorch Dataset for WAF evaluation."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


class WAFEvaluator:
    """Evaluator for trained WAF model."""
    
    def __init__(self, model_path: str, config: TrainingConfig):
        self.config = config
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(self.model_path))
        self.model = DistilBertForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def load_test_data(self):
        """Load test dataset."""
        data_dir = Path(self.config.data_dir)
        test_df = pd.read_csv(data_dir / "test.csv")
        
        
        # Load unseen test data
        unseen_path = Path("data/raw/unseen_requests.csv")
        if unseen_path.exists():
          unseen_df = pd.read_csv(unseen_path)
          test_df = pd.concat([test_df, unseen_df], ignore_index=True)

        
        # Use encoded_text if available
        text_column = "encoded_text" if "encoded_text" in test_df.columns else "text"
        
        texts = test_df[text_column].fillna("").tolist()
        labels = test_df["label"].map(self.config.label_map).tolist()
        
        return texts, labels, test_df
    
    
    def evaluate(self, texts, labels):
        """Run evaluation on test set."""
        print("\nTokenizing test data...")
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Create dataset and loader
        dataset = WAFDataset(encodings, labels)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Evaluate
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("Running evaluation...")
        with torch.no_grad():
            for batch in tqdm(loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def generate_report(self, predictions, labels, probabilities, output_dir: Path):
        """Generate comprehensive evaluation report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Classification Report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            labels,
            predictions,
            target_names=list(self.config.id2label.values()),
            digits=4
        )
        print(report)
        
        # Save report
        with open(output_dir / "classification_report.txt", 'w') as f:
            f.write(report)
        
        # 2. Confusion Matrix
        print("\nGenerating confusion matrix...")
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.config.id2label.values()),
            yticklabels=list(self.config.id2label.values())
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
        plt.close()
        print(f"✓ Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
        
        # 3. Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(self.config.num_labels))
        )
        
        metrics_df = pd.DataFrame({
            'Class': list(self.config.id2label.values()),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        print("\n" + "="*60)
        print("PER-CLASS METRICS")
        print("="*60)
        print(metrics_df.to_string(index=False))
        
        metrics_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
        
        # 4. Confidence distribution
        print("\nGenerating confidence distribution plots...")
        
        max_probs = np.max(probabilities, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Overall confidence distribution
        axes[0].hist(max_probs, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Overall Prediction Confidence Distribution')
        axes[0].axvline(x=0.6, color='orange', linestyle='--', label='Medium threshold (0.6)')
        axes[0].axvline(x=0.85, color='red', linestyle='--', label='High threshold (0.85)')
        axes[0].legend()
        
        # Confidence by correctness
        correct_mask = predictions == labels
        correct_confidences = max_probs[correct_mask]
        incorrect_confidences = max_probs[~correct_mask]
        
        axes[1].hist(correct_confidences, bins=30, alpha=0.7, label='Correct', edgecolor='black')
        axes[1].hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', edgecolor='black')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Distribution by Correctness')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "confidence_distribution.png", dpi=300)
        plt.close()
        print(f"✓ Saved confidence distribution to {output_dir / 'confidence_distribution.png'}")
        
        # 5. False positive/negative analysis
        print("\nAnalyzing false positives and negatives...")
        
        # False positives: Predicted as attack but actually SAFE
        safe_label = self.config.label_map["SAFE"]
        fp_mask = (labels == safe_label) & (predictions != safe_label)
        fp_count = np.sum(fp_mask)
        fp_rate = fp_count / np.sum(labels == safe_label) if np.sum(labels == safe_label) > 0 else 0
        
        # False negatives: Predicted as SAFE but actually attack
        fn_mask = (labels != safe_label) & (predictions == safe_label)
        fn_count = np.sum(fn_mask)
        fn_rate = fn_count / np.sum(labels != safe_label) if np.sum(labels != safe_label) > 0 else 0
        
        error_analysis = {
            "false_positives": int(fp_count),
            "false_positive_rate": float(fp_rate),
            "false_negatives": int(fn_count),
            "false_negative_rate": float(fn_rate),
            "total_errors": int(np.sum(predictions != labels)),
            "total_samples": len(labels),
            "accuracy": float(accuracy_score(labels, predictions))
        }
        
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        print(f"False Positives: {fp_count} ({fp_rate*100:.2f}%)")
        print(f"False Negatives: {fn_count} ({fn_rate*100:.2f}%)")
        print(f"Total Errors: {error_analysis['total_errors']}")
        print(f"Accuracy: {error_analysis['accuracy']*100:.2f}%")
        
        with open(output_dir / "error_analysis.json", 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # 6. Summary statistics
        summary = {
            "model_path": str(self.model_path),
            "test_samples": len(labels),
            "accuracy": float(accuracy_score(labels, predictions)),
            "weighted_precision": float(precision.mean()),
            "weighted_recall": float(recall.mean()),
            "weighted_f1": float(f1.mean()),
            "avg_confidence": float(max_probs.mean()),
            "median_confidence": float(np.median(max_probs)),
            "min_confidence": float(max_probs.min()),
            "max_confidence": float(max_probs.max())
        }
        
        with open(output_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        
        return summary
    
    def run_evaluation(self, output_dir: str = "models/evaluation"):
        """Run complete evaluation pipeline."""
        output_dir = Path(output_dir)
        
        # Load test data
        texts, labels, test_df = self.load_test_data()
        print(f"Loaded {len(texts)} test samples")
        
        # Run evaluation
        predictions, labels, probabilities = self.evaluate(texts, labels)
        
        # Generate report
        summary = self.generate_report(predictions, labels, probabilities, output_dir)
        
        return summary


if __name__ == "__main__":
    config = TrainingConfig()
    
    # Evaluate best model
    model_path = "models/checkpoints/best_model"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_v2.py")
    else:
        evaluator = WAFEvaluator(model_path, config)
        evaluator.run_evaluation()