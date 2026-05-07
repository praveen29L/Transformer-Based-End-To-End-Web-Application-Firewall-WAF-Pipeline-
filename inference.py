"""
Enhanced Inference Module with Confidence Scores

Features:
- Confidence score extraction
- Batch inference support
- Thread-safe model loading
- Detailed prediction metadata
"""

import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from pathlib import Path
from typing import Dict, List, Union, Optional
import time


class WAFClassifier:
    """
    Enhanced WAF classifier with confidence scores and batch support.
    """
    
    def __init__(self, model_path: str = "models/checkpoints/best_model"):
        """
        Initialize the WAF classifier.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please train the model first using train_v2.py"
            )
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(self.model_path))
        self.model = DistilBertForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Label mappings
        self.id2label = {
            0: "SAFE",
            1: "SQLI",
            2: "XSS",
            3: "PATH_TRAVERSAL",
            4: "COMMAND_INJECTION"
        }
        
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        print(f"Model loaded successfully on {self.device}")
    
    def classify(self, text: str) -> Dict:
        """
        Classify a single HTTP request.
        
        Args:
            text: Request text to classify
        
        Returns:
            Dictionary containing:
                - label: Predicted class label
                - confidence: Confidence score (0-1)
                - all_scores: Dictionary of all class probabilities
                - inference_time_ms: Inference time in milliseconds
        """
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
        
        # Get prediction
        confidence, pred_idx = torch.max(probs, dim=0)
        predicted_label = self.id2label[pred_idx.item()]
        
        # Get all class scores
        all_scores = {
            self.id2label[i]: float(probs[i].item())
            for i in range(len(self.id2label))
        }
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "label": predicted_label,
            "confidence": float(confidence.item()),
            "all_scores": all_scores,
            "inference_time_ms": round(inference_time, 2)
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """
        Classify multiple HTTP requests in batch.
        
        Args:
            texts: List of request texts to classify
        
        Returns:
            List of prediction dictionaries
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Process results
        results = []
        for i in range(len(texts)):
            prob_vector = probs[i]
            confidence, pred_idx = torch.max(prob_vector, dim=0)
            predicted_label = self.id2label[pred_idx.item()]
            
            all_scores = {
                self.id2label[j]: float(prob_vector[j].item())
                for j in range(len(self.id2label))
            }
            
            results.append({
                "label": predicted_label,
                "confidence": float(confidence.item()),
                "all_scores": all_scores
            })
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(texts)
        
        # Add timing info to results
        for result in results:
            result["inference_time_ms"] = round(avg_time, 2)
        
        return results
    
    def get_attack_probability(self, text: str) -> float:
        """
        Get probability that the request is an attack (any non-SAFE class).
        
        Args:
            text: Request text to classify
        
        Returns:
            Probability (0-1) that request is malicious
        """
        result = self.classify(text)
        
        # Sum probabilities of all attack classes
        attack_prob = sum(
            score for label, score in result["all_scores"].items()
            if label != "SAFE"
        )
        
        return attack_prob


# Global classifier instance (lazy-loaded)
_classifier = None


def get_classifier(model_path: str = "models/checkpoints/best_model") -> WAFClassifier:
    """
    Get or create the global classifier instance.
    
    This function provides a singleton pattern for the classifier,
    which is useful for FastAPI dependency injection.
    
    Args:
        model_path: Path to the trained model
    
    Returns:
        WAFClassifier instance
    """
    global _classifier
    
    if _classifier is None:
        _classifier = WAFClassifier(model_path)
    
    return _classifier


def classify_request(text: str, model_path: str = "models/checkpoints/best_model") -> Dict:
    """
    Convenience function for single request classification.
    
    Args:
        text: Request text to classify
        model_path: Path to the trained model
    
    Returns:
        Prediction dictionary
    """
    classifier = get_classifier(model_path)
    return classifier.classify(text)


if __name__ == "__main__":
    # Test the classifier
    print("="*60)
    print("Testing WAF Classifier")
    print("="*60)
    
    # Check if model exists
    model_path = "models/checkpoints/best_model"
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using: python models/train_v2.py")
        exit(1)
    
    # Initialize classifier
    classifier = WAFClassifier(model_path)
    
#     # Test cases
#     test_cases = [
#     # Benign traffic
#     ("GET /products?id=123&sort=price", "SAFE"),
#     ("POST /login username=john&password=hello123", "SAFE"),
#     ("GET /search?q=laptop+bags", "SAFE"),

#     # SQL Injection (variants)
#     ("GET /login?user=admin'/**/OR/**/'1'='1", "SQLI"),
#     ("GET /item?id=10 UNION SELECT username,password FROM users", "SQLI"),
#     ("POST /auth user=admin' OR 'x'='x' --", "SQLI"),

#     # XSS (variants)
#     ("GET /search?q=<ScRipT>alert(document.cookie)</ScRipT>", "XSS"),
#     ("GET /comment?msg=<img src=x onerror=alert(1)>", "XSS"),
#     ("POST /feedback message=<svg/onload=alert(1)>", "XSS"),

#     # Path Traversal (encoded + mixed)
#     ("GET /file?name=..%2f..%2f..%2fetc%2fpasswd", "PATH_TRAVERSAL"),
#     ("GET /download?file=../../../../windows/system32/config", "PATH_TRAVERSAL"),
#     ("GET /read?path=../config/.env", "PATH_TRAVERSAL"),

#     # Command Injection (variants)

#     ("POST /run cmd=$(id)", "COMMAND_INJECTION"),
#     ("GET /exec?cmd=ls;cat /etc/passwd", "COMMAND_INJECTION"),
#     # SQLi – unseen variants
#     ("GET /login?user=admin'/**/OR/**/'1'='1", "SQLI"),
#     ("GET /login?u=admin') OR (SELECT 1 FROM dual)--", "SQLI"),
    
#      # XSS – unseen variants
#     ("GET /search?q=<ScRipT>alert(document.domain)</ScRipT>", "XSS"),
#     ("GET /search?q=<svg/onload=confirm(1)>", "XSS"),

#     # Path Traversal – unseen variants
#     ("GET /file?path=..%2f..%2f..%2fetc%2fpasswd", "PATH_TRAVERSAL"),
#     ("GET /download?file=....//....//etc/passwd", "PATH_TRAVERSAL"),

#     # Command Injection – unseen variants
#     ("GET /ping?x=127.0.0.1|whoami", "COMMAND_INJECTION"),
#     ("POST /run cmd=$(id)", "COMMAND_INJECTION"),
# ]
    test_cases = [
    # Standard GET requests with benign parameters
    ("GET /products?id=456&category=electronics", "SAFE"),
    ("GET /search?q=wireless+mouse", "SAFE"),
    ("GET /user/profile?user_id=789", "SAFE"),
    ("GET /api/data?start=2023-01-01&end=2023-12-31", "SAFE"),
    ("GET /images?file=logo.png", "SAFE"),
    ("GET /weather?city=Paris&units=metric", "SAFE"),
    ("GET /news?category=technology&page=3", "SAFE"),
    ("GET /contact?name=Jane%20Doe", "SAFE"),  # Safe URL-encoded space
    
    # Standard POST requests with benign form data
    ("POST /login username=alice&password=SecurePass123!", "SAFE"),
    ("POST /register email=user@example.com&username=bob&password=NewPass2024", "SAFE"),
    ("POST /comment post_id=101&text=Excellent+resource+for+learners", "SAFE"),
    ("POST /order items=notebook,pen&quantity=1,2&shipping=express", "SAFE"),
    ("POST /feedback rating=4&comments=Fast+delivery+and+great+service", "SAFE"),
    ("POST /upload filename=quarterly_report.pdf", "SAFE"),
    ("POST /settings theme=light&notifications=email", "SAFE"),
    
    # Additional realistic benign patterns
    ("GET /checkout?session=abc123xyz&cart_id=789", "SAFE"),
    ("GET /support/ticket?ref=TKT-2024-98765", "SAFE"),
    ("POST /payment method=credit_card&amount=49.99", "SAFE"),
    ("GET /blog/post?slug=modern-web-development-tips", "SAFE"),
    ("POST /newsletter subscribe=yes&email=news@example.org", "SAFE")
]
    
    print("\n" + "="*60)
    print("Single Request Classification")
    print("="*60)
    
    for request_text, expected_label in test_cases:
        result = classifier.classify(request_text)
        
        is_correct = "✓" if result["label"] == expected_label else "✗"
        
        print(f"\n{is_correct} Request: {request_text[:60]}")
        print(f"  Predicted: {result['label']} (confidence: {result['confidence']:.3f})")
        print(f"  Expected: {expected_label}")
        print(f"  Inference time: {result['inference_time_ms']} ms")
    
    # Test batch classification
    print("\n" + "="*60)
    print("Batch Classification")
    print("="*60)
    
    batch_texts = [req for req, _ in test_cases]
    batch_results = classifier.classify_batch(batch_texts)
    
    print(f"\nProcessed {len(batch_texts)} requests in batch")
    for i, (result, (_, expected)) in enumerate(zip(batch_results, test_cases)):
        is_correct = "✓" if result["label"] == expected else "✗"
        print(f"{is_correct} Request {i+1}: {result['label']} (conf: {result['confidence']:.3f})")