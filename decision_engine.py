"""
Confidence-Aware Decision Engine

Makes allow/block decisions based on model predictions and confidence scores.
"""

from typing import Dict
from config import WAFConfig


class DecisionEngine:
    """
    Makes WAF decisions based on prediction confidence.
    
    Decision zones:
    - High confidence attack (>= 0.85): BLOCK
    - Medium confidence attack (0.60-0.85): ALLOW_WITH_LOG
    - Low confidence (<0.60): ALLOW_WITH_FLAG
    - SAFE prediction: ALLOW
    """
    
    def __init__(self, config: WAFConfig):
        self.config = config
        self.high_threshold = config.high_confidence_threshold
        self.medium_threshold = config.medium_confidence_threshold
    
    def decide(self, prediction: Dict) -> Dict:
        """
        Make allow/block decision based on prediction.
        
        Args:
            prediction: Dictionary with 'label' and 'confidence' from classifier
        
        Returns:
            Dictionary with 'action' and 'reason'
        """
        label = prediction["label"]
        confidence = prediction["confidence"]
        
        # Safe prediction - always allow
        if label == "SAFE":
            return {
                "action": "ALLOW",
                "reason": "Benign traffic detected",
                "confidence_level": "high" if confidence >= self.high_threshold else "medium"
            }
        
        # Attack detected - decision based on confidence
        if confidence >= self.high_threshold:
            # High confidence attack - block
            if self.config.block_on_high_confidence:
                return {
                    "action": "BLOCK",
                    "reason": f"High confidence {label} attack detected",
                    "confidence_level": "high",
                    "attack_type": label
                }
            else:
                return {
                    "action": "ALLOW_WITH_LOG",
                    "reason": f"High confidence {label} attack (blocking disabled)",
                    "confidence_level": "high",
                    "attack_type": label
                }
        
        elif confidence >= self.medium_threshold:
            # Medium confidence attack - log but allow with warning
            return {
                "action": "ALLOW_WITH_LOG",
                "reason": f"Medium confidence {label} attack detected",
                "confidence_level": "medium",
                "attack_type": label,
                "warning": "Possible attack - monitoring recommended"
            }
        
        else:
            # Low confidence - allow with flag for manual review
            return {
                "action": "ALLOW_WITH_FLAG",
                "reason": f"Low confidence {label} detection",
                "confidence_level": "low",
                "attack_type": label,
                "flag": "manual_review_recommended"
            }
    
    def should_block(self, decision: Dict) -> bool:
        """
        Check if request should be blocked.
        
        Args:
            decision: Decision dictionary from decide()
        
        Returns:
            True if request should be blocked
        """
        return decision.get("action") == "BLOCK"
    
    def get_threat_level(self, decision: Dict) -> str:
        """
        Get threat level from decision.
        
        Args:
            decision: Decision dictionary
        
        Returns:
            Threat level: "critical", "high", "medium", "low", or "none"
        """
        action = decision.get("action")
        confidence_level = decision.get("confidence_level")
        
        if action == "BLOCK":
            return "critical"
        elif action == "ALLOW_WITH_LOG" and confidence_level == "high":
            return "high"
        elif action == "ALLOW_WITH_LOG":
            return "medium"
        elif action == "ALLOW_WITH_FLAG":
            return "low"
        else:
            return "none"


if __name__ == "__main__":
    # Test decision engine
    config = WAFConfig()
    engine = DecisionEngine(config)
    
    test_predictions = [
        {"label": "SAFE", "confidence": 0.95},
        {"label": "SQLI", "confidence": 0.92},
        {"label": "XSS", "confidence": 0.75},
        {"label": "PATH_TRAVERSAL", "confidence": 0.55},
        {"label": "COMMAND_INJECTION", "confidence": 0.30},
    ]
    
    print("Decision Engine Test")
    print("="*60)
    
    for pred in test_predictions:
        decision = engine.decide(pred)
        threat = engine.get_threat_level(decision)
        
        print(f"\nPrediction: {pred['label']} (confidence: {pred['confidence']:.2f})")
        print(f"  Action: {decision['action']}")
        print(f"  Reason: {decision['reason']}")
        print(f"  Threat Level: {threat}")
        if "warning" in decision:
            print(f"  Warning: {decision['warning']}")
        if "flag" in decision:
            print(f"  Flag: {decision['flag']}")