"""
Structured JSON Logger for WAF

Logs all WAF decisions in structured JSON format.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import uuid


class WAFLogger:
    """Structured logger for WAF events."""
    
    def __init__(self, log_file: str = "logs/waf.jsonl", enabled: bool = True):
        self.log_file = Path(log_file)
        self.enabled = enabled
        
        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "allowed_requests": 0,
            "flagged_requests": 0,
            "total_latency_ms": 0.0
        }
    
    def log_request(self,
                   request_data: Dict,
                   prediction: Dict,
                   decision: Dict,
                   latency_ms: float,
                   request_id: Optional[str] = None):
        """
        Log a WAF request decision.
        
        Args:
            request_data: Parsed request data
            prediction: Model prediction
            decision: Decision engine output
            latency_ms: Total processing latency
            request_id: Optional request ID
        """
        if not self.enabled:
            return
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "client_ip": request_data.get("client_ip", "unknown"),
            "method": request_data.get("method"),
            "path": request_data.get("path"),
            "query": request_data.get("query", ""),
            "prediction": {
                "label": prediction.get("label"),
                "confidence": round(prediction.get("confidence", 0), 4),
                "inference_time_ms": round(prediction.get("inference_time_ms", 0), 2)
            },
            "decision": {
                "action": decision.get("action"),
                "reason": decision.get("reason"),
                "confidence_level": decision.get("confidence_level"),
                "threat_level": decision.get("threat_level", "unknown")
            },
            "latency_ms": round(latency_ms, 2),
            "user_agent": request_data.get("user_agent", "")[:200]  # Truncate
        }
        
        # Add attack type if present
        if "attack_type" in decision:
            log_entry["decision"]["attack_type"] = decision["attack_type"]
        
        # Add warning/flag if present
        if "warning" in decision:
            log_entry["decision"]["warning"] = decision["warning"]
        if "flag" in decision:
            log_entry["decision"]["flag"] = decision["flag"]
        
        # Write to file (append mode)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")
        
        # Update statistics
        self._update_stats(decision, latency_ms)
    
    def _update_stats(self, decision: Dict, latency_ms: float):
        """Update internal statistics."""
        self.stats["total_requests"] += 1
        self.stats["total_latency_ms"] += latency_ms
        
        action = decision.get("action")
        if action == "BLOCK":
            self.stats["blocked_requests"] += 1
        elif action in ["ALLOW", "ALLOW_WITH_LOG"]:
            self.stats["allowed_requests"] += 1
        elif action == "ALLOW_WITH_FLAG":
            self.stats["flagged_requests"] += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats["total_requests"] > 0:
            stats["avg_latency_ms"] = round(
                stats["total_latency_ms"] / stats["total_requests"], 2
            )
            stats["block_rate"] = round(
                stats["blocked_requests"] / stats["total_requests"] * 100, 2
            )
        else:
            stats["avg_latency_ms"] = 0
            stats["block_rate"] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "allowed_requests": 0,
            "flagged_requests": 0,
            "total_latency_ms": 0.0
        }
    
    def read_recent_logs(self, n: int = 100) -> list:
        """
        Read the most recent N log entries.
        
        Args:
            n: Number of entries to read
        
        Returns:
            List of log entry dictionaries
        """
        if not self.log_file.exists():
            return []
        
        entries = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Get last N lines
                for line in lines[-n:]:
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        return entries


# Singleton instance
waf_logger = WAFLogger()


if __name__ == "__main__":
    # Test logger
    logger = WAFLogger("logs/test_waf.jsonl")
    
    # Sample data
    request_data = {
        "client_ip": "192.168.1.100",
        "method": "GET",
        "path": "/login",
        "query": "user=admin' OR '1'='1",
        "user_agent": "Mozilla/5.0"
    }
    
    prediction = {
        "label": "SQLI",
        "confidence": 0.94,
        "inference_time_ms": 42.5
    }
    
    decision = {
        "action": "BLOCK",
        "reason": "High confidence SQLI attack detected",
        "confidence_level": "high",
        "threat_level": "critical",
        "attack_type": "SQLI"
    }
    
    # Log the request
    logger.log_request(request_data, prediction, decision, latency_ms=45.2)
    
    print("Logged sample request")
    print("\nStats:", json.dumps(logger.get_stats(), indent=2))
    
    # Read recent logs
    recent = logger.read_recent_logs(n=5)
    print(f"\nRecent logs ({len(recent)} entries):")
    for entry in recent:
        print(json.dumps(entry, indent=2))