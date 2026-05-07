"""
Context-Enriched HTTP Request Encoder

This module transforms raw HTTP requests into semantically meaningful text
representations that are optimized for transformer-based attack detection.

Key innovations:
1. Attack keyword extraction (SQL, XSS, path traversal, command injection patterns)
2. Special character frequency analysis
3. URL encoding normalization
4. Payload structure analysis
"""

import re
from typing import Dict, List, Optional
from urllib.parse import unquote_plus


class RequestEncoder:
    """Encodes HTTP requests into context-enriched text for ML classification."""
    
    # Attack pattern dictionaries
    SQL_KEYWORDS = [
        'UNION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
        'ALTER', 'EXEC', 'EXECUTE', 'DECLARE', 'CAST', 'CONVERT',
        'SLEEP', 'BENCHMARK', 'WAITFOR', 'DELAY'
    ]
    
    SQL_OPERATORS = ["'", '"', '--', '/*', '*/', ';', '=', 'OR', 'AND']
    
    XSS_KEYWORDS = [
        'script', 'onerror', 'onload', 'onclick', 'alert', 'prompt',
        'confirm', 'eval', 'document', 'cookie', 'iframe', 'svg',
        'img', 'body', 'input', 'javascript:', 'vbscript:'
    ]
    
    PATH_TRAVERSAL_PATTERNS = ['../', '..\\ ', '%2e%2e', '%252e', 'etc/passwd', 'boot.ini', 'windows/system32']
    
    COMMAND_INJECTION_KEYWORDS = [
        'ls', 'cat', 'whoami', 'id', 'uname', 'netstat', 'ifconfig',
        'wget', 'curl', 'chmod', 'rm', 'sh', 'bash', 'cmd', 'powershell',
        'system', 'exec', 'passthru', 'shell_exec'
    ]
    
    COMMAND_OPERATORS = ['|', '&', ';', '`', '$', '(', ')', '{', '}']
    
    def __init__(self):
        self.sql_pattern = re.compile(r'\b(' + '|'.join(self.SQL_KEYWORDS) + r')\b', re.IGNORECASE)
        self.xss_pattern = re.compile(r'<[^>]*>|' + '|'.join(re.escape(k) for k in self.XSS_KEYWORDS), re.IGNORECASE)
        self.path_pattern = re.compile('|'.join(re.escape(p) for p in self.PATH_TRAVERSAL_PATTERNS), re.IGNORECASE)
        self.cmd_pattern = re.compile(r'\b(' + '|'.join(self.COMMAND_INJECTION_KEYWORDS) + r')\b', re.IGNORECASE)
    
    def encode(self, 
               method: str,
               path: str,
               query: str = "",
               body: str = "",
               headers: Optional[Dict[str, str]] = None) -> str:
        """
        Encode HTTP request into context-enriched text.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: URL path
            query: Query string
            body: Request body
            headers: Request headers (optional)
        
        Returns:
            Context-enriched text representation
        """
        # Normalize URL encoding
        path = unquote_plus(path) if path else ""
        query = unquote_plus(query) if query else ""
        body = unquote_plus(body) if body else ""
        
        # Combine all payload parts
        full_payload = f"{path} {query} {body}"
        
        # Extract attack indicators
        attack_indicators = self._extract_attack_indicators(full_payload)
        
        # Analyze special characters
        special_chars = self._analyze_special_chars(full_payload)
        
        # Calculate payload metrics
        payload_length = len(full_payload)
        
        # Build context-enriched representation
        parts = [
            f"ATTACK_INDICATORS: {', '.join(attack_indicators) if attack_indicators else 'none'}",
            f"SPECIAL_CHARS: {special_chars}",
            f"METHOD: {method.upper()}",
            f"PATH: {path}",
        ]
        
        if query:
            parts.append(f"QUERY: {query}")
        
        if body:
            parts.append(f"BODY: {body[:500]}")  # Limit body length
        
        # Add relevant headers
        if headers:
            content_type = headers.get('content-type', headers.get('Content-Type', ''))
            if content_type:
                parts.append(f"CONTENT_TYPE: {content_type}")
        
        parts.append(f"PAYLOAD_LENGTH: {payload_length} chars")
        
        return "\n".join(parts)
    
    def _extract_attack_indicators(self, payload: str) -> List[str]:
        """Extract attack-related keywords and patterns."""
        indicators = []
        
        # SQL injection indicators
        sql_matches = self.sql_pattern.findall(payload)
        if sql_matches:
            indicators.extend([f"SQL:{m}" for m in sql_matches[:5]])  # Limit to 5
        
        # Check for SQL operators
        for op in self.SQL_OPERATORS:
            if op in payload:
                indicators.append(f"SQL_OP:{op}")
        
        # XSS indicators
        if self.xss_pattern.search(payload):
            indicators.append("XSS_PATTERN")
            if '<script' in payload.lower():
                indicators.append("XSS:script_tag")
            if 'onerror' in payload.lower() or 'onload' in payload.lower():
                indicators.append("XSS:event_handler")
        
        # Path traversal indicators
        if self.path_pattern.search(payload):
            indicators.append("PATH_TRAVERSAL")
            if '../' in payload or '..\\' in payload:
                indicators.append("TRAVERSAL:dotdot")
        
        # Command injection indicators
        cmd_matches = self.cmd_pattern.findall(payload)
        if cmd_matches:
            indicators.extend([f"CMD:{m}" for m in cmd_matches[:5]])
        
        for op in self.COMMAND_OPERATORS:
            if op in payload:
                indicators.append(f"CMD_OP:{op}")
        
        return indicators[:15]  # Limit total indicators
    
    def _analyze_special_chars(self, payload: str) -> str:
        """Analyze frequency of special characters."""
        special_char_counts = {}
        
        chars_to_check = ["'", '"', '-', '=', '<', '>', '(', ')', ';', '&', '|', '%']
        
        for char in chars_to_check:
            count = payload.count(char)
            if count > 0:
                special_char_counts[char] = count
        
        if not special_char_counts:
            return "none"
        
        # Format as "char:count, char:count"
        return ", ".join([f"{char}:{count}" for char, count in sorted(special_char_counts.items())])
    
    def encode_simple(self, text: str) -> str:
        """
        Encode a simple text payload (for backward compatibility with existing dataset).
        
        Args:
            text: Raw request text (e.g., "GET /login?user=admin' OR '1'='1")
        
        Returns:
            Context-enriched text representation
        """
        # Parse simple format: "METHOD path query"
        parts = text.strip().split(None, 1)
        
        if len(parts) == 0:
            method = "GET"
            rest = ""
        elif len(parts) == 1:
            method = parts[0]
            rest = ""
        else:
            method = parts[0]
            rest = parts[1]
        
        # Split path and query/body
        if '?' in rest:
            path, query = rest.split('?', 1)
        elif ' ' in rest:
            # Could be POST with body
            path_parts = rest.split(None, 1)
            path = path_parts[0] if len(path_parts) > 0 else "/"
            query = ""
            body = path_parts[1] if len(path_parts) > 1 else ""
            return self.encode(method, path, query, body)
        else:
            path = rest if rest else "/"
            query = ""
        
        return self.encode(method, path, query)


# Singleton instance for easy import
encoder = RequestEncoder()


if __name__ == "__main__":
    # Test cases
    encoder = RequestEncoder()
    
    # Test 1: Benign request
    print("=== Benign Request ===")
    print(encoder.encode("GET", "/products", "id=123"))
    print()
    
    # Test 2: SQL Injection
    print("=== SQL Injection ===")
    print(encoder.encode("GET", "/login", "user=admin' OR '1'='1"))
    print()
    
    # Test 3: XSS
    print("=== XSS Attack ===")
    print(encoder.encode("GET", "/search", "q=<script>alert(1)</script>"))
    print()
    
    # Test 4: Path Traversal
    print("=== Path Traversal ===")
    print(encoder.encode("GET", "/download", "file=../../etc/passwd"))
    print()
    
    # Test 5: Command Injection
    print("=== Command Injection ===")
    print(encoder.encode("GET", "/ping", "host=8.8.8.8 && cat /etc/passwd"))
    print()
    
    # Test 6: Simple format (backward compatibility)
    print("=== Simple Format ===")
    print(encoder.encode_simple("GET /login?user=admin' OR 1=1 --"))