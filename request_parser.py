"""
HTTP Request Parser

Parses and enriches HTTP requests for WAF classification.
"""

import sys
sys.path.append(".")

from fastapi import Request
from typing import Dict, Optional
from data.request_encoder import RequestEncoder


class RequestParser:
    """Parses HTTP requests into context-enriched text."""
    
    def __init__(self):
        self.encoder = RequestEncoder()
    
    async def parse_request(self, request: Request, full_path: str) -> Dict:
        """
        Parse FastAPI request into structured format.
        
        Args:
            request: FastAPI Request object
            full_path: Full URL path
        
        Returns:
            Dictionary with parsed request data
        """
        # Extract method
        method = request.method
        
        # Extract path
        path = f"/{full_path}" if full_path else "/"
        
        # Extract query string
        query = str(request.url.query) if request.url.query else ""
        
        # Extract body (if present)
        try:
            body = await request.body()
            body_text = body.decode("utf-8", errors="ignore") if body else ""
        except:
            body_text = ""
        
        # Extract headers (selected ones)
        headers = {
            "content-type": request.headers.get("content-type", ""),
            "user-agent": request.headers.get("user-agent", ""),
            "referer": request.headers.get("referer", "")
        }
        
        # Extract client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Generate context-enriched text
        enriched_text = self.encoder.encode(
            method=method,
            path=path,
            query=query,
            body=body_text,
            headers=headers
        )
        
        return {
            "method": method,
            "path": path,
            "query": query,
            "body": body_text[:500],  # Limit body length
            "client_ip": client_ip,
            "user_agent": headers.get("user-agent", ""),
            "content_type": headers.get("content-type", ""),
            "enriched_text": enriched_text
        }


# Singleton instance
request_parser = RequestParser()