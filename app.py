"""
Enhanced Transformer-Based AI WAF
Production-ready WAF with path filtering and clean logging.
"""

import sys
sys.path.append(".")

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from datetime import datetime
import csv
import threading
import time
import uvicorn
from contextlib import asynccontextmanager

# WAF components
from waf.config import WAFConfig, waf_config
from waf.request_parser import RequestParser
from waf.decision_engine import DecisionEngine
from waf.logger import WAFLogger

# Model inference
from models.inference import WAFClassifier


# Global instances
classifier = None
parser = None
engine = None
logger = None


# -----------------------------
# STARTUP INITIALIZATION
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, parser, engine, logger

    print("=" * 60)
    print("Initializing Transformer-Based WAF")
    print("=" * 60)

    parser = RequestParser()
    engine = DecisionEngine(waf_config)

    logger = WAFLogger(
        log_file=waf_config.log_file,
        enabled=waf_config.enable_logging
    )
    # Start dataset processing in background
    threading.Thread(target=process_dataset, daemon=True).start()

    try:
        classifier = WAFClassifier(model_path=waf_config.model_path)
        print("✓ WAF initialized successfully")
    except FileNotFoundError as e:
        print(f"✗ Model not found: {e}")
        classifier = None

    print("=" * 60)
    yield


app = FastAPI(
    title="Transformer-Based AI WAF",
    version="2.0.0",
    lifespan=lifespan
)

app.mount("/ui", StaticFiles(directory="frontend"), name="ui")


# -----------------------------
# HELPER: IGNORE SYSTEM PATHS
# -----------------------------
IGNORED_PATH_PREFIXES = [
    "ui/",
    "docs",
    "openapi.json",
    "favicon.ico",
    "health",
    "stats",
    "logs",
    "hybridaction"   # 👈 removes your unwanted logs
]


def should_ignore(full_path: str) -> bool:
    return any(full_path.startswith(prefix) for prefix in IGNORED_PATH_PREFIXES)


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
async def root():
    return RedirectResponse(url="/ui/index.html")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if classifier else "degraded",
        "model_loaded": classifier is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_statistics():
    if logger is None:
        return {"error": "Logger not initialized"}

    stats = logger.get_stats()
    stats["timestamp"] = datetime.now().isoformat()
    return stats


@app.get("/logs/recent")
async def get_recent_logs(limit: int = 50):
    if logger is None:
        return {"error": "Logger not initialized"}

    limit = min(limit, 500)
    logs = logger.read_recent_logs(n=limit)

    return {
        "count": len(logs),
        "logs": logs
    }


# -----------------------------
# MAIN WAF INSPECTION ROUTE
# -----------------------------
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def waf_inspection(full_path: str, request: Request):

    # Ignore internal/system paths
    if should_ignore(full_path):
        return JSONResponse(
            status_code=200,
            content={"status": "IGNORED"}
        )

    start_time = time.time()

    if classifier is None:
        return JSONResponse(
            status_code=503,
            content={"status": "ERROR", "message": "Model not loaded"}
        )

    try:
        # Parse request
        request_data = await parser.parse_request(request, full_path)

        # Classify
        prediction = classifier.classify(request_data["enriched_text"])

        # Decide
        decision = engine.decide(prediction)
        decision["threat_level"] = engine.get_threat_level(decision)

        total_latency = (time.time() - start_time) * 1000

        # Log only real inspected traffic
        logger.log_request(
            request_data=request_data,
            prediction=prediction,
            decision=decision,
            latency_ms=total_latency
        )

        # Block if needed
        if engine.should_block(decision):
            return JSONResponse(
                status_code=403,
                content={
                    "status": "BLOCKED",
                    "reason": decision["reason"],
                    "attack_type": decision.get("attack_type"),
                    "confidence": prediction["confidence"],
                    "threat_level": decision["threat_level"]
                }
            )

        # Otherwise allow
        return JSONResponse(
            status_code=200,
            content={
                "status": decision["action"],
                "analysis": {
                    "prediction": prediction["label"],
                    "confidence": round(prediction["confidence"], 3),
                    "threat_level": decision["threat_level"]
                },
                "latency_ms": round(total_latency, 2)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "message": str(e)}
        )






def process_dataset(file_path="requests.csv"):
    global classifier, parser, engine, logger

    if classifier is None:
        print("Model not loaded. Skipping dataset processing.")
        return

    print("📊 Processing dataset...")

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            try:
                # Assume dataset has column like: request or text
                request_text = row.get("request") or row.get("text") or ""

                if not request_text:
                    continue

                # Simulate parsed request
                request_data = {
                    "method": "GET",
                    "path": "/dataset",
                    "query": request_text,
                    "body": "",
                    "enriched_text": request_text
                }

                # 🔥 MODEL ONLY (no rules)
                prediction = classifier.classify(request_text)

                # Decision
                decision = engine.decide(prediction)
                decision["threat_level"] = engine.get_threat_level(decision)

                # Log it
                logger.log_request(
                    request_data=request_data,
                    prediction=prediction,
                    decision=decision,
                    latency_ms=0
                )

            except Exception as e:
                print("Dataset row error:", e)

    print("✅ Dataset processing completed")
# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning" , access_log=False)
