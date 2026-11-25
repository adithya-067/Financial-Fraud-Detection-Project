"""FastAPI inference API."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .schemas import TransactionRequest, BatchRequest
from src.predict import load_model, predict_one
from src.config import MODEL_DIR, REPORT_DIR, PLOT_DIR, DATA_PATH

app=FastAPI(title="Intelligent Financial Fraud Detection API",version="1.0.0",description="Academic prototype for credit card fraud risk scoring.")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

def artifact():
    try:return load_model()
    except FileNotFoundError as exc: raise HTTPException(status_code=503,detail=str(exc))

@app.get("/")
def root(): return {"name":app.title,"status":"ok","docs":"/docs"}

@app.get("/health")
def health(): return {"status":"ok","model_available":(MODEL_DIR/"fraud_pipeline.joblib").exists()}

@app.get("/setup-status")
def setup_status():
    """Return project readiness without exposing local file paths."""
    return {"api_available": True,
            "dataset_available": DATA_PATH.is_file(),
            "model_available": (MODEL_DIR/"fraud_pipeline.joblib").is_file(),
            "metrics_available": (REPORT_DIR/"test_metrics.json").is_file()}

@app.get("/model-info")
def model_info():
    model=artifact(); return {k:v for k,v in model.get("metadata",{}).items() if k!="demo_samples"}

@app.get("/demo-samples")
def demo_samples():
    model=artifact(); samples=model.get("metadata",{}).get("demo_samples")
    if not samples: raise HTTPException(404,"Dataset demonstration samples are unavailable.")
    return samples

@app.get("/metrics")
def metrics():
    p=REPORT_DIR/"test_metrics.json"
    if not p.exists(): raise HTTPException(404,"Evaluation metrics are unavailable until training completes.")
    import json
    return json.loads(p.read_text(encoding="utf-8"))

@app.get("/visualizations")
def visualizations():
    return {"items":[{"name":p.stem,"url":f"/visualizations/{p.name}"} for p in sorted(PLOT_DIR.glob("*.png"))]}

@app.get("/visualizations/{name}")
def visualization(name:str):
    from fastapi.responses import FileResponse
    p=PLOT_DIR/name
    if Path(name).name!=name or not p.is_file() or p.suffix.lower()!=".png": raise HTTPException(404,"Visualization not found")
    return FileResponse(p,media_type="image/png")

@app.post("/predict")
def predict(request:TransactionRequest):
    model=artifact()
    try:return predict_one(request.features,model)
    except ValueError as exc: raise HTTPException(422,str(exc))

@app.post("/predict/batch")
def predict_batch(request:BatchRequest):
    model=artifact(); results=[]
    for i,row in enumerate(request.transactions):
        try: results.append({"row":i+1,**predict_one(row,model)})
        except ValueError as exc: raise HTTPException(422,f"Transaction {i+1}: {exc}")
    return {"predictions":results,"count":len(results)}
