from fastapi import HTTPException
from backend.main import root, health, setup_status, predict, predict_batch
from backend.schemas import TransactionRequest, BatchRequest

def test_health():
    result=health()
    assert result["status"]=="ok" and "model_available" in result

def test_root():
    assert root()["docs"]=="/docs"

def test_setup_status_is_safe_and_reports_readiness():
    result=setup_status()
    assert result["api_available"] is True
    assert isinstance(result["dataset_available"], bool)
    assert isinstance(result["model_available"], bool)
    assert not any("path" in key.lower() for key in result)

def test_missing_model_behaviour():
    if not health()["model_available"]:
        row={k:0 for k in ["Time",*[f"V{i}" for i in range(1,29)],"Amount"]}
        try: predict(TransactionRequest(features=row))
        except HTTPException as exc: assert exc.status_code==503
        else: raise AssertionError("Missing model did not produce HTTP 503")
        try: predict_batch(BatchRequest(transactions=[row]))
        except HTTPException as exc: assert exc.status_code==503
        else: raise AssertionError("Missing model did not produce HTTP 503")
