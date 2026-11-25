from pydantic import BaseModel, Field

class TransactionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Exactly the 30 model feature names and finite numeric values")

class BatchRequest(BaseModel):
    transactions: list[dict[str, float]] = Field(..., min_length=1, max_length=500)
