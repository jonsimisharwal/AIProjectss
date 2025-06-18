from pydantic import BaseModel

class UserData(BaseModel):
    pages_visited: int
    time_spent: float
    days_since_signup: int
    is_logged_in: bool

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
