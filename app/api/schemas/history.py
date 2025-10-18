from pydantic import BaseModel

class HistoryResponse(BaseModel):
    """A specific history in the users galaxy instance"""
    id: str
    name: str