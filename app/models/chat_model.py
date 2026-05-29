from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class Chat(BaseModel):
    user_id: Optional[str] = None
    conversation_id: str
    role: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
