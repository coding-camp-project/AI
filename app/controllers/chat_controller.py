from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from bson import ObjectId

import app.services.chat_service as chat_service

# TODO: ntar migrasi ke mongodb kalo be udh siap
fake_db = []

async def get_db():
    yield fake_db

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

async def send_chat_message(request: ChatRequest, db=Depends(get_db)):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")
        
    try:
        result = await chat_service.handle_chat_message(
            db=db,
            message=request.message.strip(),
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        return {
            "success": True,
            "reply": result["reply"],
            "conversation_id": result["conversation_id"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_chat_history(conversation_id: str, db=Depends(get_db)):
    if not conversation_id or not conversation_id.strip():
        raise HTTPException(status_code=400, detail="Conversation ID is required.")
        
    try:
        history = await chat_service.get_conversation_history(db, conversation_id)
        
        for msg in history:
            if "_id" in msg:
                msg["_id"] = str(msg["_id"])
                
        return {
            "success": True,
            "data": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
