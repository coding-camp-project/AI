from fastapi import APIRouter
from app.controllers.chat_controller import send_chat_message, get_chat_history

router = APIRouter(prefix="/chat", tags=["Chat"])

router.post("/")(send_chat_message)
router.get("/history/{conversation_id}")(get_chat_history)
