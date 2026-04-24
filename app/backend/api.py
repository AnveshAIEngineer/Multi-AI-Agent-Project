from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import List
from app.core.ai_agent import get_response_from_ai_agents
from app.config.settings import settings
from app.common.logger import get_logger
from app.common.custom_execption import CustomException

logger = get_logger(__name__)

app = FastAPI(title="MULTI AI AGENT")

class RequestState(BaseModel):
    model_name:str
    system_prompt:str
    messages:List[str]
    allow_search: bool

@app.post("/chat")
async def chat_endpoint(request:RequestState):
    logger.info(f"Recieved request for the model : {request.model_name}")

    if request.model_name not in settings.ALLOWED_MODEL_NAMES:
        logger.warning("Invalid model name")
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    try:
        response = get_response_from_ai_agents(
            request.model_name,
            request.messages,
            request.allow_search,
            request.system_prompt
        )
    
        logger.info(f"Sucessfully got Response from AI Agent {request.model_name}")
        return {"response" : response}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error("Some Error occured during response generation")
        raise HTTPException(
            status_code=500, 
            detail = str(CustomException("Failed to get AI response",error_detail=e))
            )
