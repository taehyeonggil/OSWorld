"""
OpenCUA VLM Server

This server provides a REST API for OpenCUA vision-language model inference.
It accepts chat completion requests with images and returns generated responses.

Usage:
    python -m mm_agents.llm_server.OpenCUA.opencua_server --model_path OpenCUA-7B --port 7908
"""

import os
import gc
import time
import base64
import io

from contextlib import asynccontextmanager
from typing import List, Literal, Union, Tuple, Optional
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
)

# Default model path and device
MODEL_DIR = os.environ.get('OPENCUA_MODEL_PATH', 'OpenCUA-7B')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global model variables
opencua_tokenizer = None
opencua_model = None
opencua_image_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends.
    """
    global opencua_tokenizer, opencua_model, opencua_image_processor
    
    logger.info(f"Loading OpenCUA model from {MODEL_DIR}...")
    opencua_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    opencua_model = AutoModel.from_pretrained(
        MODEL_DIR, 
        torch_dtype="auto", 
        device_map="auto", 
        trust_remote_code=True
    )
    opencua_image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    logger.info("Model loaded successfully!")
    
    yield
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    """A Pydantic model representing a model card."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ImageUrl(BaseModel):
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = Union[TextContent, ImageUrlContent]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion"]
    choices: List[ChatCompletionResponseChoice]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    model_card = ModelCard(id="opencua")
    return ModelList(data=[model_card])


def extract_images_from_messages(messages: List[ChatMessageInput]) -> List[Image.Image]:
    """
    Extract PIL Image objects from messages.
    
    Args:
        messages: List of chat messages that may contain base64-encoded images.
    
    Returns:
        List of PIL Image objects in order they appear in messages.
    """
    img_list = []
    
    for message in messages:
        content = message.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if image_url.startswith("data:image/png;base64,"):
                        base64_encoded_image = image_url.split("data:image/png;base64,")[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(io.BytesIO(image_data))
                        img_list.append(image)
                    elif image_url.startswith("data:image/jpeg;base64,"):
                        base64_encoded_image = image_url.split("data:image/jpeg;base64,")[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(io.BytesIO(image_data))
                        img_list.append(image)
    
    return img_list


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using OpenCUA model."""
    global opencua_tokenizer, opencua_model, opencua_image_processor
    
    if opencua_model is None or opencua_tokenizer is None or opencua_image_processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported yet")
    
    try:
        # Extract images from messages
        img_list = extract_images_from_messages(request.messages)
        
        # Apply chat template to get input_ids
        input_ids = opencua_tokenizer.apply_chat_template(
            request.messages, 
            tokenize=True, 
            add_generation_prompt=True
        )
        
        # Preprocess images
        if img_list:
            info = opencua_image_processor.preprocess(images=img_list)
            pixel_values = torch.tensor(info['pixel_values']).to(
                dtype=torch.bfloat16, 
                device=opencua_model.device
            )
            grid_thws = torch.tensor(info['image_grid_thw'])
        else:
            pixel_values = None
            grid_thws = None
        
        input_ids = torch.tensor([input_ids]).to(opencua_model.device)
        
        # Generate response
        with torch.no_grad():
            if pixel_values is not None and grid_thws is not None:
                generated_ids = opencua_model.generate(
                    input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.0,
                )
            else:
                generated_ids = opencua_model.generate(
                    input_ids,
                    max_new_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.0,
                )
        
        prompt_len = input_ids.shape[1]
        generated_ids = generated_ids[:, prompt_len:]
        
        response = opencua_tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Create response
        message = ChatMessageResponse(
            role="assistant",
            content=response,
        )
        
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        
        # Estimate token usage (rough approximation)
        prompt_tokens = len(input_ids[0])
        completion_tokens = len(generated_ids[0])
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        return ChatCompletionResponse(
            model=request.model,
            choices=[choice_data],
            object="chat.completion",
            usage=usage
        )
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenCUA VLM Server")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=MODEL_DIR,
        help="Path to the OpenCUA model directory"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7908,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    
    args = parser.parse_args()

    # There is no need to declare 'MODEL_DIR' as global here,
    # because this is the main module scope (not inside a function).
    MODEL_DIR = args.model_path

    logger.info(f"Starting OpenCUA VLM Server on {args.host}:{args.port}")
    logger.info(f"Model path: {MODEL_DIR}")
    
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
