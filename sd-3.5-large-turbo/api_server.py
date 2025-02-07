from fastapi import FastAPI
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
import io
import base64

MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"

app = FastAPI()

# Initialize the model
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

class Query(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(query: Query):
    image = pipe(query.prompt).images[0]
    
    # Convert PIL Image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image": img_str}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)