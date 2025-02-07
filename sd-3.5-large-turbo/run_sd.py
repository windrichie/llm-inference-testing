import torch
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"

# Initialize the model
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")

# Set a fixed seed for reproducibility
# generator = torch.Generator("cuda").manual_seed(40)

# Generate an image
prompt = "A serene landscape with mountains and a lake at sunset, highly detailed, professional photography, 8k resolution"
images = pipe(
    prompt = [prompt] * 8,
    width=512,
    height=512,
    num_inference_steps=4,
    guidance_scale=0.0,
    # generator=generator
).images

# Save the image
i = 0
for image in images:
    image.save(f"output/generated_image{i}.png")
    i += 1
print("Images generated.")