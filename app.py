import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime
from PIL import Image
from IPython.display import display

# ----------------------------
# Setup Device
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# Load Stable Diffusion Model
# ----------------------------
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

print("✅ Model loaded successfully!")

# ----------------------------
# Image Generation Function
# ----------------------------
def generate_image(prompt):
    try:
        print("\n🎨 Generating image... Please wait...")

        result = pipe(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=30
        )

        image = result.images[0]

        # Save with timestamp
        filename = datetime.now().strftime("generated_%Y%m%d_%H%M%S.png")
        image.save(filename)

        print("✅ Image saved as:", filename)

        return image

    except Exception as e:
        print("❌ Error:", e)
        return None


# ----------------------------
# Continuous Generator Loop
# ----------------------------
print("\n==== AI Image Generator (Stable Diffusion) ====")
print("Type 'exit' or 'quit' to stop\n")

while True:

    prompt = input("\n📝 Enter your prompt:\n> ")

    if prompt.lower() in ["exit", "quit", "stop"]:
        print("\n👋 Stopping generator. Goodbye!")
        break

    if not prompt.strip():
        print("⚠️ Prompt cannot be empty!")
        continue

    img = generate_image(prompt)

    if img:
        display(img)
