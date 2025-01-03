from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Specify the directory to save the model
model_dir = "./clipseg_model"

# Load the model and processor from Hugging Face
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Save the model and processor locally
processor.save_pretrained(model_dir)
model.save_pretrained(model_dir)

print(f"Model and processor saved successfully in '{model_dir}'!")
