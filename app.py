from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import os

# Load the model and processor from the local directory
processor = CLIPSegProcessor.from_pretrained("./clipseg_model")
model = CLIPSegForImageSegmentation.from_pretrained("./clipseg_model")
print("Model and processor loaded successfully!")

def segment_image(image_path, prompt):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Get segmentation logits
    with torch.no_grad():
        outputs = model(**inputs)
    
    segmentation_logits = outputs.logits
    segmentation_mask = torch.sigmoid(segmentation_logits).cpu().numpy()[0, :, :]  # Extract 2D mask

    # Apply a threshold to create a binary mask
    threshold = 0.5
    binary_mask = (segmentation_mask > threshold).astype(np.uint8)

    # Highlight the segmented area
    mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(image.size)
    mask = np.array(mask_image)
    highlighted_image = np.array(image)
    highlighted_image[mask == 0] = [0, 0, 0]  # Black out unsegmented areas
    
    return image, Image.fromarray(highlighted_image)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image and text prompt
        image_file = request.files["image"]
        prompt = request.form["prompt"]

        if image_file and prompt:
            # Save the uploaded image locally
            image_path = os.path.join("static", image_file.filename)
            image_file.save(image_path)

            # Perform segmentation
            original_image, segmented_image = segment_image(image_path, prompt)

            # Save results
            original_image_path = os.path.join("static", "original_image.jpg")
            segmented_image_path = os.path.join("static", "segmented_image.jpg")
            original_image.save(original_image_path)
            segmented_image.save(segmented_image_path)

            return render_template("index.html", original_image="static/original_image.jpg", segmented_image="static/segmented_image.jpg")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
