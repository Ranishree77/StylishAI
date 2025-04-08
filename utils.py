# utils.py
from PIL import Image
from collections import Counter
import torch
from inputs import tops, bottoms, dresses, footwear  # Import tops, bottoms, footwear from inputs.py
import rembg
from rembg import remove
import numpy as np
from sklearn.cluster import KMeans
import io
from PIL import Image
import matplotlib.pyplot as plt


def remove_background(image_path):
    """Removes background from the input image using rembg."""
    with open(image_path, "rb") as f:
        input_image = f.read()
    output_image = remove(input_image)  # Ensure remove() returns bytes

    # Convert output_image to bytes if it's not already
    if isinstance(output_image, Image.Image):
        with io.BytesIO() as output_buffer:
            output_image.save(output_buffer, format="PNG")
            output_image = output_buffer.getvalue()
    image = Image.open(io.BytesIO(output_image))  # Convert to PIL image
    
    return image

def get_dominant_color_kmeans(image, resize_size=(300, 300), k=3):
    """Extracts the dominant color using K-Means clustering, ignoring shadows and transparency."""
    try:
        img = image.convert("RGBA")  # Ensure image has alpha channel
        img = img.resize(resize_size)
        pixels = np.array(img.getdata())

        # Keep only non-transparent pixels
        pixels = np.array([rgb[:3] for rgb in pixels if rgb[3] > 0])

        # Remove dark/shadow pixels
        pixels = np.array([rgb for rgb in pixels if sum(rgb) > 150])  # R+G+B > 150

        if len(pixels) == 0:
            print("No valid color pixels found!")
            return (0, 0, 0)

        # Apply K-Means to find dominant colors
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return tuple(map(int, dominant_color))  # Convert to integer RGB
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def classify_image_clip(image_path, processor, model, clothing_types, occasions, seasons, materials):
    """
    Classifies an image using the FashionCLIP model.
    """
    try:
        image = Image.open(image_path).convert("RGB")

        # Classify clothing type
        inputs = processor(text=clothing_types, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
        clothing_type = clothing_types[probs.argmax().item()]

        # Classify occasion
        inputs = processor(text=occasions, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
        occasion = occasions[probs.argmax().item()]

        # Classify season
        inputs = processor(text=seasons, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
        season = seasons[probs.argmax().item()]

        # Detect material
        inputs = processor(text=materials, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
        material = materials[probs.argmax().item()]

        # Map clothing type to broader category
        category = "Top" if clothing_type in tops else "Bottom" if clothing_type in bottoms else "Dress" if clothing_type in dresses else "Footwear" if clothing_type in footwear else "Other"  # Add dresses
        
        #extracting dominating color form image
        # Remove background first
        background_removed_image = remove_background(image_path)

        # Extract dominant color from the background-removed image
        dominant_color = get_dominant_color_kmeans(background_removed_image)
        image = Image.open(image_path).convert("RGB")

        return clothing_type, category, occasion, season, material, dominant_color
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None, None, None, None