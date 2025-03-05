# utils.py
from PIL import Image
from collections import Counter
import torch
from inputs import tops, bottoms  # Import tops and bottoms from inputs.py

def get_dominant_color(image_path, resize_size=(150, 150)):
    """
    Extracts the single most dominant color from an image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(resize_size)
        pixels = list(img.getdata())
        color_counts = Counter(pixels)
        dominant_color = color_counts.most_common(1)[0][0]
        return dominant_color
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def classify_image_clip(image_path, processor, model, clothing_types, occasions, seasons, materials):
    """
    Classifies an image using the CLIP model.
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
        category = "Top" if clothing_type in tops else "Bottom" if clothing_type in bottoms else "Other"
        dominant_color = get_dominant_color(image_path)

        return clothing_type, category, occasion, season, material, dominant_color
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None, None, None, None
