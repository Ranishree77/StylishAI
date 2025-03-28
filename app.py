from flask import Flask, request, jsonify
import os
import pandas as pd
import torch
from PIL import Image
from models import model, processor
from inputs import clothing_types, occasions, seasons, materials, compatibility_prompts
from outfit_analyzer import OutfitCompatibilityAnalyzer
from utils import classify_image_clip

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
image_folder = os.path.join(os.getcwd(), "Images")
csv_file = os.path.join(os.getcwd(), "files")

def classify_and_analyze(images):
    results = []
    for image_data in images:
        image_path = os.path.join(image_folder, image_data['filename'])
        image = Image.open(image_path)
        image.save(image_path)

        clothing_type, category, occasion, season, material, dominant_color = classify_image_clip(
            image_path, processor, model, clothing_types, occasions, seasons, materials
        )

        results.append({
            "image_name": image_data['filename'],
            "Clothing_Type": clothing_type,
            "Category": category,
            "Occasion": occasion,
            "Season": season,
            "Material": material,
            "Dominant Color": dominant_color
        })
    
    df = pd.read_csv(os.path.join(csv_file, "Classified.csv"))
    image_features = torch.load("image_features.pt") if os.path.exists("image_features.pt") else {}
    analyzer = OutfitCompatibilityAnalyzer(df, processor, model, compatibility_prompts, image_features, device)
    best_outfits = analyzer.find_best_matches(results[0]['Occasion'] if results else 'Partywear')
    
    outfit_combinations = {}
    if best_outfits:
        for item, matches in best_outfits:
            if matches is None:
                # Standalone dress (no bottoms needed)
                outfit_combinations[item] = []
            else:
                # Tops with compatible bottoms
                outfit_combinations[item] = [bottom for bottom, score in matches]
    else:
        return jsonify({"error": "No suitable outfits found"}), 404
    
    return jsonify({
        "classification": results,
        "outfit_combinations": outfit_combinations
    })

@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        image_urls = request.get_json()
        if not isinstance(image_urls, list):
            return jsonify({"error": "Input must be a list of image URLs"}), 400
        
        if not image_urls:
            return jsonify({"error": "No image URLs provided"}), 400
        
        result = classify_and_analyze(image_urls)
        if "error" in result:
            return jsonify({"error": result["error"]}), result.get("status", 400)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
