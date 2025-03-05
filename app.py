# app.py
import os
import pandas as pd
from PIL import Image
from models import model, processor  # Import from models.py
from inputs import clothing_types, occasions, seasons, materials, compatibility_prompts  # Import from inputs.py
from utils import get_dominant_color, classify_image_clip  # Import from utils.py
from outfit_analyzer import OutfitCompatibilityAnalyzer  # Import from outfit_analyzer.py

# Load classified images and analyze
image_folder = os.path.join(os.getcwd(), "Images")
csv_file = os.path.join(os.getcwd(), "files")
results = []

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if os.path.isfile(image_path):
        clothing_type, category, occasion, season, material, dominant_color = classify_image_clip(
            image_path, processor, model, clothing_types, occasions, seasons, materials
        )
        if clothing_type:
            results.append({
                "image_path": image_path,
                "image_name": image_name,
                "Clothing_Type": clothing_type,
                "Category": category,
                "Occasion": occasion,
                "Season": season,
                "Material": material,
                "Dominant Color": dominant_color
            })

# Convert results to DataFrame
df = pd.DataFrame(results)
df.to_csv(csv_file + "/Classified.csv", index=False)
print("Classification complete. Results saved to Classified.csv")

# Initialize OutfitCompatibilityAnalyzer
analyzer = OutfitCompatibilityAnalyzer(df, processor, model, compatibility_prompts)

# User-selected occasion
user_selected_occasion = "Partywear"

# Retrieve the top and best matches
best_outfits = analyzer.find_best_matches(user_selected_occasion)

# Display results
if best_outfits:
    for idx, (top, best_matches) in enumerate(best_outfits, 1):
        print(f"\n--- Outfit {idx} ---")
        print(f"Top Image Information:")
        print(top)
        Image.open(top['image_path']).resize((256, 256)).show

        print("\nBest Matched Bottoms:")
        for bottom, score in best_matches:
            print(f"  Bottom Information:")
            print(bottom)
            print(f"   Match Score: {score:.2f}")
            Image.open(bottom['image_path']).resize((256, 256)).show()
else:
    print("No results to display.")

