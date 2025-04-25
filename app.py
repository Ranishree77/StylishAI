from flask import Flask, request, jsonify
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import logging
import os
import torch
from models import model, processor
from models import model_fc, processor_fc
from inputs import clothing_types, occasions, seasons, materials, compatibility_prompts
from outfit_analyzer import OutfitCompatibilityAnalyzer  # Assuming this class is defined in outfit_analyzer.py
from utils import classify_image_clip
import traceback
from typing import List, Dict, Union, Optional
import re
import tempfile
from flask import has_app_context #only for testing

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
image_folder = os.path.join(os.getcwd(), "Images")
csv_file = os.path.join(os.getcwd(), "files")

def normalize_firebase_url(url: str) -> str:
    """Normalize Firebase Storage URLs to consistent format."""
    # Remove port 443 if present
    url = url.replace(':443', '')

    # If the URL is already in the correct format, return as-is
    if 'firebasestorage.googleapis.com/v0/b/' in url and '?alt=media' in url:
        return url

    return url  # Return unchanged if not a Firebase URL needing normalization

def validate_image_url(url: str) -> bool:
    """Validate the image URL format including Firebase Storage URLs."""
    if not isinstance(url, str):
        return False

    # Firebase Storage URL pattern
    firebase_pattern = (
        r'^https:\/\/([a-zA-Z0-9\-]+\.)?firebasestorage\.googleapis\.com(:443)?'
        r'\/v0\/b\/[^\/]+\/o\/[^\/]+\.(jpg|jpeg|png|webp)\?alt=media(&token=[^&]+)?$'
    )

    # Standard URL check
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')

    # Check Firebase pattern
    if re.fullmatch(firebase_pattern, url, re.IGNORECASE):
        return True

    # Fall back to standard URL check
    return (url.startswith(('http://', 'https://')) and
            any(url.lower().endswith(ext) for ext in supported_extensions))

def download_image(url: str) -> Optional[Image.Image]:
    """Download and validate an image from URL with Firebase support."""
    try:
        # Normalize Firebase URLs first
        url = normalize_firebase_url(url)

        if not validate_image_url(url):
            raise ValueError(f"Invalid URL format: {url}")

        headers = {'User-Agent': 'Mozilla/5.0'}  # Some servers require user-agent
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Check content type (more flexible for Firebase)
        content_type = response.headers.get('Content-Type', '')
        if not (content_type.startswith('image/') or
                'octet-stream' in content_type or
                'application/octet-stream' in content_type):
            raise ValueError(f"Invalid content type: {content_type}")

        image = Image.open(BytesIO(response.content))

        # Basic image validation
        if image.width < 50 or image.height < 50:
            raise ValueError("Image dimensions too small")

        return image

    except Exception as e:
        logger.error(f"Failed to download image {url}: {str(e)}", exc_info=True)
        return None

def classify_and_analyze_url(image_urls: List[str]) -> Dict[str, Union[List, Dict, str]]:
    """Classify images from URLs and analyze outfit compatibility."""
    results = []
    analyzed_items = []
    errors = []

    if not image_urls:
        return {"error": "No image URLs provided"}

    for url in image_urls:
        try:
            if not validate_image_url(url):
                raise ValueError(f"Invalid image URL format: {url}")

            image = download_image(url)
            if not image:
                raise ValueError("Failed to download image")

            with BytesIO() as img_byte_arr:
                image.save(img_byte_arr, format="JPEG", quality=85)
                img_byte_arr.seek(0)

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    tmp_file.write(img_byte_arr.read())
                    temp_file_path = tmp_file.name

                try:
                    classification = classify_image_clip(
                        temp_file_path, processor, model,
                        clothing_types, occasions, seasons, materials
                    )
                finally:
                    os.remove(temp_file_path) # Clean up the temporary file

                if not classification or len(classification) != 6:
                    raise ValueError("Invalid classification results")

                clothing_type, category, occasion, season, material, dominant_color = classification

            item_data = {
                "image_url": url,  # Keep the original URL here
                "image_path": url,  # Also store as image_path but keep as URL
                "Clothing_Type": clothing_type,
                "Category": category,
                "Occasion": occasion or 'Casual',
                "Season": season,
                "Material": material,
                "Dominant_Color": str(dominant_color)
            }

            results.append(item_data)
            analyzed_items.append(item_data)  # Use the same item_data

        except Exception as e:
            error_msg = f"Failed to process {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            continue

    if not analyzed_items:
        return {
            "error": "All images failed processing",
            "details": errors,
            "success": False
        }

    try:
        current_df = pd.DataFrame(analyzed_items)

        # Ensure we have at least one valid category
        valid_categories = ['Top', 'Bottom', 'Dress', 'Footwear'] # Include Footwear
        if not any(cat in current_df['Category'].values for cat in valid_categories):
            logger.warning("No tops, bottoms, dresses, or footwear found in analyzed items")
            return {
                "classification": results,
                "outfit_combinations": {},
                "errors": errors,
                "warning": "No tops, bottoms, or dresses found",
                "success": True
            }

        analyzer = OutfitCompatibilityAnalyzer(
            classified_df=current_df,
            clip_processor=processor,
            clip_model=model,
            compatibility_prompts=compatibility_prompts,
            image_download_function=download_image  # Pass the download function
        )

        all_occasions = current_df['Occasion'].unique().tolist()
        best_outfits = []

        for occasion in all_occasions:
            try:
                occasion_outfits = analyzer.find_best_matches(occasion)
                if occasion_outfits:
                    best_outfits.extend(occasion_outfits)
            except Exception as e:
                logger.warning(f"Failed to find matches for occasion {occasion}: {str(e)}")
                continue

        if not best_outfits:
            logger.info("No occasion-specific outfits found, trying generic matching")
            best_outfits = analyzer.find_best_matches(None) or []

        outfit_combinations = {}
        for item, matches in best_outfits:
            try:
                item_url = item.get('image_path', '') if isinstance(item, dict) else getattr(item, 'image_path', '')
                if not item_url:
                    continue

                if matches is None:
                    # Case 1: Standalone Dress
                    outfit_combinations[item_url] = []
                elif isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    second_item, score = matches[0]
                    second_url = second_item.get('image_path', '') if isinstance(second_item, dict) else getattr(second_item, 'image_path', '')

                    if item.get('Category', '') == 'Dress':
                        # Case 2: Dress + Footwear
                        outfit_combinations[item_url] = [second_url]
                    else:
                        # Case 3: Top + Bottom
                        outfit_combinations[item_url] = [
                        match[0].get('image_path', '') if isinstance(match[0], dict)
                        else getattr(match[0], 'image_path', '')
                        for match in (matches or []) if match and match[0]
                        ]
                elif isinstance(matches[0], tuple) and len(matches[0]) == 3:
                    # Case 4: Top + Bottom + Footwear
                    outfit_combinations[item_url] = [
                        [
                            #item_url,
                            match[0].get('image_path', ''),  # bottom
                            match[1].get('image_path', '')   # footwear
                        ] for match in matches
                    ]

            except Exception as e:
                logger.warning(f"Failed to process outfit combination: {str(e)}")
                continue

        return {
            "classification": results,
            "outfit_combinations": outfit_combinations,
            "errors": errors if errors else None,
            "success": True
        }

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return {
            "error": f"Analysis failed: {str(e)}",
            "details": errors,
            "classification": results if results else None,
            "success": False
        }

def classify_and_analyze_local(images: List[Dict]) -> Dict:
    """Classify local images and analyze outfit compatibility."""
    results = []
    for image_data in images:
        image_path = os.path.join(image_folder, image_data['filename'])
        try:
            image = Image.open(image_path)
            image.save(image_path)  # Ensure the image is properly saved/read
        except FileNotFoundError:
            logger.error(f"Image not found: {image_path}")
            continue
        except Exception as e:
            logger.error(f"Error opening image {image_path}: {e}")
            continue

        try:
            clothing_type, category, occasion, season, material, dominant_color = classify_image_clip(
                image_path, processor_fc, model_fc, clothing_types, occasions, seasons, materials
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
        except Exception as e:
            logger.error(f"Error classifying {image_path}: {e}")
            continue
    
    if not results:
        return jsonify({"error": "No images classified successfully"}), 400

    try:
        df = pd.read_csv(os.path.join(csv_file, "Classified.csv"))
        image_features = torch.load("image_features.pt") if os.path.exists("image_features.pt") else {}
        analyzer = OutfitCompatibilityAnalyzer(df, processor, model, compatibility_prompts, image_features, device) # type: ignore

        #just for testing #remove later
        #occasion_to_analyze = 'Casual'
        occasion_to_analyze = results[0]['Occasion'] if results else 'Partywear'
        best_outfits = analyzer.find_best_matches(occasion_to_analyze)

        outfit_combinations = {}
        if best_outfits:
            for item, matches in best_outfits:
                try:
                    item_url = item.get('image_path') if isinstance(item, dict) else getattr(item, 'image_path', '')
                    if not item_url:
                        continue

                    if matches is None:
                        # Case 1: Standalone Dress
                        outfit_combinations[item_url] = []
                    elif isinstance(matches[0], tuple) and len(matches[0]) == 2:
                        # Case 2: Two-item combinations (Top + Bottom or Dress + Footwear)
                        second_items = [
                        match[0].get('image_path') if isinstance(match[0], dict)
                        else getattr(match[0], 'image_path', '')
                        for match in matches
                            ]
                        outfit_combinations[item_url] = second_items
                    elif isinstance(matches[0], tuple) and len(matches[0]) == 3:
                        # Case 3: Three-item combinations (Top + Bottom + Footwear)
                        combo_urls = [
                            [
                                match[0].get('image_path') if isinstance(match[0], dict)
                                else getattr(match[0], 'image_path', ''),
                                match[1].get('image_path') if isinstance(match[1], dict)
                                else getattr(match[1], 'image_path', ''),
                                match[2].get('image_path', '') if isinstance(match[2], dict) 
                                else getattr(match[2], 'image_path', '')
                            ]
                            for match in matches
                        ]
                        outfit_combinations[item_url] = combo_urls
                except Exception as e:
                    logger.warning(f"Error building outfit for item {item}: {e}")
        
        else:
            logger.info(f"No suitable outfits found for occasion: {occasion_to_analyze}")
            return jsonify({"classification": results, "outfit_combinations": {}}), 200

        return jsonify({
            "classification": results,
            "outfit_combinations": outfit_combinations
        })

    except FileNotFoundError:
        logger.error("Classified.csv not found.")
        return jsonify({"error": "Could not load classified data."}), 500
    except Exception as e:
        logger.error(f"Error during outfit analysis: {e}", exc_info=True)
        return jsonify({"error": f"Error during outfit analysis: {str(e)}"}), 500

@app.route('/process_images', methods=['POST'])
def process_images():
    try:
        if request.is_json:
            data = request.get_json()  # Get the dictionary
            if isinstance(data, dict) and "images" in data and isinstance(data["images"], list) and all(isinstance(url, str) for url in data["images"]):
                image_urls = data["images"]  # Extract the list of URLs
                logger.info(f"Processing {len(image_urls)} images from URLs")
                result = classify_and_analyze_url(image_urls)
                return jsonify(result)
            else:
                return jsonify({"error": "Input must be a JSON with an 'images' key containing a list of image URLs"}), 400
        elif request.files:
            images = []
            for filename, file in request.files.items():
                if filename.startswith('image'):
                    filepath = os.path.join(image_folder, file.filename)
                    file.save(filepath)
                    images.append({'filename': file.filename})
            if images:
                logger.info(f"Processing {len(images)} local images")
                result = classify_and_analyze_local(images)
                return result
            else:
                return jsonify({"error": "No image files uploaded"}), 400
        else:
            return jsonify({"error": "Invalid request format. Expecting JSON with image URLs or image files."}), 400
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(csv_file, exist_ok=True)
    if not os.path.exists(os.path.join(csv_file, "Classified.csv")):
        pd.DataFrame(columns=['image_name', 'Clothing_Type', 'Category', 'Occasion', 'Season', 'Material', 'Dominant Color']).to_csv(os.path.join(csv_file, "Classified.csv"), index=False)
    app.run(host='0.0.0.0', port=5000, debug=True)