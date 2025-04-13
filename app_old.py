from flask import Flask, request, jsonify
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import logging
from models import model, processor
from inputs import clothing_types, occasions, seasons, materials, compatibility_prompts
from outfit_analyzer import OutfitCompatibilityAnalyzer  # Assuming this class is defined in outfit_analyzer.py
from utils import classify_image_clip
import traceback
from typing import List, Dict, Union, Optional
import re

app = Flask(_name_)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(_name_)

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

def classify_and_analyze(image_urls: List[str]) -> Dict[str, Union[List, Dict, str]]:
    """Classify images and analyze outfit compatibility."""
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

                classification = classify_image_clip(
                    img_byte_arr, processor, model,
                    clothing_types, occasions, seasons, materials
                )

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
        valid_categories = ['Top', 'Bottom', 'Dress']
        if not any(cat in current_df['Category'].values for cat in valid_categories):
            logger.warning("No tops, bottoms, or dresses found in analyzed items")
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

                outfit_combinations[item_url] = [
                    match[0].get('image_path', '') if isinstance(match[0], dict)
                    else getattr(match[0], 'image_path', '')
                    for match in (matches or [])
                    if match and match[0]
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

@app.route('/process_images', methods=['POST'])
def process_images():
    """Endpoint to process images and suggest outfit combinations."""
    try:
        logger.info("Received request to process images")

        data = request.get_json()
        if not data or 'images' not in data:
            logger.warning("Missing images data in request")
            return jsonify({
                "error": "Missing images data",
                "success": False
            }), 400

        image_urls = data['images']
        if not isinstance(image_urls, list):
            logger.warning("Images data is not a list")
            return jsonify({
                "error": "Images should be a list of URLs",
                "success": False
            }), 400

        if not image_urls:
            logger.warning("Empty images list provided")
            return jsonify({
                "error": "No images provided",
                "success": False
            }), 400

        # Validate URLs before processing
        invalid_urls = [url for url in image_urls if not validate_image_url(url)]
        if invalid_urls:
            logger.warning(f"Invalid URLs detected: {invalid_urls}")
            return jsonify({
                "error": f"Invalid image URLs: {invalid_urls[:3]}...",
                "success": False
            }), 400

        logger.info(f"Processing {len(image_urls)} images")
        result = classify_and_analyze(image_urls)

        if not result.get("success", False):
            status_code = 400 if result.get("error") else 200
            return jsonify(result), status_code

        return jsonify(result)

    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "success": False
        }), 500

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)