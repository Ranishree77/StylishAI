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
from outfit_analyzer import OutfitCompatibilityAnalyzer  # Ensure this class is correctly implemented
from utils import classify_image_clip
import traceback
from typing import List, Dict, Union, Optional
import re
import tempfile
import asyncio
import aiohttp

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
model.to(device)
# processor.to(device)
model_fc.to(device)
# processor_fc.to(device)
image_folder = os.path.join(os.getcwd(), "Images")
csv_file = os.path.join(os.getcwd(), "files")

def normalize_firebase_url(url: str) -> str:
    """Normalize Firebase Storage URLs to consistent format."""
    url = url.replace(':443', '')
    if 'firebasestorage.googleapis.com/v0/b/' in url and '?alt=media' in url:
        return url
    return url

def validate_image_url(url: str) -> bool:
    """Validate the image URL format including Firebase Storage URLs."""
    if not isinstance(url, str):
        return False
    firebase_pattern = (
        r'^https:\/\/([a-zA-Z0-9\-]+\.)?firebasestorage\.googleapis\.com(:443)?'
        r'\/v0\/b\/[^\/]+\/o\/[^\/]+\.(jpg|jpeg|png|webp)\?alt=media(&token=[^&]+)?$'
    )
    supported_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    if re.fullmatch(firebase_pattern, url, re.IGNORECASE):
        return True
    return (url.startswith(('http://', 'https://')) and
            any(url.lower().endswith(ext) for ext in supported_extensions))

async def download_image_async(session: aiohttp.ClientSession, url: str) -> Optional[Image.Image]:
    """Asynchronously download and validate an image from URL."""
    try:
        url = normalize_firebase_url(url)
        if not validate_image_url(url):
            raise ValueError(f"Invalid URL format: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with session.get(url, headers=headers, timeout=60) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if not (content_type.startswith('image/') or
                    'octet-stream' in content_type or
                    'application/octet-stream' in content_type):
                raise ValueError(f"Invalid content type: {content_type}")
            content = await response.read()
            image = Image.open(BytesIO(content))
            if image.width < 50 or image.height < 50:
                raise ValueError("Image dimensions too small")
            return image
    except Exception as e:
        logger.error(f"Failed to download image {url}: {str(e)}", exc_info=True)
        return None

async def classify_and_analyze_url_async(image_urls: List[str], processor, model, clothing_types, occasions, seasons, materials, device, compatibility_prompts) -> Dict[str, Union[List, Dict, str, bool, None]]:
    """Classify images from URLs and analyze outfit compatibility asynchronously."""
    results = []
    analyzed_items = []
    errors = []

    if not image_urls:
        return {"error": "No image URLs provided", "success": False}

    async with aiohttp.ClientSession() as session:
        async def process_single_url(session, url):
            try:
                if not validate_image_url(url):
                    raise ValueError(f"Invalid image URL format: {url}")

                image = await download_image_async(session, url)
                if not image:
                    raise ValueError("Failed to download image")

                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format="JPEG", quality=85)
                img_byte_arr.seek(0)
                image_bytes = img_byte_arr.read() # Read the bytes here

                tmp_file = None
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                        tmp_file = tf
                        tmp_file.write(image_bytes) # Now writing the pre-read bytes
                        temp_file_path = tmp_file.name

                    classification = classify_image_clip(
                        temp_file_path, processor, model,
                        clothing_types, occasions, seasons, materials, device=device
                    )

                    if not classification or len(classification) != 6:
                        raise ValueError("Invalid classification results")

                    clothing_type, category, occasion, season, material, dominant_color = classification

                    item_data = {
                        "image_url": url,
                        "image_path": url,
                        "Clothing_Type": clothing_type,
                        "Category": category,
                        "Occasion": occasion or 'Casual',
                        "Season": season,
                        "Material": material,
                        "Dominant_Color": str(dominant_color)
                    }
                    return item_data, None
                finally:
                    if tmp_file:
                        tmp_file.close()
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            except ValueError as ve:
                error_msg = f"Processing error for {url}: {str(ve)}"
                logger.error(error_msg)
                return None, error_msg
            except Exception as e:
                error_msg = f"Failed to process {url}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return None, error_msg

        tasks = [process_single_url(session, url) for url in image_urls]
        processing_results = await asyncio.gather(*tasks)

    for item_data, error in processing_results:
        if item_data:
            results.append(item_data)
            analyzed_items.append(item_data)
        if error:
            errors.append(error)

    if not analyzed_items:
        return {
            "error": "All images failed processing",
            "details": errors,
            "success": False
        }

    try:
        current_df = pd.DataFrame(analyzed_items)
        valid_categories = ['Top', 'Bottom', 'Dress', 'Footwear']
        if not any(cat in current_df['Category'].values for cat in valid_categories):
            logger.warning("No tops, bottoms, dresses, or footwear found in analyzed items")
            return {
                "classification": results,
                "outfit_combinations": {},
                "errors": errors if errors else None,
                "warning": "No tops, bottoms, dresses, or footwear found",
                "success": True
            }

        analyzer = OutfitCompatibilityAnalyzer(
            classified_df=current_df,
            clip_processor=processor,
            clip_model=model,
            compatibility_prompts=compatibility_prompts,
            image_download_function=download_image_async
        )

        all_occasions = current_df['Occasion'].unique().tolist()
        best_outfits = []

        for occasion in all_occasions:
            try:
                occasion_outfits = await analyzer.find_best_matches(occasion)
                if occasion_outfits:
                    best_outfits.extend(occasion_outfits)
            except Exception as e:
                logger.warning(f"Failed to find matches for occasion {occasion}: {str(e)}")
                continue

        if not best_outfits:
            logger.info("No occasion-specific outfits found, trying generic matching")
            best_outfits = await analyzer.find_best_matches(None) or []

        outfit_combinations = {}
        for item, matches in best_outfits:
            try:
                item_url = item.get('image_path', '') if isinstance(item, dict) else getattr(item, 'image_path', '')
                if not item_url:
                    continue

                if matches is None:
                    outfit_combinations[item_url] = []
                elif isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    second_item, score = matches[0]
                    second_url = second_item.get('image_path', '') if isinstance(second_item, dict) else getattr(second_item, 'image_path', '')
                    if item.get('Category', '') in ['Dress', 'Footwear']:
                        outfit_combinations[item_url] = [second_url] if second_url else []
                    else:
                        outfit_combinations[item_url] = [
                            match[0].get('image_path', '') if isinstance(match[0], dict)
                            else getattr(match[0], 'image_path', '')
                            for match in (matches or []) if match and match[0]
                        ]
                elif isinstance(matches[0], tuple) and len(matches[0]) == 3:
                    outfit_combinations[item_url] = [
                        [
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
    analyzed_items = []
    errors = []
    local_image_paths = []

    for image_data in images:
        image_path = os.path.join(image_folder, image_data['filename'])
        local_image_paths.append(image_path)
        try:
            image = Image.open(image_path)
            image.save(image_path)
        except FileNotFoundError:
            logger.error(f"Image not found: {image_path}")
            errors.append(f"Image not found: {image_data['filename']}")
            continue
        except Exception as e:
            logger.error(f"Error opening image {image_path}: {e}")
            errors.append(f"Error opening image {image_data['filename']}: {e}")
            continue

        try:
            clothing_type, category, occasion, season, material, dominant_color = classify_image_clip(
                image_path, processor_fc, model_fc, clothing_types, occasions, seasons, materials, device=device
            )

            item_data = {
                "image_name": image_data['filename'],
                "image_path": image_path,
                "Clothing_Type": clothing_type,
                "Category": category,
                "Occasion": occasion,
                "Season": season,
                "Material": material,
                "Dominant_Color": str(dominant_color)
            }
            results.append(item_data)
            analyzed_items.append(item_data)
        except Exception as e:
            logger.error(f"Error classifying {image_path}: {e}")
            errors.append(f"Error classifying {image_data['filename']}: {e}")
            continue

    if not analyzed_items:
        return jsonify({"error": "No images classified successfully", "details": errors}), 400

    try:
        current_df = pd.DataFrame(analyzed_items)
        valid_categories = ['Top', 'Bottom', 'Dress', 'Footwear']
        if not any(cat in current_df['Category'].values for cat in valid_categories):
            logger.warning("No tops, bottoms, dresses, or footwear found in analyzed items")
            return jsonify({
                "classification": results,
                "outfit_combinations": {},
                "errors": errors,
                "warning": "No tops, bottoms, or dresses found"
            }), 200

        analyzer = OutfitCompatibilityAnalyzer(
            classified_df=current_df,
            clip_processor=processor,
            clip_model=model,
            compatibility_prompts=compatibility_prompts,
            image_download_function=None
        )

        all_occasions = current_df['Occasion'].unique().tolist()
        best_outfits = []

        for occasion in all_occasions:
            try:
                occasion_outfits = analyzer.find_best_matches(occasion) # Assuming find_best_matches is synchronous for local analysis
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
                    outfit_combinations[item_url] = []
                elif isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    second_item, score = matches[0]
                    second_url = second_item.get('image_path', '') if isinstance(second_item, dict) else getattr(second_item, 'image_path', '')
                    if item.get('Category', '') in ['Dress', 'Footwear']:
                        outfit_combinations[item_url] = [second_url] if second_url else []
                    else:
                        outfit_combinations[item_url] = [
                            match[0].get('image_path', '') if isinstance(match[0], dict)
                            else getattr(match[0], 'image_path', '')
                            for match in (matches or []) if match and match[0]
                        ]
                elif isinstance(matches[0], tuple) and len(matches[0]) == 3:
                    outfit_combinations[item_url] = [
                        [
                            match[0].get('image_path', ''),  # bottom
                            match[1].get('image_path', '')   # footwear
                        ] for match in matches
                    ]

            except Exception as e:
                logger.warning(f"Failed to process outfit combination: {str(e)}")
                continue

        return jsonify({
            "classification": results,
            "outfit_combinations": outfit_combinations,
            "errors": errors if errors else None
        })

    except Exception as e:
        logger.error(f"Error during outfit analysis: {e}", exc_info=True)
        return jsonify({"error": f"Error during outfit analysis: {str(e)}", "details": errors}), 500

@app.route('/process_images', methods=['POST'])
async def process_images():
    try:
        if request.is_json:
            data = request.get_json()
            if isinstance(data, dict) and "images" in data and isinstance(data["images"], list) and all(isinstance(url, str) for url in data["images"]):
                image_urls = data["images"]
                logger.info(f"Processing {len(image_urls)} images from URLs (async)")
                result = await classify_and_analyze_url_async(
                    image_urls,
                    processor=processor,
                    model=model,
                    clothing_types=clothing_types,
                    occasions=occasions,
                    seasons=seasons,
                    materials=materials,
                    device=device,
                    compatibility_prompts=compatibility_prompts
                )
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