import random
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class OutfitCompatibilityAnalyzer:
    def __init__(self, classified_df, clip_processor, clip_model, compatibility_prompts, image_download_function):
        self.df = classified_df.copy()
        self.tops = self.df[self.df['Category'] == 'Top'].copy()
        self.bottoms = self.df[self.df['Category'] == 'Bottom'].copy()
        self.dresses = self.df[self.df['Category'] == 'Dress'].copy()  # Add dresses
        self.footwears = self.df[self.df['Category'] == 'Footwear'].copy()  # Add footwear
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.compatibility_prompts = compatibility_prompts
        self.download_image = image_download_function

    def find_best_matches(self, occasion):
        # Filter items based on occasion
        filtered_tops = self.tops[self.tops['Occasion'] == occasion].copy()
        filtered_bottoms = self.bottoms[self.bottoms['Occasion'] == occasion].copy()
        filtered_dresses = self.dresses[self.dresses['Occasion'] == occasion].copy()  # Filter dresses
        filtered_footwear = self.footwears[self.footwears['Occasion'] == occasion].copy() # Filter footwear

        # Check if there are any relevant items for the selected occasion
        if filtered_tops.empty and filtered_bottoms.empty and filtered_dresses.empty and filtered_footwear.empty:
            logger.info(f"No tops, bottoms, dresses, or footwear found for the occasion: {occasion}")
            return []

        recommendations = []

        # 1. Dresses + Footwear
        if not filtered_dresses.empty and not filtered_footwear.empty:
            for _, dress in filtered_dresses.iterrows():
                scores = []
                for _, shoe in filtered_footwear.iterrows():
                    score = self._calculate_compatibility(dress.to_dict(), shoe.to_dict(), outfit_type="dress_footwear")
                    scores.append((shoe.to_dict(), score))
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:2]
                recommendations.append((dress.to_dict(), sorted_scores))

        # 2. Standalone Dresses (if no footwear to pair)
        elif not filtered_dresses.empty:
            for _, dress in filtered_dresses.iterrows():
                recommendations.append((dress.to_dict(), None))

        # 3. Top + Bottom
        if not filtered_tops.empty and not filtered_bottoms.empty:
            if len(filtered_tops) < 3:
                logger.info(f"Not enough tops found for the occasion: {occasion}. Need at least 3, found {len(filtered_tops)}")
            else:
                try:
                    random_top_indices = random.sample(range(len(filtered_tops)), min(3, len(filtered_tops)))
                    selected_tops = [filtered_tops.iloc[i].to_dict() for i in random_top_indices]

                    for top in selected_tops:
                        scores = []
                        for _, bottom in filtered_bottoms.iterrows():
                            score = self._calculate_compatibility(top, bottom.to_dict())
                            scores.append((bottom.to_dict(), score))
                        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
                        recommendations.append((top, sorted_scores))
                except Exception as e:
                    logger.error(f"Error generating top-bottom recommendations for occasion {occasion}: {e}", exc_info=True)

        def get_recommendation_score(recommendation):
            item, matches = recommendation
            if matches is None:  # Standalone dress
                return 1.0
            elif isinstance(matches, list) and matches:
                return np.mean([score for _, score in matches])
            return 0.0

        recommendations.sort(key=get_recommendation_score, reverse=True)
        return recommendations[:5]

    def _calculate_compatibility(self, item1, item2, outfit_type="top_bottom"):
        image_score = self._get_visual_compatibility_score(item1['image_path'], item2['image_path'])
        text_score = self._get_text_compatibility_score(item1, item2, outfit_type)

        if outfit_type == "dress_footwear":
            return 0.7 * image_score + 0.3 * text_score
        else:
            return 0.6 * image_score + 0.4 * text_score
        # Add logics for bottom + footwear , top+bottom+footwear later

    def _get_visual_compatibility_score(self, path1, path2):
        try:
            img1 = self._load_image_from_url(path1)
            img2 = self._load_image_from_url(path2)
            if img1 is None or img2 is None:
                return 0.0
            combined_image = self._create_combined_image(img1, img2)
            image_inputs = self.clip_processor(images=combined_image, return_tensors="pt", padding=True)
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_inputs = self.clip_processor(text=["a fashionable outfit", "an unfashionable outfit"], return_tensors="pt", padding=True)
            text_features = self.clip_model.get_text_features(**text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            return similarity[0][0].item()
        except Exception as e:
            logger.error(f"Error calculating visual compatibility for {path1} and {path2}: {e}", exc_info=True)
            return 0.0

    def _load_image_from_url(self, image_url):
        try:
            image = self.download_image(image_url)
            return image.resize((256, 256)) if image else None
        except Exception as e:
            logger.error(f"Error loading image from {image_url}: {e}", exc_info=True)
            return None

    def _create_combined_image(self, img1, img2):
        combined = Image.new('RGB', (512, 256))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (256, 0))
        return combined

    def _get_text_compatibility_score(self, item1, item2, outfit_type="top_bottom"):
        scores = []
        try:
            img1 = self._load_image_from_url(item1['image_path'])
            img2 = self._load_image_from_url(item2['image_path'])
            if img1 is None or img2 is None:
                return 0.0

            combined_image = self._create_combined_image(img1, img2)
            relevant_prompts = self.compatibility_prompts.get(outfit_type, [])

            for prompt_template in relevant_prompts:
                if outfit_type == "dress_footwear":
                    prompt = prompt_template.format(
                        dress_material=item1.get('Material', 'unknown'),
                        dress_color=item1.get('Dominant_Color', 'unknown'),
                        footwear_material=item2.get('Material', 'unknown'),
                        footwear_color=item2.get('Dominant_Color', 'unknown')
                    )
                else:  # Default: Top + Bottom
                    prompt = prompt_template.format(
                        top_material=item1.get('Material', 'unknown'),
                        bottom_material=item2.get('Material', 'unknown'),
                        top_color=item1.get('Dominant_Color', 'unknown'),
                        bottom_color=item2.get('Dominant_Color', 'unknown')
                    )

                inputs = self.clip_processor(
                    text=[prompt, "unfashionable combination"],
                    images=combined_image,
                    return_tensors="pt",
                    padding=True
                )
                outputs = self.clip_model(**inputs)
                scores.append(outputs.logits_per_image.softmax(dim=1)[0][0].item())
            return np.mean(scores) if scores else 0.0
        except Exception as e:
            logger.error(f"Error calculating text compatibility for items: {e}", exc_info=True)
            return 0.0