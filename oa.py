import random
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import logging

logger = logging.getLogger(_name_)

class OutfitCompatibilityAnalyzer:
    def _init_(self, classified_df, clip_processor, clip_model, compatibility_prompts, image_download_function):
        self.df = classified_df
        self.tops = self.df[self.df['Category'] == 'Top'].copy()
        self.bottoms = self.df[self.df['Category'] == 'Bottom'].copy()
        self.dresses = self.df[self.df['Category'] == 'Dress'].copy()  # Add dresses
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.compatibility_prompts = compatibility_prompts
        self.download_image = image_download_function

    def find_best_matches(self, occasion):
        # Filter tops, bottoms, and dresses based on occasion
        filtered_tops = self.tops[self.tops['Occasion'] == occasion].copy()
        filtered_bottoms = self.bottoms[self.bottoms['Occasion'] == occasion].copy()
        filtered_dresses = self.dresses[self.dresses['Occasion'] == occasion].copy()  # Filter dresses

        # Check if there are any items for the selected occasion
        if filtered_tops.empty and filtered_bottoms.empty and filtered_dresses.empty:
            logger.info(f"No tops, bottoms, or dresses found for the occasion: {occasion}")
            return []

        # Initialize a list to store all recommendations
        recommendations = []

        # Add dresses to recommendations
        if not filtered_dresses.empty:
            for _, dress in filtered_dresses.iterrows():
                recommendations.append((dress.to_dict(), None))  # Dresses don't need pairing

        # Add top-bottom pairs to recommendations
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

        # Sort all recommendations by score (for dresses, use a default score of 1.0)
        def get_recommendation_score(recommendation):
            if recommendation[1] is None:  # Dress
                return 1.0  # Default score for dresses
            elif recommendation[1]:  # Top-bottom pair with scores
                return np.mean([score for _, score in recommendation[1]])
            else:
                return 0.0 # No compatible bottom found

        recommendations.sort(key=get_recommendation_score, reverse=True)
        return recommendations[:5]  # Return top 5 recommendations (dresses and top-bottom pairs)

    def _calculate_compatibility(self, top, bottom):
        image_score = self._get_visual_compatibility_score(top['image_path'], bottom['image_path'])
        text_score = self._get_text_compatibility_score(top, bottom)
        return 0.6 * image_score + 0.4 * text_score

    def _get_visual_compatibility_score(self, top_path, bottom_path):
        try:
            top_img = self._load_image_from_url(top_path)
            bottom_img = self._load_image_from_url(bottom_path)
            if top_img is None or bottom_img is None:
                return 0.0
            combined_image = self._create_combined_image(top_img, bottom_img)
            image_inputs = self.clip_processor(images=combined_image, return_tensors="pt", padding=True)
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_inputs = self.clip_processor(text=["a fashionable outfit", "an unfashionable outfit"], return_tensors="pt", padding=True)
            text_features = self.clip_model.get_text_features(**text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            return similarity[0][0].item()
        except Exception as e:
            logger.error(f"Error calculating visual compatibility for {top_path} and {bottom_path}: {e}", exc_info=True)
            return 0.0

    def _load_image_from_url(self, image_url):
        try:
            image = self.download_image(image_url)
            return image.resize((256, 256)) if image else None
        except Exception as e:
            logger.error(f"Error loading image from {image_url}: {e}", exc_info=True)
            return None

    def _create_combined_image(self, top_img, bottom_img):
        combined = Image.new('RGB', (512, 256))
        combined.paste(top_img, (0, 0))
        combined.paste(bottom_img, (256, 0))
        return combined

    def _get_text_compatibility_score(self, top, bottom):
        scores = []
        try:
            top_img = self._load_image_from_url(top['image_path'])
            bottom_img = self._load_image_from_url(bottom['image_path'])
            if top_img is None or bottom_img is None:
                return 0.0

            combined_image = self._create_combined_image(top_img, bottom_img)

            for prompt_template in self.compatibility_prompts:
                prompt = prompt_template.format(
                    top_material=top.get('Material', 'unknown'),
                    bottom_material=bottom.get('Material', 'unknown'),
                    top_color=top.get('Dominant_Color', 'unknown'),
                    bottom_color=bottom.get('Dominant_Color', 'unknown')
                    # dress_material=dress['Material'],
                    # dress_color=dress['dress_color']
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
            logger.error(f"Error calculating text compatibility for top: {top.get('image_path', 'N/A')} and bottom: {bottom.get('image_path', 'N/A')}: {e}", exc_info=True)
            return 0.0