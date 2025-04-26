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

    async def find_best_matches(self, occasion):
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
                    score = await self._calculate_compatibility(dress.to_dict(), shoe.to_dict(), outfit_type="dress_footwear")
                    scores.append((shoe.to_dict(), score))
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:2]
                recommendations.append((dress.to_dict(), sorted_scores))

        # 2. Standalone Dresses (if no footwear to pair)
        elif not filtered_dresses.empty:
            for _, dress in filtered_dresses.iterrows():
                recommendations.append((dress.to_dict(), None))

        # 3. Top + Bottom + Footwear
        if not filtered_tops.empty and not filtered_bottoms.empty and not filtered_footwear.empty:
            try:
                # Get sample tops - up to 3 if available, but work with whatever we have (even just 1)
                top_count = len(filtered_tops)
                if top_count > 0:
                    # If we have more than 3 tops, randomly sample 3
                    if top_count > 3:
                        random_top_indices = random.sample(range(len(filtered_tops)), min(3, len(filtered_tops)))
                        selected_tops = [filtered_tops.iloc[i].to_dict() for i in random_top_indices]
                    # Otherwise use all available tops
                    else:
                        selected_tops = [filtered_tops.iloc[i].to_dict() for i in range(top_count)]

                    for top in selected_tops:
                        # First find compatible bottoms
                        bottom_scores = []
                        for _, bottom in filtered_bottoms.iterrows():
                            bottom_dict = bottom.to_dict()
                            score = await self._calculate_compatibility(top, bottom_dict, outfit_type="top_bottom")
                            bottom_scores.append((bottom_dict, score))

                        # Get the top 2 most compatible bottoms
                        top_bottoms = sorted(bottom_scores, key=lambda x: x[1], reverse=True)[:2]

                        # For each top-bottom pair, find compatible footwear
                        for bottom_item, bottom_score in top_bottoms:
                            footwear_scores = []
                            for _, shoe in filtered_footwear.iterrows():
                                shoe_dict = shoe.to_dict()
                                # Calculate compatibility score for the three-piece outfit
                                three_piece_score = await self._calculate_three_piece_compatibility(
                                    top, bottom_item, shoe_dict
                                )
                                footwear_scores.append((shoe_dict, three_piece_score))

                            # Get the top 2 most compatible footwear options for best matched "top+bottom" pair
                            top_footwear = sorted(footwear_scores, key=lambda x: x[1], reverse=True)[:2]

                            # Create the three-piece recommendation
                            # Store as a tuple: (top, [(bottom, footwear1, score1), (bottom, footwear2, score2), ...])
                            three_piece_recs = []
                            for shoe_item, three_piece_score in top_footwear:
                                three_piece_recs.append((bottom_item, shoe_item, three_piece_score))

                            if three_piece_recs:
                                recommendations.append((top, three_piece_recs))

            except Exception as e:
                logger.error(f"Error generating top-bottom-footwear recommendations for occasion {occasion}: {e}", exc_info=True)

        # 3. Top + Bottom (Also generate Top + Bottom outfits (even if footwear exists))
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
                            score = await self._calculate_compatibility(top, bottom.to_dict())
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
                if all(len(match) == 3 for match in matches):  # Three-piece outfit (top+bottom+footwear)
                    return np.mean([score for _, _, score in matches])
                else:# Two-piece outfit (dress+footwear or top+bottom)
                    return np.mean([score for _, score in matches])
            return 0.0

        recommendations.sort(key=get_recommendation_score, reverse=True)
        return recommendations[:5]

    async def _calculate_compatibility(self, item1, item2, outfit_type="top_bottom"):
        image_score = await self._get_visual_compatibility_score(item1['image_path'], item2['image_path'])
        text_score = await self._get_text_compatibility_score(item1, item2, outfit_type)

        if outfit_type == "dress_footwear":
            return 0.7 * image_score + 0.3 * text_score
        else:
            return 0.6 * image_score + 0.4 * text_score

    #new function for top+bottom+footwear compatibility check
    #determine how well three pieces of clothing (top, bottom, and footwear) work together as a complete outfit.
    async def _calculate_three_piece_compatibility(self, top, bottom, footwear):
        """Calculate compatibility for a three-piece outfit (top + bottom + footwear)"""
        try:
            # Calculate pairwise compatibility scores
            top_bottom_score = await self._calculate_compatibility(top, bottom, "top_bottom")
            bottom_footwear_score = await self._calculate_compatibility(bottom, footwear, "bottom_footwear")
            top_footwear_score = await self._calculate_compatibility(top, footwear, "top_footwear")

            # Calculate visual compatibility score for all three pieces together
            visual_score = await self._get_visual_compatibility_score(
                top['image_path'],
                bottom['image_path'],
                footwear['image_path']
            )

            # Get text compatibility score for the three-piece outfit
            text_score = await self._get_text_compatibility_score(
                top,
                bottom,
                "top_bottom_footwear",
                item3=footwear
            )

            # Calculate final score - weighted average of all compatibility metrics
            final_score = (
                0.2 * top_bottom_score +
                0.1 * bottom_footwear_score +
                0.1 * top_footwear_score +
                0.3 * visual_score +
                0.3 * text_score
            )
            return final_score

        except Exception as e:
            logger.error(f"Error calculating three-piece compatibility: {e}", exc_info=True)
            return 0.0

    #accomodated to include three piece outfit #review once
    async def _get_visual_compatibility_score(self, path1, path2, path3=None):
        """Get visual compatibility score for 2-3 clothing items"""
        #review this function how image will be loaded for each outfit type
        try:
            img1 = await self._load_image_from_url(path1)
            img2 = await self._load_image_from_url(path2)
            if img1 is None or img2 is None:
                return 0.0
            # Load the third image if provided
            img3 = None
            if path3:
                img3 = await self._load_image_from_url(path3)
                if img3 is None:
                    return 0.0

            combined_image = self._create_combined_image(img1, img2, img3)
            image_inputs = self.clip_processor(images=combined_image, return_tensors="pt", padding=True)
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_inputs = self.clip_processor(text=["a fashionable outfit", "an unfashionable outfit"], return_tensors="pt", padding=True)
            text_features = self.clip_model.get_text_features(**text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            return similarity[0][0].item()
        except Exception as e:
            logger.error(f"Error calculating visual compatibility for: {e}", exc_info=True)
            return 0.0

    async def _load_image_from_url(self, image_url):
        try:
            image = await self.download_image(None, image_url) # Pass None for session here, assuming download_image handles it
            return image.resize((256, 256)) if image else None
        except Exception as e:
            logger.error(f"Error loading image from {image_url}: {e}", exc_info=True)
            return None

    #updated to accomodate 3 piece outfit type
    def _create_combined_image(self, img1, img2, img3=None):
        """Create a combined image from 2-3 component images"""
        if img3 is None:
            # Original case: two images side by side
            combined = Image.new('RGB', (512, 256))
            combined.paste(img1, (0, 0))
            combined.paste(img2, (256, 0))
        else:
            # New case: three images side by side
            combined = Image.new('RGB', (768, 256))
            combined.paste(img1, (0, 0))
            combined.paste(img2, (256, 0))
            combined.paste(img3, (512, 0))
        return combined

    #defaults to top_bottom outfit type
    async def _get_text_compatibility_score(self, item1, item2, outfit_type="top_bottom", item3=None):
        scores = []
        try:
            # Load images
            img1 = await self._load_image_from_url(item1['image_path'])
            img2 = await self._load_image_from_url(item2['image_path'])
            if img1 is None or img2 is None:
                return 0.0

            img3 = None
            if item3:
                img3 = await self._load_image_from_url(item3['image_path'])
                if img3 is None:
                    return 0.0

            # Create combined image based on number of items
            combined_image = self._create_combined_image(img1, img2, img3)
            relevant_prompts = self.compatibility_prompts.get(outfit_type, [])

            for prompt_template in relevant_prompts:
                if outfit_type == "dress_footwear":
                    prompt = prompt_template.format(
                        dress_material=item1.get('Material', 'unknown'),
                        dress_color=item1.get('Dominant_Color', 'unknown'),
                        footwear_material=item2.get('Material', 'unknown'),
                        footwear_color=item2.get('Dominant_Color', 'unknown')
                    )
                elif outfit_type == "bottom_footwear": #new
                    prompt = prompt_template.format(
                        bottom_material=item1.get('Material', 'unknown'),
                        bottom_color=item1.get('Dominant_Color', 'unknown'),
                        footwear_material=item2.get('Material', 'unknown'),
                        footwear_color=item2.get('Dominant_Color', 'unknown')
                    )
                elif outfit_type == "top_footwear": #new
                    prompt = prompt_template.format(
                        top_material=item1.get('Material', 'unknown'),
                        top_color=item1.get('Dominant_Color', 'unknown'),
                        footwear_material=item2.get('Material', 'unknown'),
                        footwear_color=item2.get('Dominant_Color', 'unknown')
                    )
                elif outfit_type == "top_bottom_footwear" and item3: #new (review logic)
                    prompt = prompt_template.format(
                        top_material=item1.get('Material', 'unknown'),
                        top_color=item1.get('Dominant_Color', 'unknown'),
                        bottom_material=item2.get('Material', 'unknown'),
                        bottom_color=item2.get('Dominant_Color', 'unknown'),
                        footwear_material=item3.get('Material', 'unknown'),
                        footwear_color=item3.get('Dominant_Color', 'unknown')
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