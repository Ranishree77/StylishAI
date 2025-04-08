import random
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class OutfitCompatibilityAnalyzer:
    def __init__(self, classified_df, clip_processor, clip_model, compatibility_prompts):
        self.df = classified_df
        self.tops = self.df[self.df['Category'] == 'Top']
        self.bottoms = self.df[self.df['Category'] == 'Bottom']
        self.dresses = self.df[self.df['Category'] == 'Dress']  # Add dresses
        self.footwears = self.df[self.df['Category'] == 'Footwear']  # Add footwear
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.compatibility_prompts = compatibility_prompts

    def find_best_matches(self, occasion):
        # Filter tops, bottoms, and dresses based on occasion
        filtered_tops = self.tops[self.tops['Occasion'] == occasion]
        filtered_bottoms = self.bottoms[self.bottoms['Occasion'] == occasion]
        filtered_dresses = self.dresses[self.dresses['Occasion'] == occasion]  # Filter dresses
        filtered_footwear = self.footwears[self.footwears['Category'] == 'Footwear'] # Filter footwear

        # Check if there are any items for the selected occasion
        if filtered_tops.empty and filtered_bottoms.empty and filtered_dresses.empty:
            print(f"No tops, bottoms, or dresses found for the occasion: {occasion}")
            return []

        # Initialize a list to store all recommendations
        recommendations = []

        #1 Add **Dresses + Footwear** to recommendations
        if not filtered_dresses.empty and not filtered_footwear.empty:
            for _, dress in filtered_dresses.iterrows():
                scores = []
                for _, shoe in filtered_footwear.iterrows():
                    # Pass 'outfit_type="dress_footwear"' to indicate it's a dress + footwear combination
                    score = self._calculate_compatibility(dress, shoe, outfit_type="dress_footwear")
                    scores.append((shoe, score))

                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:2]
                recommendations.append((dress, sorted_scores))

        # Add dresses to recommendations
        elif not filtered_dresses.empty:
            for _, dress in filtered_dresses.iterrows():
                recommendations.append((dress, None))  # Dresses don't need pairing

        # Add top-bottom pairs to recommendations
        if not filtered_tops.empty and not filtered_bottoms.empty:
            if len(filtered_tops) < 3:
                print(f"Not enough tops found for the occasion: {occasion}. Need at least 3, found {len(filtered_tops)}")
            else:
                random_top_indices = random.sample(range(len(filtered_tops)), 3)
                selected_tops = [filtered_tops.iloc[i] for i in random_top_indices]

                for top in selected_tops:
                    scores = []
                    for _, bottom in filtered_bottoms.iterrows():
                        score = self._calculate_compatibility(top, bottom)
                        scores.append((bottom, score))

                    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
                    recommendations.append((top, sorted_scores))

        # Sort all recommendations by score (for dresses, use a default score of 1.0)
        def get_recommendation_score(recommendation):
            if recommendation[1] is None:  # Dress
                return 1.0  # Default score for dresses
            elif isinstance(recommendation[1], list):  # If paired (Dress+Footwear or Top+Bottom)
                return np.mean([score for _, score in recommendation[1]])  # Average score
            return 0  # Fallback (should not happen)

        recommendations.sort(key=get_recommendation_score, reverse=True)
        return recommendations[:5]  # Return top 5 recommendations (dresses and top-bottom pairs)

    def _calculate_compatibility(self, item1, item2, outfit_type="top_bottom"):
        image_score = self._get_visual_compatibility_score(item1['image_path'], item2['image_path'])
        text_score = self._get_text_compatibility_score(item1, item2)

        if outfit_type == "dress_footwear":
            # Adjust weightage for Dress + Footwear
            return 0.7 * image_score + 0.3 * text_score
        else:
            # Default: Top + Bottom
            return 0.6 * image_score + 0.4 * text_score
        #Add logics for bottom + footwear , top+bottom+footwear later

    def _get_visual_compatibility_score(self, top_path, bottom_path):
        combined_image = self._create_combined_image(top_path, bottom_path)
        image_inputs = self.clip_processor(images=combined_image, return_tensors="pt", padding=True)
        image_features = self.clip_model.get_image_features(**image_inputs)
        text_inputs = self.clip_processor(text=["a fashionable outfit", "an unfashionable outfit"], return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        return similarity[0][0].item()

    def _create_combined_image(self, top_path, bottom_path):
        top_img = Image.open(top_path).resize((256, 256))
        bottom_img = Image.open(bottom_path).resize((256, 256))
        combined = Image.new('RGB', (512, 256))
        combined.paste(top_img, (0, 0))
        combined.paste(bottom_img, (256, 0))
        return combined

    def _get_text_compatibility_score(self, item1, item2, outfit_type="top_bottom"):
        scores = []

        # Retrieve the relevant prompts for the given outfit_type
        # Make sure the outfit_type is always passed exactly as it appears in the compatibility_prompts keys
        relevant_prompts = self.compatibility_prompts.get(outfit_type, [])

        # Loop over filtered prompts
        for prompt_template in relevant_prompts:
            if outfit_type == "dress_footwear":
                prompt = prompt_template.format(
                    dress_material=item1['Material'],
                    dress_color=item1['Dominant Color'],
                    footwear_material=item2['Material'],
                    footwear_color=item2['Dominant Color']
                )

            else: # Default: Top + Bottom
                prompt = prompt_template.format(
                    top_material=item1['Material'],
                    bottom_material=item2['Material'],
                    top_color=item1['Dominant Color'],
                    bottom_color=item2['Dominant Color']
                )

            inputs = self.clip_processor(
                text=[prompt, "unfashionable combination"],
                images=self._create_combined_image(item1['image_path'], item2['image_path']),
                return_tensors="pt",
                padding=True
            )
            outputs = self.clip_model(**inputs)
            scores.append(outputs.logits_per_image.softmax(dim=1)[0][0].item())
        return np.mean(scores)
