import random
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class OutfitCompatibilityAnalyzer:
    def __init__(self, classified_df, clip_processor, clip_model, compatibility_prompts):
        self.df = classified_df
        self.tops = self.df[self.df['Category'] == 'Top']
        self.bottoms = self.df[self.df['Category'] == 'Bottom']
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.compatibility_prompts = compatibility_prompts

    def find_best_matches(self, occasion):
        filtered_tops = self.tops[self.tops['Occasion'] == occasion]
        filtered_bottoms = self.bottoms[self.bottoms['Occasion'] == occasion]

        if filtered_tops.empty or filtered_bottoms.empty:
            print(f"No tops or bottoms found for the occasion: {occasion}")
            return []

        if len(filtered_tops) < 3:
            print(f"Not enough tops found for the occasion: {occasion}. Need at least 3, found {len(filtered_tops)}")
            return []

        random_top_indices = random.sample(range(len(filtered_tops)), 3)
        selected_tops = [filtered_tops.iloc[i] for i in random_top_indices]

        best_outfits = []
        for top in selected_tops:
            scores = []
            for _, bottom in filtered_bottoms.iterrows():
                score = self._calculate_compatibility(top, bottom)
                scores.append((bottom, score))

            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            best_outfits.append((top, sorted_scores))

        def average_outfit_score(outfit):
            return np.mean([score for _, score in outfit[1]])

        best_outfits.sort(key=average_outfit_score, reverse=True)
        return best_outfits[:2]

    def _calculate_compatibility(self, top, bottom):
        image_score = self._get_visual_compatibility_score(top['image_path'], bottom['image_path'])
        text_score = self._get_text_compatibility_score(top, bottom)
        return 0.6 * image_score + 0.4 * text_score

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

    def _get_text_compatibility_score(self, top, bottom):
        scores = []
        for prompt_template in self.compatibility_prompts:
            prompt = prompt_template.format(
                top_material=top['Material'],
                bottom_material=bottom['Material'],
                top_color=top['Dominant Color'],
                bottom_color=bottom['Dominant Color']
            )
            inputs = self.clip_processor(
                text=[prompt, "unfashionable combination"],
                images=self._create_combined_image(top['image_path'], bottom['image_path']),
                return_tensors="pt",
                padding=True
            )
            outputs = self.clip_model(**inputs)
            scores.append(outputs.logits_per_image.softmax(dim=1)[0][0].item())
        return np.mean(scores)

        