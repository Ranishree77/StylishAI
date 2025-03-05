from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForZeroShotImageClassification

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load model directly fashion clip
processor_fc = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model_fc = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
