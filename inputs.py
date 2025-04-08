# Define classification labels
clothing_types = ["Top", "Jean", "Dress", "Bag", "Shoes", "Sneakers", "Jacket", "Sweater", "Cardigan", "Skirt", "T-shirt", "Hoodie", "Trousers", "Shirt", "Coat", "Boots", "Blouse", "Jumpsuit", "Sandals", "Ballerinas", "Necklace", "Earrings", "Scarf", "Other accessories", "Leggings", "Joggers", "Running Shorts", "Shorts", "Tights", "Track Pants","Loafers","Mules","Oxfords","Derby","Espadrilles","Flip-Flops","Clogs","Platform Shoes","Wedges","Ankle Boots","Chelsea Boots","Knee-High Boots","Combat Boots","Hiking Boots","Slip-Ons","Clutch","Belt","Gloves","Sunglass","Headwear","Hat","Cap","Beanie","Jewelry","Bracelet","Anklet","Pendant","Choker","Wallet","Handbag","Tote Bag","Backpack","Crossbody Bag","Shoulder Bag","Satchel","Duffel Bag","Messenger Bag"]
occasions = ["Casual", "Formal", "Sportswear", "Partywear", "Travel"]
seasons = ["Summer", "Winter", "Spring", "Fall"]
tops = {"Top", "T-shirt", "Blouse", "Hoodie", "Sweater", "Jacket", "Cardigan", "Shirt", "Coat"}
bottoms = {"Jean", "Trousers", "Skirt", "Joggers", "Leggings", "Shorts", "Running Shorts", "Tights", "Track Pants"}
dresses = {"Dress", "Jumpsuit"}
footwear = {"Sneakers", "Shoes", "Boots", "Sandals", "Ballerinas", "Loafers", "Mules","Oxfords","Derby","Espadrilles","Flip-Flops","Clogs","Platform Shoes","Wedges","Ankle Boots","Chelsea Boots","Knee-High Boots","Combat Boots","Hiking Boots","Slip-Ons"}
accessories = {"Bag", "Scarf", "Necklace", "Earrings" , "Clutch", "Belt", "Gloves", "Sunglass","Headwear","Hat","Cap","Beanie","Jewelry","Bracelet","Anklet","Pendant","Choker","Wallet","Handbag","Tote Bag","Backpack","Crossbody Bag","Shoulder Bag","Satchel","Duffel Bag","Messenger Bag","Other accessories"}
materials = ["cotton", "denim", "silk", "wool", "leather", "linen", "polyester", "nylon", "velvet", "suede", "rayon", "chiffon", "spandex", "canvas", "corduroy", "satin", "tweed", "fleece", "acrylic", "cashmere", "viscose", "microfiber", "modal", "terrycloth", "bamboo", "hemp", "merino", "alpaca", "tulle", "neoprene", "lycra", "jacquard", "gabardine", "tencel", "organza", "mesh", "georgette", "seersucker", "poplin", "taffeta", "lace"]
dresses = {"Dress", "Jumpsuit"}  # Ensure this is defined

# Outfit rules based on category
OUTFIT_RULES = {
    "top_bottom": ["top", "bottom"],
    "top_bottom_footwear": ["top", "bottom", "footwear"], #Add this rule
    "dress_footwear": ["dress", "footwear"], #Add this rule
    "dress_footwear_accessory": ["dress", "footwear", "accessory"],  # Add this rule
}


#Defining prompts - outfit based or universal prompts
top_bottom_prompts = [
    #prompts with color, material + general outfit type prompt
    "A {top_material} {top_color} top paired with a {bottom_material} {bottom_color} bottom",
    "A stylish combination of a {top_material} {top_color} top and a {bottom_material} {bottom_color} bottom",
    "A fashionable outfit with complementary {top_color} and {bottom_color} colors",
    "A textured {top_material} {top_color} top with a {bottom_material} {bottom_color} bottom",
    "A coordinated {top_color} top and {bottom_color} bottom ensemble",
    "An outfit with clashing {top_color} and {bottom_color} colors",
    "A combination of {top_color} and {bottom_color} that does not follow color theory",
    "An outfit with incompatible {top_material} and {bottom_material} materials",
    "A poorly coordinated outfit with mismatched colors and materials",
    "An outfit that lacks harmony in color and material",
    "A disjointed combination of {top_color} and {bottom_color}",
    "An outfit with a jarring mix of {top_material} and {bottom_material}",
    "An outfit that uses contrasting colors in a pleasing way",
    "A bold outfit with a mix of contrasting patterns",
    "A visually dynamic combination of patterns",
    "A textured outfit with diverse materials",
    "A chic mix of smooth and rough textures",
    "An outfit with materials that complement each other",
    "A sleek and sophisticated monochromatic outfit",
]

dress_footwear_prompts = [
    "A fashionable {dress_material} {dress_color} dress paired with {footwear_material} shoes",
    "A stylish combination of a {dress_material} {dress_color} dress with {footwear_material} shoes",
    "An elegant {dress_material} dress in {dress_color} paired with trendy {footwear_material} shoes",
    "A chic {dress_material} {dress_color} dress and {footwear_material} footwear combination",
    "A well-coordinated {dress_color} dress with matching {footwear_material} shoes",
    "A stunning {dress_material} {dress_color} dress paired with classic {footwear_material} shoes",
    "A bold and fashionable {dress_color} dress with {footwear_material} footwear",
    "A sophisticated {dress_material} {dress_color} dress complemented by {footwear_material} shoes",
    "An elegant {dress_color} dress and {footwear_material} shoes ensemble",
    "A trendy {dress_material} {dress_color} dress with a modern twist, matched with {footwear_material} footwear",
    "A glamorous {dress_color} dress featuring {footwear_material} shoes for the perfect finish",
    "A contemporary {dress_material} dress in {dress_color} paired with {footwear_material} footwear",
    "A vintage-inspired {dress_material} {dress_color} dress combined with {footwear_material} shoes",
    "A luxurious {dress_material} {dress_color} dress with matching {footwear_material} shoes for a chic look",
    "A minimalist {dress_material} dress in {dress_color} paired with simple {footwear_material} shoes",
    "An effortlessly chic {dress_material} dress in {dress_color} with stylish {footwear_material} shoes",
    "A feminine {dress_color} dress complemented by delicate {footwear_material} footwear",
    "A playful {dress_material} {dress_color} dress and comfortable {footwear_material} shoes combination",
    "A luxurious {dress_material} dress in {dress_color}, paired with sophisticated {footwear_material} shoes",
    "A romantic {dress_material} {dress_color} dress matched with soft {footwear_material} footwear"
    #general prompt
    "A perfect pairing of a chic dress with stylish shoes",
    "An elegant dress and footwear combination for a fashionable look",
    "A trendy dress with a complementary pair of shoes",
    "A graceful dress matched with the perfect footwear for an elegant touch",
    "A beautifully coordinated dress and shoes combination for any occasion",
    "A classy dress paired with sophisticated footwear",
    "A fashionable dress and shoes combo to elevate your style",
    "A seamless ensemble of a dress and shoes that creates a polished look",
    "A well-coordinated dress with stylish footwear that completes the outfit",
    "A fashionable look with a stunning dress and eye-catching shoes",
    "A sleek dress paired with footwear for a flawless finish",
    "An effortlessly stylish dress and footwear combination for a chic look",
    "A fashionable and functional pairing of a dress and matching shoes",
    "A timeless dress complemented by fashionable shoes"
]

universal_prompts = [
    "A well-put-together and stylish ensemble",
    "A fashionable and well-coordinated outfit",
    "A chic and harmonious outfit",
    "A classic and fashionable combination",
    "An outfit that looks thoughtfully assembled",
    "A stylish and balanced outfit"
]

#can be extended later as we add more outfits
compatibility_prompts = {
    "top_bottom": top_bottom_prompts + universal_prompts,
    "dress_footwear": dress_footwear_prompts + universal_prompts
}
