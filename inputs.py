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
    "bottom_footwear": ["bottom", "footwear"],  # Add this
    "top_footwear": ["top", "footwear"]
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

top_bottom_footwear_prompts = [
    "A {top_material} {top_color} top with {bottom_material} {bottom_color} bottom and {footwear_material} {footwear_color} shoes",
    "A stylish ensemble with a {top_color} top, {bottom_color} bottom, and complementary {footwear_color} footwear",
    "A coordinated outfit with {top_material} top, {bottom_material} bottom, and {footwear_material} shoes",
    "A fashionable three-piece outfit with matching colors and textures",
    "A complete outfit with harmonious {top_color}, {bottom_color}, and {footwear_color} color palette",
    "A balanced casual look with {top_material} top, {bottom_material} bottom, and {footwear_material} footwear",
    "An elegant combination of {top_color} top, {bottom_color} bottom, and {footwear_color} shoes",
    "A sophisticated outfit with textured {top_material} top, {bottom_material} bottom, and {footwear_material} footwear",
    "A trendy look combining {top_color} upper wear, {bottom_color} lower wear, and {footwear_color} footwear",
    "A cohesive outfit with complementary materials from top to shoes"
]

bottom_footwear_prompts = [
    "A {bottom_material} {bottom_color} bottom paired with {footwear_material} {footwear_color} footwear",
    "A stylish combination of {bottom_material} {bottom_color} bottoms and {footwear_material} {footwear_color} shoes",
    "A fashionable pairing of {bottom_color} bottom wear with {footwear_color} footwear",
    "A coordinated ensemble with {bottom_material} bottoms and {footwear_material} shoes",
    "A balanced look with {bottom_color} bottoms and {footwear_color} footwear",
    "A complementary pairing of {bottom_material} {bottom_color} pants and {footwear_material} shoes"
]

top_footwear_prompts = [
    "A {top_material} {top_color} top paired with {footwear_material} {footwear_color} footwear",
    "A stylish combination of {top_material} {top_color} top and {footwear_material} {footwear_color} shoes",
    "A fashionable pairing of {top_color} top wear with {footwear_color} footwear",
    "A coordinated ensemble with {top_material} tops and {footwear_material} shoes",
    "A balanced look with {top_color} top and {footwear_color} footwear",
    "A complementary pairing of {top_material} {top_color} top and {footwear_material} shoes"
]

top_footwear_general_prompts = [
    "A well-coordinated look from top to toe",
    "A stylish pairing that brings harmony between topwear and footwear",
    "A carefully selected top and footwear combo that stands out",
    "A sleek and modern match of topwear and shoes",
    "An effortlessly fashionable top and shoe ensemble",
    "A refined combination of top and footwear for a balanced look",
    "An elevated style pairing that connects topwear with the perfect shoes",
    "A smart and polished outfit focusing on topwear and footwear coordination",
    "A chic head-to-foot pairing with an emphasis on top and shoes",
    "An eye-catching combo where topwear and footwear complement each other beautifully"
]

bottom_footwear_general_prompts = [
    "A cohesive bottom and footwear pairing that defines the outfit's base",
    "A stylish bottomwear and shoe combination that anchors the look",
    "A modern take on combining bottoms with coordinating footwear",
    "An outfit foundation with perfectly matched bottoms and shoes",
    "A minimal yet stylish approach to matching pants and footwear",
    "A grounded outfit pairing that balances color and shape in bottoms and shoes",
    "A strong lower-body statement with bottoms and footwear in sync",
    "A clean, refined bottom and shoe duo that completes the look",
    "A thoughtfully matched pair of bottoms and shoes that work in harmony",
    "A dynamic bottom-footwear combo that adds depth to the outfit"
]

top_bottom_footwear_general_prompts = [
    "A complete and stylish outfit with a well-matched top, bottom, and footwear",
    "An effortlessly coordinated look from top to toe",
    "A balanced outfit that brings harmony between the top, bottom, and shoes",
    "An all-around chic outfit combining a fashionable top, bottom, and matching footwear",
    "A trendy full-body look thats stylishly put together with top, bottom, and footwear",
    "A head-to-toe outfit that shows thoughtful color and material coordination",
    "An eye-catching ensemble where top, bottom, and footwear work together seamlessly",
    "A polished three-piece look that ties everything together with flair",
    "A cohesive outfit designed for both comfort and style, featuring matching top, bottom, and footwear",
    "A stylishly layered look with attention to detail from the top to the shoes"
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
    "dress_footwear": dress_footwear_prompts + universal_prompts,
    "top_bottom_footwear": top_bottom_footwear_prompts + top_bottom_footwear_general_prompts,
    "bottom_footwear": bottom_footwear_prompts + bottom_footwear_general_prompts,  # Add this
    "top_footwear": top_footwear_prompts + top_footwear_general_prompts  # Add this
}