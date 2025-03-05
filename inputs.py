# Define classification labels
clothing_types = ["Top", "Jean", "Dress", "Bag", "Shoes", "Sneakers", "Jacket", "Sweater", "Cardigan", "Skirt", "T-shirt", "Hoodie", "Trousers", "Shirt", "Coat", "Boots", "Blouse", "Jumpsuit", "Sandals", "Ballerinas", "Necklace", "Earrings", "Scarf", "Other accessories", "Leggings", "Joggers", "Running Shorts", "Shorts", "Tights", "Track Pants"]
occasions = ["Casual", "Formal", "Sportswear", "Partywear", "Travel"]
seasons = ["Summer", "Winter", "Spring", "Fall"]
tops = {"Top", "T-shirt", "Blouse", "Hoodie", "Sweater", "Jacket", "Cardigan", "Shirt", "Coat"}
bottoms = {"Jean", "Trousers", "Skirt", "Joggers", "Leggings", "Shorts", "Running Shorts", "Tights", "Track Pants"}
dresses = {"Dress", "Jumpsuit"}
footwear = {"Sneakers", "Shoes", "Boots", "Sandals", "Ballerinas"}
accessories = {"Bag", "Scarf", "Necklace", "Earrings", "Other accessories"}
materials = ["cotton", "denim", "silk", "wool", "leather", "linen", "polyester", "nylon", "velvet", "suede", "rayon", "chiffon", "spandex", "canvas", "corduroy", "satin", "tweed", "fleece", "acrylic", "cashmere", "viscose", "microfiber", "modal", "terrycloth", "bamboo", "hemp", "merino", "alpaca", "tulle", "neoprene", "lycra", "jacquard", "gabardine", "tencel", "organza", "mesh", "georgette", "seersucker", "poplin", "taffeta", "lace"]

#Outfit rules based on category #not using currently but need to figure out few cases here
OUTFIT_RULES = {
    "top_bottom": ["top", "bottom"],
    "top_bottom_footwear": ["top", "bottom", "footwear"],
    "dress_footwear_accessory": ["dress", "footwear", "accessory"],
}

compatibility_prompts = [
    "A {top_material} {top_color} top paired with a {bottom_material} {bottom_color} bottom",
    "A stylish combination of a {top_material} {top_color} top and a {bottom_material} {bottom_color} bottom",
    "A fashionable outfit with complementary {top_color} and {bottom_color} colors",
    "A textured {top_material} {top_color} top with a {bottom_material} {bottom_color} bottom",
    "A stylish combination with complementary colors",
    "A well-coordinated outfit with balanced colors",
    "Fashionable outfit with complementary colors",
    "An outfit that uses contrasting colors in a pleasing way",
    "A color-coordinated outfit that looks harmonious together",
    "A bold outfit with a mix of contrasting patterns",
    "A visually dynamic combination of patterns",
    "An outfit where patterns enhance each other",
    "An outfit that creatively combines different patterns",
    "A sleek and sophisticated monochromatic outfit",
    "A monochromatic style that is both chic and modern",
    "A coordinated {top_color} top and {bottom_color} bottom ensemble",
    "A textured outfit with diverse materials",
    "A chic mix of smooth and rough textures",
    "An interesting and stylish blend of various textures",
    "A fashion-forward outfit with multiple textures",
    "An outfit with materials that complement each other",
    "A harmonious blend of fabrics",
    "A look that is carefully coordinated in terms of material",
    "An outfit that looks cohesive in its use of materials",
    "A well-put-together and stylish ensemble",
    "A fashionable and well-coordinated outfit",
    "A chic and harmonious outfit",
    "An outfit that looks thoughtfully assembled",
    "A stylish and balanced outfit",
    "A classic and fashionable combination",
    # "An outfit with clashing {top_pattern} and {bottom_pattern} patterns",
    # "A combination of {top_texture} and {bottom_texture} textures that do not complement each other",
    # "An outfit with a mismatched mix of {top_pattern} and {bottom_pattern}",
    # "An outfit with mismatched styles: {top_style} top and {bottom_style} bottom",
    # "A combination of {top_cultural_context} and {bottom_cultural_context} that does not align",
    "A mismatched outfit with clashing {top_color} and {bottom_color} colors",
    "An outfit with clashing {top_color} and {bottom_color} colors",
    "A combination of {top_color} and {bottom_color} that does not follow color theory",
    "An outfit with incompatible {top_material} and {bottom_material} materials",
    "A poorly coordinated outfit with mismatched colors and materials",
    "An outfit that lacks harmony in color and material",
    "A disjointed combination of {top_color} and {bottom_color}",
    "An outfit with a jarring mix of {top_material} and {bottom_material}"
]
