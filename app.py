import streamlit as st
from ultralytics import YOLO
import os
import cv2
from PIL import Image
import numpy as np

# Path to the model file
model_path = "/Users/cliq-tech/Desktop/AI waste management/runs/detect/waste_detection_x/weights/best.pt"
model = YOLO(model_path)

# Assert the model path exists
assert os.path.exists(model_path), "Model file not found!"

# Waste suggestions database with waste types
waste_suggestions = {
    "battery": {
        "type": "hazardous",
        "disposal": [
            "1. Drop off at battery recycling centers or hazardous waste facilities like Agbogbloshie Market.",
            "2. Donate to NGOs or schools for proper disposal education campaigns.",
            "3. Partner with e-waste collection drives in Accra or Tema.",
            "4. Seal in non-reactive containers to prevent leakage before disposal.",
            "5. Participate in exchange programs offering rewards for recycling batteries."
        ],
        "reuse": [
            "1. Use for school experiments to teach basic circuits and electricity.",
            "2. Repurpose in training centers to demonstrate proper e-waste handling.",
            "3. Combine multiple dead batteries for non-critical low-power devices like LED flashlights.",
            "4. Use in local art installations to create awareness of e-waste.",
            "5. Engage artisans to craft sculptures or decor from batteries."
        ]
    },
    "can": {
        "type": "recyclable",
        "disposal": [
            "1. Wash and recycle in metal bins.",
            "2. Donate to scrap collectors for industrial reuse.",
            "3. Melt for small-scale aluminum or tin casting in workshops.",
            "4. Use for roadside clean-up campaigns and recycling drives.",
            "5. Contribute to upcycling centers in urban areas like Kumasi."
        ],
        "reuse": [
            "1. Convert into kitchen organizers for utensils or spice containers.",
            "2. Craft into musical instruments such as rattles or drumsticks.",
            "3. Use in schools for simple science experiments, such as sound amplifiers.",
            "4. Create custom jewelry boxes or personal storage containers.",
            "5. Decorate as lanterns or lampshades for local festivals."
        ]
    },
    "cardboard_bowl": {
        "type": "recyclable",
        "disposal": [
            "1. Compost in backyard gardens for soil improvement.",
            "2. Offer to farmers as biodegradable seedling trays.",
            "3. Drop at paper recycling depots in major cities.",
            "4. Provide to vendors needing lightweight disposable bowls.",
            "5. Use as temporary containers for waste segregation at events."
        ],
        "reuse": [
            "1. Cut and use for creative children’s craft activities.",
            "2. Shred for chicken coop bedding in poultry farms.",
            "3. Line pots or trays for additional insulation in plant germination.",
            "4. Transform into masks or other props for traditional dances.",
            "5. Use as scoops for measuring grains or flour in kitchens."
        ]
    },
    "cardboard_box": {
        "type": "recyclable",
        "disposal": [
            "1. Flatten and donate to small businesses for packaging.",
            "2. Recycle at local depots collecting bulk cardboard.",
            "3. Compost in community gardens for soil amendment.",
            "4. Use in flood-prone areas to raise items off the ground temporarily.",
            "5. Burn in controlled environments for safe energy generation."
        ],
        "reuse": [
            "1. Convert into playhouses or dollhouses for children.",
            "2. Use in market stalls for displaying goods.",
            "3. Repurpose into furniture such as low stools or shelves.",
            "4. Insulate walls or roofs in informal settlements.",
            "5. Create large-scale murals or displays in community art projects."
        ]
    },
    "chemical_plastic_bottle": {
        "type": "hazardous",
        "disposal": [
            "1. Triple-rinse and return to designated hazardous waste centers.",
            "2. Store securely until e-waste collection drives.",
            "3. Give to companies specializing in industrial plastic recycling.",
            "4. Label and drop at chemical suppliers for take-back programs.",
            "5. Prevent leaks by sealing caps before disposal."
        ],
        "reuse": [
            "1. After thorough cleaning, use as watering cans for flower gardens.",
            "2. Convert into insect traps by cutting and adding bait.",
            "3. Use for storing non-edible household items like nails or screws.",
            "4. Repurpose as sturdy funnels for oil or detergent.",
            "5. Decorate and turn into durable gift boxes."
        ]
    },
    "chemical_plastic_gallon": {
        "type": "hazardous",
        "disposal": [
            "1. Wash thoroughly and deliver to hazardous waste centers.",
            "2. Repurpose for industrial recycling through companies like Nelplast Ghana.",
            "3. Store away from sunlight to avoid chemical reactions until proper disposal.",
            "4. Partner with local recycling initiatives accepting large plastic containers.",
            "5. Engage in government programs aimed at reducing industrial waste."
        ],
        "reuse": [
            "1. Use as durable buckets for collecting rainwater.",
            "2. Repurpose as protective casings for delicate equipment during transport.",
            "3. Clean and use for storing tools in workshops.",
            "4. Turn into large feeders for livestock on farms.",
            "5. Turn into large-scale murals or displays in community art projects."
        ]
    },
    "chemical_spray_can": {
        "type": "hazardous",
        "disposal": [
            "1. Empty cans can be recycled as metal waste.",
            "2. Label as hazardous waste and deliver to appropriate disposal sites.",
            "3. Use specialized e-waste programs for proper handling.",
            "4. Engage with community clean-up events to collect and manage spray cans.",
            "5. Partner with paint companies offering spray-can recycling programs."
        ],
        "reuse": [
            "1. Convert empty cans into unique art pieces or sculptures.",
            "2. Use as mini storage containers after removing nozzles.",
            "3. Create customized stamps or painting tools for creative projects.",
            "4. Turn into wind chimes or mobiles for decoration.",
            "5. Use as educational props to teach about aerosol dynamics."
        ]
    },
    "light_bulb": {
        "type": "hazardous",
        "disposal": [
            "1. Deliver to light bulb recycling programs in urban areas.",
            "2. Store in secure packaging until e-waste collection drives.",
            "3. Give to artisans who repurpose glass and metal parts.",
            "4. Avoid breaking to minimize mercury release.",
            "5. Recycle through municipal e-waste programs if available."
        ],
        "reuse": [
            "1. Create mini terrariums for indoor plants.",
            "2. Use for decorative oil lamps during festivals or events.",
            "3. Turn into unique flower vases or candle holders.",
            "4. Craft hanging decorations for weddings or cultural gatherings.",
            "5. Repurpose in schools for practical science experiments."
        ]
    },
    "paint_bucket": {
        "type": "recyclable",
        "disposal": [
            "1. Drop empty buckets at industrial waste collection points.",
            "2. Scrape off dried paint and recycle as metal or plastic.",
            "3. Use in construction projects for temporary water storage.",
            "4. Provide to artisans for storage of tools or raw materials.",
            "5. Contribute to community clean-up drives collecting large waste."
        ],
        "reuse": [
            "1. Repurpose as compost bins for organic kitchen waste.",
            "2. Convert into sturdy planters for trees or large plants.",
            "3. Use for mixing concrete or mortar on construction sites.",
            "4. Decorate and use as toy bins for children.",
            "5. Craft into drums for traditional music performances."
        ]
    },
    "plastic_bag": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle at collection points for soft plastics.",
            "2. Donate to organizations making eco-bricks.",
            "3. Use as protective covers for items in storage.",
            "4. Contribute to craft groups weaving plastic mats or bags.",
            "5. Avoid burning to prevent harmful emissions."
        ],
        "reuse": [
            "1. Weave into durable shopping bags or purses.",
            "2. Use for waterproof lining in shelters or storage units.",
            "3. Cut into strips and braid into ropes or decorative strings.",
            "4. Layer for insulating cold storage containers.",
            "5. Use in craft workshops for children’s art projects."
        ]
    },
    "plastic_bottle": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle with local waste collection companies.",
            "2. Contribute to eco-building projects turning bottles into bricks.",
            "3. Offer to schools teaching sustainable practices.",
            "4. Use in DIY community waste management systems.",
            "5. Avoid disposing in open fields to reduce plastic pollution."
        ],
        "reuse": [
            "1. Transform into drip irrigation systems for farms.",
            "2. Cut and use as seedling starters for crops.",
            "3. Make floating devices for small-scale fishing.",
            "4. Fill with sand to create weights for exercise or construction.",
            "5. Use as water dispensers for handwashing stations."
        ]
    },
    "plastic_bottle_cap": {
        "type": "recyclable",
        "disposal": [
            "1. Collect and send to specialized recycling centers for hard plastics.",
            "2. Contribute to craft or school projects as raw materials.",
            "3. Donate to NGOs repurposing plastics into new items.",
            "4. Use in community clean-up drives for sorting small recyclables.",
            "5. Avoid scattering as they can block drainage systems."
        ],
        "reuse": [
            "1. Create mosaics or wall art with colorful patterns.",
            "2. Use as counters for board games or classroom learning tools.",
            "3. Drill holes and repurpose as decorative beads or jewelry.",
            "4. Attach to walls as hooks for hanging lightweight items.",
            "5. Use in craft workshops to teach children about recycling."
        ]
    },
    "plastic_box": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle through facilities that accept rigid plastics.",
            "2. Offer to vendors needing storage for small market goods.",
            "3. Donate to schools for organizing stationery or supplies.",
            "4. Use as part of eco-brick structures in local communities.",
            "5. Avoid incineration to prevent harmful plastic emissions."
        ],
        "reuse": [
            "1. Use for organizing personal items, such as jewelry or tools.",
            "2. Repurpose as containers for traditional herbs or medicines.",
            "3. Turn into mini greenhouses for nurturing seedlings.",
            "4. Use for stacking and storing farm produce like eggs or gari.",
            "5. Paint and decorate as gift boxes for local ceremonies."
        ]
    },
    "plastic_cultery": {
        "type": "non-recyclable",
        "disposal": [
            "1. Drop at hard plastic recycling centers if available.",
            "2. Encourage switching to biodegradable cutlery to reduce waste.",
            "3. Offer to artisans repurposing plastics for decorative crafts.",
            "4. Use for community recycling initiatives focused on hard plastics.",
            "5. Avoid dumping in water bodies where they harm marine life."
        ],
        "reuse": [
            "1. Use in gardening for planting seeds or aerating soil.",
            "2. Repurpose as tools for small DIY repairs around the home.",
            "3. Combine with other materials to create wind chimes or ornaments.",
            "4. Use as makeshift utensils for stirring non-food substances like paint.",
            "5. Include in craft workshops for sculpting and model-building."
        ]
    },
    "plastic_cup": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle through hard plastic collection programs.",
            "2. Donate to local drink vendors for reuse in serving beverages.",
            "3. Participate in plastic take-back initiatives by waste management firms.",
            "4. Avoid burning, as it releases toxic fumes harmful to health.",
            "5. Offer to community craft projects focusing on recycling."
        ],
        "reuse": [
            "1. Cut and use as seedling pots for crops like tomatoes or cocoa.",
            "2. Turn into decorative pen holders for desks.",
            "3. Use as scoops for measuring grains or small items in the kitchen.",
            "4. Paint and stack as modular art pieces for community displays.",
            "5. Repurpose as molds for making small concrete items like candle holders."
        ]
    },
    "plastic_cup_lid": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle with small hard plastics if facilities accept them.",
            "2. Collect and contribute to art programs repurposing plastics.",
            "3. Offer to schools for use in educational projects.",
            "4. Use in community sorting programs for waste management.",
            "5. Avoid littering to prevent clogging of urban drainage systems."
        ],
        "reuse": [
            "1. Use as coasters for small cups or glasses.",
            "2. Repurpose as teaching aids for children learning shapes or colors.",
            "3. Create small decorative items like painted ornaments or mini wall hangings.",
            "4. Stack and glue to create lightweight building blocks for crafts.",
            "5. Use as templates for drawing or cutting circular shapes."
        ]
    },
    "reuseable_paper": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle in clean paper bins or facilities accepting mixed papers.",
            "2. Shred and compost for use in urban gardening or farming.",
            "3. Donate to schools for rough work or practice materials.",
            "4. Provide to artisans for papier-mâché projects or recycled crafts.",
            "5. Offer to vendors wrapping fragile market goods like eggs or glassware."
        ],
        "reuse": [
            "1. Use for gift wrapping or making envelopes.",
            "2. Create DIY paper decorations like origami or garlands.",
            "3. Repurpose into handmade paper for cards or crafts.",
            "4. Layer sheets in gardening beds to suppress weeds.",
            "5. Use as filler material when packaging fragile items."
        ]
    },
    "scrap_paper": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle at facilities accepting mixed or lower-grade papers.",
            "2. Shred and use as mulch or compost material in gardens.",
            "3. Burn in controlled environments for energy generation.",
            "4. Offer to local crafters for upcycling into decorative items.",
            "5. Use in community clean-ups to collect and sort biodegradable waste."
        ],
        "reuse": [
            "1. Use for creating papier-mâché art or handmade paper.",
            "2. Cut into strips for basket weaving or other decorative crafts.",
            "3. Use for packing fragile goods like ceramics or glass.",
            "4. Turn into kindling for starting traditional fires or ovens.",
            "5. Decorate and repurpose as bookmarks or handmade tags."
        ]
    },
    "scrap_plastic": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle through specialized facilities processing mixed plastics.",
            "2. Offer to NGOs turning plastics into eco-friendly building materials.",
            "3. Contribute to eco-brick projects making walls or structures.",
            "4. Sort and donate to local artisans crafting recycled goods.",
            "5. Avoid mixing with food waste to keep the recycling process efficient."
        ],
        "reuse": [
            "1. Melt and mold into small items like coasters or keychains.",
            "2. Use for insulating walls or roofs in informal settlements.",
            "3. Repurpose into makeshift waterproof containers for tools.",
            "4. Turn into parts for DIY crafts or teaching aids.",
            "5. Use for creating small garden tools like plant labels or markers."
        ]
    },
    "snack_bag": {
        "type": "recyclable",
        "disposal": [
            "1. Recycle through soft plastic recycling programs where available.",
            "2. Collect and donate to eco-brick or upcycling initiatives.",
            "3. Avoid burning as it releases harmful toxins into the air.",
            "4. Offer to craft groups making recycled products.",
            "5. Participate in beach or community clean-ups to collect and recycle."
        ],
        "reuse": [
            "1. Use to make wallets, coin purses, or small pouches.",
            "2. Weave into mats, rugs, or shopping bags.",
            "3. Cut and use as decorative panels in art projects.",
            "4. Repurpose for insulating containers to keep items cool.",
            "5. Use in mixed media art to create textured surfaces or collages."
        ]
    },
    "stick": {
        "type": "non-recyclable",
        "disposal": [
            "1. Compost if the stick is natural wood.",
            "2. Burn in controlled environments for firewood or energy.",
            "3. Offer to artisans for carving or traditional crafts.",
            "4. Use as part of community tree-planting projects.",
            "5. Donate to schools for creative projects like making models."
        ],
        "reuse": [
            "1. Use as stakes for supporting plants in gardens.",
            "2. Carve into simple tools like spoons or small handles.",
            "3. Repurpose as firewood for traditional cooking stoves.",
            "4. Use as structural components in small rural fences.",
            "5. Decorate and use as props for traditional dances or ceremonies."
        ]
    },
    "straw": {
        "type": "non-recyclable",
        "disposal": [
            "1. Recycle if accepted locally, especially by plastic collection programs.",
            "2. Donate to artists creating recycled plastic crafts.",
            "3. Avoid throwing into water bodies to protect aquatic life.",
            "4. Use in clean-up events to separate small plastics from general waste.",
            "5. Sort and deliver to upcycling workshops focused on soft plastics."
        ],
        "reuse": [
            "1. Weave into baskets or small mats for decoration.",
            "2. Use in craft projects for making jewelry or toys.",
            "3. Turn into plant markers for identifying garden crops.",
            "4. Combine with other materials to create wind chimes.",
            "5. Use for teaching geometry or building small models in schools."
        ]
    }
}






st.markdown(
    """
    <style>
    /* General body style */
    .stApp {
        background-color: #004d00; /* Deep Green Background */
        color: white;
        font-family: 'Poppins', sans-serif;
        padding: 0;
        margin: 0;
    }

    /* Header with glowing effect */
    h1 {
        text-align: center;
        font-size: 3em;
        color: #76ff03;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0px 0px 15px rgba(118, 255, 3, 0.7);
        animation: floatHeader 2s ease-in-out infinite;
    }

    @keyframes floatHeader {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }

    /* Style for the input fields */
    .stFileUploader, .stSelectbox, .stCameraInput, .stButton>button {
        background-color: #1b5e20;
        color: white;
        border-radius: 15px;
        font-size: 1.2em;
        padding: 12px 25px;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        border: 2px solid #76ff03;
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.3);
    }

    .stFileUploader:hover, .stSelectbox:hover, .stCameraInput:hover, .stButton>button:hover {
        background-color: #4caf50;
        box-shadow: 0px 10px 30px rgba(0, 255, 0, 0.5);
        transform: scale(1.05);
    }

    /* Focused style for the inputs (when clicked) */
    .stFileUploader input:focus, .stSelectbox input:focus, .stCameraInput input:focus, .stButton>button:focus {
        border-color: #76ff03;
        outline: none;
    }

    /* Output text styling */
    .output-text {
        background-color: rgba(1, 50, 32, 0.85);
        color: #76ff03;
        padding: 20px;
        border-radius: 20px;
        border: 3px solid #76ff03;
        font-size: 1.4em;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        animation: floatOutputText 3s ease-in-out infinite;
    }

    @keyframes floatOutputText {
        0% { transform: translateY(0); }
        50% { transform: translateY(10px); }
        100% { transform: translateY(0); }
    }

    /* Image styling */
    .stImage img {
        border-radius: 20px;
        border: 4px solid #76ff03;
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease-in-out;
    }

    .stImage img:hover {
        transform: scale(1.05);
        box-shadow: 0px 10px 30px rgba(0, 255, 0, 0.5);
    }

    /* File Uploader Border */
    .stFileUploader {
        border: 3px solid #76ff03;
        border-radius: 10px;
        padding: 20px;
        background-color: #1b5e20;
    }

    /* Add some margin for spacing */
    .stFileUploader input {
        background-color: #1b5e20;
        border: none;
        padding: 10px;
        color: white;
    }

    /* Label for the file uploader */
    .stFileUploader label {
        font-size: 1.2em;
        font-weight: 600;
        color: #76ff03;
    }

    /* Select Box Border */
    .stSelectbox select {
        border: 2px solid #76ff03;
        border-radius: 10px;
        padding: 10px;
        background-color: #1b5e20;
        color: white;
    }

    .stSelectbox select:hover {
        background-color: #4caf50;
        box-shadow: 0px 10px 30px rgba(0, 255, 0, 0.5);
        transform: scale(1.05);
    }

    /* Camera Input Border */
    .stCameraInput input {
        border: 2px solid #76ff03;
        border-radius: 10px;
        padding: 10px;
        background-color: #1b5e20;
        color: white;
    }

    /* General Button Styling */
    .stButton>button {
        background-color: #76ff03;
        border-radius: 10px;
        color: white;
        font-size: 1.2em;
        font-weight: 600;
        padding: 12px 25px;
        transition: background-color 0.3s ease-in-out;
        border: 2px solid #76ff03;
    }

    .stButton>button:hover {
        background-color: #4caf50;
        box-shadow: 0px 10px 30px rgba(0, 255, 0, 0.5);
        transform: scale(1.05);
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)



# Function to detect waste type from an image
def detect_waste(image):
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()
    
    if len(detections) == 0:
        return "No waste detected", "", "", "", "", ""
    
    # Assuming the first detection is the most relevant
    class_id = int(detections[0][5])
    class_name = results[0].names[class_id]
    
    waste_info = waste_suggestions.get(class_name, {})
    waste_type = waste_info.get("type", "Unknown")
    disposal_methods = "\n".join(waste_info.get("disposal", []))
    reuse_ideas = "\n".join(waste_info.get("reuse", []))
    
    return class_name, waste_type, "Disposal Methods:", disposal_methods, "Reuse Ideas:", reuse_ideas

# Streamlit app
st.title("🌱 Waste Detection and Eco-Friendly Suggestions 🌍")
st.write("Upload an image or take a picture to identify waste and get eco-friendly disposal and reuse suggestions.")

# Option to choose input method
input_method = st.selectbox("Choose Input Method", ["Upload Image", "Take Picture"])

if input_method == "Upload Image":
    # Image input
    image_input = st.file_uploader("Upload Waste Image", type=["jpg", "jpeg", "png"])
    
    if image_input is not None:
        image = Image.open(image_input)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Waste"):
            detected_waste_type, waste_category, disposal_methods_header, disposal_methods, reuse_ideas_header, reuse_ideas = detect_waste(image)
            
            st.write(f"**Detected Waste Type:** {detected_waste_type}")
            st.write(f"**Waste Category:** {waste_category}")
            st.write(f"**{disposal_methods_header}**")
            st.write(disposal_methods)
            st.write(f"**{reuse_ideas_header}**")
            st.write(reuse_ideas)

elif input_method == "Take Picture":
    # Take a picture using Streamlit's camera input
    picture = st.camera_input("Take a Picture")
    
    if picture is not None:
        image = Image.open(picture)
        st.image(image, caption="Captured Image", use_column_width=True)
        
        if st.button("Detect Waste from Picture"):
            detected_waste_type, waste_category, disposal_methods_header, disposal_methods, reuse_ideas_header, reuse_ideas = detect_waste(image)
            
            st.write(f"**Detected Waste Type:** {detected_waste_type}")
            st.write(f"**Waste Category:** {waste_category}")
            st.write(f"**{disposal_methods_header}**")
            st.write(disposal_methods)
            st.write(f"**{reuse_ideas_header}**")
            st.write(reuse_ideas)
