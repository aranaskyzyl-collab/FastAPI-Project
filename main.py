# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
from app.predict import predict_realtime

app = FastAPI()

class StreamFrame(BaseModel):
    image: str

# Structured Advice Dictionary
DISEASE_GUIDE = {
    "Bacterial leaf blight_NSIC Rc 18": {
        "management": "Drain the field immediately to reduce humidity. Avoid excess Nitrogen application.",
        "treatment": "Spray Copper-based fungicides (e.g., Copper Hydroxide) or Bactericides containing Streptomycin."
    },
    "Brown spot_NSIC Rc 18": {
        "management": "Improve soil nutrients. Ensure balanced N-P-K application.",
        "treatment": "Apply Mancozeb or Iprodione if spots are widespread."
    },
    "Brown spot_NSIC Rc 402": {
        "management": "Address Potassium deficiency. Ensure field is not submerged for too long.",
        "treatment": "Foliar spray of Carbendazim or Propiconazole."
    },
    "Brown spot_NSIC Rc 436": {
        "management": "Use clean, certified seeds. Use moderate nitrogen levels.",
        "treatment": "Apply Fungicides like Edifenphos or Benomyl."
    },
    "Leaf blight_NISC Rc 436": {
        "management": "Remove infected straw/stubble from previous harvest. Improve drainage.",
        "treatment": "Copper Oxychloride or validamycin spray."
    },
    "Leaf blight_NSIC Rc 402": {
        "management": "Keep water levels low (2-5cm). Avoid wounding the plants during weeding.",
        "treatment": "Apply Bismerthiazol or Zinc-based treatments."
    },
    "Sheath blight_NSIC Rc 402": {
        "management": "Increase plant spacing to allow airflow. Remove 'weeds' that act as hosts.",
        "treatment": "Use Hexaconazole or Validamycin A targeting the base of the plant."
    },
    "Healthy": {
        "management": "Maintain regular weeding and monitoring.",
        "treatment": "No chemical application needed. Keep it up!"
    }
}

@app.post("/stream")
async def stream_predict(data: StreamFrame):
    try:
        header, encoded = data.image.split(",", 1) if "," in data.image else (None, data.image)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run your CNN-GAT Model
        result = predict_realtime(image)
        
        # Extract disease name from model output
        disease_name = result.get("disease", "Unknown")
        
        # Fetch advice from our guide
        guide = DISEASE_GUIDE.get(disease_name, {
            "management": "Align leaf clearly for better detection.",
            "treatment": "N/A"
        })
        
        # Update result with new fields
        result["management_advice"] = guide["management"]
        result["treatment"] = guide["treatment"]
        
        return result
        
    except Exception as e:
        return {"disease": "Error", "confidence": 0, "management_advice": str(e), "treatment": "N/A"}
