# main.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import base64
import io
from PIL import Image
from app.predict import predict_realtime

app = FastAPI()

class StreamFrame(BaseModel):
    image: str

# ADD THIS: This fixes the "Not Found" error when opening the link
@app.get("/")
async def root():
    return {
        "project": "Hybrid CNN-GAT Rice Disease Detection",
        "status": "Online",
        "endpoint": "/stream (POST only)"
    }

@app.post("/stream")
async def stream_predict(data: StreamFrame):
    try:
        header, encoded = data.image.split(",", 1) if "," in data.image else (None, data.image)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        result = predict_realtime(image)
        
        advice_map = {
            "Bacterial leaf blight_NSIC Rc 18": "Management: Drain field to reduce humidity and stop Nitrogen application.\nTreatment: Spray Bactericides like Copper Hydroxide or Streptomycin Sulfate.",
            "Brown spot_NSIC Rc 18": "Management: Correct soil nutrient deficiencies by applying balanced N-P-K.\nTreatment: Apply fungicides such as Mancozeb or Iprodione if infection persists.",
            "Brown spot_NSIC Rc 402": "Management: Ensure field is not water-stressed; provide proper drainage.\nTreatment: Foliar spray with Propiconazole or Carbendazim to limit spore spread.",
            "Brown spot_NSIC Rc 436": "Management: Use Manganese-rich fertilizers and improve soil quality.\nTreatment: Apply Benomyl or Edifenphos if more than 25% of the leaf area is affected.",
            "Healthy": "Status: Plant is vigorous.\nAdvice: Maintain current irrigation schedule and continue weekly monitoring.",
            "Leaf blight_NISC Rc 436": "Management: Remove weeds that act as alternate hosts and keep water level at 2-3cm.\nTreatment: Use Copper-based fungicides like Copper Oxychloride.",
            "Leaf blight_NSIC Rc 402": "Management: Avoid wounding plants during cultivation and reduce Nitrogen.\nTreatment: Spray Streptomycin sulfate or Bismerthiazol if rapid spreading occurs.",
            "Sheath blight_NSIC Rc 402": "Management: Increase plant spacing for better ventilation and remove infected straw.\nTreatment: Apply systemic fungicides like Hexaconazole or Validamycin A."
        }
        
        result["management"] = advice_map.get(result["disease"], "Scanning... Please align leaf clearly.")
        return result
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"disease": "Error", "confidence": 0, "management": "Check backend connection."}
