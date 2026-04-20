from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware # Add this
from pydantic import BaseModel
import base64
import io
from PIL import Image
from app.predict import predict_realtime

app = FastAPI()

# --- FIX 1: Add CORS for Flutter Web/Android/iOS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)

class StreamFrame(BaseModel):
    image: str

# --- FIX 2: Add a Home Route ---
@app.get("/")
async def root():
    return {"status": "success", "message": "Rice Disease API is Running"}

@app.post("/stream")
async def stream_predict(data: StreamFrame):
    try:
        # Your existing logic...
        header, encoded = data.image.split(",", 1) if "," in data.image else (None, data.image)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        result = predict_realtime(image)
        
        advice_map = {
            "Bacterial leaf blight_NSIC Rc 18": "Variety Rc18: Drain fields, apply Potash (MOP), and avoid top-dressing Nitrogen during outbreaks.",
            "Brown spot_NSIC Rc 18": "Variety Rc18: Improve soil fertility. Apply balanced N-P-K fertilizer and check for soil acidity.",
            "Brown spot_NSIC Rc 402": "Variety Rc402: Apply Manganese-rich fertilizer. Ensure proper drainage to reduce humidity.",
            "Brown spot_NSIC Rc 436": "Variety Rc436: Use certified seeds for next season. Apply fungicides like Mancozeb if spots cover 25% of leaf.",
            "Healthy": "Plant looks strong! Maintain current irrigation and monitor for pests weekly.",
            "Leaf blight_NISC Rc 436": "Variety Rc436: Use Copper-based fungicides. Keep water level at 2-3cm and remove weeds nearby.",
            "Leaf blight_NSIC Rc 402": "Variety Rc402: Reduce Nitrogen usage. Spray Streptomycin sulfate if infection spreads rapidly.",
            "Sheath blight_NSIC Rc 402": "Variety Rc402: Increase plant spacing for better airflow. Apply Hexaconazole or Carbendazim at the base."
        }
        
        result["management"] = advice_map.get(result["disease"], "Scanning... Please align leaf clearly.")
        return result
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"disease": "Error", "confidence": 0, "management": f"Error: {str(e)}"}
