import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from app.model_loader import model, CLASS_LABELS, DEVICE

# Optimized transformation for rice leaf analysis
# We resize to 256 first and then crop to 224 to maintain the aspect ratio 
# of the lesions, preventing the "squishing" that lowers confidence.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_realtime(image: Image.Image):
    """
    Performs inference on a single leaf image using the Hybrid CNN-GAT model.
    """
    try:
        # CRITICAL: Set model to evaluation mode. 
        # This disables dropout and uses running averages for Batchnorm.
        # Without this, confidence often drops to 40-50%.
        model.eval()

        # Ensure image is in RGB (removes alpha channel from PNGs if present)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess and move to the correct device (CPU or CUDA)
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Disable gradient calculation for faster inference and lower memory usage
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Apply Softmax to convert raw logits into probability percentages
            probabilities = F.softmax(outputs, dim=1)
            
            # Get the highest probability and its corresponding class index
            confidence, predicted = torch.max(probabilities, 1)

        # Convert tensor index to string to match CLASS_LABELS keys
        idx = str(predicted.item())
        label = CLASS_LABELS.get(idx, f"Unknown Class {idx}")

        # Return results formatted for your Flutter/Web frontend
        return {
            "disease": label,
            "confidence": round(confidence.item() * 100, 1),
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {
            "disease": "Error during detection",
            "confidence": 0.0,
            "status": "error",
            "message": str(e)
        }