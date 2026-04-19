import torch
import json
import os
import sys
from .hybrid_model import HybridCNNGAT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")

LABEL_PATH = os.path.join(MODEL_DIR, "classes.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_hybrid_cnn_gat.pth")

# --- Load Labels ---
with open(LABEL_PATH, "r") as f:
    CLASS_LABELS = json.load(f)
    if isinstance(CLASS_LABELS, list):
        CLASS_LABELS = {str(i): label for i, label in enumerate(CLASS_LABELS)}

# --- Initialize Model ---
model = HybridCNNGAT(num_classes=len(CLASS_LABELS))
model.to(DEVICE)

# --- DEBUG: Weight Check (Before) ---
print("\n--- WEIGHT VERIFICATION ---")
try:
    initial_weights = model.classifier[1].weight[0][:5].clone().detach()
    print(f"Weights BEFORE loading: {initial_weights}")
except Exception as e:
    print(f"Could not print initial weights: {e}")

# --- Load Weights ---
try:
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        if name == "edge_index" or "num_batches_tracked" in name:
            continue
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    
    # --- DEBUG: Weight Check (After) ---
    loaded_weights = model.classifier[1].weight[0][:5].detach()
    print(f"Weights AFTER loading:  {loaded_weights}")

    # Compare them
    if torch.equal(initial_weights, loaded_weights):
        print("⚠️ WARNING: Weights did NOT change! The model is not learning from your file.")
    else:
        print("✅ SUCCESS: Weights updated. The model has loaded your training data.")
    print("---------------------------\n")

except Exception as e:
    print(f"❌ Critical Error during loading: {e}")
    sys.exit(1)

model.eval()