import logging
from predictor import NexusPredictor

logging.basicConfig(level=logging.INFO)

print("Forcing a retraining of the entire Predictor ensemble (XGBoost + LSTM)...")
pred = NexusPredictor()

# Force standard train loop
success, progress, promotion = pred.train()

print(f"Train success: {success}")
print(f"Promotion details: {promotion}")
