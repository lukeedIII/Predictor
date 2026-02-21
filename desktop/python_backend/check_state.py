import pickle
with open('f:/Predictor/desktop/python_backend/models/ensemble_state_v3.pkl', 'rb') as f:
    state = pickle.load(f)
print(f"Items: {state.keys()}")
print(f"Last Validation Accuracy: {state.get('last_validation_accuracy')}")
