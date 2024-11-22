# test.py
import torch
import glob
from train import SimpleCNN

# Load the latest model file dynamically
model_files = glob.glob("model_*.pt")
if not model_files:
    raise FileNotFoundError("No model file found. Please run train.py first.")
latest_model = max(model_files, key=lambda x: x)

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load(latest_model))
model.eval()

# Test if the model has 10 output units
assert model.fc2.out_features == 10, f"Model output shape is {model.fc2.out_features}, expected 10."

# Test if the model can process 28x28 input without any issues
input_shape = (1, 1, 28, 28)
test_input = torch.randn(input_shape)
try:
    model(test_input)
    print("Model successfully processed 28x28 input.")
except Exception as e:
    raise AssertionError(f"Model failed to process 28x28 input: {e}")