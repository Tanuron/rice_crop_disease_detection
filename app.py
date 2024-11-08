from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import traceback
# from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import resnet50, ResNet50_Weights

app = Flask(__name__)
CORS(app)  # Allow CORS

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        # Load the pre-trained ResNet-50 model
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze the parameters in the base layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer to match the number of classes
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

# Load the dataset to get class names
dataset = datasets.ImageFolder(root=r'C:\Users\DELL\Documents\miniproject\RiceLeafDiseaseImages')
class_names = dataset.classes

# Load the model
try:
    model = CustomResNet50(num_classes=len(class_names))
    model.load_state_dict(torch.load(r"C:\Users\DELL\Documents\rice\models\myresnetmodel.pth", map_location=torch.device('cpu')), strict=False)
    model.eval()
except Exception as e:
    print("Error loading model:", str(e))
    traceback.print_exc()

# Define the transformation
transforms = transforms.Compose([
    transforms.Resize(556),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
@app.route('/')
def home():
    return "Welcome to the Rice Prediction API. Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    try:
        img = Image.open(file.stream)
        img = transforms(img)
        img = img.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(img)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        prediction_index = predicted.item()
        prediction_name = class_names[prediction_index]
        confidence_percentage = confidence.item() * 100

        return jsonify({'prediction': prediction_name, 'confidence': confidence_percentage})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
