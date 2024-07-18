import torch
from PIL import Image
from model import KAN
import torchvision.transforms as transforms


input_dim = 224 * 224 * 3  # For 224x224 RGB images
hidden_dim = 256
output_dim = 2  # Number of classes
model_save_path = 'kan_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer(model, image_path, device, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
    
    return predicted.item()


test_image_path = r"D:\windowsps_backup\weapon_det_13_9\oth_for_FE\AM_15_c3578ef2e982bd48_13451.jpg"
loaded_model = load_model(KAN(input_dim, hidden_dim, output_dim), model_save_path, device)
prediction = infer(loaded_model, test_image_path, device, transform)
print(f"Predicted class for test image: {prediction}")