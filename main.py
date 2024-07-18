import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from train import train
from evaluate import evaluate
from model import KAN
from torch.multiprocessing import cpu_count

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Main execution
def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    input_dim = 224 * 224 * 3  # For 224x224 RGB images
    hidden_dim = 256
    output_dim = 2  # Number of classes
    model_save_path = 'kan_model.pth'

    # Multiprocessing settings
    num_workers = cpu_count()  # Use all available CPU cores
    pin_memory = True  # Enables faster data transfer to CUDA devices

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load custom dataset
    train_dataset = CustomDataset(root_dir=r"D:\windowsps_backup\weapon_det_13_9\train_cls_weapon\train", transform=transform)
    test_dataset = CustomDataset(root_dir=r"D:\windowsps_backup\weapon_det_13_9\train_cls_weapon\val", transform=transform)
    print("Image Transformation done")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    print("Image dataset done")
    # Initialize the model
    model = KAN(input_dim, hidden_dim, output_dim).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    # for epoch in range(num_epochs):
    #     train(model, train_loader, criterion, optimizer, device)
    #     accuracy = evaluate(model, test_loader, device)
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}')
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


    save_model(model, model_save_path)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()