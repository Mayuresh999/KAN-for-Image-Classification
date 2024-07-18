from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.file_list = self._create_file_list()

    def _create_file_list(self):
        file_list = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in tqdm(os.listdir(class_dir), desc="Getting files per class"):
                file_list.append((os.path.join(class_dir, filename), class_idx))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label