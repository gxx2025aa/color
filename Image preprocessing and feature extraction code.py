# utils/image_processing.py
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

class ImagePreprocessor:
    """Image preprocessing class"""
    
    def __init__(self, image_size=256):
        self.image_size = image_size
        
        # Image conversion pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.grayscale_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def rgb_to_opponent(self, image_tensor):
        """
       Convert RGB image to opposite color space
        Args:
            image_tensor: [batch_size, 3, H, W]
        Returns:
            opponent_tensor: [batch_size, 3, H, W]
        """
        r, g, b = image_tensor[:, 0:1, :, :], image_tensor[:, 1:2, :, :], image_tensor[:, 2:3, :, :]
        
        # Opposite color space conversion
        o1 = (r - g) / np.sqrt(2)  # Red-green
        o2 = (r + g - 2 * b) / np.sqrt(6)  # Yellow-Blue 
        o3 = (r + g + b) / np.sqrt(3)  # brightness
        
        return torch.cat([o1, o2, o3], dim=1)
    
    def opponent_to_rgb(self, opponent_tensor):
        """
        Convert the opposite color space back to RGB
        Args:
            opponent_tensor: [batch_size, 3, H, W]
        Returns:
            rgb_tensor: [batch_size, 3, H, W]
        """
        o1, o2, o3 = opponent_tensor[:, 0:1, :, :], opponent_tensor[:, 1:2, :, :], opponent_tensor[:, 2:3, :, :]
        
        r = (o1 / np.sqrt(2)) + (o2 / np.sqrt(6)) + (o3 / np.sqrt(3))
        g = (-o1 / np.sqrt(2)) + (o2 / np.sqrt(6)) + (o3 / np.sqrt(3))
        b = (-2 * o2 / np.sqrt(6)) + (o3 / np.sqrt(3))
        
        return torch.cat([r, g, b], dim=1)
    
    def extract_color_features(self, image_tensor):
        """
        Extract color features
        Args:
            image_tensor: [batch_size, 3, H, W]
        Returns:
            color_features: [batch_size, feature_dim]
        """
        # Convert to HSV space to extract color features
        image_np = (image_tensor.cpu().numpy().transpose(0, 2, 3, 1) * 127.5 + 127.5).astype(np.uint8)
        
        color_features = []
        for img in image_np:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Calculate color histogram features
            hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
            
            feature = np.concatenate([hist_h, hist_s, hist_v])
            color_features.append(feature)
        
        return torch.tensor(np.array(color_features), dtype=torch.float32)
    
    def adjust_brightness_contrast(self, image_tensor, brightness=0, contrast=0):
        """
        Adjust brightness and contrast
        Args:
            image_tensor: Input image tensor
            brightness: Brightness adjustment value
            contrast: Contrast adjustment value
        """
        if brightness != 0:
            image_tensor = image_tensor + brightness
            
        if contrast != 0:
            image_tensor = image_tensor * (1 + contrast)
            
        return torch.clamp(image_tensor, -1, 1)

# utils/dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class AdvertisingPatternDataset(Dataset):
    """Advertising pattern data set"""
    
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get all image files
        self.image_files = []
        for file in os.listdir(data_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(os.path.join(data_dir, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        # Load Image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Generate grayscale image
        grayscale = torch.mean(image, dim=0, keepdim=True).repeat(3, 1, 1)
        
        return {
            'grayscale': grayscale,
            'color': image,
            'file_path': image_path
        }

def create_data_loaders(data_dir, batch_size=16, image_size=256):
    """Create Data Loader"""
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = AdvertisingPatternDataset(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    return dataloader