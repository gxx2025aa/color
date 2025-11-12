# model/manns_stylegan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MemoryAugmentedNetwork(nn.Module):
    """Memory enhancement neural network module"""
    def __init__(self, memory_size=512, memory_dim=256):
        super(MemoryAugmentedNetwork, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Memory matrix
        self.memory_keys = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Initialization memory
        nn.init.kaiming_uniform_(self.memory_keys)
        nn.init.kaiming_uniform_(self.memory_values)
        
    def forward(self, query, scene_threshold=0.8):
        """
        Memory query process
        Args:
            query: Query characteristics [batch_size, feature_dim]
            scene_threshold: Scene detection threshold
        """
        batch_size = query.size(0)
        
        # Calculate cosine similarity
        query_norm = F.normalize(query, p=2, dim=1)
        memory_norm = F.normalize(self.memory_keys, p=2, dim=1)
        
        similarity = torch.matmul(query_norm, memory_norm.t())  # [batch_size, memory_size]
        
        # Get the most similar memory item
        weights = F.softmax(similarity * 10, dim=1)  # Temperature parameters=10
        output = torch.matmul(weights, self.memory_values)  # [batch_size, memory_dim]
        
        # Scene detection（formula7）
        if batch_size > 1:
            scene_diff = torch.norm(query[1:] - query[:-1], dim=1)
            same_scene = scene_diff < scene_threshold
        else:
            same_scene = torch.tensor([True])
            
        return output, weights, same_scene

class StyleGANGenerator(nn.Module):
    """Improved StyleGAN generator"""
    def __init__(self, latent_dim=512, image_size=256):
        super(StyleGANGenerator, self).__init__()
        
        # Map Network
        self.mapping_network = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )
        
        # Synthetic network
        self.synthesis_network = nn.ModuleList([
            self._make_synthesis_block(512, 512, 4),
            self._make_synthesis_block(512, 256, 8),
            self._make_synthesis_block(256, 128, 16),
            self._make_synthesis_block(128, 64, 32),
            self._make_synthesis_block(64, 32, 64),
            self._make_synthesis_block(32, 16, 128),
            self._make_synthesis_block(16, 8, 256),
        ])
        
        self.to_rgb = nn.Conv2d(8, 3, 1)
        
    def _make_synthesis_block(self, in_channels, out_channels, size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, latent_code):
        # Style vector extraction（formula2）
        style_vector = self.mapping_network(latent_code)
        
        # Initial constant input
        batch_size = latent_code.size(0)
        x = torch.ones(batch_size, 512, 4, 4).to(latent_code.device)
        
        # Forward propagation of synthetic network
        for layer in self.synthesis_network:
            x = layer(x)
            
        # Convert to RGB
        output = self.to_rgb(x)
        output = torch.tanh(output)  # Normalized to[-1,1]
        
        return output

class MANNsStyleGAN(nn.Module):
    """Complete MANNS StyleGAN model"""
    def __init__(self, latent_dim=512, image_size=256, memory_size=512):
        super(MANNsStyleGAN, self).__init__()
        
        # Feature extraction network
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last full connection layer
        
        # Memory network
        self.memory_network = MemoryAugmentedNetwork(
            memory_size=memory_size, 
            memory_dim=512
        )
        
        # StyleGAN generator
        self.generator = StyleGANGenerator(latent_dim, image_size)
        
        # Color feature extraction network
        self.color_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256)
        )
        
    def forward(self, grayscale_image, color_hint=None):
        # Extract spatial features
        spatial_features = self.resnet(grayscale_image)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        
        # Memory query
        memory_output, attention_weights, same_scene = self.memory_network(spatial_features)
        
        # Generate potential code by combining memory characteristics
        combined_features = spatial_features + memory_output
        latent_code = combined_features
        
        # Generate Color Image
        colored_image = self.generator(latent_code)
        
        return colored_image, attention_weights, same_scene

class Discriminator(nn.Module):
    """Discriminator network"""
    def __init__(self, image_size=256):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            # input: 3 x 256 x 256
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 128 x 128
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 16 x 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=0),  # 13 x 13
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)