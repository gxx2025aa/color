# utils/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_psnr(original, generated):
    """
    Calculate peak signal-to-noise ratio (PSNR) - formula9
    Args:
        original: original image [H, W, C] or [B, C, H, W]
        generated: Generate image [H, W, C] or [B, C, H, W]
    Returns:
        psnr_value: PSNRvalue
    """
    if len(original.shape) == 3:
        original = original.permute(2, 0, 1).unsqueeze(0)
        generated = generated.permute(2, 0, 1).unsqueeze(0)
    
    mse = F.mse_loss(original, generated)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0  # The image has been normalized to[0,1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(original, generated):
    """
   Calculate structural similarity index (SSIM) - formula10, 11
    Args:
        original: original image [H, W, C]
        generated: Generate image [H, W, C]
    Returns:
        ssim_value: SSIMvalue
    """
    # Convert to numpy array
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(generated):
        generated = generated.cpu().numpy()
    
    # Make sure the image is within [0,1]
    original = np.clip(original, 0, 1)
    generated = np.clip(generated, 0, 1)
    
    # Calculate multichannelSSIM
    if len(original.shape) == 3 and original.shape[2] == 3:
        ssim_values = []
        for channel in range(3):
            channel_ssim = ssim(original[:, :, channel], 
                               generated[:, :, channel],
                               data_range=1.0)
            ssim_values.append(channel_ssim)
        return np.mean(ssim_values)
    else:
        return ssim(original, generated, data_range=1.0)

def calculate_color_difference(image1, image2, method='ciede2000'):
    """
    Calculate color differences
    Args:
        image1, image2: input image [H, W, C]
        method: Color difference calculation method
    Returns:
        color_diff: Average color difference
    """
    if torch.is_tensor(image1):
        image1 = (image1.cpu().numpy() * 255).astype(np.uint8)
    if torch.is_tensor(image2):
        image2 = (image2.cpu().numpy() * 255).astype(np.uint8)
    
    # Convert to Lab color space
    lab1 = cv2.cvtColor(image1, cv2.COLOR_RGB2Lab)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_RGB2Lab)
    
    # calculate Delta E
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
    
    return np.mean(delta_e)

def calculate_perceptual_similarity(original, generated):
    """
    Calculate perceived similarity index (PSI)
    Args:
        original: original image
        generated: Generate image
    Returns:
        psi_value: PSIvalue
    """
    # Using LPIPS or other perceived similarity measures
    # SSIM is used here as an alternative
    return calculate_ssim(original, generated)

def evaluate_coloring_quality(original_images, generated_images):
    """
    Comprehensive evaluation of coloring quality
    Args:
        original_images: Original image list
        generated_images: Generate image list
    Returns:
        metrics_dict: Evaluation index dictionary
    """
    psnr_values = []
    ssim_values = []
    color_diff_values = []
    psi_values = []
    
    for orig, gen in zip(original_images, generated_images):
        psnr_values.append(calculate_psnr(orig, gen))
        ssim_values.append(calculate_ssim(orig, gen))
        color_diff_values.append(calculate_color_difference(orig, gen))
        psi_values.append(calculate_perceptual_similarity(orig, gen))
    
    return {
        'PSNR_mean': np.mean(psnr_values),
        'PSNR_std': np.std(psnr_values),
        'SSIM_mean': np.mean(ssim_values),
        'SSIM_std': np.std(ssim_values),
        'Color_Difference_mean': np.mean(color_diff_values),
        'Color_Difference_std': np.std(color_diff_values),
        'PSI_mean': np.mean(psi_values),
        'PSI_std': np.std(psi_values)
    }

# training/trainer.py
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os

class MANNsStyleGANTrainer:
    """MANNs-StyleGAN Trainer"""
    
    def __init__(self, model, discriminator, train_loader, val_loader, device):
        self.model = model
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # optimizer
        self.g_optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # loss function 
        self.adversarial_loss = torch.nn.BCELoss()
        self.reconstruction_loss = torch.nn.L1Loss()
        
        # Recorder
        self.writer = SummaryWriter('logs/manns_stylegan')
        
    def train_epoch(self, epoch):
        """train an epoch"""
        self.model.train()
        self.discriminator.train()
        
        for batch_idx, batch in enumerate(self.train_loader):
            grayscale = batch['grayscale'].to(self.device)
            color = batch['color'].to(self.device)
            batch_size = grayscale.size(0)
            
            # True and false labels
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # Training discriminator
            self.d_optimizer.zero_grad()
            
            # Real image loss
            real_output = self.discriminator(color)
            d_loss_real = self.adversarial_loss(real_output, real_labels)
            
            # Generated image loss
            fake_images, _, _ = self.model(grayscale)
            fake_output = self.discriminator(fake_images.detach())
            d_loss_fake = self.adversarial_loss(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()
            
            # Training generator
            self.g_optimizer.zero_grad()
            
            fake_output = self.discriminator(fake_images)
            g_loss_adv = self.adversarial_loss(fake_output, real_labels)
            g_loss_rec = self.reconstruction_loss(fake_images, color)
            
            g_loss = g_loss_adv + 100 * g_loss_rec  # Reconstruction of loss weight
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Record loss
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] '
                      f'D_loss: {d_loss.item():.6f} G_loss: {g_loss.item():.6f}')
                
                self.writer.add_scalar('Loss/Discriminator', d_loss.item(), 
                                     epoch * len(self.train_loader) + batch_idx)
                self.writer.add_scalar('Loss/Generator', g_loss.item(),
                                     epoch * len(self.train_loader) + batch_idx)
    
    def validate(self, epoch):
        """validation model"""
        self.model.eval()
        val_metrics = {
            'psnr': [], 'ssim': [], 'color_diff': []
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                grayscale = batch['grayscale'].to(self.device)
                color = batch['color'].to(self.device)
                
                fake_images, _, _ = self.model(grayscale)
                
                # Calculate evaluation index
                for i in range(fake_images.size(0)):
                    orig = color[i].cpu()
                    gen = fake_images[i].cpu()
                    
                    val_metrics['psnr'].append(calculate_psnr(orig, gen))
                    val_metrics['ssim'].append(calculate_ssim(orig, gen))
                    val_metrics['color_diff'].append(calculate_color_difference(orig, gen))
        
        # Record average index
        for metric_name, values in val_metrics.items():
            avg_value = np.mean(values)
            self.writer.add_scalar(f'Validation/{metric_name}', avg_value, epoch)
            print(f'Validation {metric_name}: {avg_value:.4f}')
        
        return val_metrics
    
    def train(self, epochs):
        """complete training process"""
        for epoch in range(epochs):
            start_time = time.time()
            
            self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            # Save model
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
    
    def save_checkpoint(self, epoch, metrics):
        """save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'metrics': metrics
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/manns_stylegan_epoch_{epoch}.pth')