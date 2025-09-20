"""
Segmentation Methods

This module contains various image segmentation methods including neural network-based approaches.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from sklearn.cluster import KMeans
from PIL import Image

from .models import UNet, DeepLabV3Segmenter


class SegmentationMethods:
    """Container class for various segmentation methods."""
    
    def __init__(self, device):
        self.device = device
        self.unet_model = None
        self.deeplabv3_model = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def initialize_unet(self):
        """Initialize U-Net model."""
        print("Initializing U-Net model...")
        self.unet_model = UNet(n_channels=3, n_classes=2).to(self.device)
        # Initialize with random weights for demonstration
        # In practice, you would load pre-trained weights here
        
    def initialize_deeplabv3(self):
        """Initialize DeepLabV3 model."""
        print("Initializing DeepLabV3 model...")
        self.deeplabv3_model = DeepLabV3Segmenter(num_classes=2).to(self.device)
        self.deeplabv3_model.eval()
    
    def segment_with_unet(self, image_tensor, original_size):
        """Segment image using U-Net."""
        if self.unet_model is None:
            self.initialize_unet()
        
        self.unet_model.eval()
        with torch.no_grad():
            output = self.unet_model(image_tensor)
            output = torch.softmax(output, dim=1)
            
            # Get the segmentation mask (argmax)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Convert to two-color image
            segmented = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            segmented[mask == 0] = [0, 0, 0]      # Background - black
            segmented[mask == 1] = [255, 255, 255]  # Foreground - white
            
            # Resize back to original size
            segmented = cv2.resize(segmented, original_size)
            
        return segmented
    
    def segment_with_deeplabv3(self, image_tensor, original_size):
        """Segment image using DeepLabV3."""
        if self.deeplabv3_model is None:
            self.initialize_deeplabv3()
        
        self.deeplabv3_model.eval()
        with torch.no_grad():
            output = self.deeplabv3_model(image_tensor)['out']
            output = torch.softmax(output, dim=1)
            
            # Get the segmentation mask
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Convert to two-color image
            segmented = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            segmented[mask == 0] = [0, 0, 0]      # Background - black
            segmented[mask == 1] = [255, 255, 255]  # Foreground - white
            
            # Resize back to original size
            segmented = cv2.resize(segmented, original_size)
            
        return segmented
    
    def segment_with_advanced_clustering(self, image_np):
        """Advanced clustering-based segmentation with neural network features."""
        # Convert to tensor for feature extraction
        image_pil = Image.fromarray(image_np)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Use pre-trained ResNet features
        resnet = models.resnet50(pretrained=True).to(self.device)
        resnet.eval()
        
        # Remove the final classification layer
        feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = F.adaptive_avg_pool2d(features, (32, 32))
            features = features.squeeze().cpu().numpy()
        
        # Reshape for clustering
        h, w = features.shape[1], features.shape[2]
        features_flat = features.reshape(features.shape[0], -1).T
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_flat)
        
        # Reshape back to spatial dimensions
        mask = labels.reshape(h, w)
        
        # Resize to original image size
        mask_resized = cv2.resize(mask.astype(np.uint8), 
                                 (image_np.shape[1], image_np.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Convert to two-color image
        segmented = np.zeros_like(image_np)
        segmented[mask_resized == 0] = [0, 0, 0]      # Background - black
        segmented[mask_resized == 1] = [255, 255, 255]  # Foreground - white
        
        return segmented
    
    def segment_with_grabcut_neural(self, image_np):
        """GrabCut algorithm enhanced with neural network guidance."""
        try:
            # Use neural network to get initial mask
            image_pil = Image.fromarray(image_np)
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            # Get rough segmentation using pre-trained model
            if self.deeplabv3_model is None:
                self.initialize_deeplabv3()
            
            with torch.no_grad():
                output = self.deeplabv3_model(image_tensor)['out']
                rough_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()
                rough_mask = cv2.resize(rough_mask.astype(np.uint8), 
                                      (image_np.shape[1], image_np.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Check if we have both foreground and background pixels
            unique_values = np.unique(rough_mask)
            if len(unique_values) < 2:
                # If neural network didn't segment properly, use simple thresholding
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                _, rough_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Initialize GrabCut mask
            mask = np.full(image_np.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
            
            # Set definite foreground and background based on rough mask
            mask[rough_mask == 1] = cv2.GC_PR_FGD  # Probable foreground
            mask[rough_mask == 0] = cv2.GC_PR_BGD  # Probable background
            
            # Ensure we have some definite foreground and background pixels
            coords = np.where(rough_mask == 1)
            if len(coords[0]) > 0:
                # Add some definite foreground pixels in the center of detected regions
                y_center = int(np.mean(coords[0]))
                x_center = int(np.mean(coords[1]))
                mask[max(0, y_center-5):min(mask.shape[0], y_center+5), 
                     max(0, x_center-5):min(mask.shape[1], x_center+5)] = cv2.GC_FGD
                
                # Add some definite background pixels at the edges
                mask[:10, :] = cv2.GC_BGD
                mask[-10:, :] = cv2.GC_BGD
                mask[:, :10] = cv2.GC_BGD
                mask[:, -10:] = cv2.GC_BGD
                
                # Create rectangle around the probable foreground
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                rect = (x_min, y_min, x_max - x_min, y_max - y_min)
            else:
                # Fallback rectangle and mask setup
                h, w = image_np.shape[:2]
                rect = (w//4, h//4, w//2, h//2)
                mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD
                mask[:h//4, :] = cv2.GC_BGD
                mask[3*h//4:, :] = cv2.GC_BGD
                mask[:, :w//4] = cv2.GC_BGD
                mask[:, 3*w//4:] = cv2.GC_BGD
            
            # Apply GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(image_np, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            # Create final mask
            final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
            
            # Convert to two-color image
            segmented = np.zeros_like(image_np)
            segmented[final_mask == 0] = [0, 0, 0]      # Background - black
            segmented[final_mask == 1] = [255, 255, 255]  # Foreground - white
            
            return segmented
            
        except Exception as e:
            print(f"    GrabCut failed, falling back to simple binary segmentation: {e}")
            # Fallback to simple thresholding
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            segmented = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            return segmented
