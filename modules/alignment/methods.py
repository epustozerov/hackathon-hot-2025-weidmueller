"""
Object Alignment Methods

This module provides methods for separating and aligning objects from segmented images.
It can handle both binary and colored segmented images to extract individual objects.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os


def calculate_rotation_angle(image_shape: tuple) -> float:
    """
    Calculate the rotation angle to orient the largest diagonal horizontally.
    
    Args:
        image_shape: Tuple of (height, width) of the image
        
    Returns:
        Rotation angle in degrees
    """
    height, width = image_shape[:2]
    
    # Calculate the angle of the diagonal from bottom-left to top-right
    # This diagonal goes from (0, height) to (width, 0)
    diagonal_angle = np.degrees(np.arctan2(height, width))
    
    # To make the diagonal horizontal, we need to rotate by -diagonal_angle
    # This will make the diagonal parallel to the x-axis
    rotation_angle = -diagonal_angle
    
    return rotation_angle


def rotate_image_to_horizontal_diagonal(image: np.ndarray) -> np.ndarray:
    """
    Rotate an image so that its largest diagonal is horizontally oriented.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Rotated image
    """
    if image is None or image.size == 0:
        return image
    
    # Calculate rotation angle
    rotation_angle = calculate_rotation_angle(image.shape)
    
    # Get image center
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Calculate new image dimensions to avoid cropping
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))
    
    # Adjust rotation matrix for new center
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated_image


class ObjectAlignment:
    """Class for aligning objects from segmented images."""
    
    def __init__(self, output_dir: str = "data/output/aligned_objects", rotate_image: bool = False):
        """
        Initialize the ObjectAlignment class.
        
        Args:
            output_dir: Directory to save aligned objects
            rotate_image: Whether to rotate images so largest diagonal is horizontal
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_image = rotate_image
        
        # Create subdirectories for different types
        self.binary_dir = self.output_dir / "binary_objects"
        self.colored_dir = self.output_dir / "colored_objects"
        self.original_dir = self.output_dir / "original_objects"
        
        for dir_path in [self.binary_dir, self.colored_dir, self.original_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def separate_objects_from_binary(self, mask_path: Path, original_path: Path = None) -> List[Dict]:
        """
        Separate objects from a binary mask and extract corresponding pixels from original image.
        
        Args:
            mask_path: Path to the binary mask image
            original_path: Path to the original image (if None, uses mask_path with .jpg extension)
            
        Returns:
            List of dictionaries containing object information
        """
        print(f"🔍 Separating objects from binary mask: {mask_path.name}")
        
        # Load mask image
        mask_img = cv2.imread(str(mask_path))
        if mask_img is None:
            print(f"❌ Could not load mask image: {mask_path}")
            return []
        
        # Determine original image path
        if original_path is None:
            # Try to find corresponding original image
            original_name = mask_path.stem.replace('_sam_points_binary', '') + '.jpg'
            original_path = mask_path.parent / original_name
            
        if not original_path.exists():
            print(f"❌ Original image not found: {original_path}")
            return []
        
        # Load original image
        original_img = cv2.imread(str(original_path))
        if original_img is None:
            print(f"❌ Could not load original image: {original_path}")
            return []
        
        print(f"📸 Using original image: {original_path.name}")
        
        # Convert mask to grayscale and create binary mask
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find connected components in the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        objects = []
        print(f"📊 Found {num_labels - 1} objects (excluding background)")
        
        # Process each object (skip background label 0)
        for i in range(1, num_labels):
            # Get object mask
            object_mask = (labels == i).astype(np.uint8) * 255
            
            # Get object statistics
            x, y, w, h, area = stats[i]
            centroid = centroids[i]
            
            # Skip very small objects
            if area < 100:
                print(f"   ⏭️  Skipping small object {i} (area: {area})")
                continue
            
            # Extract object from original image using the mask
            object_img = original_img[y:y+h, x:x+w].copy()
            object_mask_cropped = object_mask[y:y+h, x:x+w]
            
            # Apply mask to object image (set background to black)
            object_img[object_mask_cropped == 0] = [0, 0, 0]
            
            # Create object info
            object_info = {
                'id': i,
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': centroid,
                'image': object_img,
                'mask': object_mask_cropped,
                'source_image': mask_path.name,
                'original_image': original_path.name
            }
            
            objects.append(object_info)
            print(f"   ✅ Object {i}: area={area}, bbox=({x},{y},{w},{h})")
        
        return objects
    
    def separate_objects_from_colored(self, image_path: Path) -> List[Dict]:
        """
        Separate objects from a colored segmented image.
        
        Args:
            image_path: Path to the colored segmented image
            
        Returns:
            List of dictionaries containing object information
        """
        print(f"🔍 Separating objects from colored image: {image_path.name}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return []
        
        # Convert to grayscale for connected components analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create binary mask (non-black pixels)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        objects = []
        print(f"📊 Found {num_labels - 1} objects (excluding background)")
        
        # Process each object (skip background label 0)
        for i in range(1, num_labels):
            # Get object mask
            object_mask = (labels == i).astype(np.uint8) * 255
            
            # Get object statistics
            x, y, w, h, area = stats[i]
            centroid = centroids[i]
            
            # Skip very small objects
            if area < 100:
                print(f"   ⏭️  Skipping small object {i} (area: {area})")
                continue
            
            # Extract object bounding box
            object_img = img[y:y+h, x:x+w]
            object_mask_cropped = object_mask[y:y+h, x:x+w]
            
            # Create object info
            object_info = {
                'id': i,
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': centroid,
                'image': object_img,
                'mask': object_mask_cropped,
                'source_image': image_path.name
            }
            
            objects.append(object_info)
            print(f"   ✅ Object {i}: area={area}, bbox=({x},{y},{w},{h})")
        
        return objects
    
    def align_object(self, object_info: Dict, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Align a single object by centering and resizing.
        
        Args:
            object_info: Dictionary containing object information
            target_size: Target size for the aligned object (width, height)
            
        Returns:
            Aligned object image
        """
        img = object_info['image']
        mask = object_info['mask']
        
        # Apply rotation if enabled
        if self.rotate_image:
            img = rotate_image_to_horizontal_diagonal(img)
        
        # Create a new image with the object centered
        aligned_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        aligned_mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        
        # Calculate scaling to fit object in target size
        h, w = img.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h) * 0.8  # 0.8 for padding
        
        # Resize object
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w > 0 and new_h > 0:
            resized_img = cv2.resize(img, (new_w, new_h))
            resized_mask = cv2.resize(mask, (new_w, new_h))
            
            # Center the object
            start_x = (target_size[0] - new_w) // 2
            start_y = (target_size[1] - new_h) // 2
            
            # Place object in center
            aligned_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
            aligned_mask[start_y:start_y+new_h, start_x:start_x+new_w] = resized_mask
        
        return aligned_img
    
    def save_aligned_object(self, object_info: Dict, aligned_img: np.ndarray, 
                           image_type: str = "binary") -> Path:
        """
        Save an aligned object to file.
        
        Args:
            object_info: Dictionary containing object information
            aligned_img: Aligned object image
            image_type: Type of image ("binary", "colored", or "original")
            
        Returns:
            Path to saved file
        """
        # Determine output directory
        if image_type == "binary":
            output_dir = self.binary_dir
        elif image_type == "colored":
            output_dir = self.colored_dir
        else:
            output_dir = self.original_dir
        
        # Create filename
        source_name = Path(object_info['source_image']).stem
        filename = f"{source_name}_object_{object_info['id']:02d}.png"
        output_path = output_dir / filename
        
        # Save image
        cv2.imwrite(str(output_path), aligned_img)
        
        return output_path
    
    def process_segmented_image(self, image_path: Path, image_type: str = "auto", original_path: Path = None) -> List[Path]:
        """
        Process a segmented image and extract all objects.
        
        Args:
            image_path: Path to the segmented image (mask)
            image_type: Type of image ("binary", "colored", or "auto")
            original_path: Path to the original image (for binary masks)
            
        Returns:
            List of paths to saved aligned objects
        """
        print(f"🚀 Processing segmented image: {image_path.name}")
        
        # Auto-detect image type if not specified
        if image_type == "auto":
            img = cv2.imread(str(image_path))
            if img is not None:
                unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
                if unique_colors <= 2:
                    image_type = "binary"
                else:
                    image_type = "colored"
            else:
                print(f"❌ Could not load image: {image_path}")
                return []
        
        # Extract objects based on type
        if image_type == "binary":
            objects = self.separate_objects_from_binary(image_path, original_path)
        else:
            objects = self.separate_objects_from_colored(image_path)
        
        if not objects:
            print(f"❌ No objects found in {image_path.name}")
            return []
        
        # Align and save each object
        saved_paths = []
        for obj in objects:
            try:
                # Align object
                aligned_img = self.align_object(obj, target_size=(256, 256))
                
                # Save aligned object
                output_path = self.save_aligned_object(obj, aligned_img, image_type)
                saved_paths.append(output_path)
                
                print(f"   💾 Saved: {output_path.name}")
                
            except Exception as e:
                print(f"   ❌ Error processing object {obj['id']}: {e}")
        
        print(f"✅ Processed {len(saved_paths)} objects from {image_path.name}")
        return saved_paths
    
    def process_directory(self, input_dir: Path) -> Dict[str, List[Path]]:
        """
        Process all segmented images in a directory.
        
        Args:
            input_dir: Directory containing segmented images
            
        Returns:
            Dictionary mapping image names to lists of saved object paths
        """
        print(f"📁 Processing directory: {input_dir}")
        
        results = {}
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(ext))
            image_files.extend(input_dir.glob(ext.upper()))
        
        if not image_files:
            print(f"❌ No images found in {input_dir}")
            return results
        
        print(f"📊 Found {len(image_files)} images to process")
        
        # Process each image
        for image_path in image_files:
            try:
                # For binary masks, try to find corresponding original image
                original_path = None
                if 'binary' in image_path.name.lower():
                    # Try to find corresponding original image
                    original_name = image_path.stem.replace('_sam_points_binary', '') + '.jpg'
                    original_path = image_path.parent / original_name
                    
                    if not original_path.exists():
                        print(f"⚠️  Original image not found for {image_path.name}, skipping...")
                        results[image_path.name] = []
                        continue
                
                saved_paths = self.process_segmented_image(image_path, original_path=original_path)
                results[image_path.name] = saved_paths
                
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                results[image_path.name] = []
        
        # Summary
        total_objects = sum(len(paths) for paths in results.values())
        print(f"\n🎉 Processing complete!")
        print(f"📊 Total objects extracted: {total_objects}")
        print(f"📂 Results saved in: {self.output_dir}")
        
        return results


def align_image(image_path: Path, mask_path: Path, output_path: Path) -> bool:
    """
    Align a single image using its mask.
    
    Args:
        image_path: Path to the original image
        mask_path: Path to the mask image
        output_path: Path to save the aligned image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image and mask
        img = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"❌ Could not load image or mask")
            return False
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"❌ No contours found in mask")
            return False
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract object
        object_img = img[y:y+h, x:x+w]
        
        # Align object (center and resize)
        aligned_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Calculate scaling
        scale = min(256 / w, 256 / h) * 0.8
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w > 0 and new_h > 0:
            resized_img = cv2.resize(object_img, (new_w, new_h))
            
            # Center the object
            start_x = (256 - new_w) // 2
            start_y = (256 - new_h) // 2
            
            aligned_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
        
        # Save aligned image
        cv2.imwrite(str(output_path), aligned_img)
        print(f"✅ Aligned image saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error aligning image: {e}")
        return False


def align_images(input_dir: Path, output_dir: Path) -> bool:
    """
    Align all images in a directory.
    
    Args:
        input_dir: Directory containing images to align
        output_dir: Directory to save aligned images
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize alignment processor
        aligner = ObjectAlignment(str(output_dir))
        
        # Process directory
        results = aligner.process_directory(input_dir)
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error aligning images: {e}")
        return False
