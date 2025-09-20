#!/usr/bin/env python3
"""
Image Matching and Overlap Optimization Application

This application performs optimal image matching by:
1. Converting images to binary representations
2. Finding the best overlap position to maximize alignment
3. Visualizing the matched images with different colors
4. Providing detailed overlap analysis and metrics

Features:
- Binary image conversion with adaptive thresholding
- Optimal positioning using template matching
- Multi-scale matching for better accuracy
- Color-coded visualization of overlaps
- Detailed overlap metrics and analysis

Configuration:
- Modify parameters in the main() function
- Set paths for input images
- Configure matching parameters
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class ImageMatcher:
    """Class for optimal image matching and overlap analysis."""
    
    def __init__(self, output_dir: str = "data/output/matching_results"):
        """
        Initialize the ImageMatcher class.
        
        Args:
            output_dir: Directory to save matching results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different outputs
        self.binary_dir = self.output_dir / "binary_images"
        self.overlay_dir = self.output_dir / "overlay_images"
        self.analysis_dir = self.output_dir / "analysis"
        
        for dir_path in [self.binary_dir, self.overlay_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_binary_image(self, image: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Convert an image to binary representation.
        
        Args:
            image: Input image as numpy array
            method: Binary conversion method ("adaptive", "otsu", "threshold")
            
        Returns:
            Binary image
        """
        if image is None or image.size == 0:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == "adaptive":
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        elif method == "otsu":
            # Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "threshold":
            # Simple thresholding
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = gray
        
        return binary
    
    def find_optimal_overlap(self, img1: np.ndarray, img2: np.ndarray, 
                           method: str = "template_matching") -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Find the optimal position to overlap two images for maximum alignment.
        Includes rotation and scaling optimization.
        
        Args:
            img1: First binary image (template)
            img2: Second binary image (search image)
            method: Matching method ("template_matching", "contour_matching", "feature_matching")
            
        Returns:
            Tuple of (best_position, overlap_score, transformed_img2)
        """
        if img1 is None or img2 is None:
            return (0, 0), 0.0, img2
        
        # First, find optimal rotation and scaling
        print("   🔄 Optimizing rotation and scaling...")
        transformed_img2, best_rotation, best_scale = self._optimize_rotation_and_scale(img1, img2)
        
        # Then find optimal position with the transformed image
        if method == "template_matching":
            position, score = self._template_matching_overlap(img1, transformed_img2)
        elif method == "contour_matching":
            position, score = self._contour_matching_overlap(img1, transformed_img2)
        elif method == "feature_matching":
            position, score = self._feature_matching_overlap(img1, transformed_img2)
        else:
            position, score = self._template_matching_overlap(img1, transformed_img2)
        
        print(f"   📐 Best rotation: {best_rotation:.1f}°, scale: {best_scale:.3f}")
        
        return position, score, transformed_img2
    
    def _optimize_rotation_and_scale(self, template: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Find optimal rotation and scaling using systematic angle testing with mask-based optimization.
        
        Args:
            template: Template binary image (or mask)
            image: Image binary image (or mask)
            
        Returns:
            Tuple of (transformed_image, best_rotation, best_scale)
        """
        print("   🎯 Systematic rotation and scale optimization...")
        
        # First, try systematic rotation testing for better global optimum
        best_score = -1
        best_transformed = image
        best_rotation = 0
        best_scale = 1.0
        best_method = "Systematic Search"
        
        # Test systematic rotations (more comprehensive than before)
        rotation_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
        scale_factors = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.0]
        
        print(f"   🔍 Testing {len(rotation_angles)} rotations × {len(scale_factors)} scales = {len(rotation_angles) * len(scale_factors)} combinations")
        
        for rotation in rotation_angles:
            for scale in scale_factors:
                # Apply transformation
                transformed = self._apply_rotation_and_scale(image, rotation, scale)
                
                if transformed is None or transformed.size == 0:
                    continue
                
                # Calculate comprehensive similarity score
                score = self._calculate_comprehensive_similarity(template, transformed)
                
                if score > best_score:
                    best_score = score
                    best_transformed = transformed
                    best_rotation = rotation
                    best_scale = scale
        
        print(f"   📊 Systematic Search: Best Score={best_score:.3f}, Rot={best_rotation:.1f}°, Scale={best_scale:.3f}")
        
        # Now refine with sophisticated methods around the best systematic result
        if best_score > 0.1:  # Only refine if we found a reasonable match
            print("   🔧 Refining with sophisticated point-based methods...")
            refined_result = self._refine_with_sophisticated_methods(template, best_transformed, best_rotation, best_scale)
            
            if refined_result[3] > best_score:  # If refinement improved the score
                best_transformed, best_rotation, best_scale, best_score = refined_result
                best_method = "Systematic + Refined"
                print(f"   ✅ Refined result: Score={best_score:.3f}, Rot={best_rotation:.1f}°, Scale={best_scale:.3f}")
            else:
                print("   ℹ️  Systematic result was already optimal")
        
        print(f"   ✅ Best method: {best_method} (Score: {best_score:.3f})")
        return best_transformed, best_rotation, best_scale
    
    def _extract_object_keypoints(self, binary_image: np.ndarray, max_points: int = 50) -> np.ndarray:
        """
        Extract sophisticated keypoints representing the object.
        
        Args:
            binary_image: Binary image of the object
            max_points: Maximum number of points to extract
            
        Returns:
            Array of keypoints (N, 2)
        """
        if binary_image is None or binary_image.size == 0:
            return np.array([])
        
        # Method 1: Contour-based critical points
        contour_points = self._extract_contour_keypoints(binary_image, max_points // 3)
        
        # Method 2: Corner detection
        corner_points = self._extract_corner_points(binary_image, max_points // 3)
        
        # Method 3: Skeleton endpoints and junctions
        skeleton_points = self._extract_skeleton_points(binary_image, max_points // 3)
        
        # Combine and filter points
        all_points = []
        if len(contour_points) > 0:
            all_points.append(contour_points)
        if len(corner_points) > 0:
            all_points.append(corner_points)
        if len(skeleton_points) > 0:
            all_points.append(skeleton_points)
        
        if not all_points:
            return np.array([])
        
        combined_points = np.vstack(all_points)
        
        # Remove duplicate points
        unique_points = self._remove_duplicate_points(combined_points, min_distance=10)
        
        # Limit to max_points using importance-based selection
        if len(unique_points) > max_points:
            unique_points = self._select_most_important_points(unique_points, binary_image, max_points)
        
        return unique_points
    
    def _extract_contour_keypoints(self, binary_image: np.ndarray, max_points: int) -> np.ndarray:
        """Extract critical points from object contours."""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.array([])
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract critical points using Douglas-Peucker algorithm
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Extract points
        points = approx_contour.reshape(-1, 2)
        
        # Add curvature-based points
        curvature_points = self._extract_high_curvature_points(largest_contour, max_points // 2)
        
        if len(curvature_points) > 0:
            points = np.vstack([points, curvature_points])
        
        return points[:max_points] if len(points) > max_points else points
    
    def _extract_high_curvature_points(self, contour: np.ndarray, max_points: int) -> np.ndarray:
        """Extract points with high curvature from contour."""
        contour = contour.reshape(-1, 2)
        
        if len(contour) < 5:
            return np.array([])
        
        # Calculate curvature at each point
        curvatures = []
        window_size = 5
        
        for i in range(len(contour)):
            # Get neighboring points
            start_idx = max(0, i - window_size)
            end_idx = min(len(contour), i + window_size + 1)
            
            if end_idx - start_idx < 3:
                curvatures.append(0)
                continue
            
            # Calculate curvature using cross product
            p1 = contour[start_idx]
            p2 = contour[i]
            p3 = contour[end_idx - 1]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            cross_product = np.cross(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm_product > 0:
                curvature = abs(cross_product) / norm_product
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        # Select points with highest curvature
        curvatures = np.array(curvatures)
        high_curvature_indices = np.argsort(curvatures)[-max_points:]
        
        return contour[high_curvature_indices]
    
    def _extract_corner_points(self, binary_image: np.ndarray, max_points: int) -> np.ndarray:
        """Extract corner points using Harris corner detection."""
        # Convert to float32 for corner detection
        float_img = binary_image.astype(np.float32)
        
        # Harris corner detection
        corners = cv2.goodFeaturesToTrack(
            float_img,
            maxCorners=max_points,
            qualityLevel=0.01,
            minDistance=10,
            useHarrisDetector=True,
            k=0.04
        )
        
        if corners is None:
            return np.array([])
        
        return corners.reshape(-1, 2)
    
    def _extract_skeleton_points(self, binary_image: np.ndarray, max_points: int) -> np.ndarray:
        """Extract critical points from object skeleton."""
        # Create skeleton using morphological operations
        skeleton = self._create_skeleton(binary_image)
        
        # Find endpoints and junctions
        endpoints = self._find_skeleton_endpoints(skeleton)
        junctions = self._find_skeleton_junctions(skeleton)
        
        # Combine points
        skeleton_points = []
        if len(endpoints) > 0:
            skeleton_points.append(endpoints)
        if len(junctions) > 0:
            skeleton_points.append(junctions)
        
        if not skeleton_points:
            return np.array([])
        
        combined = np.vstack(skeleton_points)
        return combined[:max_points] if len(combined) > max_points else combined
    
    def _create_skeleton(self, binary_image: np.ndarray) -> np.ndarray:
        """Create skeleton using morphological operations."""
        # Zhang-Suen skeletonization algorithm (simplified)
        img = binary_image.copy()
        img[img == 255] = 1
        
        skeleton = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            # Opening
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, opened)
            eroded = cv2.erode(img, element)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            
            if cv2.countNonZero(img) == 0:
                break
        
        skeleton[skeleton == 1] = 255
        return skeleton
    
    def _find_skeleton_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """Find endpoints in skeleton."""
        # Kernel for endpoint detection
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Convolve skeleton with kernel
        convolved = cv2.filter2D(skeleton, -1, kernel)
        
        # Endpoints have exactly one neighbor (convolved value = 11)
        endpoints = np.where((convolved == 11) & (skeleton == 255))
        
        if len(endpoints[0]) == 0:
            return np.array([])
        
        return np.column_stack((endpoints[1], endpoints[0]))  # (x, y) format
    
    def _find_skeleton_junctions(self, skeleton: np.ndarray) -> np.ndarray:
        """Find junction points in skeleton."""
        # Kernel for junction detection
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        # Convolve skeleton with kernel
        convolved = cv2.filter2D(skeleton, -1, kernel)
        
        # Junctions have 3 or more neighbors (convolved value >= 13)
        junctions = np.where((convolved >= 13) & (skeleton == 255))
        
        if len(junctions[0]) == 0:
            return np.array([])
        
        return np.column_stack((junctions[1], junctions[0]))  # (x, y) format
    
    def _remove_duplicate_points(self, points: np.ndarray, min_distance: float = 10) -> np.ndarray:
        """Remove duplicate points within minimum distance."""
        if len(points) == 0:
            return points
        
        unique_points = [points[0]]
        
        for point in points[1:]:
            distances = np.linalg.norm(np.array(unique_points) - point, axis=1)
            if np.min(distances) > min_distance:
                unique_points.append(point)
        
        return np.array(unique_points)
    
    def _select_most_important_points(self, points: np.ndarray, binary_image: np.ndarray, max_points: int) -> np.ndarray:
        """Select most important points based on geometric properties."""
        if len(points) <= max_points:
            return points
        
        # Calculate importance score for each point
        importance_scores = []
        
        for point in points:
            # Distance from centroid
            centroid = np.mean(points, axis=0)
            dist_from_center = np.linalg.norm(point - centroid)
            
            # Local gradient magnitude
            x, y = int(point[0]), int(point[1])
            if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
                # Sample local neighborhood
                x1, x2 = max(0, x-2), min(binary_image.shape[1], x+3)
                y1, y2 = max(0, y-2), min(binary_image.shape[0], y+3)
                local_patch = binary_image[y1:y2, x1:x2]
                gradient_mag = np.std(local_patch.astype(float))
            else:
                gradient_mag = 0
            
            # Combine metrics
            importance = dist_from_center * 0.3 + gradient_mag * 0.7
            importance_scores.append(importance)
        
        # Select top points
        importance_scores = np.array(importance_scores)
        top_indices = np.argsort(importance_scores)[-max_points:]
        
        return points[top_indices]
    
    def _apply_rotation_and_scale(self, image: np.ndarray, rotation: float, scale: float) -> np.ndarray:
        """
        Apply rotation and scaling to an image.
        
        Args:
            image: Input image
            rotation: Rotation angle in degrees
            scale: Scaling factor
            
        Returns:
            Transformed image
        """
        if image is None or image.size == 0:
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D(center, rotation, scale)
        
        # Calculate new dimensions
        cos_angle = abs(M[0, 0])
        sin_angle = abs(M[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Apply transformation
        transformed = cv2.warpAffine(image, M, (new_w, new_h), 
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return transformed
    
    def _calculate_distance_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate similarity based on overall distance between images.
        
        Args:
            img1: First binary image
            img2: Second binary image
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize images to same size for comparison
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use the smaller dimensions to avoid padding issues
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        img1_resized = cv2.resize(img1, (target_w, target_h))
        img2_resized = cv2.resize(img2, (target_w, target_h))
        
        # Convert to float for distance calculation
        img1_float = img1_resized.astype(np.float32) / 255.0
        img2_float = img2_resized.astype(np.float32) / 255.0
        
        # Calculate various distance metrics
        # 1. Euclidean distance
        euclidean_dist = np.sqrt(np.sum((img1_float - img2_float) ** 2))
        euclidean_sim = 1.0 / (1.0 + euclidean_dist / (target_h * target_w))
        
        # 2. Manhattan distance
        manhattan_dist = np.sum(np.abs(img1_float - img2_float))
        manhattan_sim = 1.0 / (1.0 + manhattan_dist / (target_h * target_w))
        
        # 3. Correlation coefficient
        correlation = np.corrcoef(img1_float.flatten(), img2_float.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        correlation_sim = max(0.0, correlation)
        
        # 4. Structural similarity (simplified)
        mean1 = np.mean(img1_float)
        mean2 = np.mean(img2_float)
        var1 = np.var(img1_float)
        var2 = np.var(img2_float)
        covar = np.mean((img1_float - mean1) * (img2_float - mean2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
               ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
        
        if np.isnan(ssim):
            ssim = 0.0
        
        # Combine metrics with weights
        combined_score = (0.3 * euclidean_sim + 
                         0.2 * manhattan_sim + 
                         0.3 * correlation_sim + 
                         0.2 * ssim)
        
        return combined_score
    
    def _calculate_comprehensive_similarity(self, template: np.ndarray, image: np.ndarray) -> float:
        """
        Calculate comprehensive similarity optimized for masks and binary images.
        
        Args:
            template: Template binary image/mask
            image: Image binary image/mask
            
        Returns:
            Comprehensive similarity score (0.0 to 1.0)
        """
        if template is None or image is None:
            return 0.0
        
        # Resize to same dimensions for comparison
        h1, w1 = template.shape[:2]
        h2, w2 = image.shape[:2]
        
        # Use the smaller dimensions to avoid padding issues
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        template_resized = cv2.resize(template, (target_w, target_h))
        image_resized = cv2.resize(image, (target_w, target_h))
        
        # Convert to float for calculations
        template_float = template_resized.astype(np.float32) / 255.0
        image_float = image_resized.astype(np.float32) / 255.0
        
        # 1. Intersection over Union (IoU) - critical for masks
        intersection = np.logical_and(template_resized > 127, image_resized > 127)
        union = np.logical_or(template_resized > 127, image_resized > 127)
        iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
        
        # 2. Dice coefficient - another overlap metric
        intersection_sum = np.sum(intersection)
        dice_score = (2.0 * intersection_sum) / (np.sum(template_resized > 127) + np.sum(image_resized > 127)) if (np.sum(template_resized > 127) + np.sum(image_resized > 127)) > 0 else 0.0
        
        # 3. Normalized cross-correlation
        template_norm = template_float - np.mean(template_float)
        image_norm = image_float - np.mean(image_float)
        
        correlation = np.sum(template_norm * image_norm) / (np.sqrt(np.sum(template_norm**2)) * np.sqrt(np.sum(image_norm**2)))
        if np.isnan(correlation):
            correlation = 0.0
        correlation = max(0.0, correlation)
        
        # 4. Structural similarity (simplified)
        mean1 = np.mean(template_float)
        mean2 = np.mean(image_float)
        var1 = np.var(template_float)
        var2 = np.var(image_float)
        covar = np.mean((template_float - mean1) * (image_float - mean2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
               ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
        
        if np.isnan(ssim):
            ssim = 0.0
        
        # 5. Edge alignment score
        template_edges = cv2.Canny(template_resized, 50, 150)
        image_edges = cv2.Canny(image_resized, 50, 150)
        
        edge_intersection = cv2.bitwise_and(template_edges, image_edges)
        edge_union = cv2.bitwise_or(template_edges, image_edges)
        edge_score = np.sum(edge_intersection) / np.sum(edge_union) if np.sum(edge_union) > 0 else 0.0
        
        # Combine metrics with weights optimized for mask comparison
        final_score = (0.35 * iou_score +      # IoU is most important for masks
                      0.25 * dice_score +      # Dice coefficient for overlap
                      0.15 * correlation +     # Cross-correlation
                      0.15 * max(0.0, ssim) + # SSIM (clamped to positive)
                      0.10 * edge_score)      # Edge alignment
        
        return final_score
    
    def _refine_with_sophisticated_methods(self, template: np.ndarray, image: np.ndarray, 
                                         base_rotation: float, base_scale: float) -> Tuple[np.ndarray, float, float, float]:
        """
        Refine the systematic search result using sophisticated point-based methods.
        
        Args:
            template: Template binary image/mask
            image: Image binary image/mask  
            base_rotation: Base rotation from systematic search
            base_scale: Base scale from systematic search
            
        Returns:
            Tuple of (transformed_image, rotation, scale, score)
        """
        # Extract keypoints for refinement
        template_points = self._extract_object_keypoints(template)
        image_points = self._extract_object_keypoints(image)
        
        if len(template_points) < 3 or len(image_points) < 3:
            # Can't refine, return original
            return image, base_rotation, base_scale, self._calculate_comprehensive_similarity(template, image)
        
        # Try refinement methods around the base result
        methods = [
            ("Procrustes Analysis", self._procrustes_alignment),
            ("ICP (Iterative Closest Point)", self._icp_alignment),
        ]
        
        best_score = -1
        best_transformed = image
        best_rotation = base_rotation
        best_scale = base_scale
        
        for method_name, method_func in methods:
            try:
                transformed, rotation, scale, score = method_func(template, image, template_points, image_points)
                
                # Bias towards results close to our systematic search result
                rotation_diff = abs(rotation - base_rotation)
                if rotation_diff > 180:
                    rotation_diff = 360 - rotation_diff
                
                scale_diff = abs(scale - base_scale)
                
                # Penalize results that are too far from systematic optimum
                proximity_bonus = max(0, 1.0 - (rotation_diff / 45.0) - (scale_diff / 0.5))
                adjusted_score = score * (0.7 + 0.3 * proximity_bonus)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_transformed = transformed
                    best_rotation = rotation
                    best_scale = scale
                    
            except Exception:
                continue
        
        return best_transformed, best_rotation, best_scale, best_score
    
    def _contour_based_alignment(self, template: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Fallback contour-based alignment method."""
        # Simple centroid alignment
        template_contours, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not template_contours or not image_contours:
            return image, 0.0, 1.0
        
        # Get centroids
        template_M = cv2.moments(template_contours[0])
        image_M = cv2.moments(image_contours[0])
        
        if template_M["m00"] == 0 or image_M["m00"] == 0:
            return image, 0.0, 1.0
        
        template_cx = template_M["m10"] / template_M["m00"]
        template_cy = template_M["m01"] / template_M["m00"]
        image_cx = image_M["m10"] / image_M["m00"]
        image_cy = image_M["m01"] / image_M["m00"]
        
        # Calculate scale from areas
        template_area = cv2.contourArea(template_contours[0])
        image_area = cv2.contourArea(image_contours[0])
        scale = np.sqrt(template_area / image_area) if image_area > 0 else 1.0
        
        # Apply transformation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        transformed = cv2.warpAffine(image, M, (w, h))
        
        return transformed, 0.0, scale
    
    def _procrustes_alignment(self, template: np.ndarray, image: np.ndarray, 
                            template_points: np.ndarray, image_points: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Procrustes analysis for optimal alignment.
        Finds optimal rotation, scaling, and translation.
        """
        if len(template_points) < 3 or len(image_points) < 3:
            raise ValueError("Insufficient points for Procrustes analysis")
        
        # Use subset of points for efficiency
        n_points = min(len(template_points), len(image_points), 20)
        template_subset = template_points[:n_points]
        image_subset = image_points[:n_points]
        
        # Center the points
        template_centroid = np.mean(template_subset, axis=0)
        image_centroid = np.mean(image_subset, axis=0)
        
        template_centered = template_subset - template_centroid
        image_centered = image_subset - image_centroid
        
        # Calculate optimal rotation using SVD
        H = np.dot(image_centered.T, template_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Calculate rotation angle
        rotation_angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        
        # Calculate optimal scale
        template_norm = np.linalg.norm(template_centered, 'fro')
        image_norm = np.linalg.norm(image_centered, 'fro')
        scale = template_norm / image_norm if image_norm > 0 else 1.0
        
        # Apply transformation to image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, scale)
        transformed = cv2.warpAffine(image, M, (w, h))
        
        # Calculate alignment score
        score = self._calculate_alignment_score(template, transformed, template_points, image_points)
        
        return transformed, rotation_angle, scale, score
    
    def _icp_alignment(self, template: np.ndarray, image: np.ndarray,
                      template_points: np.ndarray, image_points: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Iterative Closest Point (ICP) alignment.
        """
        max_iterations = 20
        tolerance = 1e-6
        
        current_points = image_points.copy()
        best_rotation = 0.0
        best_scale = 1.0
        
        for iteration in range(max_iterations):
            # Find closest points
            correspondences = self._find_closest_points(template_points, current_points)
            
            if len(correspondences) < 3:
                break
            
            # Extract corresponding points
            template_corr = np.array([template_points[i] for i, j in correspondences])
            image_corr = np.array([current_points[j] for i, j in correspondences])
            
            # Calculate transformation using Procrustes
            template_centroid = np.mean(template_corr, axis=0)
            image_centroid = np.mean(image_corr, axis=0)
            
            template_centered = template_corr - template_centroid
            image_centered = image_corr - image_centroid
            
            # SVD for rotation
            H = np.dot(image_centered.T, template_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            # Calculate incremental rotation and scale
            inc_rotation = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            
            template_norm = np.linalg.norm(template_centered, 'fro')
            image_norm = np.linalg.norm(image_centered, 'fro')
            inc_scale = template_norm / image_norm if image_norm > 0 else 1.0
            
            # Update total transformation
            best_rotation += inc_rotation
            best_scale *= inc_scale
            
            # Apply transformation to current points
            cos_r = np.cos(np.radians(inc_rotation))
            sin_r = np.sin(np.radians(inc_rotation))
            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            current_points = (current_points - image_centroid) @ rotation_matrix.T * inc_scale + template_centroid
            
            # Check convergence
            if abs(inc_rotation) < tolerance and abs(inc_scale - 1.0) < tolerance:
                break
        
        # Apply final transformation to image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_rotation, best_scale)
        transformed = cv2.warpAffine(image, M, (w, h))
        
        # Calculate final score
        score = self._calculate_alignment_score(template, transformed, template_points, current_points)
        
        return transformed, best_rotation, best_scale, score
    
    def _find_closest_points(self, template_points: np.ndarray, image_points: np.ndarray) -> List[Tuple[int, int]]:
        """Find closest point correspondences."""
        correspondences = []
        
        for i, template_point in enumerate(template_points):
            distances = np.linalg.norm(image_points - template_point, axis=1)
            closest_idx = np.argmin(distances)
            correspondences.append((i, closest_idx))
        
        return correspondences
    
    def _ransac_affine_alignment(self, template: np.ndarray, image: np.ndarray,
                               template_points: np.ndarray, image_points: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        RANSAC-based affine transformation estimation.
        """
        max_iterations = 100
        min_samples = 3
        threshold = 10.0
        
        if len(template_points) < min_samples or len(image_points) < min_samples:
            raise ValueError("Insufficient points for RANSAC")
        
        best_inliers = 0
        best_transform = None
        best_rotation = 0.0
        best_scale = 1.0
        
        for _ in range(max_iterations):
            # Randomly sample points
            sample_indices = np.random.choice(len(template_points), min_samples, replace=False)
            template_sample = template_points[sample_indices]
            
            # Find closest points in image
            image_sample = []
            for template_pt in template_sample:
                distances = np.linalg.norm(image_points - template_pt, axis=1)
                closest_idx = np.argmin(distances)
                image_sample.append(image_points[closest_idx])
            
            image_sample = np.array(image_sample)
            
            # Estimate affine transformation
            try:
                transform_matrix = cv2.getAffineTransform(
                    image_sample.astype(np.float32),
                    template_sample.astype(np.float32)
                )
                
                # Extract rotation and scale from transformation matrix
                a, b = transform_matrix[0, 0], transform_matrix[0, 1]
                rotation = np.degrees(np.arctan2(b, a))
                scale = np.sqrt(a*a + b*b)
                
                # Count inliers
                inliers = 0
                for template_pt in template_points:
                    # Find closest point in image
                    distances = np.linalg.norm(image_points - template_pt, axis=1)
                    closest_image_pt = image_points[np.argmin(distances)]
                    
                    # Transform image point
                    transformed_pt = cv2.transform(
                        np.array([[closest_image_pt]], dtype=np.float32),
                        transform_matrix
                    )[0][0]
                    
                    # Check if it's close to template point
                    if np.linalg.norm(transformed_pt - template_pt) < threshold:
                        inliers += 1
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_transform = transform_matrix
                    best_rotation = rotation
                    best_scale = scale
                    
            except:
                continue
        
        if best_transform is None:
            raise ValueError("RANSAC failed to find valid transformation")
        
        # Apply best transformation to image
        h, w = image.shape[:2]
        transformed = cv2.warpAffine(image, best_transform, (w, h))
        
        # Calculate score based on inlier ratio
        score = best_inliers / len(template_points)
        
        return transformed, best_rotation, best_scale, score
    
    def _coherent_point_drift(self, template: np.ndarray, image: np.ndarray,
                            template_points: np.ndarray, image_points: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Simplified Coherent Point Drift alignment.
        """
        # This is a simplified version - full CPD requires more complex implementation
        max_iterations = 15
        w = 0.1  # Noise weight
        
        # Initialize transformation parameters
        rotation = 0.0
        scale = 1.0
        translation = np.zeros(2)
        
        current_image_points = image_points.copy()
        
        for iteration in range(max_iterations):
            # E-step: Calculate correspondence probabilities
            probabilities = self._calculate_correspondence_probabilities(
                template_points, current_image_points, w
            )
            
            # M-step: Update transformation parameters
            # Simplified update using weighted centroids
            weights = np.sum(probabilities, axis=0)
            if np.sum(weights) == 0:
                break
            
            # Weighted centroids
            template_centroid = np.sum(template_points * weights[:len(template_points), np.newaxis], axis=0) / np.sum(weights[:len(template_points)])
            image_centroid = np.sum(current_image_points * weights[:len(current_image_points), np.newaxis], axis=0) / np.sum(weights[:len(current_image_points)])
            
            # Update translation
            translation = template_centroid - image_centroid
            
            # Center points
            template_centered = template_points - template_centroid
            image_centered = current_image_points - image_centroid
            
            # Update rotation and scale using weighted SVD
            try:
                H = np.zeros((2, 2))
                total_weight = 0
                
                for i in range(min(len(template_points), len(image_points))):
                    weight = probabilities[i, i] if i < len(probabilities) else 0.1
                    H += weight * np.outer(image_centered[i], template_centered[i])
                    total_weight += weight
                
                if total_weight > 0:
                    H /= total_weight
                    
                    U, S, Vt = np.linalg.svd(H)
                    R = np.dot(Vt.T, U.T)
                    
                    if np.linalg.det(R) < 0:
                        Vt[-1, :] *= -1
                        R = np.dot(Vt.T, U.T)
                    
                    # Update rotation
                    rotation = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    
                    # Update scale (simplified)
                    template_var = np.var(template_centered, axis=0).sum()
                    image_var = np.var(image_centered, axis=0).sum()
                    scale = np.sqrt(template_var / image_var) if image_var > 0 else 1.0
                
            except:
                break
            
            # Apply current transformation to image points
            cos_r = np.cos(np.radians(rotation))
            sin_r = np.sin(np.radians(rotation))
            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            current_image_points = (image_points - image_centroid) @ rotation_matrix.T * scale + template_centroid
        
        # Apply final transformation to image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation, scale)
        M[0, 2] += translation[0]
        M[1, 2] += translation[1]
        
        transformed = cv2.warpAffine(image, M, (w, h))
        
        # Calculate final score
        score = self._calculate_alignment_score(template, transformed, template_points, current_image_points)
        
        return transformed, rotation, scale, score
    
    def _calculate_correspondence_probabilities(self, template_points: np.ndarray, 
                                             image_points: np.ndarray, w: float) -> np.ndarray:
        """Calculate correspondence probabilities for CPD."""
        n_template = len(template_points)
        n_image = len(image_points)
        
        probabilities = np.zeros((n_template, n_image))
        
        for i, template_pt in enumerate(template_points):
            for j, image_pt in enumerate(image_points):
                distance = np.linalg.norm(template_pt - image_pt)
                probabilities[i, j] = np.exp(-distance**2 / (2 * w**2))
        
        # Normalize
        row_sums = np.sum(probabilities, axis=1, keepdims=True)
        probabilities = probabilities / (row_sums + 1e-10)
        
        return probabilities
    
    def _calculate_alignment_score(self, template: np.ndarray, transformed_image: np.ndarray,
                                 template_points: np.ndarray, image_points: np.ndarray) -> float:
        """Calculate alignment quality score."""
        # Combine multiple metrics
        
        # 1. Point-based score
        if len(template_points) > 0 and len(image_points) > 0:
            point_distances = []
            for template_pt in template_points:
                distances = np.linalg.norm(image_points - template_pt, axis=1)
                min_distance = np.min(distances)
                point_distances.append(min_distance)
            
            avg_point_distance = np.mean(point_distances)
            point_score = 1.0 / (1.0 + avg_point_distance / 50.0)  # Normalize
        else:
            point_score = 0.0
        
        # 2. Image-based score
        image_score = self._calculate_distance_similarity(template, transformed_image)
        
        # 3. Overlap score
        overlap_score = self._calculate_overlap_score(template, transformed_image, (0, 0))
        
        # Combine scores
        final_score = 0.4 * point_score + 0.3 * image_score + 0.3 * overlap_score
        
        return final_score
    
    def _template_matching_overlap(self, template: np.ndarray, image: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Find optimal overlap using template matching.
        
        Args:
            template: Template binary image
            image: Search binary image
            
        Returns:
            Tuple of (best_position, overlap_score)
        """
        # Use normalized cross correlation for better results
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Return the best position and confidence score
        return max_loc, max_val
    
    def _contour_matching_overlap(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Find optimal overlap using contour matching.
        
        Args:
            img1: First binary image
            img2: Second binary image
            
        Returns:
            Tuple of (best_position, overlap_score)
        """
        # Find contours in both images
        contours1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours1 or not contours2:
            return (0, 0), 0.0
        
        # Get the largest contour from each image
        contour1 = max(contours1, key=cv2.contourArea)
        contour2 = max(contours2, key=cv2.contourArea)
        
        # Calculate moments
        M1 = cv2.moments(contour1)
        M2 = cv2.moments(contour2)
        
        if M1["m00"] == 0 or M2["m00"] == 0:
            return (0, 0), 0.0
        
        # Calculate centroids
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        cx2 = int(M2["m10"] / M2["m00"])
        cy2 = int(M2["m01"] / M2["m00"])
        
        # Calculate offset to align centroids
        offset_x = cx1 - cx2
        offset_y = cy1 - cy2
        
        # Calculate overlap score
        overlap_score = self._calculate_overlap_score(img1, img2, (offset_x, offset_y))
        
        return (offset_x, offset_y), overlap_score
    
    def _feature_matching_overlap(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Find optimal overlap using feature matching.
        
        Args:
            img1: First binary image
            img2: Second binary image
            
        Returns:
            Tuple of (best_position, overlap_score)
        """
        # Convert binary images to 3-channel for feature detection
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1_color, None)
        kp2, des2 = sift.detectAndCompute(img2_color, None)
        
        if des1 is None or des2 is None:
            return (0, 0), 0.0
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return (0, 0), 0.0
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return (0, 0), 0.0
        
        # Calculate translation from homography
        translation_x = H[0, 2]
        translation_y = H[1, 2]
        
        # Calculate overlap score
        overlap_score = self._calculate_overlap_score(img1, img2, (int(translation_x), int(translation_y)))
        
        return (int(translation_x), int(translation_y)), overlap_score
    
    def _calculate_overlap_score(self, img1: np.ndarray, img2: np.ndarray, offset: Tuple[int, int]) -> float:
        """
        Calculate overlap score between two images with given offset.
        
        Args:
            img1: First binary image
            img2: Second binary image
            offset: (x, y) offset to apply to img2
            
        Returns:
            Overlap score (0.0 to 1.0)
        """
        offset_x, offset_y = offset
        
        # Create a larger canvas to accommodate both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate canvas size
        canvas_h = max(h1, h2 + abs(offset_y))
        canvas_w = max(w1, w2 + abs(offset_x))
        
        # Create canvas
        canvas1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas2 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        
        # Place images on canvas
        canvas1[:h1, :w1] = img1
        
        # Place img2 with offset
        start_y = max(0, offset_y)
        start_x = max(0, offset_x)
        end_y = min(canvas_h, h2 + offset_y)
        end_x = min(canvas_w, w2 + offset_x)
        
        if start_y < end_y and start_x < end_x:
            img2_y_start = max(0, -offset_y)
            img2_x_start = max(0, -offset_x)
            img2_y_end = img2_y_start + (end_y - start_y)
            img2_x_end = img2_x_start + (end_x - start_x)
            
            canvas2[start_y:end_y, start_x:end_x] = img2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
        
        # Calculate overlap
        intersection = cv2.bitwise_and(canvas1, canvas2)
        union = cv2.bitwise_or(canvas1, canvas2)
        
        if np.sum(union) > 0:
            overlap_score = np.sum(intersection) / np.sum(union)
        else:
            overlap_score = 0.0
        
        return overlap_score
    
    def create_overlay_visualization(self, img1: np.ndarray, img2: np.ndarray, 
                                   offset: Tuple[int, int], colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Create a color-coded overlay visualization of two images.
        
        Args:
            img1: First binary image
            img2: Second binary image
            offset: (x, y) offset to apply to img2
            colors: Tuple of (color1, color2) for the two images
            
        Returns:
            Color-coded overlay image
        """
        if colors is None:
            colors = ((0, 255, 0), (255, 0, 0))  # Green and Red
        
        color1, color2 = colors
        offset_x, offset_y = offset
        
        # Create a larger canvas to accommodate both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate canvas size
        canvas_h = max(h1, h2 + abs(offset_y))
        canvas_w = max(w1, w2 + abs(offset_x))
        
        # Create 3-channel canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Place img1 in green
        canvas[:h1, :w1] = np.where(img1[..., np.newaxis] > 0, color1, [0, 0, 0])
        
        # Place img2 with offset in red
        start_y = max(0, offset_y)
        start_x = max(0, offset_x)
        end_y = min(canvas_h, h2 + offset_y)
        end_x = min(canvas_w, w2 + offset_x)
        
        if start_y < end_y and start_x < end_x:
            img2_y_start = max(0, -offset_y)
            img2_x_start = max(0, -offset_x)
            img2_y_end = img2_y_start + (end_y - start_y)
            img2_x_end = img2_x_start + (end_x - start_x)
            
            img2_region = img2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
            canvas_region = canvas[start_y:end_y, start_x:end_x]
            
            # Overlay img2 in red, with yellow for overlap
            mask = img2_region > 0
            canvas_region[mask] = np.where(
                canvas_region[mask].any(axis=1, keepdims=True),  # If there's already green
                [255, 255, 0],  # Yellow for overlap
                color2  # Red for img2 only
            )
        
        return canvas
    
    def match_images(self, img1_path: Path, img2_path: Path, 
                    binary_method: str = "adaptive", 
                    matching_method: str = "template_matching",
                    masks_dir: Path = None) -> Dict:
        """
        Match two images and find optimal overlap.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            binary_method: Method for binary conversion
            matching_method: Method for finding optimal overlap
            
        Returns:
            Dictionary containing matching results
        """
        print(f"🔍 Matching images: {img1_path.name} and {img2_path.name}")
        
        # Load images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            print(f"❌ Could not load one or both images")
            return {}
        
        print(f"📐 Image 1 shape: {img1.shape}")
        print(f"📐 Image 2 shape: {img2.shape}")
        
        # Load or create binary images/masks
        binary1 = None
        binary2 = None
        
        if masks_dir and masks_dir.exists():
            # Try to load corresponding masks for object images
            mask1_path = masks_dir / f"{img1_path.stem}_mask.png"
            mask2_path = masks_dir / f"{img2_path.stem}_mask.png"
            
            # Load mask for image 1 if available
            if mask1_path.exists():
                print(f"🎭 Loading mask for {img1_path.name}...")
                binary1 = cv2.imread(str(mask1_path), cv2.IMREAD_GRAYSCALE)
                print(f"   📐 Mask 1 shape: {binary1.shape}")
            else:
                print(f"🔄 No mask found for {img1_path.name}, converting to binary...")
                binary1 = self.create_binary_image(img1, binary_method)
            
            # Load mask for image 2 if available
            if mask2_path.exists():
                print(f"🎭 Loading mask for {img2_path.name}...")
                binary2 = cv2.imread(str(mask2_path), cv2.IMREAD_GRAYSCALE)
                print(f"   📐 Mask 2 shape: {binary2.shape}")
            else:
                print(f"🔄 No mask found for {img2_path.name}, converting to binary...")
                binary2 = self.create_binary_image(img2, binary_method)
        else:
            print("🔄 Converting images to binary...")
            binary1 = self.create_binary_image(img1, binary_method)
            binary2 = self.create_binary_image(img2, binary_method)
        
        # Save binary images for reference
        binary1_path = self.binary_dir / f"{img1_path.stem}_binary.png"
        binary2_path = self.binary_dir / f"{img2_path.stem}_binary.png"
        cv2.imwrite(str(binary1_path), binary1)
        cv2.imwrite(str(binary2_path), binary2)
        
        # Find optimal overlap with rotation and scaling
        print(f"🎯 Finding optimal overlap using {matching_method}...")
        best_position, overlap_score, transformed_binary2 = self.find_optimal_overlap(binary1, binary2, matching_method)
        
        print(f"📍 Best position: {best_position}")
        print(f"📊 Overlap score: {overlap_score:.3f}")
        
        # Create overlay visualization using transformed image
        print("🎨 Creating overlay visualization...")
        overlay = self.create_overlay_visualization(binary1, transformed_binary2, best_position)
        
        # Save overlay image
        overlay_path = self.overlay_dir / f"{img1_path.stem}_vs_{img2_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        
        # Create detailed analysis
        analysis = self._create_detailed_analysis(binary1, binary2, best_position, overlap_score)
        
        # Save analysis results
        analysis_path = self.analysis_dir / f"{img1_path.stem}_vs_{img2_path.stem}_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Create summary visualization
        summary_path = self.overlay_dir / f"{img1_path.stem}_vs_{img2_path.stem}_summary.png"
        self._create_summary_visualization(img1, img2, binary1, transformed_binary2, overlay, 
                                         best_position, overlap_score, summary_path)
        
        results = {
            'image1_path': str(img1_path),
            'image2_path': str(img2_path),
            'binary1_path': str(binary1_path),
            'binary2_path': str(binary2_path),
            'overlay_path': str(overlay_path),
            'analysis_path': str(analysis_path),
            'summary_path': str(summary_path),
            'best_position': best_position,
            'overlap_score': overlap_score,
            'binary_method': binary_method,
            'matching_method': matching_method,
            'analysis': analysis
        }
        
        return results
    
    def _create_detailed_analysis(self, binary1: np.ndarray, binary2: np.ndarray, 
                                offset: Tuple[int, int], overlap_score: float) -> Dict:
        """
        Create detailed analysis of the matching results.
        
        Args:
            binary1: First binary image
            binary2: Second binary image
            offset: Best offset found
            overlap_score: Overlap score
            
        Returns:
            Dictionary with detailed analysis
        """
        # Calculate various metrics
        area1 = np.sum(binary1 > 0)
        area2 = np.sum(binary2 > 0)
        
        # Calculate overlap at best position
        offset_x, offset_y = offset
        overlap_area = self._calculate_overlap_area(binary1, binary2, offset)
        
        # Calculate coverage metrics
        coverage1 = overlap_area / area1 if area1 > 0 else 0
        coverage2 = overlap_area / area2 if area2 > 0 else 0
        
        # Calculate alignment quality
        alignment_quality = (coverage1 + coverage2) / 2
        
        analysis = {
            'overlap_score': float(overlap_score),
            'best_offset': {'x': offset_x, 'y': offset_y},
            'image_areas': {
                'image1_pixels': int(area1),
                'image2_pixels': int(area2)
            },
            'overlap_metrics': {
                'overlap_area': int(overlap_area),
                'coverage_image1': float(coverage1),
                'coverage_image2': float(coverage2),
                'alignment_quality': float(alignment_quality)
            },
            'image_dimensions': {
                'image1': {'height': int(binary1.shape[0]), 'width': int(binary1.shape[1])},
                'image2': {'height': int(binary2.shape[0]), 'width': int(binary2.shape[1])}
            }
        }
        
        return analysis
    
    def _calculate_overlap_area(self, img1: np.ndarray, img2: np.ndarray, offset: Tuple[int, int]) -> int:
        """
        Calculate the actual overlap area between two images.
        
        Args:
            img1: First binary image
            img2: Second binary image
            offset: (x, y) offset to apply to img2
            
        Returns:
            Number of overlapping pixels
        """
        offset_x, offset_y = offset
        
        # Create canvas for both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        canvas_h = max(h1, h2 + abs(offset_y))
        canvas_w = max(w1, w2 + abs(offset_x))
        
        canvas1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas2 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        
        # Place images
        canvas1[:h1, :w1] = img1
        
        start_y = max(0, offset_y)
        start_x = max(0, offset_x)
        end_y = min(canvas_h, h2 + offset_y)
        end_x = min(canvas_w, w2 + offset_x)
        
        if start_y < end_y and start_x < end_x:
            img2_y_start = max(0, -offset_y)
            img2_x_start = max(0, -offset_x)
            img2_y_end = img2_y_start + (end_y - start_y)
            img2_x_end = img2_x_start + (end_x - start_x)
            
            canvas2[start_y:end_y, start_x:end_x] = img2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
        
        # Calculate intersection
        intersection = cv2.bitwise_and(canvas1, canvas2)
        return int(np.sum(intersection > 0))
    
    def _create_summary_visualization(self, img1: np.ndarray, img2: np.ndarray, 
                                    binary1: np.ndarray, binary2: np.ndarray, 
                                    overlay: np.ndarray, offset: Tuple[int, int], 
                                    overlap_score: float, output_path: Path):
        """
        Create a comprehensive summary visualization.
        
        Args:
            img1: Original first image
            img2: Original second image
            binary1: Binary first image
            binary2: Binary second image
            overlay: Overlay visualization
            offset: Best offset found
            overlap_score: Overlap score
            output_path: Path to save summary image
        """
        # Create a 2x3 grid visualization
        h, w = 300, 400  # Standard size for each panel
        
        # Resize images to standard size
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        binary1_resized = cv2.resize(binary1, (w, h))
        binary2_resized = cv2.resize(binary2, (w, h))
        overlay_resized = cv2.resize(overlay, (w, h))
        
        # Convert binary images to 3-channel for consistent stacking
        if len(binary1_resized.shape) == 2:
            binary1_resized = cv2.cvtColor(binary1_resized, cv2.COLOR_GRAY2BGR)
        if len(binary2_resized.shape) == 2:
            binary2_resized = cv2.cvtColor(binary2_resized, cv2.COLOR_GRAY2BGR)
        
        # Create top row: original images
        top_row = np.hstack([img1_resized, img2_resized])
        
        # Create middle row: binary images
        middle_row = np.hstack([binary1_resized, binary2_resized])
        
        # Create bottom row: overlay and info
        info_panel = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(info_panel, f"Offset: {offset}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, f"Score: {overlap_score:.3f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, "Green: Image 1", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_panel, "Red: Image 2", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(info_panel, "Yellow: Overlap", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        bottom_row = np.hstack([overlay_resized, info_panel])
        
        # Combine all rows
        summary = np.vstack([top_row, middle_row, bottom_row])
        
        # Add labels
        cv2.putText(summary, "Original Images", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(summary, "Binary Images", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(summary, "Overlay & Analysis", (10, 2*h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(str(output_path), summary)


def process_all_pairs(templates_dir: Path, objects_dir: Path, output_dir: Path, 
                     binary_method: str = "adaptive", matching_method: str = "contour_matching",
                     masks_dir: Path = None):
    """
    Process all pairs between templates and objects.
    
    Args:
        templates_dir: Directory containing template images
        objects_dir: Directory containing object images
        output_dir: Output directory for results
        binary_method: Binary conversion method
        matching_method: Matching method
    """
    print("🎯 Batch Image Matching and Overlap Optimization Application")
    print("=" * 70)
    print(f"📁 Templates directory: {templates_dir}")
    print(f"📁 Objects directory: {objects_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🔧 Binary method: {binary_method}")
    print(f"🔧 Matching method: {matching_method}")
    print()
    
    # Check if directories exist
    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        sys.exit(1)
    
    if not objects_dir.exists():
        print(f"❌ Objects directory not found: {objects_dir}")
        sys.exit(1)
    
    # Find template images
    template_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    template_files = []
    for ext in template_extensions:
        template_files.extend(templates_dir.glob(ext))
        template_files.extend(templates_dir.glob(ext.upper()))
    
    # Find object images
    object_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    object_files = []
    for ext in object_extensions:
        object_files.extend(objects_dir.glob(ext))
        object_files.extend(objects_dir.glob(ext.upper()))
    
    if not template_files:
        print(f"❌ No template images found in {templates_dir}")
        sys.exit(1)
    
    if not object_files:
        print(f"❌ No object images found in {objects_dir}")
        sys.exit(1)
    
    print(f"📊 Found {len(template_files)} templates and {len(object_files)} objects")
    print(f"🔄 Processing {len(template_files) * len(object_files)} pairs...")
    print()
    
    # Initialize matcher
    matcher = ImageMatcher(str(output_dir))
    
    all_results = {}
    total_pairs = len(template_files) * len(object_files)
    processed_pairs = 0
    
    try:
        for template_path in template_files:
            print(f"🔍 Processing template: {template_path.name}")
            print("-" * 50)
            
            template_results = {}
            
            for object_path in object_files:
                processed_pairs += 1
                print(f"   [{processed_pairs}/{total_pairs}] Matching with: {object_path.name}")
                
                try:
                    # Match template with object
                    results = matcher.match_images(template_path, object_path, binary_method, matching_method, masks_dir)
                    
                    if results:
                        template_results[object_path.name] = results
                        print(f"   ✅ Overlap score: {results['overlap_score']:.3f}, Position: {results['best_position']}")
                    else:
                        print(f"   ❌ Matching failed")
                        template_results[object_path.name] = None
                        
                except Exception as e:
                    print(f"   ❌ Error matching {object_path.name}: {e}")
                    template_results[object_path.name] = None
            
            all_results[template_path.name] = template_results
            print()
        
        # Save comprehensive results
        _save_batch_results(all_results, output_dir)
        
        print("=" * 70)
        print("🎉 Batch Image Matching Complete!")
        print(f"📊 Processed {processed_pairs} pairs")
        print(f"📂 Results saved in: {output_dir}")
        print("🖼️  Check overlay images and analysis files for detailed results")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during batch matching: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _save_batch_results(all_results: Dict, output_dir: Path):
    """
    Save comprehensive batch results.
    
    Args:
        all_results: Dictionary of all matching results
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"batch_matching_results_{timestamp}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean results for JSON serialization
    clean_results = {}
    for template_name, template_results in all_results.items():
        clean_results[template_name] = {}
        for object_name, object_data in template_results.items():
            if object_data is not None:
                clean_results[template_name][object_name] = {
                    'best_position': object_data['best_position'],
                    'overlap_score': float(object_data['overlap_score']),
                    'binary_method': object_data['binary_method'],
                    'matching_method': object_data['matching_method'],
                    'overlay_path': object_data['overlay_path'],
                    'analysis_path': object_data['analysis_path'],
                    'summary_path': object_data['summary_path'],
                    'analysis': object_data['analysis']
                }
            else:
                clean_results[template_name][object_name] = None
    
    with open(results_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"💾 Batch results saved to: {results_file}")
    
    # Generate summary report
    _generate_batch_summary_report(clean_results, output_dir / f"batch_summary_{timestamp}.txt")


def _generate_batch_summary_report(results: Dict, report_path: Path):
    """
    Generate a summary report of the batch matching results.
    
    Args:
        results: Cleaned results dictionary
        report_path: Path to save the summary report
    """
    with open(report_path, 'w') as f:
        f.write("Batch Image Matching and Overlap Optimization Report\n")
        f.write("=" * 60 + "\n\n")
        
        total_templates = len(results)
        total_objects = sum(len(template_results) for template_results in results.values())
        successful_matches = sum(1 for template_results in results.values() 
                               for object_data in template_results.values() 
                               if object_data is not None)
        
        f.write(f"Total Templates: {total_templates}\n")
        f.write(f"Total Objects: {total_objects}\n")
        f.write(f"Successful Matches: {successful_matches}\n")
        f.write(f"Success Rate: {successful_matches/total_objects*100:.1f}%\n\n")
        
        # Find best matches for each template
        for template_name, template_results in results.items():
            f.write(f"Template: {template_name}\n")
            f.write("-" * 40 + "\n")
            
            # Filter successful matches
            successful_results = {obj_name: obj_data for obj_name, obj_data in template_results.items() 
                               if obj_data is not None}
            
            if not successful_results:
                f.write("No successful matches.\n\n")
                continue
            
            # Sort by overlap score
            sorted_results = sorted(successful_results.items(), 
                                  key=lambda x: x[1]['overlap_score'], reverse=True)
            
            f.write("Best Matches (sorted by overlap score):\n")
            for i, (obj_name, obj_data) in enumerate(sorted_results[:5]):  # Top 5 matches
                f.write(f"  {i+1}. {obj_name}: Score={obj_data['overlap_score']:.3f}, "
                       f"Position={obj_data['best_position']}\n")
            
            f.write("\n")
    
    print(f"📊 Batch summary report saved to: {report_path}")


def main():
    """Main function with hardcoded parameters."""
    # Configuration parameters - modify these as needed
    templates_dir = Path('data/datasets/templates')  # Directory containing template images
    objects_dir = Path('data/datasets/separated_objects')  # Directory containing cut object images
    masks_dir = Path('data/datasets/separated_objects/masks')  # Directory containing object masks
    output_dir = Path('data/output/matching_results')  # Output directory for results
    binary_method = "adaptive"  # Binary conversion method
    matching_method = "contour_matching"  # Matching method
    
    # Test with a single pair first to verify sophisticated alignment
    print("🧪 Testing sophisticated point-based alignment with single pair...")
    
    # Initialize matcher
    matcher = ImageMatcher(str(output_dir))
    
    # Test single pair
    template_path = templates_dir / 'template_1.png'
    object_path = objects_dir / '20250919_223238_object_07.png'
    
    if template_path.exists() and object_path.exists():
        print(f"🔍 Testing: {template_path.name} vs {object_path.name}")
        results = matcher.match_images(template_path, object_path, binary_method, matching_method, masks_dir)
        
        if results:
            print(f"✅ Test successful! Overlap score: {results['overlap_score']:.3f}")
            print("🚀 Proceeding with full batch processing...")
            
            # Clear results and run full batch
            import shutil
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all pairs
            process_all_pairs(templates_dir, objects_dir, output_dir, binary_method, matching_method, masks_dir)
        else:
            print("❌ Test failed, check implementation")
    else:
        print("❌ Test files not found, proceeding with full batch")
        # Process all pairs
        process_all_pairs(templates_dir, objects_dir, output_dir, binary_method, matching_method, masks_dir)


if __name__ == "__main__":
    main()
