"""
Image Comparison and Analysis Methods

This module provides methods for:
- Creating point clouds from images
- Edge detection and segmentation
- Image comparison using various metrics
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def create_point_cloud(image: np.ndarray, method: str = "contour") -> np.ndarray:
    """
    Create a point cloud from an image representing object boundaries.
    
    Args:
        image: Input image as numpy array
        method: Method to extract points ("contour", "edge", "corner")
        
    Returns:
        Array of points representing the object
    """
    if image is None or image.size == 0:
        return np.array([])
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    points = []
    
    if method == "contour":
        # Extract contour points
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Simplify contour to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points.extend(approx.reshape(-1, 2))
    
    elif method == "edge":
        # Extract edge points
        edges = cv2.Canny(gray, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))
        # Sample points to reduce density
        if len(edge_points) > 1000:
            indices = np.random.choice(len(edge_points), 1000, replace=False)
            points = edge_points[indices]
        else:
            points = edge_points
    
    elif method == "corner":
        # Extract corner points
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            points = corners.reshape(-1, 2)
    
    return np.array(points)


def edge_segmentation(image: np.ndarray, method: str = "canny") -> np.ndarray:
    """
    Apply edge detection to find object boundaries.
    
    Args:
        image: Input image as numpy array
        method: Edge detection method ("canny", "sobel", "laplacian", "adaptive")
        
    Returns:
        Edge-detected image
    """
    if image is None or image.size == 0:
        return image
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method == "canny":
        # Standard Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
    
    elif method == "sobel":
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges)
    
    elif method == "laplacian":
        # Laplacian edge detection
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
    
    elif method == "adaptive":
        # Adaptive edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mean_val = np.mean(blur)
        lower_thresh = max(0, mean_val - 30)
        upper_thresh = min(255, mean_val + 30)
        edges = cv2.Canny(blur, int(lower_thresh), int(upper_thresh))
    
    else:
        edges = gray
    
    return edges


def compare_images(image1: np.ndarray, image2: np.ndarray, metrics: List[str] = None) -> Dict[str, float]:
    """
    Compare two images using various similarity metrics.
    
    Args:
        image1: First image
        image2: Second image
        metrics: List of metrics to calculate (default: all available)
        
    Returns:
        Dictionary of similarity scores
    """
    if image1 is None or image2 is None:
        return {}
    
    if metrics is None:
        metrics = ['ssim', 'mse', 'psnr', 'ncc', 'histogram', 'edge']
    
    results = {}
    
    # Ensure images are the same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # Convert to grayscale for some metrics
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
        
    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2
    
    # Calculate requested metrics
    if 'ssim' in metrics:
        results['ssim'] = _calculate_ssim(gray1, gray2)
    
    if 'mse' in metrics:
        results['mse'] = _calculate_mse(gray1, gray2)
    
    if 'psnr' in metrics:
        results['psnr'] = _calculate_psnr(gray1, gray2)
    
    if 'ncc' in metrics:
        results['ncc'] = _calculate_ncc(gray1, gray2)
    
    if 'histogram' in metrics:
        results['histogram'] = _calculate_histogram_similarity(gray1, gray2)
    
    if 'edge' in metrics:
        results['edge'] = _calculate_edge_similarity(gray1, gray2)
    
    return results


def _calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2)
    except ImportError:
        # Manual SSIM calculation
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim


def _calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)


def _calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = _calculate_mse(img1, img2)
    if mse > 0:
        return 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        return float('inf')


def _calculate_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Normalized Cross Correlation."""
    return cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0][0]


def _calculate_histogram_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate histogram correlation."""
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def _calculate_edge_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate edge-based similarity."""
    edges1 = edge_segmentation(img1, "canny")
    edges2 = edge_segmentation(img2, "canny")
    
    # Calculate intersection over union of edges
    intersection = cv2.bitwise_and(edges1, edges2)
    union = cv2.bitwise_or(edges1, edges2)
    
    if np.sum(union) > 0:
        return np.sum(intersection) / np.sum(union)
    else:
        return 0.0


def find_best_template_match(object_image: np.ndarray, template_dir: Path, 
                           metric: str = "ssim") -> Tuple[Path, Dict[str, float]]:
    """
    Find the best matching template for an object image.
    
    Args:
        object_image: Object image to match
        template_dir: Directory containing template images
        metric: Metric to use for comparison ("ssim", "psnr", "edge")
        
    Returns:
        Tuple of (best_template_path, comparison_metrics)
    """
    if not template_dir.exists():
        return None, {}
    
    # Find template images
    template_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    template_files = []
    for ext in template_extensions:
        template_files.extend(template_dir.glob(ext))
        template_files.extend(template_dir.glob(ext.upper()))
    
    if not template_files:
        return None, {}
    
    best_score = -float('inf')
    best_template = None
    best_metrics = {}
    
    for template_path in template_files:
        template_img = cv2.imread(str(template_path))
        if template_img is None:
            continue
        
        # Calculate comparison metrics
        metrics = compare_images(object_image, template_img)
        
        # Select score based on metric
        if metric == "ssim":
            score = metrics.get('ssim', 0)
        elif metric == "psnr":
            score = metrics.get('psnr', 0)
        elif metric == "edge":
            score = metrics.get('edge', 0)
        else:
            score = metrics.get('ssim', 0)
        
        if score > best_score:
            best_score = score
            best_template = template_path
            best_metrics = metrics
    
    return best_template, best_metrics