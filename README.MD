# PHinterland Hack - Object Detection & Classification

Our solution for the Weidmüller challenge at Hinterland of Things 2025. We built a computer vision pipeline that finds, aligns, and classifies objects from images.

The system has four main parts: **`app_segmentation.py`** extracts objects from raw images, **`app_alignment.py`** cuts and rotates them properly, **`app_matching.py`** tests 216 different rotation/scale combinations to find the best alignment, and **`app_classifier.py`** uses neural networks to classify the results. We use standard CV metrics like IoU, SSIM, and keypoint matching.

Main features: mask-based object detection, automatic rotation correction, keypoint extraction from object contours and corners, and similarity scoring. Outputs include aligned images, JSON results, and comparison visualizations - useful for quality control and automated inspection.