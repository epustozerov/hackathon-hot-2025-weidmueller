#!/usr/bin/env python3
"""
Flask Web Application for Image Segmentation and Object Alignment

This web application provides an interface for users to:
1. Upload images
2. Process them through SAM 2 segmentation
3. Cut and align individual objects
4. Display results in an interactive web interface
"""

import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import cv2
import numpy as np

# Import our processing modules
from modules.segmentation_sam import SAMImageSegmenter
from modules.alignment.methods import ObjectAlignment
from modules.comparison.comparison import find_best_template_match
from app_matching import ImageMatcher
from app_classifier import ObjectClassifier

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configuration
UPLOAD_FOLDER = Path('web_uploads')
RESULTS_FOLDER = Path('web_results')
TEMPLATES_FOLDER = Path('data/datasets/templates')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Two-Stage Object Filtering Configuration
# Stage 1: Early filtering (at segmentation) - Skip objects < 10,000 pixels
EARLY_FILTER_MIN_AREA = 1000  # Implemented in modules/alignment/methods.py

# Stage 2: Late filtering (before display) - Show only massive objects
MIN_OBJECT_AREA = 1000    # Minimum area in pixels for an object to be considered "massive"
MIN_OBJECT_WIDTH = 15       # Minimum width in pixels
MIN_OBJECT_HEIGHT = 15      # Minimum height in pixels

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)
TEMPLATES_FOLDER.mkdir(exist_ok=True)

# Global processing instances
segmenter = None
aligner = None
matcher = None
classifier = None


@app.after_request
def after_request(response):
    """Add security headers to all responses."""
    # Content Security Policy
    csp = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net https://code.jquery.com; "
        "style-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline'; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self';"
    )
    response.headers['Content-Security-Policy'] = csp
    
    # Other security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large. Please select a file smaller than 50MB.',
        'max_size': '50MB'
    }), 413


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file too large error from werkzeug."""
    return jsonify({
        'error': 'File too large. Please select a file smaller than 50MB.',
        'max_size': '50MB'
    }), 413


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_processors():
    """Initialize the segmentation, alignment, and template matching processors."""
    global segmenter, aligner, matcher, classifier
    try:
        print("🔧 Initializing SAM 2.0 segmenter (small model for balanced performance)...")
        
        # Use SAM 2.0 small model for balanced performance (better accuracy than tiny)
        selected_model = 'sam2_hiera_small'
        
        print(f"🤖 Using SAM 2.0 small model: {selected_model}")
        print(f"📋 Configured for balanced accuracy and speed")
        
        # Verify the SAM 2.0 small model exists in the expected location
        weights_dir = Path('models/segmentation_sam/weights')
        model_path = weights_dir / 'sam2_hiera_small.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"SAM 2.0 model not found at: {model_path}")
        
        print(f"📍 Model path: {model_path}")
        print(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Initialize SAM segmenter (it will automatically use the organized directory structure)
        segmenter = SAMImageSegmenter(
            model_name=selected_model,
            input_folder=str(UPLOAD_FOLDER),
            output_folder=str(RESULTS_FOLDER / 'segmentation')
        )
        print("✅ SAM 2.0 segmenter initialized")
        
        print("🔧 Initializing object aligner...")
        aligner = ObjectAlignment(
            output_dir=str(RESULTS_FOLDER / 'alignment'),
            rotate_image=True
        )
        print("✅ Object aligner initialized")
        
        print("🔧 Initializing template matcher...")
        matcher = ImageMatcher(str(RESULTS_FOLDER / 'matching'))
        print("✅ Template matcher initialized")
        
        print("🔧 Initializing object classifier...")
        classifier = ObjectClassifier(
            output_dir=str(RESULTS_FOLDER / 'classification'),
            masks_dir=TEMPLATES_FOLDER / 'masks'
        )
        print("✅ Object classifier initialized")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing processors: {e}")
        return False


def make_json_serializable(obj):
    """Convert numpy types and other non-JSON serializable types to Python native types."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):  # numpy array - handle before checking for .item()
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):  # numpy scalar types
        return obj.item()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'item') and hasattr(obj, 'shape') and obj.shape == ():  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # other array-like objects
        return obj.tolist()
    else:
        return obj


class ImageProcessor:
    """Handles the complete image processing pipeline."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = RESULTS_FOLDER / session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.masks_dir = self.session_dir / 'masks'
        self.objects_dir = self.session_dir / 'objects'
        self.templates_dir = self.session_dir / 'templates'
        self.matching_dir = self.session_dir / 'matching'
        self.classification_dir = self.session_dir / 'classification'
        
        self.masks_dir.mkdir(exist_ok=True)
        self.objects_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        self.matching_dir.mkdir(exist_ok=True)
        self.classification_dir.mkdir(exist_ok=True)
    
    def process_image(self, image_path: Path, points: List[tuple] = None) -> Dict:
        """
        Process an uploaded image through segmentation and alignment.
        
        Args:
            image_path: Path to the uploaded image
            points: Optional list of (x, y) points for guided segmentation
            
        Returns:
            Dictionary with processing results
        """
        try:
            print(f"🚀 Processing image: {image_path.name}")
            
            # Step 1: Segmentation - Generate mask images
            print("🎯 Step 1: Running SAM segmentation...")
            
            # Create temporary segmentation output directory
            temp_seg_dir = self.session_dir / 'temp_segmentation'
            temp_seg_dir.mkdir(exist_ok=True)
            
            # Configure segmenter to use our session directory
            original_output_folder = segmenter.output_folder
            segmenter.output_folder = temp_seg_dir
            
            try:
                if points:
                    # Use point-based segmentation with provided points
                    labels = [1] * len(points)  # All foreground
                    seg_result = segmenter.segment_with_points(
                        image_path, points, labels, save_results=True
                    )
                else:
                    # Use automatic segmentation (everything mode) for binary segmentation only
                    seg_result = segmenter.methods.segment_everything(
                        str(image_path), points_per_side=32
                    )
                    
                    # For everything mode, save only binary results (no colored variations)
                    if seg_result.get('success', False):
                        save_result = segmenter.methods.save_segmentation_results(
                            str(image_path),
                            seg_result,
                            temp_seg_dir,
                            save_types=['binary']  # Only binary mask needed
                        )
                        if save_result.get('success', False):
                            print(f"✅ Binary segmentation mask saved")
                        else:
                            print(f"⚠️ Failed to save binary segmentation results")
            finally:
                # Restore original output folder
                segmenter.output_folder = original_output_folder
            
            if not seg_result.get('success', False):
                return {
                    'success': False,
                    'error': f"Segmentation failed: {seg_result.get('error', 'Unknown error')}"
                }
            
            # Find the generated binary mask file
            image_stem = image_path.stem
            prompt_type = seg_result.get('prompt_type', 'everything')
            
            # Look for the binary mask file
            binary_mask_pattern = f"{image_stem}_sam_{prompt_type}_binary.png"
            binary_mask_path = temp_seg_dir / binary_mask_pattern
            
            if not binary_mask_path.exists():
                # Try alternative patterns
                possible_patterns = [
                    f"{image_stem}_sam_points_binary.png",
                    f"{image_stem}_sam_everything_binary.png",
                    f"{image_stem}_binary.png"
                ]
                
                for pattern in possible_patterns:
                    test_path = temp_seg_dir / pattern
                    if test_path.exists():
                        binary_mask_path = test_path
                        break
                
                if not binary_mask_path.exists():
                    return {
                        'success': False,
                        'error': f"Binary mask not found. Expected: {binary_mask_pattern}"
                    }
            
            # Copy binary segmentation result to masks directory
            mask_files = []
            suffix = 'binary'
            src_pattern = f"{image_stem}_sam_{prompt_type}_{suffix}.png"
            src_path = temp_seg_dir / src_pattern
            
            if not src_path.exists():
                # Try alternative patterns
                alt_patterns = [
                    f"{image_stem}_sam_points_{suffix}.png",
                    f"{image_stem}_sam_everything_{suffix}.png"
                ]
                for alt_pattern in alt_patterns:
                    alt_path = temp_seg_dir / alt_pattern
                    if alt_path.exists():
                        src_path = alt_path
                        break
            
            if src_path.exists():
                dst_filename = f"mask_binary.png"
                dst_path = self.masks_dir / dst_filename
                shutil.copy2(src_path, dst_path)
                mask_files.append({'filename': dst_filename, 'path': str(dst_path), 'type': 'binary'})
                print(f"✅ Saved binary mask: {dst_filename}")
            
            print(f"✅ Generated binary segmentation mask")
            
            # Step 2: Object cutting and alignment using the binary mask (massive objects only)
            print("✂️ Step 2: Cutting and aligning massive objects from binary mask...")
            print(f"   📊 Two-stage filtering: Early (≥{EARLY_FILTER_MIN_AREA}px) + Late (≥{MIN_OBJECT_AREA}px)")
            object_paths = []
            
            # Use global configuration for large object filtering
            
            if binary_mask_path.exists():
                try:
                    # Use ObjectAlignment to extract and align objects from binary mask
                    all_objects = aligner.separate_objects_from_binary(binary_mask_path, image_path)
                    
                    print(f"🔍 Found {len(all_objects)} objects after early filtering, applying late filtering for massive objects...")
                    
                    massive_objects = []
                    filtered_count = 0
                    
                    for obj_info in all_objects:
                        # Get object dimensions
                        area = obj_info.get('area', 0)
                        bbox = obj_info.get('bbox', [0, 0, 0, 0])
                        
                        # Convert numpy types to Python native types
                        if hasattr(area, 'item'):
                            area = area.item()
                        else:
                            area = int(area)
                            
                        if bbox is not None and len(bbox) >= 4:
                            # Calculate width and height from bbox [x, y, w, h]
                            width = int(bbox[2])
                            height = int(bbox[3])
                        else:
                            width = height = 0
                        
                        # Filter for massive objects only (Stage 2 filtering)
                        if (area >= MIN_OBJECT_AREA and 
                            width >= MIN_OBJECT_WIDTH and 
                            height >= MIN_OBJECT_HEIGHT):
                            massive_objects.append(obj_info)
                            print(f"✅ Massive object: area={area}, size={width}x{height}")
                        else:
                            filtered_count += 1
                            print(f"⚪ Filtered medium object: area={area}, size={width}x{height}")
                    
                    print(f"📊 Final result: {len(massive_objects)} massive objects, filtered {filtered_count} medium objects")
                    
                    # Process only the massive objects
                    for j, obj_info in enumerate(massive_objects):
                        # Align the object
                        aligned_img = aligner.align_object(obj_info)
                        
                        if aligned_img is not None:
                            # Save aligned object
                            obj_filename = f"massive_object_{j:02d}.png"
                            obj_path = self.objects_dir / obj_filename
                            
                            cv2.imwrite(str(obj_path), aligned_img)
                            
                            # Save aligned object mask
                            if 'mask' in obj_info and obj_info['mask'] is not None:
                                mask_filename = f"massive_object_{j:02d}_mask.png"
                                mask_path = self.objects_dir / mask_filename
                                
                                # Apply the same alignment process to the mask
                                try:
                                    mask = obj_info['mask']
                                    target_size = (256, 256)
                                    
                                    # Apply rotation if enabled (same as object)
                                    if aligner.rotate_image:
                                        from modules.alignment.methods import rotate_image_to_horizontal_diagonal
                                        # Use the mask itself to determine rotation angle
                                        mask = rotate_image_to_horizontal_diagonal(mask, mask)
                                    
                                    # Create aligned mask with same process as align_object
                                    aligned_mask = np.zeros(target_size, dtype=np.uint8)
                                    
                                    # Calculate scaling (same as object alignment)
                                    h, w = mask.shape[:2]
                                    scale = min(target_size[0] / w, target_size[1] / h) * 0.8
                                    
                                    # Resize mask
                                    new_w = int(w * scale)
                                    new_h = int(h * scale)
                                    
                                    if new_w > 0 and new_h > 0:
                                        resized_mask = cv2.resize(mask, (new_w, new_h))
                                        
                                        # Center the mask
                                        start_x = (target_size[0] - new_w) // 2
                                        start_y = (target_size[1] - new_h) // 2
                                        
                                        # Place mask in center
                                        aligned_mask[start_y:start_y+new_h, start_x:start_x+new_w] = resized_mask
                                    
                                    cv2.imwrite(str(mask_path), aligned_mask)
                                    print(f"✅ Saved aligned object mask: {mask_filename}")
                                    
                                except Exception as e:
                                    print(f"⚠️ Error aligning mask, saving original: {e}")
                                    cv2.imwrite(str(mask_path), obj_info['mask'])
                                    print(f"✅ Saved original mask: {mask_filename}")
                            
                            # Convert numpy types to Python native types for JSON serialization
                            area = obj_info.get('area', 0)
                            bbox = obj_info.get('bbox', [0, 0, 0, 0])
                            
                            # Ensure area is a Python int
                            if hasattr(area, 'item'):  # numpy scalar
                                area = area.item()
                            else:
                                area = int(area)
                            
                            # Ensure bbox values are Python ints
                            if bbox is not None:
                                bbox = [int(x.item() if hasattr(x, 'item') else x) for x in bbox]
                            else:
                                bbox = [0, 0, 0, 0]
                            
                            object_paths.append({
                                'path': str(obj_path),  # Convert Path to string
                                'filename': obj_filename,
                                'object_index': int(j),
                                'area': area,
                                'bbox': bbox,
                                'is_massive_object': True
                            })
                            print(f"✅ Extracted massive object {j}: {obj_filename} (area: {area})")
                
                except Exception as e:
                    print(f"⚠️ Error processing binary mask: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"✅ Extracted {len(object_paths)} massive objects total")
            
            # Step 3: Template matching and classification (if templates exist and objects were extracted)
            template_results = {}
            classification_results = {}
            
            if len(object_paths) > 0 and TEMPLATES_FOLDER.exists():
                print("🎯 Step 3: Running template matching and classification...")
                
                # Check if templates exist
                template_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    template_files.extend(TEMPLATES_FOLDER.glob(ext))
                
                if template_files:
                    print(f"📋 Found {len(template_files)} templates for matching")
                    
                    try:
                        # Process each extracted object
                        for obj_info in object_paths:
                            obj_path = Path(obj_info['path'])
                            obj_filename = obj_info['filename']
                            
                            print(f"   🔍 Matching {obj_filename} against templates...")
                            
                            # Load object image
                            obj_img = cv2.imread(str(obj_path))
                            if obj_img is None:
                                continue
                            
                            # Find best template match using simple comparison
                            best_template, best_metrics = find_best_template_match(
                                obj_img, TEMPLATES_FOLDER, metric="ssim"
                            )
                            
                            if best_template:
                                print(f"   ✅ Best match: {best_template.name} (SSIM: {best_metrics.get('ssim', 0):.3f})")
                                
                                # Run sophisticated matching with the best template
                                matching_results = matcher.match_images(
                                    best_template, obj_path, 
                                    binary_method="adaptive", 
                                    matching_method="contour_matching",
                                    masks_dir=TEMPLATES_FOLDER / 'masks'
                                )
                                
                                if matching_results:
                                    # Copy matching results to session directory
                                    overlay_filename = f"{obj_path.stem}_vs_{best_template.stem}_overlay.png"
                                    src_overlay = Path(matcher.output_dir) / 'overlay_images' / overlay_filename
                                    dst_overlay = self.matching_dir / overlay_filename
                                    
                                    if src_overlay.exists():
                                        shutil.copy2(src_overlay, dst_overlay)
                                    
                                    template_results[obj_filename] = {
                                        'best_template': best_template.name,
                                        'ssim_score': best_metrics.get('ssim', 0),
                                        'overlap_score': matching_results.get('overlap_score', 0),
                                        'best_position': matching_results.get('best_position', [0, 0]),
                                        'overlay_image': overlay_filename if dst_overlay.exists() else None
                                    }
                                
                                # Run classification with the object
                                try:
                                    classification_result = classifier.compare_with_templates(obj_path, TEMPLATES_FOLDER)
                                    
                                    if classification_result:
                                        # Save classification results to session directory
                                        classification_results[obj_filename] = classification_result
                                        
                                        # Copy edge comparison images
                                        for template_name, template_data in classification_result.items():
                                            edge_filename = f"{obj_path.stem}_vs_{template_name}_edges.png"
                                            aligned_filename = f"{obj_path.stem}_vs_{template_name}_aligned.png"
                                            
                                            src_edge = classifier.edges_dir / edge_filename
                                            src_aligned = classifier.similarity_dir / aligned_filename
                                            
                                            dst_edge = self.classification_dir / edge_filename
                                            dst_aligned = self.classification_dir / aligned_filename
                                            
                                            if src_edge.exists():
                                                shutil.copy2(src_edge, dst_edge)
                                            if src_aligned.exists():
                                                shutil.copy2(src_aligned, dst_aligned)
                                
                                except Exception as e:
                                    print(f"   ⚠️ Classification failed for {obj_filename}: {e}")
                            else:
                                print(f"   ❌ No suitable template match found for {obj_filename}")
                    
                    except Exception as e:
                        print(f"⚠️ Error in template matching: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("📋 No templates found in templates directory")
            else:
                print("📋 Skipping template matching (no objects or templates)")
            
            # Clean up temporary segmentation directory
            try:
                shutil.rmtree(temp_seg_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temp directory: {e}")
            
            # Prepare results
            result = {
                'success': True,
                'session_id': self.session_id,
                'num_masks': len(mask_files),
                'num_objects': len(object_paths),
                'num_template_matches': len(template_results),
                'num_classifications': len(classification_results),
                'masks': mask_files,
                'objects': object_paths,
                'template_matches': template_results,
                'classifications': classification_results,
                'original_image': str(image_path),
                'segmentation_info': {
                    'prompt_type': prompt_type,
                    'points_used': len(points) if points else 0,
                    'model': segmenter.model_name,
                    'segmentation_type': 'binary_only'
                },
                'template_info': {
                    'templates_available': len(template_files) if 'template_files' in locals() else 0,
                    'matching_enabled': TEMPLATES_FOLDER.exists()
                }
            }
            
            # Ensure all data is JSON serializable
            result = make_json_serializable(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in image processing: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}"
            }


@app.route('/')
def index():
    """Main page with upload interface."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, TIFF'}), 400
    
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        
        upload_path = UPLOAD_FOLDER / safe_filename
        file.save(str(upload_path))
        
        # Get optional points from request
        points = []
        if 'points' in request.form:
            try:
                points_data = request.form['points']
                if points_data:
                    # Parse points in format "x1,y1;x2,y2;..."
                    for point_str in points_data.split(';'):
                        if point_str.strip():
                            x, y = map(int, point_str.split(','))
                            points.append((x, y))
            except Exception as e:
                print(f"⚠️ Error parsing points: {e}")
                points = []
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'filename': safe_filename,
            'message': 'File uploaded successfully. Processing will start shortly.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/process/<session_id>/<filename>')
def process_image(session_id, filename):
    """Process the uploaded image."""
    try:
        # Check if processors are initialized
        if not segmenter or not aligner:
            if not init_processors():
                return jsonify({'error': 'Failed to initialize processors'}), 500
        
        upload_path = UPLOAD_FOLDER / filename
        if not upload_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # Get optional points
        points = []
        if 'points' in request.args:
            try:
                points_str = request.args['points']
                for point_str in points_str.split(';'):
                    if point_str.strip():
                        x, y = map(int, point_str.split(','))
                        points.append((x, y))
            except:
                points = []
        
        # Process image
        processor = ImageProcessor(session_id)
        result = processor.process_image(upload_path, points)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/results/<session_id>')
def show_results(session_id):
    """Display processing results."""
    session_dir = RESULTS_FOLDER / session_id
    if not session_dir.exists():
        return render_template('error.html', error='Session not found'), 404
    
    # Get masks, objects, and template results
    masks_dir = session_dir / 'masks'
    objects_dir = session_dir / 'objects'
    matching_dir = session_dir / 'matching'
    classification_dir = session_dir / 'classification'
    
    masks = []
    if masks_dir.exists():
        # Look for binary mask file only
        mask_files = list(masks_dir.glob('*.png'))
        for mask_file in mask_files:
            mask_info = {'filename': mask_file.name, 'type': 'binary'}
            masks.append(mask_info)
        
        # Sort masks for consistent display
        masks.sort(key=lambda x: x['filename'])
    
    objects = []
    if objects_dir.exists():
        objects = [f.name for f in objects_dir.glob('*.png')]
        objects.sort()  # Sort for consistent display
    
    # Get template matching results
    matching_files = []
    if matching_dir.exists():
        matching_files = [f.name for f in matching_dir.glob('*_overlay.png')]
        matching_files.sort()
    
    # Get classification results
    classification_files = []
    if classification_dir.exists():
        edge_files = [f.name for f in classification_dir.glob('*_edges.png')]
        aligned_files = [f.name for f in classification_dir.glob('*_aligned.png')]
        classification_files = {
            'edges': sorted(edge_files),
            'aligned': sorted(aligned_files)
        }
    
    return render_template('results.html', 
                         session_id=session_id,
                         masks=masks,
                         objects=objects,
                         matching_files=matching_files,
                         classification_files=classification_files)


@app.route('/image/<session_id>/<image_type>/<filename>')
def serve_image(session_id, image_type, filename):
    """Serve processed images."""
    session_dir = RESULTS_FOLDER / session_id
    
    if image_type == 'masks':
        image_path = session_dir / 'masks' / filename
    elif image_type == 'objects':
        image_path = session_dir / 'objects' / filename
    elif image_type == 'matching':
        image_path = session_dir / 'matching' / filename
    elif image_type == 'classification':
        image_path = session_dir / 'classification' / filename
    else:
        return 'Invalid image type', 404
    
    if not image_path.exists():
        return 'Image not found', 404
    
    return send_file(str(image_path))


@app.route('/download/<session_id>/<download_type>')
def download_results(session_id, download_type):
    """Download template matching results or classification reports."""
    session_dir = RESULTS_FOLDER / session_id
    if not session_dir.exists():
        return 'Session not found', 404
    
    if download_type == 'template_matching':
        # Create a ZIP file with all template matching results
        import zipfile
        import tempfile
        
        matching_dir = session_dir / 'matching'
        if not matching_dir.exists() or not list(matching_dir.glob('*')):
            return 'No template matching results found', 404
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.close()
        
        try:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all matching files
                for file_path in matching_dir.glob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)
                
                # Add matching results JSON if it exists
                results_json = session_dir / 'template_matching_results.json'
                if results_json.exists():
                    zipf.write(results_json, 'template_matching_results.json')
            
            return send_file(temp_zip.name, 
                           as_attachment=True, 
                           download_name=f'template_matching_results_{session_id}.zip',
                           mimetype='application/zip')
        
        except Exception as e:
            return f'Error creating ZIP file: {str(e)}', 500
    
    elif download_type == 'classification_report':
        # Create a ZIP file with all classification results
        import zipfile
        import tempfile
        import json
        
        classification_dir = session_dir / 'classification'
        if not classification_dir.exists() or not list(classification_dir.glob('*')):
            return 'No classification results found', 404
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.close()
        
        try:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all classification files
                for file_path in classification_dir.glob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.name)
                
                # Generate and add classification summary report
                summary_data = {
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'files': {
                        'edge_comparisons': [f.name for f in classification_dir.glob('*_edges.png')],
                        'aligned_comparisons': [f.name for f in classification_dir.glob('*_aligned.png')]
                    },
                    'summary': f"Classification results for session {session_id}"
                }
                
                # Create temporary JSON file
                temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
                json.dump(summary_data, temp_json, indent=2)
                temp_json.close()
                
                zipf.write(temp_json.name, 'classification_summary.json')
                
                # Clean up temp JSON
                import os
                os.unlink(temp_json.name)
            
            return send_file(temp_zip.name, 
                           as_attachment=True, 
                           download_name=f'classification_report_{session_id}.zip',
                           mimetype='application/zip')
        
        except Exception as e:
            return f'Error creating ZIP file: {str(e)}', 500
    
    else:
        return 'Invalid download type', 404


@app.route('/api/status')
def api_status():
    """API endpoint to check system status."""
    return jsonify({
        'status': 'ready' if segmenter and aligner and matcher and classifier else 'initializing',
        'segmenter_ready': segmenter is not None,
        'aligner_ready': aligner is not None,
        'matcher_ready': matcher is not None,
        'classifier_ready': classifier is not None,
        'templates_available': TEMPLATES_FOLDER.exists() and len(list(TEMPLATES_FOLDER.glob('*.png'))) > 0
    })


if __name__ == '__main__':
    print("🚀 Starting Flask Web Application for Image Processing")
    print("=" * 60)
    
    # Initialize processors
    print("🔧 Initializing processing modules...")
    if init_processors():
        print("✅ All processors initialized successfully")
        print(f"📁 Upload folder: {UPLOAD_FOLDER}")
        print(f"📁 Results folder: {RESULTS_FOLDER}")
        print("🌐 Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to initialize processors")
        print("Please check your SAM 2 installation and model weights")
        exit(1)