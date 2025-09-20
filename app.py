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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configuration
UPLOAD_FOLDER = Path('web_uploads')
RESULTS_FOLDER = Path('web_results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Large Object Filtering Configuration - VERY LARGE OBJECTS ONLY
MIN_OBJECT_AREA = 50000    # Minimum area in pixels for an object to be considered "very large"
MIN_OBJECT_WIDTH = 200     # Minimum width in pixels
MIN_OBJECT_HEIGHT = 200    # Minimum height in pixels

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# Global processing instances
segmenter = None
aligner = None


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
    """Initialize the segmentation and alignment processors."""
    global segmenter, aligner
    try:
        print("🔧 Initializing SAM 2.0 segmenter (largest model for binary segmentation)...")
        
        # Use SAM 2.0 large model for maximum accuracy binary segmentation (better compatibility)
        selected_model = 'sam2_hiera_large'
        
        print(f"🤖 Using SAM 2.0 largest model: {selected_model}")
        print(f"📋 Configured for binary segmentation only")
        
        # Verify the SAM 2.0 model exists in the expected location
        weights_dir = Path('models/segmentation_sam/weights')
        model_path = weights_dir / 'sam2_hiera_large.pt'
        
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
        print("✅ SAM 2.1 segmenter initialized")
        
        print("🔧 Initializing object aligner...")
        aligner = ObjectAlignment(
            output_dir=str(RESULTS_FOLDER / 'alignment'),
            rotate_image=True
        )
        print("✅ Object aligner initialized")
        
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
        self.masks_dir.mkdir(exist_ok=True)
        self.objects_dir.mkdir(exist_ok=True)
    
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
            
            # Step 2: Object cutting and alignment using the binary mask (very large objects only)
            print("✂️ Step 2: Cutting and aligning very large objects from binary mask...")
            object_paths = []
            
            # Use global configuration for large object filtering
            
            if binary_mask_path.exists():
                try:
                    # Use ObjectAlignment to extract and align objects from binary mask
                    all_objects = aligner.separate_objects_from_binary(binary_mask_path, image_path)
                    
                    print(f"🔍 Found {len(all_objects)} total objects, filtering for very large objects...")
                    
                    very_large_objects = []
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
                        
                        # Filter for very large objects only
                        if (area >= MIN_OBJECT_AREA and 
                            width >= MIN_OBJECT_WIDTH and 
                            height >= MIN_OBJECT_HEIGHT):
                            very_large_objects.append(obj_info)
                            print(f"✅ Very large object: area={area}, size={width}x{height}")
                        else:
                            filtered_count += 1
                            print(f"⚪ Filtered smaller object: area={area}, size={width}x{height}")
                    
                    print(f"📊 Keeping {len(very_large_objects)} very large objects, filtered {filtered_count} smaller objects")
                    
                    # Process only the very large objects
                    for j, obj_info in enumerate(very_large_objects):
                        # Align the object
                        aligned_img = aligner.align_object(obj_info)
                        
                        if aligned_img is not None:
                            # Save aligned object
                            obj_filename = f"very_large_object_{j:02d}.png"
                            obj_path = self.objects_dir / obj_filename
                            
                            cv2.imwrite(str(obj_path), aligned_img)
                            
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
                                'is_very_large_object': True
                            })
                            print(f"✅ Extracted very large object {j}: {obj_filename} (area: {area})")
                
                except Exception as e:
                    print(f"⚠️ Error processing binary mask: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"✅ Extracted {len(object_paths)} very large objects total")
            
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
                'masks': mask_files,
                'objects': object_paths,
                'original_image': str(image_path),
                'segmentation_info': {
                    'prompt_type': prompt_type,
                    'points_used': len(points) if points else 0,
                    'model': segmenter.model_name,
                    'segmentation_type': 'binary_only'
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
    
    # Get masks and objects
    masks_dir = session_dir / 'masks'
    objects_dir = session_dir / 'objects'
    
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
    
    return render_template('results.html', 
                         session_id=session_id,
                         masks=masks,
                         objects=objects)


@app.route('/image/<session_id>/<image_type>/<filename>')
def serve_image(session_id, image_type, filename):
    """Serve processed images."""
    session_dir = RESULTS_FOLDER / session_id
    
    if image_type == 'masks':
        image_path = session_dir / 'masks' / filename
    elif image_type == 'objects':
        image_path = session_dir / 'objects' / filename
    else:
        return 'Invalid image type', 404
    
    if not image_path.exists():
        return 'Image not found', 404
    
    return send_file(str(image_path))


@app.route('/api/status')
def api_status():
    """API endpoint to check system status."""
    return jsonify({
        'status': 'ready' if segmenter and aligner else 'initializing',
        'segmenter_ready': segmenter is not None,
        'aligner_ready': aligner is not None
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