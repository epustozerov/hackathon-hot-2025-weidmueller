/**
 * Index page JavaScript - handles file upload and processing
 */

// Store selected points
let selectedPoints = [];
let currentSessionId = null;
let currentFilename = null;

$(document).ready(function() {
    // Check system status
    checkSystemStatus();
    
    // File input change handler
    $('#file-input').change(function() {
        const file = this.files[0];
        if (file) {
            previewImage(file);
        }
    });
    
    // Form submit handler
    $('#upload-form').submit(function(e) {
        e.preventDefault();
        uploadAndProcess();
    });
    
    // Clear points handler
    $('#clear-points').click(function() {
        selectedPoints = [];
        updatePointsDisplay();
    });
});

function checkSystemStatus() {
    $.get('/api/status')
        .done(function(data) {
            if (data.status === 'ready') {
                $('#status-card').hide();
                $('#upload-card').show();
            } else {
                $('#status-text').text('System is initializing...');
                setTimeout(checkSystemStatus, 2000);
            }
        })
        .fail(function() {
            $('#status-text').text('System unavailable');
            $('#status-spinner').hide();
        });
}

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        $('#image-preview').attr('src', e.target.result);
        $('#image-preview-container').show();
        $('#point-selection').show();
        
        // Add click handler for point selection
        $('#image-preview').off('click').on('click', function(event) {
            const rect = this.getBoundingClientRect();
            const x = Math.round((event.clientX - rect.left) * (this.naturalWidth / rect.width));
            const y = Math.round((event.clientY - rect.top) * (this.naturalHeight / rect.height));
            
            selectedPoints.push({x: x, y: y});
            updatePointsDisplay();
        });
    };
    reader.readAsDataURL(file);
}

function updatePointsDisplay() {
    const container = $('#selected-points');
    container.empty();
    
    if (selectedPoints.length === 0) {
        container.html('<small class="text-muted">No points selected</small>');
        return;
    }
    
    selectedPoints.forEach(function(point, index) {
        container.append(`
            <span class="badge bg-primary me-1 mb-1">
                Point ${index + 1}: (${point.x}, ${point.y})
                <button type="button" class="btn-close btn-close-white ms-1" 
                        data-point-index="${index}"></button>
            </span>
        `);
    });
    
    // Add event handlers for remove buttons
    $('[data-point-index]').click(function() {
        const index = parseInt($(this).data('point-index'));
        removePoint(index);
    });
}

function removePoint(index) {
    selectedPoints.splice(index, 1);
    updatePointsDisplay();
}

function uploadAndProcess() {
    const formData = new FormData();
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a file');
        return;
    }
    
    // Validate file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
        showError('File size must be less than 50MB');
        return;
    }
    
    formData.append('file', file);
    
    // Add points if any
    if (selectedPoints.length > 0) {
        const pointsStr = selectedPoints.map(p => `${p.x},${p.y}`).join(';');
        formData.append('points', pointsStr);
    }
    
    // Show processing
    $('#upload-card').hide();
    $('#processing-card').show();
    updateProgress(20, 'Uploading file...');
    
    // Upload file
    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.success) {
                currentSessionId = response.session_id;
                currentFilename = response.filename;
                updateProgress(40, 'File uploaded successfully');
                
                // Start processing
                setTimeout(() => processImage(), 1000);
            } else {
                showError(response.error || 'Upload failed');
            }
        },
        error: function(xhr) {
            const response = xhr.responseJSON;
            if (xhr.status === 413) {
                showError('File too large. Please select a file smaller than 50MB.');
            } else {
                showError(response ? response.error : 'Upload failed');
            }
        }
    });
}

function processImage() {
    updateProgress(60, 'Running segmentation...');
    
    let url = `/process/${currentSessionId}/${currentFilename}`;
    if (selectedPoints.length > 0) {
        const pointsStr = selectedPoints.map(p => `${p.x},${p.y}`).join(';');
        url += `?points=${encodeURIComponent(pointsStr)}`;
    }
    
    $.get(url)
        .done(function(response) {
            if (response.success) {
                updateProgress(100, 'Processing complete!');
                showResults(response);
            } else {
                showError(response.error || 'Processing failed');
            }
        })
        .fail(function(xhr) {
            const response = xhr.responseJSON;
            showError(response ? response.error : 'Processing failed');
        });
}

function updateProgress(percent, message) {
    $('#progress-bar').css('width', percent + '%');
    
    // Update messages to quality control terminology
    const qcMessages = {
        'Uploading file...': 'Uploading part image...',
        'File uploaded successfully': 'Image uploaded successfully',
        'Running segmentation...': 'Analyzing and filtering major parts...',
        'Processing complete!': 'Major parts analysis complete!'
    };
    
    const qcMessage = qcMessages[message] || message;
    $('#processing-status').text(qcMessage);
}

function showResults(data) {
    $('#processing-card').hide();
    $('#mask-count').text(data.num_masks);
    $('#object-count').text(data.num_objects);
    $('#view-results-btn').attr('href', `/results/${data.session_id}`);
    $('#results-preview').show();
}

function showError(message) {
    $('#processing-card').hide();
    $('#error-message').text(message);
    $('#error-alert').show();
}
