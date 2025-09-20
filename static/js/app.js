/**
 * Image Processor Web App JavaScript
 * Handles file uploads, image processing, and UI interactions
 */

// Global variables
let processingTimeout;
let statusCheckInterval;

$(document).ready(function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize other components
    initializeFileUpload();
    initializeImagePreview();
    initializeErrorHandling();
});

/**
 * Initialize file upload functionality
 */
function initializeFileUpload() {
    // Drag and drop support
    const fileInput = document.getElementById('file-input');
    const uploadCard = document.getElementById('upload-card');
    
    if (!fileInput || !uploadCard) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadCard.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    uploadCard.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        uploadCard.classList.add('border-primary');
    }
    
    function unhighlight(e) {
        uploadCard.classList.remove('border-primary');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            $(fileInput).trigger('change');
        }
    }
}

/**
 * Initialize image preview functionality
 */
function initializeImagePreview() {
    // Add click-to-zoom for gallery images
    $(document).on('click', '.gallery-image', function() {
        const imageSrc = $(this).attr('src');
        const imageTitle = $(this).attr('alt');
        
        if (imageSrc && imageTitle) {
            showImageModal(imageSrc, imageTitle);
        }
    });
}

/**
 * Show image in modal
 */
function showImageModal(imageSrc, imageTitle) {
    $('#modalImage').attr('src', imageSrc);
    $('#imageModalLabel').text(imageTitle);
    $('#downloadLink').attr('href', imageSrc);
    $('#imageModal').modal('show');
}

/**
 * Initialize error handling
 */
function initializeErrorHandling() {
    // Global AJAX error handler
    $(document).ajaxError(function(event, xhr, settings, thrownError) {
        console.error('AJAX Error:', {
            url: settings.url,
            status: xhr.status,
            error: thrownError,
            response: xhr.responseText
        });
        
        // Don't show error for status checks
        if (settings.url.includes('/api/status')) {
            return;
        }
        
        let errorMessage = 'An unexpected error occurred';
        
        try {
            const response = JSON.parse(xhr.responseText);
            if (response.error) {
                errorMessage = response.error;
            }
        } catch (e) {
            // Use default message
        }
        
        showNotification('Error', errorMessage, 'danger');
    });
}

/**
 * Show notification toast
 */
function showNotification(title, message, type = 'info') {
    const toastHtml = `
        <div class="toast align-items-center text-white bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${title}:</strong> ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    // Create toast container if it doesn't exist
    if (!document.getElementById('toast-container')) {
        $('body').append('<div id="toast-container" class="toast-container position-fixed top-0 end-0 p-3"></div>');
    }
    
    const $toast = $(toastHtml);
    $('#toast-container').append($toast);
    
    const toast = new bootstrap.Toast($toast[0], {
        autohide: true,
        delay: 5000
    });
    
    toast.show();
    
    // Remove toast element after it's hidden
    $toast.on('hidden.bs.toast', function() {
        $(this).remove();
    });
}

/**
 * Validate file before upload
 */
function validateFile(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff'];
    const maxSize = 50 * 1024 * 1024; // 50MB
    
    if (!allowedTypes.includes(file.type)) {
        showNotification('Invalid File', 'Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, TIFF)', 'warning');
        return false;
    }
    
    if (file.size > maxSize) {
        showNotification('File Too Large', 'File size must be less than 50MB', 'warning');
        return false;
    }
    
    return true;
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Show loading state
 */
function showLoading(element, message = 'Loading...') {
    const $element = $(element);
    const loadingHtml = `
        <div class="loading-overlay">
            <div class="text-center">
                <div class="spinner-border loading-spinner text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="mt-2">${message}</div>
            </div>
        </div>
    `;
    
    $element.css('position', 'relative').append(loadingHtml);
}

/**
 * Hide loading state
 */
function hideLoading(element) {
    $(element).find('.loading-overlay').remove();
}

/**
 * Debounce function
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
}

/**
 * Animate count up
 */
function animateCount(element, target, duration = 1000) {
    const $element = $(element);
    const start = parseInt($element.text()) || 0;
    const increment = (target - start) / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= target) || (increment < 0 && current <= target)) {
            current = target;
            clearInterval(timer);
        }
        $element.text(Math.floor(current));
    }, 16);
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Copied', 'Text copied to clipboard', 'success');
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            showNotification('Error', 'Failed to copy text', 'danger');
        });
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            document.execCommand('copy');
            showNotification('Copied', 'Text copied to clipboard', 'success');
        } catch (err) {
            console.error('Fallback copy failed: ', err);
            showNotification('Error', 'Failed to copy text', 'danger');
        }
        document.body.removeChild(textArea);
    }
}

/**
 * Check if element is in viewport
 */
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

/**
 * Smooth scroll to element
 */
function scrollToElement(element, offset = 0) {
    const $element = $(element);
    if ($element.length) {
        $('html, body').animate({
            scrollTop: $element.offset().top - offset
        }, 500);
    }
}

/**
 * Initialize lazy loading for images
 */
function initializeLazyLoading() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });

        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    } else {
        // Fallback for browsers without IntersectionObserver
        document.querySelectorAll('img[data-src]').forEach(img => {
            img.src = img.dataset.src;
        });
    }
}

/**
 * Format timestamp
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Generate random ID
 */
function generateId(length = 8) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}

/**
 * Handle window resize events
 */
$(window).on('resize', debounce(function() {
    // Handle responsive adjustments
    const width = $(window).width();
    
    if (width < 768) {
        // Mobile adjustments
        $('.gallery-image').css('height', '120px');
    } else if (width < 1200) {
        // Tablet adjustments
        $('.gallery-image').css('height', '150px');
    } else {
        // Desktop adjustments
        $('.gallery-image').css('height', '200px');
    }
}, 250));

/**
 * Initialize keyboard shortcuts
 */
$(document).on('keydown', function(e) {
    // Escape key to close modals
    if (e.key === 'Escape') {
        $('.modal').modal('hide');
    }
    
    // Ctrl+Enter to submit forms
    if (e.ctrlKey && e.key === 'Enter') {
        const activeForm = $('form:visible').first();
        if (activeForm.length) {
            activeForm.submit();
        }
    }
});

// Export functions for global use
window.ImageProcessor = {
    showNotification,
    validateFile,
    formatFileSize,
    showLoading,
    hideLoading,
    animateCount,
    copyToClipboard,
    scrollToElement,
    showImageModal
};
