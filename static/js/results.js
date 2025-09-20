/**
 * Results page JavaScript - handles gallery interactions
 */

$(document).ready(function() {
    // Handle image modal
    $('.gallery-image').click(function() {
        const imageSrc = $(this).data('image-src');
        const imageTitle = $(this).data('image-title');
        
        $('#modalImage').attr('src', imageSrc);
        $('#imageModalLabel').text(imageTitle);
        $('#downloadLink').attr('href', imageSrc);
    });
    
    // Add hover effects
    $('.gallery-image').hover(
        function() {
            $(this).css('transform', 'scale(1.05)');
            $(this).css('transition', 'transform 0.2s');
        },
        function() {
            $(this).css('transform', 'scale(1)');
        }
    );
    
    // Initialize lazy loading if needed
    if (typeof ImageProcessor !== 'undefined' && ImageProcessor.initializeLazyLoading) {
        ImageProcessor.initializeLazyLoading();
    }
});
