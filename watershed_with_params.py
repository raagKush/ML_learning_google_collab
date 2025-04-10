import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from matplotlib.widgets import Slider
from matplotlib.widgets import Slider, CheckButtons

# Load your image here (example):
# img = cv2.imread('your_image.jpg')

# Initial thresholding using Otsu's method
ret, thld_img = cv2.threshold(img[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thld_img = 255 - thld_img  # Inverted binarization by default

# Function to update the image based on slider values and checkbox state
def update(val):
    # Get the updated parameter values from sliders
    multiplier = slider_multiplier.val
    kernel_size_open = int(slider_kernel_open.val)
    kernel_size_dilate = int(slider_kernel_dilate.val)
    dilate_iterations = int(slider_dilate_iterations.val)
    otsu_threshold = slider_otsu.val
    reverse_binarize = checkbox_reverse_binarization.get_status()[0]  # Get checkbox state

    # Reverse the binarization if checkbox is checked
    if reverse_binarize:
        thld_img_reversed = 255 - thld_img
    else:
        thld_img_reversed = thld_img

    # Recreate opening operation with updated kernel size
    kernel_open = np.ones((kernel_size_open, kernel_size_open), np.uint8)
    opening = cv2.morphologyEx(thld_img_reversed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Recalculate sure background with updated dilation kernel size and iterations
    kernel_dilate = np.ones((kernel_size_dilate, kernel_size_dilate), np.uint8)
    sure_bg = cv2.dilate(opening, kernel_dilate, iterations=dilate_iterations)

    # Distance Transform to get sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, multiplier * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Subtract sure foreground from sure background to get the region of interest (ROI)
    roi = cv2.subtract(sure_bg, sure_fg)

    # Connected Components to label the sure foreground
    ret2, markers = cv2.connectedComponents(sure_fg)

    # Mark sure foreground as 1 and the ROI as 0
    markers = markers + 1
    markers[roi == 255] = 0

    # Apply Watershed algorithm with the updated markers
    markers_watershed = markers.copy()
    markers_watershed = cv2.watershed(img, markers_watershed)
    
    # Color the labeled regions using label2rgb
    colored_overlay = label2rgb(markers_watershed, bg_label=1)  # Background label set to 0 to avoid coloring the background
    
    # If reverse binarization is selected, show the boundary in white
    if reverse_binarize:
        img_copy = colored_overlay  # Already colored
        img_copy[markers_watershed == -1] = [255, 255, 255]  # Boundary in white
    else:
        img_copy = colored_overlay  # Already colored
        img_copy[markers_watershed == -1] = [0, 0, 255]  # Boundary in red

    # Update the display
    ax_segmentation.imshow(img_copy)
    ax_segmentation.set_title(f"Segmentation (Multiplier: {multiplier:.2f})")
    
    # Update the small plots (input image, sure_bg, and sure_fg)
    ax_input.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_input.set_title("Input Image")
    ax_input.axis('off')

    ax_sure_bg.imshow(sure_bg, cmap='gray')
    ax_sure_bg.set_title("Sure Background")
    ax_sure_bg.axis('off')

    ax_sure_fg.imshow(sure_fg, cmap='gray')
    ax_sure_fg.set_title("Sure Foreground")
    ax_sure_fg.axis('off')

# Create the figure and axis
fig, (ax_segmentation, ax_input, ax_sure_bg, ax_sure_fg) = plt.subplots(1, 4, figsize=(12, 6))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3)

# Show the initial image
ax_segmentation.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax_segmentation.set_title("Adjust Parameters for Segmentation")


# Add sliders for each parameter
ax_slider_multiplier = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider_multiplier = Slider(ax_slider_multiplier, 'Multiplier', 0.1, 1.0, valinit=0.5, valstep=0.01)

ax_slider_kernel_open = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider_kernel_open = Slider(ax_slider_kernel_open, 'Kernel Size Open', 3, 15, valinit=6, valstep=2)

ax_slider_kernel_dilate = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider_kernel_dilate = Slider(ax_slider_kernel_dilate, 'Kernel Size Dilate', 3, 15, valinit=3, valstep=2)

ax_slider_dilate_iterations = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider_dilate_iterations = Slider(ax_slider_dilate_iterations, 'Dilate Iterations', 1, 10, valinit=3, valstep=1)

ax_slider_otsu = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider_otsu = Slider(ax_slider_otsu, 'Otsu Threshold', 0.1, 1.0, valinit=0.5, valstep=0.01)

# Add checkbox to reverse binarization
ax_checkbox_reverse_binarization = plt.axes([0.1, 0.3, 0.03,0.04], facecolor='lightgoldenrodyellow')
checkbox_reverse_binarization = CheckButtons(ax_checkbox_reverse_binarization, ['Reverse Binarization'], [False])

# Attach the update function to the sliders and checkbox
slider_multiplier.on_changed(update)
slider_kernel_open.on_changed(update)
slider_kernel_dilate.on_changed(update)
slider_dilate_iterations.on_changed(update)
slider_otsu.on_changed(update)
checkbox_reverse_binarization.on_clicked(update)

# Show the plot
plt.show()
