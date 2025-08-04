import cv2
import numpy as np
import math

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def point_in_box(point, box):
    """Check if a point is inside a bounding box"""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def get_patches_in_boxes(patch_centers, boxes):
    """Get all patch centers that are inside any of the boxes"""
    patches_in_boxes = []
    for center in patch_centers:
        for box in boxes:
            if point_in_box(center, box):
                patches_in_boxes.append(center)
                break
    return patches_in_boxes


def find_nearest_patch_centers(click_x, click_y, patch_centers, threshold_val):
    """Find all patch centers within threshold distance of the clicked point"""
    nearby_centers = []
    nearby_indices = []
    
    for i, (center_x, center_y) in enumerate(patch_centers):
        dist = distance((click_x, click_y), (center_x, center_y))
        if dist < threshold_val:
            nearby_centers.append((center_x, center_y))
            nearby_indices.append(i)
    
    return nearby_centers, nearby_indices


def labeling_session(raw_depth_clean, patch_centers, default_threshold=10, logger=None):
    """
    Conducts a labeling session to select good and bad grasp patches from depth image.
    
    Args:
        raw_depth_clean (np.ndarray): Cleaned depth image.
        patch_centers (list): List of (x, y) coordinates for all patches.
        default_threshold (int): Default threshold for selecting nearby patches.
        logger (DataLabelingLogger, optional): Logger instance for recording session details.
    
    Returns:
        good_patch_center (list): List of (x, y) coordinates for good grasp locations.
        bad_patch_center (list): List of (x, y) coordinates for bad grasp locations.
    """
    if logger:
        logger.log_info("Starting interactive labeling session...")
    else:
        print("Starting labeling session...")
        
    # Local variables for this session
    object_boxes = []
    good_patch_centers = []
    current_box = None
    drawing_box = False
    box_start = None
    threshold = default_threshold
    
    # Track user interactions for logging
    total_clicks = 0
    add_clicks = 0
    remove_clicks = 0
    
    def threshold_callback(val):
        """Callback function for threshold trackbar"""
        nonlocal threshold
        threshold = val
        if logger:
            logger.log_user_interaction("Threshold Changed", f"New threshold: {threshold} pixels")
        else:
            print(f"Selection threshold updated to: {threshold} pixels")
    
    # Stage 1: Box drawing mouse callback
    def on_mouse_box_drawing(event, x, y, flags, param):
        nonlocal drawing_box, box_start, current_box, object_boxes
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_box = True
            box_start = (x, y)
            current_box = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing_box:
                current_box = (box_start[0], box_start[1], x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing_box:
                drawing_box = False
                if current_box:
                    # Ensure box coordinates are in correct order
                    x1, y1, x2, y2 = current_box
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    final_box = (x1, y1, x2, y2)
                    object_boxes.append(final_box)
                    
                    box_area = (x2 - x1) * (y2 - y1)
                    if logger:
                        logger.log_user_interaction("Object Box Created", 
                                                   f"Box {len(object_boxes)}: ({x1},{y1}) to ({x2},{y2}), area: {box_area} pixels")
                    else:
                        print(f"Added object box: {final_box}")
                    current_box = None
    
    # Stage 2: Grasp labeling mouse callback
    def on_mouse_grasp_labeling(event, x, y, flags, param):
        nonlocal good_patch_centers, total_clicks, add_clicks, remove_clicks
        patches_in_boxes = param  # Get patches that are inside boxes
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click to add
            total_clicks += 1
            nearby_centers, nearby_indices = find_nearest_patch_centers(x, y, patches_in_boxes, threshold)
            
            if nearby_centers:
                added_count = 0
                for center in nearby_centers:
                    if center not in good_patch_centers:
                        good_patch_centers.append(center)
                        added_count += 1
                
                if added_count > 0:
                    add_clicks += 1
                    if logger:
                        logger.log_user_interaction("Good Grasps Added", 
                                                   f"Added {added_count} patches at ({x}, {y}), total good: {len(good_patch_centers)}")
                    else:
                        print(f"Added {added_count} good grasp(s) near ({x}, {y})")
                else:
                    if logger:
                        logger.log_user_interaction("Duplicate Selection", 
                                                   f"All {len(nearby_centers)} patches near ({x}, {y}) already selected")
                    else:
                        print(f"All {len(nearby_centers)} point(s) near ({x}, {y}) already selected")
        
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to remove
            total_clicks += 1
            nearby_centers, nearby_indices = find_nearest_patch_centers(x, y, patches_in_boxes, threshold)
            
            if nearby_centers:
                removed_count = 0
                for center in nearby_centers:
                    if center in good_patch_centers:
                        good_patch_centers.remove(center)
                        removed_count += 1
                
                if removed_count > 0:
                    remove_clicks += 1
                    if logger:
                        logger.log_user_interaction("Good Grasps Removed", 
                                                   f"Removed {removed_count} patches at ({x}, {y}), total good: {len(good_patch_centers)}")
                    else:
                        print(f"Removed {removed_count} good grasp(s) near ({x}, {y})")
    
    # Convert depth image to display format (0-255)
    display_depth = ((raw_depth_clean - raw_depth_clean.min()) / 
                    (raw_depth_clean.max() - raw_depth_clean.min()) * 255).astype(np.uint8)
    
    # Convert to BGR for opencv display
    display_img = cv2.applyColorMap(display_depth, cv2.COLORMAP_VIRIDIS)
    
    if logger:
        logger.log_info("=== STAGE 1: Object Detection ===")
    else:
        print("=== STAGE 1: Object Detection ===")
    print("- Draw boxes around objects by clicking and dragging")
    print("- Press 'c' to clear all boxes")
    print("- Press SPACE to proceed to grasp labeling")
    print("- Press ESC to exit")
    
    # Stage 1: Object box drawing
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection", 1600, 1200)
    cv2.setMouseCallback("Object Detection", on_mouse_box_drawing)
    
    while True:
        # Create a copy of the display image
        temp_img = display_img.copy()
        
        # Draw all patch centers as small gray dots
        for center_x, center_y in patch_centers:
            cv2.circle(temp_img, (int(center_x), int(center_y)), 2, (128, 128, 128), -1)
        
        # Draw completed boxes
        for box in object_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw current box being drawn
        if drawing_box and current_box:
            x1, y1, x2, y2 = current_box
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text instructions
        cv2.putText(temp_img, f"Objects detected: {len(object_boxes)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(temp_img, "Draw boxes around objects | 'c': clear | SPACE: next | ESC: exit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the image
        cv2.imshow("Object Detection", temp_img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key - exit
            cv2.destroyAllWindows()
            if logger:
                logger.log_user_interaction("Session Cancelled", "User pressed ESC during object detection")
            return [], []  # Return empty lists if user exits
        elif key == ord('c'):  # Clear all boxes
            if object_boxes:
                cleared_count = len(object_boxes)
                object_boxes = []
                if logger:
                    logger.log_user_interaction("Boxes Cleared", f"Cleared {cleared_count} object boxes")
                else:
                    print("Cleared all boxes")
        elif key == ord(' '):  # Space - proceed to next stage
            if object_boxes:
                break
            else:
                warning_msg = "Please draw at least one object box before proceeding"
                if logger:
                    logger.log_warning(warning_msg)
                else:
                    print(warning_msg)
    
    cv2.destroyAllWindows()
    
    # Get patches that are inside the drawn boxes
    patches_in_boxes = get_patches_in_boxes(patch_centers, object_boxes)
    
    if logger:
        logger.log_object_detection(len(object_boxes))
        logger.log_patches_in_boxes(len(patch_centers), len(patches_in_boxes))
    else:
        print(f"\nFound {len(patches_in_boxes)} patch centers inside object boxes")
    
    if not patches_in_boxes:
        error_msg = "No patch centers found inside boxes. Exiting."
        if logger:
            logger.log_error(error_msg)
        else:
            print(error_msg)
        return [], []
    
    if logger:
        logger.log_info("=== STAGE 2: Grasp Labeling ===")
    else:
        print("\n=== STAGE 2: Grasp Labeling ===")
    print("- Left click on blue dots to select good grasp points (they turn green)")
    print("- Right click on green dots to remove them")
    print("- Only patches inside object boxes can be selected")
    print("- Press ESC when finished labeling")
    
    # Stage 2: Grasp point labeling
    cv2.namedWindow("Grasp Labeling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grasp Labeling", 1600, 1200)
    cv2.setMouseCallback("Grasp Labeling", on_mouse_grasp_labeling, patches_in_boxes)
    
    # Create threshold trackbar
    cv2.createTrackbar("Selection Threshold", "Grasp Labeling", threshold, 100, threshold_callback)
    
    while True:
        # Create a copy of the display image
        temp_img = display_img.copy()
        
        # Draw object boxes
        for box in object_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw all patch centers as small gray dots
        for center_x, center_y in patch_centers:
            cv2.circle(temp_img, (int(center_x), int(center_y)), 2, (128, 128, 128), -1)
        
        # Draw selected good grasp points as green circles
        for center_x, center_y in good_patch_centers:
            cv2.circle(temp_img, (int(center_x), int(center_y)), 8, (0, 255, 0), 2)
            cv2.circle(temp_img, (int(center_x), int(center_y)), 3, (0, 255, 0), -1)
        
        # Add text instructions
        cv2.putText(temp_img, f"Good grasps: {len(good_patch_centers)}/{len(patches_in_boxes)} | Threshold: {threshold}px", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(temp_img, "Left click: select | Right click: remove | ESC: finish | Slider: threshold", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the image
        cv2.imshow("Grasp Labeling", temp_img)
        
        # Check for ESC key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()
    
    # Process results
    bad_patch_center = [center for center in patches_in_boxes if center not in good_patch_centers]
    
    # Log final results
    if logger:
        logger.log_info("=== LABELING COMPLETE ===")
        logger.log_info(f"User interaction summary:")
        logger.log_info(f"  Total clicks: {total_clicks}")
        logger.log_info(f"  Add clicks: {add_clicks}")
        logger.log_info(f"  Remove clicks: {remove_clicks}")
        logger.log_info(f"  Object boxes drawn: {len(object_boxes)}")
        logger.log_grasp_labeling_results(len(good_patch_centers), len(bad_patch_center), len(patch_centers))
    else:
        print(f"\n=== LABELING COMPLETE ===")
        print(f"Total patch centers: {len(patch_centers)}")
        print(f"Patch centers in object boxes: {len(patches_in_boxes)}")
        print(f"Good grasp points selected: {len(good_patch_centers)}")
        print(f"Percentage of good grasps: {(len(good_patch_centers) / len(patch_centers) * 100):.1f}%")
    
    return good_patch_centers, bad_patch_center


def extract_patches_by_centers(patches, patch_centers, good_patch_centers, bad_patch_centers, logger=None):
    """
    Extract good and bad patches based on labeled patch centers
    
    Args:
        patches: List/array of all extracted patches
        patch_centers: List of (x, y) coordinates for all patches
        good_patch_centers: List of (x, y) coordinates for good grasp locations
        bad_patch_centers: List of (x, y) coordinates for bad grasp locations
        logger (DataLabelingLogger, optional): Logger instance
    
    Returns:
        good_patches: Array of patches at good grasp locations
        bad_patches: Array of patches at bad grasp locations
        good_indices: Indices of good patches in original patches array
        bad_indices: Indices of bad patches in original patches array
    """
    if logger:
        logger.log_info(f"Extracting patches from {len(patch_centers)} total patches...")
        logger.log_info(f"Good centers: {len(good_patch_centers)}")
        logger.log_info(f"Bad centers: {len(bad_patch_centers)}")
    else:
        print(f"Extracting patches from {len(patch_centers)} total patches...")
        print(f"Good centers: {len(good_patch_centers)}")
        print(f"Bad centers: {len(bad_patch_centers)}")
    
    # Convert to numpy arrays for easier comparison
    patch_centers = np.array(patch_centers)
    good_centers = np.array(good_patch_centers)
    bad_centers = np.array(bad_patch_centers)
    patches = np.array(patches)
    
    good_indices = []
    bad_indices = []
    
    # Find indices of good patches
    for good_center in good_centers:
        # Find matching patch center (exact coordinate match)
        matches = np.where((patch_centers[:, 0] == good_center[0]) & 
                          (patch_centers[:, 1] == good_center[1]))[0]
        if len(matches) > 0:
            good_indices.append(matches[0])
        else:
            warning_msg = f"Good center {good_center} not found in patch centers"
            if logger:
                logger.log_warning(warning_msg)
            else:
                print(f"Warning: {warning_msg}")
    
    # Find indices of bad patches
    for bad_center in bad_centers:
        # Find matching patch center (exact coordinate match)
        matches = np.where((patch_centers[:, 0] == bad_center[0]) & 
                          (patch_centers[:, 1] == bad_center[1]))[0]
        if len(matches) > 0:
            bad_indices.append(matches[0])
        else:
            warning_msg = f"Bad center {bad_center} not found in patch centers"
            if logger:
                logger.log_warning(warning_msg)
            else:
                print(f"Warning: {warning_msg}")
    
    # Extract the corresponding patches
    good_patches = patches[good_indices] if good_indices else np.array([])
    bad_patches = patches[bad_indices] if bad_indices else np.array([])
    
    if logger:
        logger.log_info(f"Successfully extracted:")
        logger.log_info(f"  Good patches: {len(good_patches)}")
        logger.log_info(f"  Bad patches: {len(bad_patches)}")
    else:
        print(f"Successfully extracted:")
        print(f"  Good patches: {len(good_patches)}")
        print(f"  Bad patches: {len(bad_patches)}")
    
    return good_patches, bad_patches, good_indices, bad_indices