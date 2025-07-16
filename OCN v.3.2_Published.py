import cv2
import numpy as np
import os
import time
from datetime import datetime
from tkinter import filedialog

# Original settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = (320, 320)
PIXELATION_BLOCKS = 5
CAPTURE_INTERVAL = 5

# Collage settings
THUMBNAIL_WIDTH = 64  # Fixed width for grid cells
THUMBNAIL_HEIGHT = 48  # Fixed height for grid cells

class VideoRecorder:
    def __init__(self, output_dir):
        """
        Initialize video recorder for saving side-by-side webcam and collage frames
        
        :param output_dir: Directory to save video recordings
        """
        self.recording = False
        self.webcam_recording = False  # New flag for webcam-only recording
        self.output_dir = output_dir
        self.video_writer = None
        self.webcam_video_writer = None  # New video writer for webcam
        self.start_time = None
        self.webcam_start_time = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def start_recording(self, frame_width, frame_height, fps=7, webcam_only=False):
        """
        Start video recording
        
        :param frame_width: Width of the combined frame
        :param frame_height: Height of the combined frame
        :param fps: Frames per second for recording
        :param webcam_only: Flag to record only webcam frame
        """
        # Stop any existing recording first
        if webcam_only:
            if self.webcam_recording:
                return
            
            # Generate unique filename for webcam recording
            filename = f"webcam_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create VideoWriter object for webcam
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.webcam_video_writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))
            
            self.webcam_recording = True
            self.webcam_start_time = time.time()
            print(f"Started webcam recording to {filepath}")
        else:
            if self.recording:
                return
            
            # Generate unique filename with timestamp
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))
            
            self.recording = True
            self.start_time = time.time()
            print(f"Started recording to {filepath}")
    
    def stop_recording(self, webcam_only=False):
        """Stop video recording and release the video writer
        
        :param webcam_only: Flag to stop only webcam recording
        """
        if webcam_only:
            if not self.webcam_recording:
                return
            
            if self.webcam_video_writer:
                self.webcam_video_writer.release()
                duration = time.time() - self.webcam_start_time
                print(f"Stopped webcam recording. Duration: {duration:.2f} seconds")
            
            self.webcam_recording = False
            self.webcam_video_writer = None
            self.webcam_start_time = None
        else:
            if not self.recording:
                return
            
            if self.video_writer:
                self.video_writer.release()
                duration = time.time() - self.start_time
                print(f"Stopped recording. Duration: {duration:.2f} seconds")
            
            self.recording = False
            self.video_writer = None
            self.start_time = None
    
    def write_frame(self, combined_frame, webcam_frame=None):
        """
        Write a frame to the video file if recording is active
        
        :param combined_frame: Side-by-side frame of webcam and collage
        :param webcam_frame: Optional separate webcam frame for webcam-only recording
        """
        if self.recording and self.video_writer:
            self.video_writer.write(combined_frame)
        
        if self.webcam_recording and self.webcam_video_writer and webcam_frame is not None:
            self.webcam_video_writer.write(webcam_frame)

class PersonDetectorWithCollage:
    def __init__(self):
        self.recording = False
        self.save_directory = None
        self.last_capture_time = time.time()
        self.pixelation_enabled = True
        self.pixelation_blocks = PIXELATION_BLOCKS
        self.color_mode = True  # True for color, False for black and white
        
        # Initialize grid dimensions
        self.grid_cols = 5
        self.grid_rows = 5
        
        # Placement mode 
        self.random_placement = False
        
        # Collage background color (0 for black, 255 for white)
        self.collage_background = 0  # Add this line to initialize the attribute
        
        # Initialize collage
        self.create_new_collage()
        
        # Setup model
        self.setup_model()
        
        # Add window dragging support
        self.dragging = False
        self.drag_start = None
        self.window_pos = (50, 50)  # Initial window position
        
        # Layout settings
        self.layout_vertical = True  # True for vertical layout, False for horizontalizontal
        
        # Collage background color (0 for black, 255 for white)
        self.collage_background = 0
        
        # Add contrast-related attributes
        self.contrast_enabled = False
        self.contrast_alpha = 1.5  # Default contrast enhancement factor
        self.contrast_beta = 10    # Default brightness adjustment
    
    def adjust_contrast(self, image):
        """
        Enhanced contrast adjustment that deepens shadows and increases highlights
        
        :param image: Input image
        :return: Contrast-adjusted image
        """
        if not self.contrast_enabled:
            return image
        
        # Convert image to floating point for precise manipulation
        image_float = image.astype(np.float32)
        
        # Normalize the image to the full range
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        
        # Scale and shift the image for enhanced contrast
        # Lower values of alpha compress the dynamic range
        # Higher values spread out the dynamic range
        alpha = self.contrast_alpha  # Contrast control (1.0-3.0)
        beta = self.contrast_beta    # Brightness control
        
        # Apply contrast and brightness
        # Formula: g(x,y) = α * f(x,y) + β
        # Where α > 1 increases contrast, β shifts the entire image intensity
        contrast_image = cv2.convertScaleAbs(
            image_float, 
            alpha=alpha,     # Contrast control (1.0 means no change)
            beta=beta        # Brightness control
        )
        
        return contrast_image
    
    def create_new_collage(self):
        """Create a new collage with current dimensions"""
        self.collage_width = self.grid_cols * THUMBNAIL_WIDTH
        self.collage_height = self.grid_rows * THUMBNAIL_HEIGHT
        self.collage = np.full((self.collage_height, self.collage_width, 3), self.collage_background, dtype=np.uint8)
        self.current_grid_position = [0, 0]  # [row, col]
        self.grid_full = False
    
    def adjust_grid_dimensions(self, delta_rows=0, delta_cols=0):
        """Adjust grid dimensions and create new collage with preserved content"""
        # Store old collage
        old_collage = self.collage.copy()
        old_rows = self.grid_rows
        old_cols = self.grid_cols
        
        # Update dimensions (minimum 1x1)
        self.grid_rows = max(1, self.grid_rows + delta_rows)
        self.grid_cols = max(1, self.grid_cols + delta_cols)
        
        # Create new collage with current background color
        temp_bg = self.collage_background  # Store current background
        self.create_new_collage()
        
        # Copy over existing content
        for row in range(min(old_rows, self.grid_rows)):
            for col in range(min(old_cols, self.grid_cols)):
                y_old = row * THUMBNAIL_HEIGHT
                x_old = col * THUMBNAIL_WIDTH
                y_new = row * THUMBNAIL_HEIGHT
                x_new = col * THUMBNAIL_WIDTH
                
                # Copy thumbnail if it exists in old collage
                if y_old + THUMBNAIL_HEIGHT <= old_collage.shape[0] and x_old + THUMBNAIL_WIDTH <= old_collage.shape[1]:
                    thumbnail = old_collage[y_old:y_old + THUMBNAIL_HEIGHT, x_old:x_old + THUMBNAIL_WIDTH]
                    # Check if thumbnail is not equal to the old background color
                    if np.any(thumbnail != temp_bg):  
                        self.collage[y_new:y_new + THUMBNAIL_HEIGHT, x_new:x_new + THUMBNAIL_WIDTH] = thumbnail
                        self.current_grid_position = [row + 1, 0] if col == self.grid_cols - 1 else [row, col + 1]
        
        # Update grid_full status
        self.grid_full = self.current_grid_position[0] >= self.grid_rows
        
    def setup_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "yolov3.weights")
        config_path = os.path.join(current_dir, "yolov3.cfg")
        
        # Check files
        for file_path in [weights_path, config_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: {file_path} not found!")
        
        # Load model
        self.net = cv2.dnn.readNet(weights_path, config_path)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("GPU acceleration enabled")
        except:
            print("Using CPU")
            
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def pixelate_region(self, image, x, y, w, h, blocks):
        region = image[y:y+h, x:x+w]
        h_region, w_region = region.shape[:2]
        
        if h_region == 0 or w_region == 0:
            return image
            
        temp = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(temp, (w_region, h_region), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = output
        return image

    def trim_to_aspect_ratio(self, image, target_width, target_height):
        """
        Crop the image to exactly match the target aspect ratio, ensuring square pixels.
        
        :param image: Input image
        :param target_width: Desired width
        :param target_height: Desired height
        :return: Cropped image with exact aspect ratio
        """
        height, width = image.shape[:2]
        target_ratio = target_width / target_height
        current_ratio = width / height
        
        # If image is too wide, crop width
        if current_ratio > target_ratio:
            new_width = int(height * target_ratio)
            start_x = (width - new_width) // 2
            return image[:, start_x:start_x + new_width]
        
        # If image is too tall, crop height
        elif current_ratio < target_ratio:
            new_height = int(width / target_ratio)
            start_y = (height - new_height) // 2
            return image[start_y:start_y + new_height, :]
        
        return image

    def add_to_collage(self, person_image):
        """
        Modified to apply contrast to collage snapshots
        """
        if self.grid_full:
            return
        
        # Apply contrast if enabled
        if self.contrast_enabled:
            person_image = self.adjust_contrast(person_image)
        
        # First, trim to match aspect ratio
        trimmed_image = self.trim_to_aspect_ratio(person_image, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
        
        # Resize to exact thumbnail size ensuring square pixels
        thumbnail = cv2.resize(trimmed_image, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), 
                                interpolation=cv2.INTER_AREA)
        
        # Convert to black and white if color_mode is False
        if not self.color_mode:
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)
        
        # Apply pixelation if enabled
        if self.pixelation_enabled:
            # Resize to pixelation size
            temp = cv2.resize(thumbnail, (self.pixelation_blocks, self.pixelation_blocks), 
                              interpolation=cv2.INTER_LINEAR)
            
            # Resize back to original size ensuring square pixels
            thumbnail = cv2.resize(temp, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Placement logic remains the same as in the original method
        if self.random_placement:
            # Random mode
            empty_positions = [
                (row, col)
                for row in range(self.grid_rows)
                for col in range(self.grid_cols)
                if np.all(
                    self.collage[
                        row * THUMBNAIL_HEIGHT:(row + 1) * THUMBNAIL_HEIGHT,
                        col * THUMBNAIL_WIDTH:(col + 1) * THUMBNAIL_WIDTH,
                    ]
                    == self.collage_background
                )
            ]
            if not empty_positions:
                self.grid_full = True
                self.save_collage()
                return
            row, col = empty_positions[np.random.randint(len(empty_positions))]
        else:
            # Linear mode
            row, col = self.current_grid_position
            if row >= self.grid_rows:
                self.grid_full = True
                self.save_collage()
                return
    
        # Add thumbnail to the grid
        y = row * THUMBNAIL_HEIGHT
        x = col * THUMBNAIL_WIDTH
        self.collage[y:y + THUMBNAIL_HEIGHT, x:x + THUMBNAIL_WIDTH] = thumbnail
        
        # Update position for linear mode
        if not self.random_placement:
            col += 1
            if col >= self.grid_cols:
                col = 0
                row += 1
            self.current_grid_position = (row, col)

    def process_frame(self, frame):
        # Apply contrast to the frame first
        processed_frame = self.adjust_contrast(frame.copy())
        
        # Convert frame to black and white if color_mode is False
        if not self.color_mode:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        blob = cv2.dnn.blobFromImage(processed_frame, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        confidences = []
        boxes = []
        height, width = processed_frame.shape[:2]
        
        # Detect persons (rest of the method remains the same as before)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id == 0 and confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = max(0, int(center_x - w/2))
                    y = max(0, int(center_y - h/2))
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)
        
        # Draw detections
        display_frame = processed_frame.copy()
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                
                if self.pixelation_enabled:
                    display_frame = self.pixelate_region(display_frame, x, y, w, h, self.pixelation_blocks)
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Person {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 2)
        
        # Draw status
        if self.recording:
            cv2.putText(display_frame, "R", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            grid_pos_text = f"{self.grid_rows}x{self.grid_cols}"
            cv2.putText(display_frame, grid_pos_text, (570, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return display_frame, boxes, indexes, confidences

    def save_collage(self):
        if self.save_directory:
            filename = f"collage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            save_path = os.path.join(self.save_directory, filename)
            cv2.imwrite(save_path, self.collage)
            print(f"Saved collage: {save_path}")
            self.create_new_collage()  # Start a new collage

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for window dragging"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Update window position
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            self.window_pos = (
                self.window_pos[0] + dx, 
                self.window_pos[1] + dy
            )
            cv2.moveWindow("Person Detection & Collage", *self.window_pos)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
    
    def create_combined_display(self, webcam_frame):
        """
        Create a combined display of webcam feed and collage.
        
        :param webcam_frame: The processed webcam frame
        :return: Combined frame
        """
        # Create a copy of the collage to ensure we don't modify the original
        collage_display = self.collage.copy()
        
        # Add a border to the collage
        collage_border_color = (0, 255, 0) if self.recording else (255, 255, 255)
        collage_display = cv2.copyMakeBorder(
            collage_display, 2, 2, 2, 2, 
            cv2.BORDER_CONSTANT, value=collage_border_color
        )
        
        # Resize collage for display based on layout
        if self.layout_vertical:
            # For vertical layout, resize collage to match webcam width
            resized_collage = cv2.resize(
                collage_display, 
                (webcam_frame.shape[1], int(collage_display.shape[0] * webcam_frame.shape[1] / collage_display.shape[1]))
            )
            # Combine vertically
            combined = np.vstack((webcam_frame, resized_collage))
        else:
            # For horizontal layout, resize collage to match webcam height
            resized_collage = cv2.resize(
                collage_display, 
                (int(collage_display.shape[1] * webcam_frame.shape[0] / collage_display.shape[0]), webcam_frame.shape[0])
            )
            # Combine horizontally
            combined = np.hstack((webcam_frame, resized_collage))
        
        return combined

def main():
    detector = PersonDetectorWithCollage()
    
    # Track fullscreen state
    fullscreen = False
    
    # Setup save directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    detector.save_directory = os.path.join(current_dir, "detected_persons")
    os.makedirs(detector.save_directory, exist_ok=True)
    
    # Create video recorder
    video_recordings_dir = os.path.join(current_dir, "video_recordings")
    video_recorder = VideoRecorder(video_recordings_dir)
    
    print("\nControls:")
    print("--------")
    print("P - Toggle pixelation")
    print("+ - Increase pixelation blocks")
    print("- - Decrease pixelation blocks")
    print("R - Start/Stop recording")
    print("V - Start/Stop video recording")
    print("Y - Start/Stop webcam-only video recording")
    print("C - Choose save directory")
    print("X - Reset collage")
    print("L - Toggle layout (vertical/horizontal)")
    print("Grid Controls:")
    print("W/S - Increase/Decrease rows")
    print("A/D - Decrease/Increase columns")
    print("O - Toggle Color/Black & White")
    print("K - Toggle Contrast")
    print("8 - Decrease Contrast")
    print("9 - Increase Contrast")
    print("F - Toggle Fullscreen")
    print("B - Toggle collage background color")
    print("T - Toggle collage placement")
    print("Esc - Exit")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Create single window
    cv2.namedWindow("Person Detection & Collage")
    
    # Set mouse callbacks for dragging
    cv2.setMouseCallback("Person Detection & Collage", detector.mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame, boxes, indexes, confidences = detector.process_frame(frame)
        
        # Create combined display
        combined_display = detector.create_combined_display(display_frame)
        
        # Show combined display
        cv2.imshow("Person Detection & Collage", combined_display)
        
        key = cv2.waitKey(1)
        if key == 27:  # Esc
            break
        elif key == ord('p'):
            detector.pixelation_enabled = not detector.pixelation_enabled
        elif key == ord('+'):
            detector.pixelation_blocks = min(50, detector.pixelation_blocks + 2)
        elif key == ord('-'):
            detector.pixelation_blocks = max(2, detector.pixelation_blocks - 2)
        elif key == ord('c'):
            detector.save_directory = filedialog.askdirectory(
                initialdir=current_dir, 
                title="Select save directory"
            )
        elif key == ord('r'):
            detector.recording = not detector.recording
        elif key == ord('v'):  # Video recording
            if video_recorder.recording:
                video_recorder.stop_recording()
            else:
                # Start recording with the combined frame size
                video_recorder.start_recording(
                    frame_width=combined_display.shape[1], 
                    frame_height=combined_display.shape[0]
                )
        elif key == ord('y'):  # Webcam-only video recording
            if video_recorder.webcam_recording:
                video_recorder.stop_recording(webcam_only=True)
            else:
                # Start recording only the webcam frame
                video_recorder.start_recording(
                    frame_width=display_frame.shape[1], 
                    frame_height=display_frame.shape[0], 
                    webcam_only=True
                )
        elif key == ord('x'):
            detector.create_new_collage()
        # Grid dimension controls
        elif key == ord('w'):  # Increase rows
            detector.adjust_grid_dimensions(delta_rows=1)
        elif key == ord('s'):  # Decrease rows
            detector.adjust_grid_dimensions(delta_rows=-1)
        elif key == ord('d'):  # Increase columns
            detector.adjust_grid_dimensions(delta_cols=1)
        elif key == ord('a'):  # Decrease columns
            detector.adjust_grid_dimensions(delta_cols=-1)
        elif key == ord('f'):  # Toggle fullscreen
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty("Person Detection & Collage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("Person Detection & Collage", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif key == ord('b'):  # Toggle background color
            detector.collage_background = 255 if detector.collage_background == 0 else 0
            detector.create_new_collage()
        elif key == ord('t'):  # Toggle random/linear placement
            detector.random_placement = not detector.random_placement
            mode = "Random" if detector.random_placement else "Linear"
            print(f"Placement mode set to {mode}")
        elif key == ord('l'):  # Toggle layout (vertical/horizontal)
            detector.layout_vertical = not detector.layout_vertical
            layout = "Vertical" if detector.layout_vertical else "Horizontal"
            print(f"Layout set to {layout}")
        elif key == ord('o'):  # New key for color mode toggle
            detector.color_mode = not detector.color_mode
            mode = "Color" if detector.color_mode else "Black & White"
            print(f"Video mode set to {mode}")
        elif key == ord('k'):  # Toggle contrast
            detector.contrast_enabled = not detector.contrast_enabled
            mode = "On" if detector.contrast_enabled else "Off"
            print(f"Contrast mode set to {mode}")
        elif key == ord('['):  # Decrease contrast
            detector.contrast_alpha = max(1.0, detector.contrast_alpha - 0.1)
            print(f"Contrast alpha: {detector.contrast_alpha:.2f}")
        elif key == ord(']'):  # Increase contrast
            detector.contrast_alpha = min(3.0, detector.contrast_alpha + 0.1)
            print(f"Contrast alpha: {detector.contrast_alpha:.2f}")

        # Video recording
        if video_recorder.recording:
            video_recorder.write_frame(combined_display)
        
        # Webcam-only recording
        if video_recorder.webcam_recording:
            video_recorder.write_frame(combined_display, webcam_frame=display_frame)

        # Capture and add to collage if recording
        current_time = time.time()
        if detector.recording and current_time - detector.last_capture_time >= CAPTURE_INTERVAL:
            detector.last_capture_time = current_time
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]
                    person_region = frame[y:y+h, x:x+w]
                    
                    # Add to collage if pixelation is enabled
                    if detector.pixelation_enabled:
                        person_region = detector.pixelate_region(
                            person_region.copy(), 
                            0, 0, w, h, 
                            detector.pixelation_blocks
                        )
                    detector.add_to_collage(person_region)
    
    # Clean up
    cap.release()
    if video_recorder.recording:
        video_recorder.stop_recording()
    if video_recorder.webcam_recording:
        video_recorder.stop_recording(webcam_only=True)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()