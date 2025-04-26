import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
import tensorflow as tf
import sqlite3
import tkinter as tk
from tkinter import simpledialog

STUDENT_ROLL_NO = None  # Set this during registration

PROCESS_EVERY_N_FRAMES = 2  # You can tweak this to 2 or 5 depending on performance

# Set gaze limits 
# GAZE_LIMIT_X_MIN = 0.4, GAZE_LIMIT_X_MAX = 0.6 → horizontal zone
# GAZE_LIMIT_Y_MIN = 0.3, GAZE_LIMIT_Y_MAX = 0.7 → vertical zone
GAZE_LIMIT_X_MIN = 0.4
GAZE_LIMIT_X_MAX = 0.6
GAZE_LIMIT_Y_MIN = 0.3
GAZE_LIMIT_Y_MAX = 0.7
# GAZE_LIMIT_X_MIN = 0.2
# GAZE_LIMIT_X_MAX = 0.55
# GAZE_LIMIT_Y_MIN = 0.2
# GAZE_LIMIT_Y_MAX = 0.4  


# Initialize MediaPipe Face Mesh with optimized settings
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
mp_objectron = mp.solutions.objectron
mp_hands = mp.solutions.hands

# Define key facial landmark indices
LEFT_EYE = 133
RIGHT_EYE = 362
NOSE_TIP = 4
MOUTH_CENTER = 13
LEFT_EAR = 234
RIGHT_EAR = 454
FOREHEAD = 151
CHIN = 199

# Eye tracking landmarks
LEFT_IRIS_CENTER = 468  # Iris center point (with refined_landmarks)
RIGHT_IRIS_CENTER = 473
LEFT_EYE_CONTOUR = [
    # Left eye contour points
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
]
RIGHT_EYE_CONTOUR = [
    # Right eye contour points
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Create snapshots directory if it doesn't exist
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

class CheatingDetectionSystem:
    def __init__(self):
        # Configuration parameters
        self.screen_gaze_threshold = 0.8  # Threshold for determining if looking at screen
        self.gaze_away_time_threshold = 2.0  # Seconds that constitute suspicious looking away
        self.face_verification_threshold = 0.8  # Similarity threshold for face verification
        
        # State variables
        self.registered_student = None  # Will store face embeddings of registered student
        self.looking_away_start_time = None
        self.total_looking_away_time = 0
        self.detected_objects = {}
        self.suspicious_events = []
        self.session_start_time = datetime.now()
        self.last_snapshot_time = time.time()
        self.snapshot_cooldown = 5  # Seconds between taking snapshots of suspicious activity
        
        # Face tracking history (for movement analysis)
        self.face_position_history = []
        self.max_history_length = 30  # Store last 30 positions
        
        # Gaze tracking history
        self.gaze_history = []
        self.max_gaze_history = 10
        
        # Load object detection model (simplified for this example)
        # In a real implementation, you'd load a proper object detection model
        self.object_detection_model = None
        try:
            # Placeholder for a real model loading
            # self.object_detection_model = tf.saved_model.load('path_to_model')
            print("Object detection model would be loaded here")
        except Exception as e:
            print(f"Failed to load object detection model: {e}")
        
        # Initialize MediaPipe Objectron for 3D object detection
        self.objectron = mp_objectron.Objectron(
            static_image_mode=False,
            max_num_objects=5,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_name='Shoe')  # We'll use this as a placeholder
        
        # Initialize hand detection
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Start logging
        self.log_file = open(f"logs/session_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}.log", "w")
        self.log_event("Session started")
    
    def register_student(self, face_landmarks, face_details):
        """Register the legitimate student's face"""
        self.registered_student = {
            'landmarks': face_landmarks,
            'details': face_details,
            'timestamp': datetime.now()
        }
        self.log_event("Student registered")
        return True
    
    def verify_student(self, face_landmarks, face_details):
        """Verify if the current face matches the registered student"""
        if self.registered_student is None:
            return False
        
        # In a real implementation, we would use proper face recognition
        # Here we'll use a simple comparison of facial metrics
        registered = self.registered_student['details']
        
        # Compare key facial metrics (simplified)
        eye_distance_diff = abs(registered['eye_distance'] - face_details['eye_distance'])
        face_width_diff = abs(registered['face_width'] - face_details['face_width'])
        face_height_diff = abs(registered['face_height'] - face_details['face_height'])
        
        # Calculate simple similarity score (0-1)
        # Lower numbers mean more similar
        if registered['eye_distance'] == 0 or face_details['eye_distance'] == 0:
            return False
            
        eye_distance_similarity = 1 - min(eye_distance_diff / registered['eye_distance'], 1.0)
        face_width_similarity = 1 - min(face_width_diff / registered['face_width'], 1.0)
        face_height_similarity = 1 - min(face_height_diff / registered['face_height'], 1.0)
        
        # Calculate overall similarity (weighted average)
        overall_similarity = (0.5 * eye_distance_similarity + 
                              0.25 * face_width_similarity + 
                              0.25 * face_height_similarity)
        
        return overall_similarity >= self.face_verification_threshold
    
    def calculate_gaze_direction(self, frame, face_landmarks, face_details, frame_shape):
        """Calculate gaze direction to determine if looking at screen"""
        h, w = frame_shape
        
        # Extract iris positions
        try:
            # If we have iris landmarks
            left_iris = (int(face_landmarks.landmark[LEFT_IRIS_CENTER].x * w),
                        int(face_landmarks.landmark[LEFT_IRIS_CENTER].y * h))
            right_iris = (int(face_landmarks.landmark[RIGHT_IRIS_CENTER].x * w),
                        int(face_landmarks.landmark[RIGHT_IRIS_CENTER].y * h))
            # print("left_iris:",left_iris," right_iris:",right_iris)
            # Calculate eye centers from contour points
            left_eye_points = [(int(face_landmarks.landmark[idx].x * w), 
                               int(face_landmarks.landmark[idx].y * h)) 
                              for idx in LEFT_EYE_CONTOUR]
            right_eye_points = [(int(face_landmarks.landmark[idx].x * w), 
                                int(face_landmarks.landmark[idx].y * h)) 
                               for idx in RIGHT_EYE_CONTOUR]
            # print("left_eye_points:",left_eye_points," right_eye_points:",right_eye_points)
            
            # Calculate eye center from contour
            left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
            right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
            
            # Calculate relative iris position within eye
            # 0.5 means centered, <0.5 means looking left, >0.5 means looking right
            left_min_x = min(p[0] for p in left_eye_points)
            left_max_x = max(p[0] for p in left_eye_points)
            left_eye_width = max(1, left_max_x - left_min_x)  # Prevent division by zero
            
            right_min_x = min(p[0] for p in right_eye_points)
            right_max_x = max(p[0] for p in right_eye_points)
            right_eye_width = max(1, right_max_x - right_min_x)  # Prevent division by zero
            
            # Calculate horizontal gaze ratio (x position within eye width)
            left_gaze_x = (left_iris[0] - left_min_x) / left_eye_width
            right_gaze_x = (right_iris[0] - right_min_x) / right_eye_width
            print("left_gaze_x: ",left_gaze_x," right_gaze_x:",right_gaze_x)
            # Similarly for vertical gaze
            left_min_y = min(p[1] for p in left_eye_points)
            left_max_y = max(p[1] for p in left_eye_points)
            left_eye_height = max(1, left_max_y - left_min_y)
            
            right_min_y = min(p[1] for p in right_eye_points)
            right_max_y = max(p[1] for p in right_eye_points)
            right_eye_height = max(1, right_max_y - right_min_y)
            
            left_gaze_y = (left_iris[1] - left_min_y) / left_eye_height
            right_gaze_y = (right_iris[1] - right_min_y) / right_eye_height
            
            # Average the gaze directions from both eyes
            gaze_x = (left_gaze_x + right_gaze_x) / 2
            gaze_y = (left_gaze_y + right_gaze_y) / 2
            
            # Store in history for smoothing
            self.gaze_history.append((gaze_x, gaze_y))
            if len(self.gaze_history) > self.max_gaze_history:
                self.gaze_history.pop(0)
            
            # Calculate smoothed gaze
            smoothed_gaze_x = sum(g[0] for g in self.gaze_history) / len(self.gaze_history)
            smoothed_gaze_y = sum(g[1] for g in self.gaze_history) / len(self.gaze_history)
            
            # Determine if looking at screen (simplified)
            # Typical eye orientation range when looking at screen is around 0.3-0.7 for both axes
            looking_at_screen = (GAZE_LIMIT_X_MIN <= smoothed_gaze_x <= GAZE_LIMIT_X_MAX and GAZE_LIMIT_Y_MIN <= smoothed_gaze_y <= GAZE_LIMIT_Y_MAX)
            
            # If the person’s iris is between 40%–60% left/right
            # AND between 30%–70% up/down →
            # → Then assume they’re looking straight at the screen → True
            # Return gaze data
            return {
                'looking_at_screen': looking_at_screen,
                'gaze_x': smoothed_gaze_x,
                'gaze_y': smoothed_gaze_y,
                'left_iris': left_iris,
                'right_iris': right_iris
            }
        
        except Exception as e:
            # Fallback method if iris detection fails
            return {
                'looking_at_screen': True,  # Default to true to avoid false positives
                'gaze_x': 0.5,
                'gaze_y': 0.5,
                'left_iris': face_details['left_eye'],
                'right_iris': face_details['right_eye']
            }
    
    def detect_objects(self, frame):
        """Detect potential suspicious objects like phones or papers"""
        # In a real implementation, you would use a trained object detection model
        # This is a placeholder that would need to be replaced
        
        # Process with MediaPipe Objectron 
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        objectron_results = self.objectron.process(image_rgb)
        
        detected_objects = []
        
        if objectron_results.detected_objects:
            for detected_object in objectron_results.detected_objects:
                # In real implementation, we'd classify the object
                # Here we'll just label it as a generic suspicious object
                detected_objects.append({
                    'type': 'unknown_object',
                    'confidence': 0.7,  # Placeholder
                    'box': [0, 0, 100, 100]  # Placeholder bounding box
                })
        
        # Process hands (hands near ear or below desk could indicate phone use)
        hands_results = self.hands.process(image_rgb)
        
        if hands_results.multi_hand_landmarks:
            frame_height, frame_width = frame.shape[:2]
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Get wrist position
                wrist_x = int(hand_landmarks.landmark[0].x * frame_width)
                wrist_y = int(hand_landmarks.landmark[0].y * frame_height)
                
                # Check for suspicious hand positions
                # Below bottom of frame (could be reaching for something under desk)
                if wrist_y > frame_height * 0.95:
                    detected_objects.append({
                        'type': 'hand_below_desk',
                        'confidence': 0.8,
                        'box': [wrist_x-50, wrist_y-50, 100, 100]
                    })
                
                # Near ear (could be using phone)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    lm_x = int(landmark.x * frame_width)
                    lm_y = int(landmark.y * frame_height)
                    
                    # Check if any hand landmark is near the ear area
                    ear_proximity_distance = 70  # pixels
                    if (abs(lm_x - face_details.get('left_ear', (0,0))[0]) < ear_proximity_distance and
                        abs(lm_y - face_details.get('left_ear', (0,0))[1]) < ear_proximity_distance) or \
                       (abs(lm_x - face_details.get('right_ear', (0,0))[0]) < ear_proximity_distance and
                        abs(lm_y - face_details.get('right_ear', (0,0))[1]) < ear_proximity_distance):
                        
                        detected_objects.append({
                            'type': 'hand_near_ear',
                            'confidence': 0.9,
                            'box': [lm_x-50, lm_y-50, 100, 100]
                        })
                        break
        
        return detected_objects
    
    def track_looking_away(self, looking_at_screen):
        """Track when and how long the student is looking away from screen"""
        current_time = time.time()
        
        if not looking_at_screen:
            # Just started looking away
            if self.looking_away_start_time is None:
                self.looking_away_start_time = current_time
                
            # Check if they've been looking away for too long
            elif (current_time - self.looking_away_start_time) > self.gaze_away_time_threshold:
                look_away_duration = current_time - self.looking_away_start_time
                self.total_looking_away_time += look_away_duration
                
                # Log this suspicious event
                if look_away_duration > self.gaze_away_time_threshold * 2:  # More severe
                    self.log_suspicious_event(f"Extended gaze deviation detected ({look_away_duration:.1f}s)", 
                                             severity="HIGH")
                else:
                    self.log_suspicious_event(f"Gaze deviation detected ({look_away_duration:.1f}s)", 
                                             severity="MEDIUM")
                
                # Reset timer for continuous tracking
                self.looking_away_start_time = current_time
                
                return {
                    'status': 'suspicious',
                    'duration': look_away_duration
                }
                
        else:  # Looking at screen
            # Reset the away timer
            if self.looking_away_start_time is not None:
                # Calculate the duration they were looking away
                look_away_duration = current_time - self.looking_away_start_time
                self.total_looking_away_time += look_away_duration
                self.looking_away_start_time = None
                
                # Only report if it exceeds threshold
                if look_away_duration > self.gaze_away_time_threshold:
                    return {
                        'status': 'returned',
                        'duration': look_away_duration
                    }
        
        return {'status': 'normal'}
    
    def log_event(self, message):
        """Log a normal event to the session log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_file.write(log_entry)
        self.log_file.flush()
        print(log_entry.strip())
    
    def log_suspicious_event(self, message, severity="LOW"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{severity}] {message}\n"
        self.log_file.write(log_entry)
        self.log_file.flush()
        print(f"ALERT: {log_entry.strip()}")

        self.suspicious_events.append({
            'timestamp': timestamp,
            'message': message,
            'severity': severity
        })

        # Save to SQLite
        try:
            conn = sqlite3.connect("students.db")
            c = conn.cursor()
            c.execute("INSERT INTO alerts (roll_no, timestamp, message, severity) VALUES (?, ?, ?, ?)",
                    (STUDENT_ROLL_NO or "UNKNOWN", timestamp, message, severity))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Failed to log alert to DB: {e}")

        # Snapshot cooldown
        current_time = time.time()
        if current_time - self.last_snapshot_time > self.snapshot_cooldown:
            self.last_snapshot_time = current_time
            return True
        return False

    
    def save_snapshot(self, frame, prefix="suspicious"):
        """Save a snapshot of the current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/{prefix}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        self.log_event(f"Snapshot saved: {filename}")
        return filename
    
    def generate_report(self):
        """Generate a summary report of the session"""
        report = "======= EXAM MONITORING REPORT =======\n"
        report += f"Session start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Session end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Total suspicious events: {len(self.suspicious_events)}\n"
        report += f"Total time looking away: {self.total_looking_away_time:.1f} seconds\n"
        report += "\n--- Suspicious Events Log ---\n"
        
        for idx, event in enumerate(self.suspicious_events, 1):
            report += f"{idx}. [{event['timestamp']}] [{event['severity']}] {event['message']}\n"
        
        report += "\n======= END OF REPORT =======\n"
        return report
    
    def close(self):
        """Close the monitoring session"""
        if hasattr(self, 'log_file') and self.log_file:
            # Write final report to log
            report = self.generate_report()
            self.log_file.write("\n\n" + report)
            
            # Close log file
            self.log_file.close()
            print("Session log closed")
            

def calculate_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # Function to extract face details
def extract_face_details(landmarks, image_shape):
        try:
            h, w = image_shape
            details = {}
            
            # Convert landmark coordinates to pixel positions
            points = {}
            for idx, landmark in enumerate(landmarks.landmark):
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                points[idx] = (px, py)
            
            # Safely get landmark points, providing defaults if missing
            left_eye_pos = points.get(LEFT_EYE, (0, 0))
            right_eye_pos = points.get(RIGHT_EYE, (0, 0))
            left_ear_pos = points.get(LEFT_EAR, (0, 0))
            right_ear_pos = points.get(RIGHT_EAR, (0, 0))
            forehead_pos = points.get(FOREHEAD, (0, 0))
            chin_pos = points.get(CHIN, (0, 0))
            nose_pos = points.get(NOSE_TIP, (0, 0))
            mouth_pos = points.get(MOUTH_CENTER, (0, 0))
            
            # Calculate eye distance (interpupillary distance)
            details['eye_distance'] = calculate_distance(left_eye_pos, right_eye_pos)
            
            # Calculate face width using ears
            details['face_width'] = calculate_distance(left_ear_pos, right_ear_pos)
            
            # Calculate face height (forehead to chin)
            details['face_height'] = calculate_distance(forehead_pos, chin_pos)
            
            # Eye positions
            details['left_eye'] = left_eye_pos
            details['right_eye'] = right_eye_pos
            
            # Ear positions
            details['left_ear'] = left_ear_pos
            details['right_ear'] = right_ear_pos
            
            # Nose and mouth positions
            details['nose'] = nose_pos
            details['mouth'] = mouth_pos
            
            # Face center approximation
            center_x = (left_eye_pos[0] + right_eye_pos[0]) // 2
            center_y = (forehead_pos[1] + chin_pos[1]) // 2
            details['face_center'] = (center_x, center_y)
            
            return details
        except Exception as e:
            print(f"Error extracting face details: {e}")
            # Return default values
            return {
                'eye_distance': 0,
                'face_width': 0,
                'face_height': 0,
                'left_eye': (0, 0),
                'right_eye': (0, 0),
                'left_ear': (0, 0),
                'right_ear': (0, 0),
                'nose': (0, 0),
                'mouth': (0, 0),
                'face_center': (0, 0)
            }
            
def main():
    print("Starting AI-Powered Cheating Detection System...")
    
    # Set up Face Mesh Detection with optimized parameters
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,  # Detect up to 10 faces
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize cheating detection system
    detector = CheatingDetectionSystem()
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Get frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        cap.release()
        return
    
    #frame_height, frame_width = test_frame.shape[:2]
    frame_width = 1920
    frame_height = 1080
    
    # Define target display size
    display_width = 1920
    display_height = 1080

    # Setup video recording
    video_filename = f"snapshots/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(video_filename, fourcc, 20.0, (display_width, display_height))

    
    print(f"Camera opened successfully. Frame size: {frame_width}x{frame_height}")
    print(f"Display size: {display_width}x{display_height}")
    print("Press 'q' to quit, 'r' to register student")

    # Status variables
    student_registered = False
    monitoring_active = False
    registration_mode = False
    
    # Main processing loop
    frame_count = 0
    start_time = time.time()
    fps = 0

    dialog_root = tk.Tk()
    dialog_root.withdraw()
    
    while True:
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame - camera may be disconnected")
                # Try to reconnect
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Could not reconnect to camera. Exiting.")
                    break
                continue
                
            # Mirror the frame horizontally for more intuitive viewing
            frame = cv2.flip(frame, 1)
            
            # Create a large output canvas (1920x1080)
            output_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            
            # Scale factor to fit the camera feed into the display
            scale_width = display_width / frame_width
            scale_height = display_height / frame_height
            scale = min(scale_width, scale_height)
            
            # Calculate new dimensions while preserving aspect ratio
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            # Calculate position to center the frame
            x_offset = (display_width - new_width) // 2
            y_offset = (display_height - new_height) // 2
            
            # Resize the frame
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Place the resized frame on the output canvas
            output_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
            
            # Convert the image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process for face mesh detection
            face_results = face_mesh.process(image_rgb)
            
            # Count faces detected
            face_count = 0
            face_details_list = []
            verified_student = False
            
            # Process face detections
            if face_results.multi_face_landmarks:
                face_count = len(face_results.multi_face_landmarks)
                
                # Log multiple faces if detected
                if face_count > 1 and monitoring_active:
                    take_snapshot = detector.log_suspicious_event(
                        f"Multiple faces detected ({face_count})", 
                        severity="HIGH"
                    )
                    if take_snapshot:
                        detector.save_snapshot(frame, "multiple_faces")
                
                for idx, face_landmarks in enumerate(face_results.multi_face_landmarks):
                    try:
                        # Extract face details from the original frame
                        face_details = extract_face_details(face_landmarks, (frame_height, frame_width))
                        face_details_list.append(face_details)
                        
                        # Verify student identity if registered
                        if student_registered:
                            if idx == 0:  # Check the first face
                                verified_student = detector.verify_student(face_landmarks, face_details)
                                
                                # Log unauthorized person if not verified during monitoring
                                if not verified_student and monitoring_active:
                                    take_snapshot = detector.log_suspicious_event(
                                        "Unauthorized person detected", 
                                        severity="HIGH"
                                    )
                                    print(gaze_data)
                                    if take_snapshot:
                                        detector.save_snapshot(frame, "unauthorized")
                        
                        # Check gaze direction (eye tracking)
                        if monitoring_active and (idx == 0) and verified_student:
                            if (frame_count % PROCESS_EVERY_N_FRAMES == 0):
                                gaze_data = detector.calculate_gaze_direction(
                                    frame, 
                                    face_landmarks, 
                                    face_details,
                                    (frame_height, frame_width)
                                )
                                print(gaze_data)
                            # Track looking away behavior
                            looking_result = detector.track_looking_away(gaze_data['looking_at_screen'])
                            
                            # Display gaze direction on frame
                            gaze_status = "Looking at screen" if gaze_data['looking_at_screen'] else "Looking away"
                            
                            cv2.putText(
                                output_frame,
                                f"Gaze: {gaze_status}",
                                (display_width - 400, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0) if gaze_data['looking_at_screen'] else (0, 0, 255),
                                2
                            )
                            
                            # Show gaze coordinates
                            cv2.putText(
                                output_frame,
                                f"Gaze X: {gaze_data['gaze_x']:.2f}, Y: {gaze_data['gaze_y']:.2f}",
                                (display_width - 400, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255),
                                1
                            )
                            
                            # Draw iris positions if available
                            if 'left_iris' in gaze_data and 'right_iris' in gaze_data:
                                left_iris = gaze_data['left_iris']
                                right_iris = gaze_data['right_iris']
                                
                                # Scale to output frame
                                left_iris_scaled = (
                                    int(left_iris[0] * scale) + x_offset,
                                    int(left_iris[1] * scale) + y_offset
                                )
                                right_iris_scaled = (
                                    int(right_iris[0] * scale) + x_offset,
                                    int(right_iris[1] * scale) + y_offset
                                )
                                
                                # Draw iris centers
                                cv2.circle(output_frame, left_iris_scaled, 3, (0, 0, 255), -1)
                                cv2.circle(output_frame, right_iris_scaled, 3, (0, 0, 255), -1)
                        
                        # Draw simplified face mesh (just contours)
                        connection_drawing = mp_drawing_styles.get_default_face_mesh_contours_style()
                        
                        # Draw directly on the output canvas with manual scaling
                        mesh_drawing = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
                        for connection in mp_face_mesh.FACEMESH_CONTOURS:
                            start_idx = connection[0]
                            end_idx = connection[1]
                            
                            if (start_idx >= len(face_landmarks.landmark) or 
                                end_idx >= len(face_landmarks.landmark)):
                                continue
                                
                            start_landmark = face_landmarks.landmark[start_idx]
                            end_landmark = face_landmarks.landmark[end_idx]
                            
                            start_x = int(start_landmark.x * frame_width * scale) + x_offset
                            start_y = int(start_landmark.y * frame_height * scale) + y_offset
                            end_x = int(end_landmark.x * frame_width * scale) + x_offset
                            end_y = int(end_landmark.y * frame_height * scale) + y_offset
                            
                            # Make sure points are within the output frame
                            if (0 <= start_x < display_width and 0 <= start_y < display_height and
                                0 <= end_x < display_width and 0 <= end_y < display_height):
                                # Use green for verified student, red for unverified
                                line_color = (0, 255, 0) if (verified_student and idx == 0) else (0, 0, 255)
                                cv2.line(output_frame, (start_x, start_y), (end_x, end_y), 
                                         line_color, 1)
                        
                        # Add face number and bounding box if we have valid face details
                        if face_details['nose'] != (0, 0):
                            # Scale nose position
                            scaled_nose = (
                                int(face_details['nose'][0] * scale) + x_offset,
                                int(face_details['nose'][1] * scale) + y_offset
                            )
                            
                            # Determine label based on verification
                            face_label = f"Student" if (verified_student and idx == 0) else f"Face {idx+1}"
                            face_color = (0, 255, 0) if (verified_student and idx == 0) else (0, 0, 255)
                            
                            # Add face label near the nose
                            cv2.putText(
                                output_frame, 
                                face_label, 
                                (scaled_nose[0], scaled_nose[1] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2
                            )
                            
                            # Draw face bounding box
                            face_center = face_details['face_center']
                            face_width = face_details['face_width']
                            face_height = face_details['face_height']
                            
                            # Make sure we have valid measurements
                            if face_width > 0 and face_height > 0:
                                # Scale to output frame
                                scaled_center = (
                                    int(face_center[0] * scale) + x_offset,
                                    int(face_center[1] * scale) + y_offset
                                )
                                scaled_width = int(face_width * scale)
                                scaled_height = int(face_height * scale)
                                
                                # Draw rectangle around face
                                top_left = (
                                    scaled_center[0] - scaled_width // 2,
                                    scaled_center[1] - scaled_height // 2
                                )
                                bottom_right = (
                                    scaled_center[0] + scaled_width // 2,
                                    scaled_center[1] + scaled_height // 2
                                )
                                # Use green for verified student, red for unverified
                                box_color = (0, 255, 0) if (verified_student and idx == 0) else (0, 0, 255)
                                cv2.rectangle(output_frame, top_left, bottom_right, box_color, 2)
                    except Exception as e:
                        print(f"Error processing face {idx}: {e}")
                        continue
            else:
                # No faces detected
                if monitoring_active:
                    take_snapshot = detector.log_suspicious_event("No face detected in frame", severity="MEDIUM")
                    if take_snapshot:
                        detector.save_snapshot(frame, "no_face")
            
            # Detect objects if monitoring is active
            if monitoring_active and verified_student and (frame_count % PROCESS_EVERY_N_FRAMES == 0):
                detected_objects = detector.detect_objects(frame)
                
                # Log and highlight suspicious objects
                for obj in detected_objects:
                    obj_type = obj['type']
                    box = obj['box']
                    
                    # Scale box coordinates to output frame
                    scaled_box = [
                        int(box[0] * scale) + x_offset,
                        int(box[1] * scale) + y_offset,
                        int(box[2] * scale),
                        int(box[3] * scale)
                    ]
                    
                    # Draw object box
                    cv2.rectangle(output_frame, 
                                 (scaled_box[0], scaled_box[1]),
                                 (scaled_box[0] + scaled_box[2], scaled_box[1] + scaled_box[3]),
                                 (0, 0, 255), 2)
                    
                    # Add object label
                    cv2.putText(output_frame, obj_type, 
                               (scaled_box[0], scaled_box[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Log suspicious object
                    take_snapshot = detector.log_suspicious_event(
                        f"Suspicious object detected: {obj_type}",
                        severity="HIGH"
                    )
                    if take_snapshot:
                        detector.save_snapshot(frame, f"object_{obj_type}")
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 10:  # Update FPS every 10 frames
                end_time = time.time()
                elapsed = end_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                frame_count = 0
                start_time = time.time()
            
            # Display system status
            cv2.putText(output_frame, "AI Cheating Detection System", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (50, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display registration status
            registration_status = "Registered" if student_registered else "Not Registered"
            cv2.putText(output_frame, f"Student: {registration_status}", (50, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (0, 255, 0) if student_registered else (0, 0, 255), 2)
            
            # Display monitoring status
            monitoring_status = "Active" if monitoring_active else "Inactive"
            cv2.putText(output_frame, f"Monitoring: {monitoring_status}", (50, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (0, 255, 0) if monitoring_active else (255, 165, 0), 2)
            
            # Display faces detected count
            cv2.putText(output_frame, f"Faces Detected: {face_count}", (50, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                        (0, 255, 0) if face_count == 1 else (0, 0, 255), 2)
            
            # Display verification status if registered
            if student_registered:
                verification_status = "Verified" if verified_student else "Not Verified"
                cv2.putText(output_frame, f"Identity: {verification_status}", (50, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                            (0, 255, 0) if verified_student else (0, 0, 255), 2)
            
            # Display suspicious event count
            if monitoring_active:
                event_count = len(detector.suspicious_events)
                cv2.putText(output_frame, f"Suspicious Events: {event_count}", (50, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                            (0, 255, 0) if event_count == 0 else (0, 0, 255), 2)
            
            # Display registration mode status if active
            if registration_mode:
                cv2.putText(output_frame, "REGISTRATION MODE ACTIVE", (display_width//2 - 200, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)
                cv2.putText(output_frame, "Look at the camera and press 'c' to confirm", 
                            (display_width//2 - 250, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            
            # Display instructions
            instruction_y = display_height - 160
            cv2.putText(output_frame, "Controls:", (50, instruction_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            instruction_y += 30
            cv2.putText(output_frame, "- Press 'r' to register student", (70, instruction_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            instruction_y += 30
            cv2.putText(output_frame, "- Press 'm' to toggle monitoring", (70, instruction_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            instruction_y += 30
            cv2.putText(output_frame, "- Press 'q' to quit", (70, instruction_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Display the most recent suspicious events
            if monitoring_active and detector.suspicious_events:
                event_y = 330
                cv2.putText(output_frame, "Recent Alerts:", (50, event_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                event_y += 30
                
                # Show last 3 events
                recent_events = detector.suspicious_events[-3:] if len(detector.suspicious_events) > 3 else detector.suspicious_events
                for event in recent_events:
                    # Truncate message if too long
                    message = event['message']
                    if len(message) > 40:
                        message = message[:37] + "..."
                        
                    cv2.putText(output_frame, f"[{event['severity']}] {message}", (70, event_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    event_y += 25
            
            # Display the frame
            cv2.namedWindow('AI Cheating Detection System', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('AI Cheating Detection System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('AI Cheating Detection System', output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
                
            elif key == ord('r'):  # Register student
                if not registration_mode:
                    registration_mode = True
                    print("Registration mode activated. Look at the camera and press 'c' to confirm.")
                else:
                    registration_mode = False
                    print("Registration mode deactivated.")
                    
            elif key == ord('c'):  # Confirm registration
                if registration_mode and face_count > 0:
                    # Use the first detected face
                    detector.register_student(face_results.multi_face_landmarks[0], face_details_list[0])
                    student_registered = True
                    registration_mode = False
                    print("Student registered successfully!")

                    # Ask user for roll number via terminal input
                    global STUDENT_ROLL_NO
                    STUDENT_ROLL_NO = simpledialog.askstring("Roll Number", "Enter student roll number:", parent=dialog_root)
                    print(f"Student registered with roll number: {STUDENT_ROLL_NO}")
                    
            elif key == ord('m'):  # Toggle monitoring
                if student_registered:
                    monitoring_active = not monitoring_active
                    if monitoring_active:
                        print("Monitoring activated!")
                        detector.log_event("Monitoring started")
                    else:
                        print("Monitoring deactivated!")
                        detector.log_event("Monitoring paused")
                else:
                    print("Please register a student first")
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            # Continue instead of exiting - this gives the program a chance to recover
            time.sleep(0.1)
            continue
    
    # Clean up
    print("Generating final report...")
    final_report = detector.generate_report()
    print("\n" + final_report)
    
    detector.close()

    video_out.write(output_frame)

    print("Releasing camera and closing windows...")
    cap.release()

    video_out.release()
    print(f"Session video saved: {video_filename}")

    cv2.destroyAllWindows()
    print("Program finished.")

# Run main function with proper error handling
if __name__ == "__main__":
    # main()
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")
        print("Program terminated due to an unrecoverable error.")