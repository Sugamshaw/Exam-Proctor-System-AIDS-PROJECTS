import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import os
import datetime
import sqlite3
import dlib
import mediapipe as mp
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import distance as dist
import json 


PROCESS_EVERY_N_FRAMES = 2  # You can tweak this to 2 or 5 depending on performance

# Set gaze limits 
GAZE_LIMIT_X_MIN = 0.4
GAZE_LIMIT_X_MAX = 0.6
GAZE_LIMIT_Y_MIN = 0.3
GAZE_LIMIT_Y_MAX = 0.7



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
            
            # Calculate eye centers from contour points
            left_eye_points = [(int(face_landmarks.landmark[idx].x * w), 
                               int(face_landmarks.landmark[idx].y * h)) 
                              for idx in LEFT_EYE_CONTOUR]
            right_eye_points = [(int(face_landmarks.landmark[idx].x * w), 
                                int(face_landmarks.landmark[idx].y * h)) 
                               for idx in RIGHT_EYE_CONTOUR]
            
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

def calculate_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
            
class ExamProctorSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Exam Proctoring System")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        self.monitoring = False
        self.recording = False
        self.create_database()
        self.alerts = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.model = None
        self.video_writer = None
        self.current_student = None
        self.session_id = None
        self.start_time = None
        self.alert_count = 0
        self.eye_ar_thresh = 0.3
        self.eye_ar_consec_frames = 3
        self.blink_counter = 0
        self.total_blinks = 0
        
        self.detector = dlib.get_frontal_face_detector()
        
        self.shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(self.shape_predictor_path):
            self.predictor = dlib.shape_predictor(self.shape_predictor_path)
        else:
            messagebox.showinfo("Missing Model", 
                               "The facial landmark predictor model is missing. Blink and gaze detection will be disabled.\n" + 
                               "Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, extract, and place in the same folder.")
            self.predictor = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,  # Set to True for image processing
            max_num_faces=1,         # Only detect one face
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define key facial landmark indices
        self.LEFT_EYE = 133
        self.RIGHT_EYE = 362
        self.NOSE_TIP = 4
        self.MOUTH_CENTER = 13
        self.LEFT_EAR = 234
        self.RIGHT_EAR = 454
        self.FOREHEAD = 151
        self.CHIN = 199
        self.load_or_train_model()
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.monitor_tab = ttk.Frame(self.notebook)
        self.enroll_tab = ttk.Frame(self.notebook)
        self.history_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.monitor_tab, text="Monitor")
        self.notebook.add(self.enroll_tab, text="Enroll Student")
        self.notebook.add(self.history_tab, text="History")
        self.notebook.add(self.settings_tab, text="Settings")
        
        self.create_monitor_tab()
        self.create_enroll_tab()
        self.create_history_tab()
        self.create_settings_tab()
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    def create_database(self):
        conn = sqlite3.connect('exam_proctor.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            enrollment_date TEXT NOT NULL,
            face_encoding BLOB
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            start_time TEXT NOT NULL,
            end_time TEXT,
            video_path TEXT,
            alert_count INTEGER DEFAULT 0,
            attention_score REAL DEFAULT 100.0,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT NOT NULL,
            image_path TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_or_train_model(self):
        if os.path.exists('facial_expression_model.h5'):
            self.model = load_model('facial_expression_model.h5')
        else:
            self.train_model()
    
    def train_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        X = np.random.rand(100, 48, 48, 1)
        y = to_categorical(np.random.randint(0, 7, 100), 7)
        
        model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
        
        model.save('facial_expression_model.h5')
        self.model = model
    
    def create_monitor_tab(self):
        self.left_frame = tk.Frame(self.monitor_tab, bg="#f0f0f0", width=800, height=700)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)
        
        self.right_frame = tk.Frame(self.monitor_tab, bg="#e0e0e0", width=400, height=700)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.right_frame.pack_propagate(False)
        
        self.video_frame = tk.Frame(self.left_frame, bg="black", width=640, height=480)
        self.video_frame.pack(padx=20, pady=20)
        
        self.control_frame = tk.Frame(self.left_frame, bg="#f0f0f0", height=180)
        self.control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.student_info_frame = tk.Frame(self.control_frame, bg="#f0f0f0")
        self.student_info_frame.pack(fill=tk.X, pady=10)
        
        self.student_label = tk.Label(
            self.student_info_frame,
            text="Student: Not Selected",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        )
        self.student_label.pack(side=tk.LEFT)
        
        self.select_student_btn = ttk.Button(
            self.student_info_frame,
            text="Select Student",
            command=self.select_student,
            width=15
        )
        self.select_student_btn.pack(side=tk.RIGHT)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack()
        
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)
        style.configure("Green.TButton", background="#4CAF50")
        style.configure("Red.TButton", background="#f44336")
        
        self.btn_frame = tk.Frame(self.control_frame, bg="#f0f0f0")
        self.btn_frame.pack(pady=20)
        
        self.start_btn = ttk.Button(
            self.btn_frame,
            text="Start Monitoring",
            command=self.start_monitoring,
            style="Green.TButton",
            width=20
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            self.btn_frame,
            text="Stop Monitoring",
            command=self.stop_monitoring,
            style="Red.TButton",
            width=20
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.status_frame = tk.Frame(self.control_frame, bg="#f0f0f0")
        self.status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Status: Not Monitoring",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#555555"
        )
        self.status_label.pack()
        
        self.alerts_frame = tk.Frame(self.right_frame, bg="#e0e0e0")
        self.alerts_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.alerts_label = tk.Label(
            self.alerts_frame,
            text="Alerts & Detections",
            font=("Arial", 16, "bold"),
            bg="#e0e0e0",
            fg="#333333"
        )
        self.alerts_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.alerts_list_frame = tk.Frame(self.alerts_frame, bg="#e0e0e0")
        self.alerts_list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(self.alerts_list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.alerts_listbox = tk.Listbox(
            self.alerts_list_frame,
            bg="white",
            fg="#333333",
            font=("Arial", 11),
            bd=0,
            highlightthickness=0,
            selectbackground="#3498db",
            yscrollcommand=self.scrollbar.set,
            height=15,  # Adjust the height as needed
            width=40    # Adjust the width as needed
        )
        self.alerts_listbox.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.alerts_listbox.yview)
        
        self.stats_frame = tk.Frame(self.right_frame, bg="#e0e0e0", height=250)
        self.stats_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.stats_label = tk.Label(
            self.stats_frame,
            text="Session Statistics",
            font=("Arial", 16, "bold"),
            bg="#e0e0e0",
            fg="#333333"
        )
        self.stats_label.pack(anchor=tk.W, pady=(0, 10))
        
        stats = [
            ("Session Duration:", "00:00:00"),
            ("Alert Count:", "0"),
            ("Attention Level:", "N/A"),
            ("Last Detection:", "N/A"),
            ("Blink Rate:", "N/A")
        ]
        
        self.stat_values = {}
        
        for key, value in stats:
            stat_frame = tk.Frame(self.stats_frame, bg="#e0e0e0")
            stat_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(
                stat_frame,
                text=key,
                font=("Arial", 12),
                bg="#e0e0e0",
                fg="#555555"
            ).pack(side=tk.LEFT)
            
            self.stat_values[key] = tk.Label(
                stat_frame,
                text=value,
                font=("Arial", 12, "bold"),
                bg="#e0e0e0",
                fg="#333333"
            )
            self.stat_values[key].pack(side=tk.RIGHT)
        
        self.clear_btn = ttk.Button(
            self.right_frame,
            text="Clear Alerts",
            command=self.clear_alerts,
            width=15
        )
        self.clear_btn.pack(pady=(0, 20))
        
        self.export_btn = ttk.Button(
            self.right_frame,
            text="Export Session Data",
            command=self.export_session_data,
            width=20
        )
        self.export_btn.pack(pady=(0, 20))
    
    def create_enroll_tab(self):
        enroll_frame = tk.Frame(self.enroll_tab, bg="#f0f0f0", padx=20, pady=20)
        enroll_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            enroll_frame,
            text="Student Enrollment",
            font=("Arial", 18, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(pady=(0, 20))
        
        form_frame = tk.Frame(enroll_frame, bg="#f0f0f0")
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        left_form = tk.Frame(form_frame, bg="#f0f0f0", width=400)
        left_form.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        right_form = tk.Frame(form_frame, bg="#f0f0f0", width=400)
        right_form.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        tk.Label(
            left_form,
            text="Student Information",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        fields = [
            ("Full Name:", "name_entry"),
            ("Student ID:", "id_entry")
        ]
        
        self.enroll_entries = {}
        
        for label_text, entry_name in fields:
            field_frame = tk.Frame(left_form, bg="#f0f0f0")
            field_frame.pack(fill=tk.X, pady=10)
            
            tk.Label(
                field_frame,
                text=label_text,
                font=("Arial", 12),
                bg="#f0f0f0",
                width=15,
                anchor=tk.W
            ).pack(side=tk.LEFT)
            
            entry = tk.Entry(
                field_frame,
                font=("Arial", 12),
                bd=1,
                relief=tk.SOLID
            )
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.enroll_entries[entry_name] = entry
        
        tk.Label(
            right_form,
            text="Face Registration",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # self.enroll_video_frame = tk.Frame(right_form, bg="black", width=320, height=240)
        self.enroll_video_frame = tk.Frame(right_form, bg="black", width=640, height=360)
        self.enroll_video_frame.pack(pady=10)
        # self.enroll_video_frame.pack_propagate(False)
        
        self.enroll_video_label = tk.Label(self.enroll_video_frame, bg="black")
        self.enroll_video_label.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = tk.Frame(right_form, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.capture_btn = ttk.Button(
            btn_frame,
            text="Capture Face",
            command=self.capture_face,
            width=15
        )
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.recapture_btn = ttk.Button(
            btn_frame,
            text="Recapture",
            command=self.recapture_face,
            width=15
        )
        self.recapture_btn.pack(side=tk.RIGHT, padx=5)
        self.recapture_btn.config(state=tk.DISABLED)
        
        self.face_status_label = tk.Label(
            right_form,
            text="Status: Face not captured",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#555555"
        )
        self.face_status_label.pack(pady=10)
        
        bottom_frame = tk.Frame(enroll_frame, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.X, pady=20)
        
        self.enroll_btn = ttk.Button(
            bottom_frame,
            text="Enroll Student",
            command=self.enroll_student,
            style="Green.TButton",
            width=20
        )
        self.enroll_btn.pack(side=tk.RIGHT)
        self.enroll_btn.config(state=tk.DISABLED)
        
        self.clear_enroll_btn = ttk.Button(
            bottom_frame,
            text="Clear Form",
            command=self.clear_enrollment_form,
            width=15
        )
        self.clear_enroll_btn.pack(side=tk.RIGHT, padx=10)
        
        self.enrollment_status = tk.Label(
            bottom_frame,
            text="",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.enrollment_status.pack(side=tk.LEFT)
        
        self.face_encoding = None
        self.face_image = None
        self.enroll_cap = None
    
    def create_history_tab(self):
        history_frame = tk.Frame(self.history_tab, bg="#f0f0f0", padx=20, pady=20)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            history_frame,
            text="Session History",
            font=("Arial", 18, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(pady=(0, 20))
        
        filter_frame = tk.Frame(history_frame, bg="#f0f0f0")
        filter_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            filter_frame,
            text="Filter by student:",
            font=("Arial", 12),
            bg="#f0f0f0"
        ).pack(side=tk.LEFT)
        
        self.student_filter = ttk.Combobox(
            filter_frame,
            font=("Arial", 12),
            width=30
        )
        self.student_filter.pack(side=tk.LEFT, padx=10)
        
        self.refresh_filter_btn = ttk.Button(
            filter_frame,
            text="Refresh List",
            command=self.load_session_history,
            width=15
        )
        self.refresh_filter_btn.pack(side=tk.RIGHT)
        
        sessions_frame = tk.Frame(history_frame, bg="#f0f0f0")
        sessions_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        sessions_table_frame = tk.Frame(sessions_frame)
        sessions_table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.sessions_tree = ttk.Treeview(
            sessions_table_frame,
            columns=("id", "student", "date", "duration", "alerts", "score"),
            show="headings",
            selectmode="browse"
        )
        
        self.sessions_tree.heading("id", text="ID")
        self.sessions_tree.heading("student", text="Student Name")
        self.sessions_tree.heading("date", text="Date")
        self.sessions_tree.heading("duration", text="Duration")
        self.sessions_tree.heading("alerts", text="Alerts")
        self.sessions_tree.heading("score", text="Attention Score")
        
        self.sessions_tree.column("id", width=50, anchor="center")
        self.sessions_tree.column("student", width=150)
        self.sessions_tree.column("date", width=150)
        self.sessions_tree.column("duration", width=100, anchor="center")
        self.sessions_tree.column("alerts", width=70, anchor="center")
        self.sessions_tree.column("score", width=100, anchor="center")
        
        scrollbar = ttk.Scrollbar(sessions_table_frame, orient="vertical", command=self.sessions_tree.yview)
        self.sessions_tree.configure(yscroll=scrollbar.set)
        
        self.sessions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        session_details_frame = tk.Frame(history_frame, bg="#f0f0f0")
        session_details_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            session_details_frame,
            text="Session Details",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        buttons_frame = tk.Frame(session_details_frame, bg="#f0f0f0")
        buttons_frame.pack(fill=tk.X, pady=10)
        
        self.view_video_btn = ttk.Button(
            buttons_frame,
            text="View Session Video",
            command=self.view_session_video,
            width=20
        )
        self.view_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.view_alerts_btn = ttk.Button(
            buttons_frame,
            text="View Session Alerts",
            command=self.view_session_alerts,
            width=20
        )
        self.view_alerts_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_report_btn = ttk.Button(
            buttons_frame,
            text="Export Session Report",
            command=self.export_session_report,
            width=20
        )
        self.export_report_btn.pack(side=tk.LEFT, padx=5)
        
        self.delete_session_btn = ttk.Button(
            buttons_frame,
            text="Delete Session",
            command=self.delete_session,
            width=15,
            style="Red.TButton"
        )
        self.delete_session_btn.pack(side=tk.RIGHT, padx=5)
        
        self.sessions_tree.bind("<ButtonRelease-1>", self.on_session_select)
        self.load_student_filter()
        self.load_session_history()
    
    def create_settings_tab(self):
        settings_frame = tk.Frame(self.settings_tab, bg="#f0f0f0", padx=20, pady=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            settings_frame,
            text="System Settings",
            font=("Arial", 18, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(pady=(0, 20))
        
        settings_categories = [
            ("Detection Thresholds", self.create_detection_settings),
            ("Alert Levels", self.create_alert_settings),
            ("Storage and Export", self.create_storage_settings),
            ("About", self.create_about_section)
        ]
        
        self.settings_notebook = ttk.Notebook(settings_frame)
        self.settings_notebook.pack(fill=tk.BOTH, expand=True)
        
        for category, create_func in settings_categories:
            tab = ttk.Frame(self.settings_notebook)
            self.settings_notebook.add(tab, text=category)
            create_func(tab)
    
    def create_detection_settings(self, parent):
        frame = tk.Frame(parent, bg="#f0f0f0", padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        settings = [
            ("Face Detection Confidence", 0.5, 0.1, 0.9, 0.1),
            ("Blink Detection Threshold", 0.3, 0.1, 0.5, 0.05),
            ("Head Pose Tolerance (degrees)", 30, 10, 45, 5),
            ("Gaze Tracking Sensitivity", 0.5, 0.1, 1.0, 0.1),
            ("Environment Scan Interval (sec)", 5, 1, 30, 1)
        ]
        
        self.detection_settings = {}
        
        for setting, default, min_val, max_val, step in settings:
            setting_frame = tk.Frame(frame, bg="#f0f0f0")
            setting_frame.pack(fill=tk.X, pady=10)
            
            tk.Label(
                setting_frame,
                text=f"{setting}:",
                font=("Arial", 12),
                bg="#f0f0f0",
                width=25,
                anchor=tk.W
            ).pack(side=tk.LEFT)
            
            value_var = tk.DoubleVar(value=default)
            self.detection_settings[setting] = value_var
            
            scale = ttk.Scale(
                setting_frame,
                from_=min_val,
                to=max_val,
                variable=value_var,
                orient=tk.HORIZONTAL,
                length=200
            )
            scale.pack(side=tk.LEFT, padx=10)
            
            value_label = tk.Label(
                setting_frame,
                textvariable=value_var,
                font=("Arial", 12),
                bg="#f0f0f0",
                width=5
            )
            value_label.pack(side=tk.LEFT)
        
        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(
            btn_frame,
            text="Save Settings",
            command=self.save_detection_settings,
            style="Green.TButton",
            width=15
        ).pack(side=tk.RIGHT)
        
        ttk.Button(
            btn_frame,
            text="Reset to Default",
            command=self.reset_detection_settings,
            width=15
        ).pack(side=tk.RIGHT, padx=10)
    
    def create_alert_settings(self, parent):
        frame = tk.Frame(parent, bg="#f0f0f0", padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        alert_levels = [
            ("Low Alert Threshold", 1),
            ("Medium Alert Threshold", 3),
            ("High Alert Threshold", 5),
            ("Critical Alert Threshold", 8)
        ]
        
        self.alert_settings = {}
        
        for setting, default in alert_levels:
            setting_frame = tk.Frame(frame, bg="#f0f0f0")
            setting_frame.pack(fill=tk.X, pady=10)
            
            tk.Label(
                setting_frame,
                text=f"{setting}:",
                font=("Arial", 12),
                bg="#f0f0f0",
                width=25,
                anchor=tk.W
            ).pack(side=tk.LEFT)
            
            value_var = tk.IntVar(value=default)
            self.alert_settings[setting] = value_var
            
            spinbox = ttk.Spinbox(
                setting_frame,
                from_=1,
                to=20,
                textvariable=value_var,
                width=5
            )
            spinbox.pack(side=tk.LEFT)
        
        notification_frame = tk.Frame(frame, bg="#f0f0f0")
        notification_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(
            notification_frame,
            text="Notification Settings",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        self.play_sound_var = tk.BooleanVar(value=True)
        sound_check = tk.Checkbutton(
            notification_frame,
            text="Play sound on alerts",
            variable=self.play_sound_var,
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        sound_check.pack(anchor=tk.W)
        
        self.save_snapshots_var = tk.BooleanVar(value=True)
        snapshot_check = tk.Checkbutton(
            notification_frame,
            text="Save snapshots for alerts",
            variable=self.save_snapshots_var,
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        snapshot_check.pack(anchor=tk.W)
        
        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(
            btn_frame,
            text="Save Settings",
            command=self.save_alert_settings,
            style="Green.TButton",
            width=15
        ).pack(side=tk.RIGHT)
        
        ttk.Button(
            btn_frame,
            text="Reset to Default",
            command=self.reset_alert_settings,
            width=15
        ).pack(side=tk.RIGHT, padx=10)
    
    def create_storage_settings(self, parent):
        frame = tk.Frame(parent, bg="#f0f0f0", padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        storage_frame = tk.Frame(frame, bg="#f0f0f0")
        storage_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            storage_frame,
            text="Storage Location:",
            font=("Arial", 12),
            bg="#f0f0f0",
            width=15,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.storage_path_var = tk.StringVar(value="./sessions")
        path_entry = tk.Entry(
            storage_frame,
            textvariable=self.storage_path_var,
            font=("Arial", 12),
            width=30
        )
        path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(
            storage_frame,
            text="Browse",
            command=self.browse_storage_path,
            width=10
        )
        browse_btn.pack(side=tk.RIGHT)
        
        video_quality_frame = tk.Frame(frame, bg="#f0f0f0")
        video_quality_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            video_quality_frame,
            text="Video Quality:",
            font=("Arial", 12),
            bg="#f0f0f0",
            width=15,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.video_quality_var = tk.StringVar(value="Medium")
        quality_options = ["Low", "Medium", "High"]
        
        quality_combo = ttk.Combobox(
            video_quality_frame,
            textvariable=self.video_quality_var,
            values=quality_options,
            width=15,
            state="readonly"
        )
        quality_combo.pack(side=tk.LEFT)
        
        retention_frame = tk.Frame(frame, bg="#f0f0f0")
        retention_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            retention_frame,
            text="Data Retention (days):",
            font=("Arial", 12),
            bg="#f0f0f0",
            width=15,
            anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.retention_var = tk.IntVar(value=30)
        
        retention_spin = ttk.Spinbox(
            retention_frame,
            from_=1,
            to=365,
            textvariable=self.retention_var,
            width=5
        )
        retention_spin.pack(side=tk.LEFT)
        
        tk.Label(
            retention_frame,
            text="(0 = keep forever)",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#777777"
        ).pack(side=tk.LEFT, padx=5)
        
        export_frame = tk.Frame(frame, bg="#f0f0f0")
        export_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(
            export_frame,
            text="Export Options",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        self.export_options = {
            "CSV": tk.BooleanVar(value=True),
            "PDF Report": tk.BooleanVar(value=True),
            "Alert Images": tk.BooleanVar(value=True),
            "Session Statistics": tk.BooleanVar(value=True)
        }
        
        for option, var in self.export_options.items():
            tk.Checkbutton(
                export_frame,
                text=option,
                variable=var,
                font=("Arial", 12),
                bg="#f0f0f0"
            ).pack(anchor=tk.W)
        
        btn_frame = tk.Frame(frame, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(
            btn_frame,
            text="Save Settings",
            command=self.save_storage_settings,
            style="Green.TButton",
            width=15
        ).pack(side=tk.RIGHT)
        
        ttk.Button(
            btn_frame,
            text="Reset to Default",
            command=self.reset_storage_settings,
            width=15
        ).pack(side=tk.RIGHT, padx=10)
    
    def create_about_section(self, parent):
        frame = tk.Frame(parent, bg="#f0f0f0", padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            frame,
            text="AI Exam Proctoring System",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(pady=(0, 10))
        
        tk.Label(
            frame,
            text="Version 1.0.0",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#333333"
        ).pack()
        
        tk.Label(
            frame,
            text="© 2025 ExamProctor AI",
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#777777"
        ).pack(pady=(5, 20))
        
        features_frame = tk.Frame(frame, bg="#f0f0f0")
        features_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            features_frame,
            text="Features:",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
            fg="#333333"
        ).pack(anchor=tk.W, pady=(0, 10))
        
        features = [
            "✓ Facial Recognition & Student Verification",
            "✓ Facial Expression Analysis",
            "✓ Gaze Tracking & Blink Detection",
            "✓ Head Pose Estimation",
            "✓ Environmental Scanning",
            "✓ Multi-Level Alert System",
            "✓ Session Recording & Reporting"
        ]
        
        for feature in features:
            tk.Label(
                features_frame,
                text=feature,
                font=("Arial", 12),
                bg="#f0f0f0",
                fg="#333333"
            ).pack(anchor=tk.W, pady=3)
    
    def on_tab_change(self, event):
        tab_id = self.notebook.index(self.notebook.select())
        
        if tab_id == 1:  # Enroll tab
            self.start_enrollment_camera()
        else:
            self.stop_enrollment_camera()
    
    def start_enrollment_camera(self):
        if self.enroll_cap is None:
            self.enroll_cap = cv2.VideoCapture(0)
            self.enrollment_camera_running = True
            threading.Thread(target=self.update_enrollment_camera, daemon=True).start()
    
    def stop_enrollment_camera(self):
        self.enrollment_camera_running = False
        if self.enroll_cap:
            self.enroll_cap.release()
            self.enroll_cap = None
        
    def update_enrollment_camera(self):
        """Continuously update the camera feed with face mesh and violations during enrollment."""
        while self.enrollment_camera_running:
            if not self.enroll_cap:
                time.sleep(0.05)
                continue

            ret, frame = self.enroll_cap.read()
            if not ret or frame is None:
                continue

            try:
                frame_height, frame_width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                display_frame = frame.copy()

                if results.multi_face_landmarks:
                    for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                        display_frame, face_details = self.draw_face_mesh(
                            frame, face_landmarks, (frame_height, frame_width)
                        )

                # Convert to PIL Image, resize, and show on GUI
                pil_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                # resized_image = pil_image.resize((320, 240), Image.LANCZOS)
                resized_image = pil_image.resize((640  , 480), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=resized_image)

                self.enroll_video_label.imgtk = imgtk
                self.enroll_video_label.configure(image=imgtk)

            except Exception as e:
                print(f"[Error] Enrollment camera update failed: {e}")

            time.sleep(0.05)

    def capture_face(self):
        """Capture face from camera using MediaPipe"""
        if self.enroll_cap:
            ret, frame = self.enroll_cap.read()
            if ret:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.face_mesh.process(rgb_frame)
                
                # Check if face is detected
                if not results.multi_face_landmarks:
                    # Try with OpenCV face cascade as backup
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                    
                    if len(faces) == 0:
                        messagebox.showwarning("No Face Detected", "Please position your face in front of the camera.")
                        return
                    elif len(faces) > 1:
                        messagebox.showwarning("Multiple Faces", "Multiple faces detected. Please ensure only one person is in frame.")
                        return
                    
                    # Use OpenCV detection
                    x, y, w, h = faces[0]
                    self.face_image = frame[y:y+h, x:x+w]
                    self.face_encoding = self.extract_face_encoding(self.face_image)
                
                    self.face_status_label.config(
                    text="Status: Face captured successfully",
                    fg="#4CAF50"
                    )
                
                    self.recapture_btn.config(state=tk.NORMAL)
                    self.enroll_btn.config(state=tk.NORMAL)
                else:
                    # Process MediaPipe results
                    if len(results.multi_face_landmarks) > 1:
                        messagebox.showwarning("Multiple Faces", "Multiple faces detected. Please ensure only one person is in frame.")
                        return
                    
                    # Get face landmarks
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Get face details
                    h, w = frame.shape[:2]
                    face_details = self.extract_face_details(face_landmarks, (h, w))
                    
                    # Extract face bounding box
                    face_center = face_details['face_center']
                    face_width = face_details['face_width']
                    face_height = face_details['face_height']
                    
                    # Add margin to face bounding box
                    margin = 0.2  # 20% margin
                    width_with_margin = int(face_width * (1 + margin))
                    height_with_margin = int(face_height * (1 + margin))
                    
                    # Calculate bounding box
                    left = max(0, face_center[0] - width_with_margin // 2)
                    top = max(0, face_center[1] - height_with_margin // 2)
                    right = min(w, face_center[0] + width_with_margin // 2)
                    bottom = min(h, face_center[1] + height_with_margin // 2)
                    
                    # Crop face from image
                    self.face_image = frame[top:bottom, left:right]
                
                    # Extract face encoding
                    self.face_encoding = self.extract_face_encoding(self.face_image)
                    
                    # Update UI
                    if self.face_status_label:
                        self.face_status_label.config(
                            text="Status: Face captured successfully",
                            fg="#4CAF50"
                        )
                    
                    if self.recapture_btn:
                        self.recapture_btn.config(state=tk.NORMAL)
                    if self.enroll_btn:
                        self.enroll_btn.config(state=tk.NORMAL)
                    
                    
                
                return self.face_encoding is not None
     
    def extract_face_details(self, landmarks, image_shape):
        """Extract face details from landmarks"""
        try:
            h, w = image_shape
            details = {}
            
            # Convert landmark coordinates to pixel positions
            points = {}
            for idx, landmark in enumerate(landmarks.landmark):
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                points[idx] = (px, py)
            
            # Get key facial points
            left_eye_pos = points.get(self.LEFT_EYE, (0, 0))
            right_eye_pos = points.get(self.RIGHT_EYE, (0, 0))
            left_ear_pos = points.get(self.LEFT_EAR, (0, 0))
            right_ear_pos = points.get(self.RIGHT_EAR, (0, 0))
            forehead_pos = points.get(self.FOREHEAD, (0, 0))
            chin_pos = points.get(self.CHIN, (0, 0))
            nose_pos = points.get(self.NOSE_TIP, (0, 0))
            mouth_pos = points.get(self.MOUTH_CENTER, (0, 0))
            
            # Calculate face measurements
            details['eye_distance'] = self.calculate_distance(left_eye_pos, right_eye_pos)
            details['face_width'] = self.calculate_distance(left_ear_pos, right_ear_pos)
            details['face_height'] = self.calculate_distance(forehead_pos, chin_pos)
            
            # Store key points
            details['left_eye'] = left_eye_pos
            details['right_eye'] = right_eye_pos
            details['left_ear'] = left_ear_pos
            details['right_ear'] = right_ear_pos
            details['nose'] = nose_pos
            details['mouth'] = mouth_pos
            
            # Calculate face center
            center_x = (left_eye_pos[0] + right_eye_pos[0]) // 2
            center_y = (forehead_pos[1] + chin_pos[1]) // 2
            details['face_center'] = (center_x, center_y)
            
            return details
        except Exception as e:
            print(f"Error extracting face details: {e}")
            return {
                'eye_distance': 0,
                'face_width': 0,
                'face_height': 0,
                'left_eye': (0, 0),
                'right_eye': (0, 0),
                'nose': (0, 0),
                'mouth': (0, 0),
                'face_center': (0, 0)
            }
    
    def extract_face_encoding(self, face_img):
        """Extract face encoding using MediaPipe landmarks"""
        try:
            if face_img is None:
                return None
            
            # Convert to RGB for MediaPipe
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(face_img_rgb)
            
            if not results.multi_face_landmarks:
                return None
            
            # Extract landmarks
            landmarks = results.multi_face_landmarks[0]
            
            # Get face details
            h, w = face_img.shape[:2]
            details = self.extract_face_details(landmarks, (h, w))
            
            # Create encoding from key measurements and points
            # This is a simplified encoding; real face recognition would use more complex features
            encoding = np.array([
                details['eye_distance'],
                details['face_width'],
                details['face_height'],
                details['left_eye'][0], details['left_eye'][1],
                details['right_eye'][0], details['right_eye'][1],
                details['nose'][0], details['nose'][1],
                details['mouth'][0], details['mouth'][1],
                details['face_center'][0], details['face_center'][1]
            ])
            
            # Normalize encoding for better comparison
            if np.sum(encoding) != 0:
                encoding = encoding / np.linalg.norm(encoding)
            
            return encoding
            
        except Exception as e:
            print(f"Error extracting face encoding: {e}")
            return None
    
    def recapture_face(self):
        """Reset and recapture face"""
        # Reset face data
        self.face_image = None
        self.face_encoding = None
        
        # Update UI
        if self.face_status_label:
            self.face_status_label.config(
                text="Status: Ready to capture face",
                fg="#FF9800"  # Orange color for ready state
            )
        
        if self.recapture_btn:
            self.recapture_btn.config(state=tk.DISABLED)
        if self.enroll_btn:
            self.enroll_btn.config(state=tk.DISABLED)
        
        # Capture new face
        self.capture_face()
    
    def draw_face_mesh(self, frame, face_landmarks, frame_dimensions, output_dimensions=None, 
                       x_offset=0, y_offset=0):
        """Draw the face mesh on the frame"""
        if output_dimensions:
            output_frame = np.zeros((output_dimensions[0], output_dimensions[1], 3), dtype=np.uint8)
            scale = min(output_dimensions[0]/frame_dimensions[0], 
                        output_dimensions[1]/frame_dimensions[1])
            display_height, display_width = output_dimensions
        else:
            output_frame = frame.copy()
            scale = 1.0
            display_height, display_width = frame_dimensions
        
        frame_height, frame_width = frame_dimensions
        
        try:
            # Extract face details
            face_details = self.extract_face_details(face_landmarks, (frame_height, frame_width))
            
            # Draw simplified face mesh (just contours)
            connection_drawing = self.mp_drawing_styles.get_default_face_mesh_contours_style()
            
            # Draw directly on the output canvas with manual scaling
            mesh_drawing = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
            for connection in self.mp_face_mesh.FACEMESH_CONTOURS:
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
                    cv2.line(output_frame, (start_x, start_y), (end_x, end_y), 
                             (0, 255, 0), 1)
            
            # Add face number and bounding box if we have valid face details
            if face_details['nose'] != (0, 0):
                # Scale nose position
                scaled_nose = (
                    int(face_details['nose'][0] * scale) + x_offset,
                    int(face_details['nose'][1] * scale) + y_offset
                )
                
                # Add face label near the nose
                cv2.putText(
                    output_frame, 
                    "", 
                    (scaled_nose[0], scaled_nose[1] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
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
                    cv2.rectangle(output_frame, top_left, bottom_right, (0, 255, 0), 2)
                    
            return output_frame, face_details
            
        except Exception as e:
            print(f"Error drawing face mesh: {e}")
            return output_frame, None
        
    def enroll_student(self):
        name = self.enroll_entries["name_entry"].get().strip()
        student_id = self.enroll_entries["id_entry"].get().strip()
        
        if not name or not student_id:
            messagebox.showwarning("Missing Information", "Please fill in all fields.")
            return
        
        if self.face_encoding is None:
            messagebox.showwarning("No Face Data", "Please capture the student's face first.")
            return
        
        try:
            enrollment_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            encoded_face = sqlite3.Binary(self.face_encoding.tobytes())
            
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO students (name, student_id, enrollment_date, face_encoding) VALUES (?, ?, ?, ?)",
                (name, student_id, enrollment_date, encoded_face)
            )
            
            conn.commit()
            conn.close()
            
            self.enrollment_status.config(
                text=f"Student {name} (ID: {student_id}) enrolled successfully!",
                fg="#4CAF50"
            )
            
            self.clear_enrollment_form()
            
        except sqlite3.IntegrityError:
            messagebox.showerror("Enrollment Error", f"Student ID {student_id} already exists.")
        except Exception as e:
            messagebox.showerror("Enrollment Error", f"An error occurred: {e}")
    
    def clear_enrollment_form(self):
        self.enroll_entries["name_entry"].delete(0, tk.END)
        self.enroll_entries["id_entry"].delete(0, tk.END)
        self.recapture_face()
    
    def select_student(self):
        self.student_select_dialog = tk.Toplevel(self.root)
        self.student_select_dialog.title("Select Student")
        self.student_select_dialog.geometry("400x400")
        self.student_select_dialog.resizable(False, False)
        self.student_select_dialog.transient(self.root)
        self.student_select_dialog.grab_set()
        
        tk.Label(
            self.student_select_dialog,
            text="Select a Student",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        search_frame = tk.Frame(self.student_select_dialog)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            search_frame,
            text="Search:",
            font=("Arial", 12)
        ).pack(side=tk.LEFT)
        
        search_var = tk.StringVar()
        search_entry = tk.Entry(
            search_frame,
            textvariable=search_var,
            font=("Arial", 12),
            width=25
        )
        search_entry.pack(side=tk.LEFT, padx=5)
        
        search_entry.bind("<KeyRelease>", lambda e: self.filter_students(search_var.get()))
        
        list_frame = tk.Frame(self.student_select_dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.students_listbox = tk.Listbox(
            list_frame,
            font=("Arial", 12),
            yscrollcommand=scrollbar.set
        )
        self.students_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.students_listbox.yview)
        
        btn_frame = tk.Frame(self.student_select_dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            btn_frame,
            text="Select",
            command=self.confirm_student_selection,
            style="Green.TButton",
            width=15
        ).pack(side=tk.RIGHT)
        
        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.student_select_dialog.destroy,
            width=15
        ).pack(side=tk.RIGHT, padx=5)
        
        self.load_students()
    
    def load_students(self):
        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, name, student_id FROM students ORDER BY name")
            students = cursor.fetchall()
            
            conn.close()
            
            self.students_data = students
            self.students_listbox.delete(0, tk.END)
            
            for student in students:
                self.students_listbox.insert(tk.END, f"{student[1]} (ID: {student[2]})")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load students: {e}")
    
    def filter_students(self, search_text):
        self.students_listbox.delete(0, tk.END)
        
        for student in self.students_data:
            if search_text.lower() in student[1].lower() or search_text in student[2]:
                self.students_listbox.insert(tk.END, f"{student[1]} (ID: {student[2]})")
    
    def confirm_student_selection(self):
        selected_idx = self.students_listbox.curselection()
        
        if not selected_idx:
            messagebox.showwarning("No Selection", "Please select a student.")
            return
            
        student_id, name, student_id_num = self.students_data[selected_idx[0]]
        self.current_student = (student_id, name, student_id_num)
        self.student_label.config(text=f"Student: {name} (ID: {student_id_num})")
        self.student_select_dialog.destroy()
    
    def start_monitoring(self):
        if not self.current_student:
            messagebox.showwarning("No Student Selected", "Please select a student before starting monitoring.")
            return
            
        if not self.monitoring:
            try: 
                self.monitoring = True
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Status: Monitoring Active", fg="#4CAF50")
                
                self.cap = cv2.VideoCapture(0)
                
                self.start_time = time.time()
                self.alert_count = 0
                self.total_blinks = 0
                
                if not os.path.exists("sessions"):
                    os.makedirs("sessions")
                    
                session_folder = f"sessions/session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(session_folder)
                
                video_path = f"{session_folder}/recording.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(video_path, fourcc, 120, (640, 480))
                
                self.session_id = self.create_session_record(video_path)
                self.root.after(10, self.monitor_video)
                
                self.update_stats()
            except Exception as e:
                print(f"error in start_monitoring: {e}")
    def monitor_video(self):
        if not self.monitoring:
            return
            
        try:
            # Import the CheatingDetectionSystem
            from videotry2 import CheatingDetectionSystem, mp_face_mesh, mp, GAZE_LIMIT_X_MIN, GAZE_LIMIT_X_MAX, GAZE_LIMIT_Y_MIN, GAZE_LIMIT_Y_MAX
            
            # Initialize the cheating detection system if not already initialized
            if not hasattr(self, 'cheating_detector'):
                self.cheating_detector = CheatingDetectionSystem()
            
            # Initialize MediaPipe Face Mesh if not already initialized
            if not hasattr(self, 'face_mesh'):
                self.face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            
            # Set up processing variables if not already set
            if not hasattr(self, 'frame_counter'):
                self.frame_counter = 0
                self.process_every_n_frames = 2
                self.looking_away_start_time = None
                self.last_gaze_status = True  # Initially assume looking at screen
                
            # Initialize suspicious events log if not present
            if not hasattr(self, 'suspicious_events'):
                self.suspicious_events = []
            
            # Process a single frame
            ret, frame = self.cap.read()
            if not ret:
                self.add_alert("Camera error - cannot read frame", "Critical")
                self.root.after(500, self.monitor_video)  # Try again after a delay
                return
                
            # Record video if writer exists
            if self.video_writer:
                self.video_writer.write(frame)
            
            # Process frame with CheatingDetectionSystem
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames == 0:
                # Process the frame
                
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]
                    
                    # Face detection processing
                    face_results = self.face_mesh.process(image_rgb)
                    
                    # Track face count for cheating detection
                    face_count = 0
                    face_detected = False
                    
                    if face_results.multi_face_landmarks:
                        face_count = len(face_results.multi_face_landmarks)
                        face_detected = True
                        # Check for multiple faces (potential cheating)
                        if face_count > 1:
                            self.add_alert(f"Multiple faces detected ({face_count}) - Possible collaboration", "Critical")
                            # Highlight frame in red to indicate critical alert
                            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                                
                            # Save snapshot if method exists
                            if hasattr(self, 'save_snapshot'):
                                self.save_snapshot(frame, "multiple_faces")
                            
                        # Process the primary face (first detected)
                        if face_count > 0:
                            primary_face = face_results.multi_face_landmarks[0]
                            
                            # Extract face details using proper method
                            
                            face_details = self.extract_face_details(primary_face, (h, w))
                            gaze_data = self.cheating_detector.calculate_gaze_direction(
                            frame, primary_face, face_details, (h, w))
                            print(gaze_data)
                            # Calculate and track gaze direction
                            
                            
                            # Get gaze status
                            looking_at_screen = gaze_data['looking_at_screen']
                            
                            # Track looking away behavior
                            looking_result = self.cheating_detector.track_looking_away(looking_at_screen)
                            
                            # Process looking away results
                            if looking_result['status'] == 'suspicious':
                                severity = "High" if looking_result.get('duration', 0) > 4 else "Medium"
                                self.add_alert(f"Extended gaze deviation ({looking_result.get('duration', 0):.1f}s)", severity)
                                cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 5)
                                
                                # Save snapshot if method exists and severity is high
                                if severity == "High" and hasattr(self, 'save_snapshot'):
                                    self.save_snapshot(frame, "gaze_deviation")
                            
                            # Display gaze information on frame
                            gaze_status_text = "Looking at screen" if looking_at_screen else "Looking away"
                            gaze_color = (0, 255, 0) if looking_at_screen else (0, 0, 255)
                            cv2.putText(frame, f"Gaze: {gaze_status_text}", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, gaze_color, 2)
                            
                            # Display gaze coordinates
                            cv2.putText(frame, f"Gaze X: {gaze_data['gaze_x']:.2f}, Y: {gaze_data['gaze_y']:.2f}", 
                                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                            
                            # Draw allowed gaze region (for debugging)
                            region_left = int(GAZE_LIMIT_X_MIN * w)
                            region_right = int(GAZE_LIMIT_X_MAX * w)
                            region_top = int(GAZE_LIMIT_Y_MIN * h)
                            region_bottom = int(GAZE_LIMIT_Y_MAX * h)
                            
                            cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), 
                                        (0, 255, 0), 1)
                            
                            # Detect objects that might indicate cheating
                            detected_objects = self.cheating_detector.detect_objects(frame)
                            
                            # Process detected objects
                            for obj in detected_objects:
                                obj_type = obj['type']
                                box = obj.get('box', [0, 0, 100, 100])
                                
                                # Draw object box
                                cv2.rectangle(frame, 
                                            (box[0], box[1]),
                                            (box[0] + box[2], box[1] + box[3]),
                                            (0, 0, 255), 2)
                                
                                # Add object label
                                cv2.putText(frame, obj_type, 
                                        (box[0], box[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                                # Generate alert for suspicious object
                                self.add_alert(f"Suspicious object detected: {obj_type}", "High")
                                
                                # Save snapshot if method exists
                                if hasattr(self, 'save_snapshot'):
                                    self.save_snapshot(frame, f"object_{obj_type}")
                            
                            # Draw face mesh (simplified)
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame, primary_face, mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                            )
                    else:
                        face_detected = False
                        self.add_alert("No face detected - Student may be absent", "High")
                        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                        
                        # Save snapshot of no face if method exists
                        if hasattr(self, 'save_snapshot'):
                            self.save_snapshot(frame, "no_face")
                except Exception as e:
                    print(f"Failed face_details: {e}")
                
            
            # Convert image for display in Tkinter
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the image in Tkinter label
            self.video_label.imgtk = imgtk  # Keep reference to prevent garbage collection
            self.video_label.configure(image=imgtk)
            
            # Schedule the next frame processing (aim for ~30fps)
            self.root.after(33, self.monitor_video)
        
        except Exception as e:
            print(f"Error in monitor_video: {e}")
            import traceback
            traceback.print_exc()
            self.add_alert(f"Monitoring error: {str(e)}", "Critical")
            
            # Try to recover monitoring if possible
            self.root.after(10, self.monitor_video)  # Try again after a delay

    def create_session_record(self, video_path):
        """Create a database record for the monitoring session"""
        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            
            # Use proper datetime import
            from datetime import datetime
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute(
                "INSERT INTO sessions (student_id, start_time, video_path) VALUES (?, ?, ?)",
                (self.current_student[0], start_time, video_path)
            )
            
            session_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return session_id
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create session record: {e}")
            return None
    
    def stop_monitoring(self):
        if self.monitoring:
            self.monitoring = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Monitoring Stopped", fg="#f44336")
            
            if self.cap:
                self.cap.release()
                
            if self.video_writer:
                self.video_writer.release()
                
            if self.session_id:
                self.update_session_record()
                
            messagebox.showinfo("Session Complete", f"Monitoring session completed.\nTotal alerts: {self.alert_count}")
            
    def update_session_record(self):
        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            
            end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            elapsed = int(time.time() - self.start_time)
            attention_score = max(0, 100 - (self.alert_count * 5))
            
            cursor.execute(
                "UPDATE sessions SET end_time = ?, alert_count = ?, attention_score = ? WHERE id = ?",
                (end_time, self.alert_count, attention_score, self.session_id)
            )
            
            conn.commit()
            conn.close()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update session record: {e}")
    
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eye_landmarks(self, shape):
        left_eye = []
        right_eye = []
        
        for i in range(36, 42):
            left_eye.append((shape.part(i).x, shape.part(i).y))
            
        for i in range(42, 48):
            right_eye.append((shape.part(i).x, shape.part(i).y))
            
        return left_eye, right_eye
          
    def add_alert(self, message, severity="Low"):
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        alert = f"[{timestamp}] [{severity}] {message}"
        
        self.alerts.append(alert)
        
        color_map = {
            "Low": "#FFA000",
            "Medium": "#F57C00",
            "High": "#D32F2F",
            "Critical": "#B71C1C"
        }
        
        self.alerts_listbox.insert(tk.END, alert)
        self.alerts_listbox.itemconfig(self.alerts_listbox.size() - 1, fg=color_map.get(severity, "#D32F2F"))
        self.alerts_listbox.yview(tk.END)
        
        self.alert_count += 1
        self.stat_values["Alert Count:"].config(text=str(self.alert_count))
        self.stat_values["Last Detection:"].config(text=timestamp)
        
        if self.session_id:
            try:
                conn = sqlite3.connect('exam_proctor.db')
                cursor = conn.cursor()
                
                cursor.execute(
                    "INSERT INTO alerts (session_id, timestamp, alert_type, message, severity) VALUES (?, ?, ?, ?, ?)",
                    (self.session_id, 
                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     message.split(" - ")[0] if " - " in message else message,
                     message,
                     severity)
                )
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                print(f"Failed to log alert: {e}")
    
    def clear_alerts(self):
        self.alerts = []
        self.alerts_listbox.delete(0, tk.END)
    
    def update_stats(self):
        if self.monitoring:
            elapsed = int(time.time() - self.start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.stat_values["Session Duration:"].config(text=time_str)
            
            blink_rate = self.total_blinks / (elapsed / 60) if elapsed > 0 else 0
            self.stat_values["Blink Rate:"].config(text=f"{blink_rate:.1f}/min")
            
            if self.alert_count == 0:
                attention = "Excellent"
                color = "#4CAF50"
            elif self.alert_count < 3:
                attention = "Good"
                color = "#8BC34A"
            elif self.alert_count < 6:
                attention = "Fair"
                color = "#FFC107"
            else:
                attention = "Poor"
                color = "#F44336"
                
            self.stat_values["Attention Level:"].config(text=attention, fg=color)
            
            self.root.after(1000, self.update_stats)
    
    def load_student_filter(self):
        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, name FROM students ORDER BY name")
            students = cursor.fetchall()
            
            conn.close()
            
            self.student_filter['values'] = ["All Students"] + [student[1] for student in students]
            self.student_filter.current(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load students: {e}")
    
    def load_session_history(self):
        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            
            self.sessions_tree.delete(*self.sessions_tree.get_children())
            
            filter_value = self.student_filter.get()
            
            if filter_value == "All Students":
                cursor.execute("""
                    SELECT 
                        s.id, st.name, s.start_time, s.end_time, s.alert_count, s.attention_score
                    FROM sessions s
                    JOIN students st ON s.student_id = st.id
                    ORDER BY s.start_time DESC
                """)
            else:
                cursor.execute("""
                    SELECT 
                        s.id, st.name, s.start_time, s.end_time, s.alert_count, s.attention_score
                    FROM sessions s
                    JOIN students st ON s.student_id = st.id
                    WHERE st.name = ?
                    ORDER BY s.start_time DESC
                """, (filter_value,))
                
            sessions = cursor.fetchall()
            
            conn.close()
            
            for session in sessions:
                session_id, name, start_time, end_time, alert_count, attention_score = session
                
                start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                date_str = start_dt.strftime("%Y-%m-%d %H:%M")
                
                if end_time:
                    end_dt = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                    end_str = end_dt.strftime("%Y-%m-%d %H:%M")
                    duration = end_dt - start_dt
                    duration_str = str(duration).split('.')[0]  # Format as HH:MM:SS
                else:
                    end_str = "Active"
                    duration_str = "Ongoing"
                
                # Format attention score to show as percentage with 2 decimal places
                if attention_score is not None:
                    att_score_str = f"{attention_score:.2f}%"
                else:
                    att_score_str = "N/A"
                
                # Insert data into the treeview
                self.sessions_tree.insert(
                    "", "end", 
                    values=(session_id, name, date_str, end_str, duration_str, alert_count, att_score_str),
                    tags=('alert' if alert_count > 3 else '')
                )
            
            # Apply color formatting based on tags
            self.sessions_tree.tag_configure('alert', background='#ffcccc')
            
            # Show message if no sessions found
            if not sessions:
                self.status_label.config(text=f"No sessions found for filter: {filter_value}")
            else:
                self.status_label.config(text=f"Loaded {len(sessions)} sessions")
                
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to load session history: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            logging.error(f"Error loading session history: {e}", exc_info=True)
            
    def view_session_video(self):
        selected_item = self.sessions_tree.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a session from the list.")
            return
        
        session_id = self.sessions_tree.item(selected_item)['values'][0]
        
        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            cursor.execute("SELECT video_path FROM sessions WHERE id = ?", (session_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and os.path.exists(result[0]):
                cap = cv2.VideoCapture(result[0])
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imshow(f"Session Video - ID {session_id}", frame)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                messagebox.showerror("Video Not Found", "The recorded video file could not be found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play session video: {e}")

      
    def view_session_alerts(self):
        selected_item = self.sessions_tree.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a session from the list.")
            return

        session_id = self.sessions_tree.item(selected_item)['values'][0]

        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, alert_type, message, severity FROM alerts WHERE session_id = ? ORDER BY timestamp", (session_id,))
            alerts = cursor.fetchall()
            conn.close()

            if not alerts:
                messagebox.showinfo("No Alerts", "No alerts were recorded in this session.")
                return

            alert_window = tk.Toplevel(self.root)
            alert_window.title("Session Alerts")
            alert_window.geometry("600x400")
            alert_window.resizable(False, False)

            tree = ttk.Treeview(alert_window, columns=("timestamp", "type", "message", "severity"), show="headings")
            tree.heading("timestamp", text="Time")
            tree.heading("type", text="Type")
            tree.heading("message", text="Message")
            tree.heading("severity", text="Severity")

            for alert in alerts:
                tree.insert("", tk.END, values=alert)

            tree.pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrieve session alerts: {e}")

    def export_session_report(self):
        selected_item = self.sessions_tree.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a session from the list.")
            return

        session_id = self.sessions_tree.item(selected_item)['values'][0]
        export_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])

        if not export_path:
            return
        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            session_data = cursor.fetchone()

            cursor.execute("SELECT timestamp, alert_type, message, severity FROM alerts WHERE session_id = ?", (session_id,))
            alerts = cursor.fetchall()

            conn.close()

            with open(export_path, "w", newline='', encoding='utf-8') as f:
                f.write("Session ID,Student ID,Start Time,End Time,Video Path,Alert Count,Attention Score\n")
                f.write(",".join(map(str, session_data)) + "\n\n")

                f.write("Alerts:\n")
                f.write("Timestamp,Alert Type,Message,Severity\n")
                for alert in alerts:
                    f.write(",".join(alert) + "\n")

            messagebox.showinfo("Export Successful", "Session report exported successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export session report: {e}")

    def delete_session(self):
        selected_item = self.sessions_tree.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a session to delete.")
            return

        confirm = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this session and all related data?")
        if not confirm:
            return

        session_id = self.sessions_tree.item(selected_item)['values'][0]

        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            conn.close()

            self.sessions_tree.delete(selected_item)
            messagebox.showinfo("Deleted", "Session deleted successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete session: {e}")
    
    def export_session_data(self):
        if not self.session_id:
            messagebox.showwarning("No Session", "No active session data to export.")
            return

        export_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Session Data"
        )
        
        if not export_path:
            return

        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM sessions WHERE id = ?", (self.session_id,))
            session_data = cursor.fetchone()

            cursor.execute("SELECT timestamp, alert_type, message, severity FROM alerts WHERE session_id = ?", (self.session_id,))
            alerts = cursor.fetchall()
            conn.close()

            with open(export_path, "w", newline='', encoding="utf-8") as f:
                f.write("Session ID,Student ID,Start Time,End Time,Video Path,Alert Count,Attention Score\n")
                f.write(",".join(map(str, session_data)) + "\n\n")

                f.write("Alerts:\n")
                f.write("Timestamp,Alert Type,Message,Severity\n")
                for alert in alerts:
                    f.write(",".join(alert) + "\n")

            messagebox.showinfo("Export Successful", "Current session data exported successfully.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export session data: {e}")

    def on_session_select(self, event):
        selected = self.sessions_tree.selection()
        if not selected:
            return

        # Optional: fetch and display session info if needed
        session_id = self.sessions_tree.item(selected)['values'][0]
        print(f"Session selected: ID {session_id}")
    
    def save_detection_settings(self):
        try:
            settings = {k: v.get() for k, v in self.detection_settings.items()}
            with open("detection_settings.json", "w") as f:
                import json
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Settings Saved", "Detection settings saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def reset_detection_settings(self):
        default_values = {
        "Face Detection Confidence": 0.5,
        "Blink Detection Threshold": 0.3,
        "Head Pose Tolerance (degrees)": 30,
        "Gaze Tracking Sensitivity": 0.5,
        "Environment Scan Interval (sec)": 5
    }
        for key, var in self.detection_settings.items():
            var.set(default_values.get(key, 0))

    def save_alert_settings(self):
        try:
            alert_thresholds = {k: v.get() for k, v in self.alert_settings.items()}
            preferences = {
                "play_sound": self.play_sound_var.get(),
                "save_snapshots": self.save_snapshots_var.get()
            }

            settings = {
                "thresholds": alert_thresholds,
                "preferences": preferences
            }

            import json
            with open("alert_settings.json", "w") as f:
                json.dump(settings, f, indent=4)

            messagebox.showinfo("Settings Saved", "Alert settings saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save alert settings: {e}")

    def reset_alert_settings(self):
        defaults = {
        "Low Alert Threshold": 1,
        "Medium Alert Threshold": 3,
        "High Alert Threshold": 5,
        "Critical Alert Threshold": 8
    }

        for key, var in self.alert_settings.items():
            var.set(defaults.get(key, 1))

        self.play_sound_var.set(True)
        self.save_snapshots_var.set(True)
    def browse_storage_path(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.storage_path_var.set(selected_dir)

    def save_storage_settings(self):
        """
        Save the storage settings from the UI to a configuration file
        and apply them to the current application instance.
        """
        storage_path = self.storage_path_var.get()
        video_quality = self.video_quality_var.get()
        retention_days = self.retention_var.get()

        if not os.path.exists(storage_path):
            try:
                os.makedirs(storage_path)
                print(f"Created directory: {storage_path}")
            except OSError as e:
                messagebox.showerror("Error", f"Failed to create directory: {e}")
                return

        export_config = {option: var.get() for option, var in self.export_options.items()}
        config = {
            "storage": {
                "path": storage_path,
                "video_quality": video_quality,
                "retention_days": retention_days,
                "export_options": export_config
            }
        }
        try:
            with open("config.json", "r") as file:
                existing_config = json.load(file)

            existing_config["storage"] = config["storage"]
            
            with open("config.json", "w") as file:
                json.dump(existing_config, file, indent=4)
        except FileNotFoundError:
            with open("config.json", "w") as file:
                json.dump(config, file, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            return
        self.config["storage"] = config["storage"]
        
        messagebox.showinfo("Success", "Storage settings saved successfully!")

    def reset_storage_settings(self):
        self.storage_path_var.set("./sessions") 
        self.video_quality_var.set("Medium")
        self.retention_var.set(30)
        self.export_options["CSV"].set(True)
        self.export_options["PDF Report"].set(True)
        self.export_options["Alert Images"].set(True)
        self.export_options["Session Statistics"].set(True)
        
        messagebox.showinfo("Settings Reset", "Storage settings have been reset to default values.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ExamProctorSystem(root)
    root.mainloop()
    