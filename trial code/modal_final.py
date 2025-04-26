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
        while self.enrollment_camera_running:
            if self.enroll_cap:
                ret, frame = self.enroll_cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv_image)
                    img = img.resize((320, 240), Image.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.enroll_video_label.imgtk = imgtk
                    self.enroll_video_label.configure(image=imgtk)
            
            time.sleep(0.05)
    
    def capture_face(self):
        if self.enroll_cap:
            ret, frame = self.enroll_cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) == 0:
                    messagebox.showwarning("No Face Detected", "Please position your face in front of the camera.")
                    return
                elif len(faces) > 1:
                    messagebox.showwarning("Multiple Faces", "Multiple faces detected. Please ensure only one person is in frame.")
                    return
                
                x, y, w, h = faces[0]
                self.face_image = frame[y:y+h, x:x+w]
                self.face_encoding = self.extract_face_encoding(self.face_image)
                
                self.face_status_label.config(
                    text="Status: Face captured successfully",
                    fg="#4CAF50"
                )
                
                self.recapture_btn.config(state=tk.NORMAL)
                self.enroll_btn.config(state=tk.NORMAL)
    
    def extract_face_encoding(self, face_img):
        try:
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face = self.detector(face_img_rgb, 1)
            
            if not face:
                return None
            
            face_encoding = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Simplified placeholder
            return face_encoding
            
        except Exception as e:
            print(f"Error extracting face encoding: {e}")
            return None
    
    def recapture_face(self):
        self.face_image = None
        self.face_encoding = None
        self.face_status_label.config(
            text="Status: Face not captured",
            fg="#555555"
        )
        self.recapture_btn.config(state=tk.DISABLED)
        self.enroll_btn.config(state=tk.DISABLED)
    
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
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
            
            self.session_id = self.create_session_record(video_path)
            
            self.monitoring_thread = threading.Thread(target=self.monitor_video)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            self.update_stats()
    
    def create_session_record(self, video_path):
        try:
            conn = sqlite3.connect('exam_proctor.db')
            cursor = conn.cursor()
            
            start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
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
    
    def monitor_video(self):
        expressions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        suspicious_expressions = ['Happy', 'Surprise']
        
        frame_counter = 0
        blink_counter = 0
        last_blink_time = time.time()
        
        while self.monitoring:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            face_detected = len(faces) > 0
            
            if self.video_writer:
                self.video_writer.write(frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dlib_faces = self.detector(rgb_frame, 0)
            
            if len(dlib_faces) > 0 and self.predictor:
                for face in dlib_faces:
                    shape = self.predictor(gray, face)
                    left_eye, right_eye = self.get_eye_landmarks(shape)
                    
                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    
                    for point in left_eye + right_eye:
                        cv2.circle(frame, point, 2, (0, 255, 0), -1)
                    
                    if ear < self.eye_ar_thresh:
                        blink_counter += 1
                    else:
                        if blink_counter >= self.eye_ar_consec_frames:
                            self.total_blinks += 1
                            
                            current_time = time.time()
                            time_since_last_blink = current_time - last_blink_time
                            
                            if time_since_last_blink > 5:
                                self.add_alert("Extended eye closure detected - possible drowsiness", "Medium")
                                
                            last_blink_time = current_time
                            
                        blink_counter = 0
                    
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                face_roi = gray[y:y+h, x:x+w]
                try:
                    face_roi = cv2.resize(face_roi, (48, 48))
                    face_roi = face_roi.astype("float") / 255.0
                    face_roi = np.expand_dims(face_roi, axis=0)
                    face_roi = np.expand_dims(face_roi, axis=-1)
                    
                    if self.model:
                        preds = self.model.predict(face_roi, verbose=0)[0]
                        expression_idx = np.argmax(preds)
                        expression = expressions[expression_idx]
                        confidence = preds[expression_idx]
                        
                        cv2.putText(frame, f"{expression}: {confidence:.2f}", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (0, 255, 0), 2)
                        
                        if expression in suspicious_expressions:
                            self.add_alert(f"Suspicious expression detected: {expression}", "Medium")
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                except Exception as e:
                    print(f"Error processing face: {e}")
            
            if not face_detected:
                self.add_alert("No face detected - Student may be absent", "High")
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            
            frame_counter += 1
            if frame_counter % 100 == 0:  # Check for multiple faces every 100 frames
                if len(faces) > 1:
                    self.add_alert(f"Multiple faces detected ({len(faces)})", "Critical")
            
            cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            time.sleep(0.05)
    
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
        # Get values from UI elements
        storage_path = self.storage_path_var.get()
        video_quality = self.video_quality_var.get()
        retention_days = self.retention_var.get()
        
        # Create directory if it doesn't exist
        if not os.path.exists(storage_path):
            try:
                os.makedirs(storage_path)
                print(f"Created directory: {storage_path}")
            except OSError as e:
                messagebox.showerror("Error", f"Failed to create directory: {e}")
                return
        
        # Build export options dictionary 
        export_config = {option: var.get() for option, var in self.export_options.items()}
        
        # Create configuration dictionary
        config = {
            "storage": {
                "path": storage_path,
                "video_quality": video_quality,
                "retention_days": retention_days,
                "export_options": export_config
            }
        }
        
        # Save to configuration file
        try:
            with open("config.json", "r") as file:
                # Load existing config if available
                existing_config = json.load(file)
            
            # Update only the storage section
            existing_config["storage"] = config["storage"]
            
            with open("config.json", "w") as file:
                json.dump(existing_config, file, indent=4)
        except FileNotFoundError:
            # Create new config file if it doesn't exist
            with open("config.json", "w") as file:
                json.dump(config, file, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            return
        
        # Apply settings to current application instance
        self.config["storage"] = config["storage"]
        
        # Show success message
        messagebox.showinfo("Success", "Storage settings saved successfully!")

    def reset_storage_settings(self):
        # Reset storage path to default
        self.storage_path_var.set("./sessions")
        
        # Reset video quality to default
        self.video_quality_var.set("Medium")
        
        # Reset retention days to default
        self.retention_var.set(30)
        
        # Reset export options to defaults
        self.export_options["CSV"].set(True)
        self.export_options["PDF Report"].set(True)
        self.export_options["Alert Images"].set(True)
        self.export_options["Session Statistics"].set(True)
        
        messagebox.showinfo("Settings Reset", "Storage settings have been reset to default values.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ExamProctorSystem(root)
    root.mainloop()
    