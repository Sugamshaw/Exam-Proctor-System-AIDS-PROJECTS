# AI Exam Proctoring System

## Overview

The AI Exam Proctoring System is a sophisticated application designed to monitor and ensure the integrity of online examinations. Leveraging advanced computer vision and machine learning techniques, the system detects potential cheating behaviors through facial recognition, gaze tracking, blink detection, head pose estimation, and environmental scanning. It provides real-time alerts, session recording, and detailed reporting for educators and administrators.

This project is built using Python and integrates libraries such as OpenCV, MediaPipe, TensorFlow, dlib, and Tkinter for a robust and user-friendly interface.

## Features

* **Facial Recognition & Student Verification:** Identifies and verifies students using facial landmarks.
* **Gaze Tracking & Blink Detection:** Monitors eye movements and blink rates to detect suspicious behavior.
* **Head Pose Estimation:** Tracks head orientation to ensure focus on the exam.
* **Environmental Scanning:** Detects suspicious objects (e.g., phones, papers) using MediaPipe Objectron and hand tracking.
* **Multi-Level Alert System:** Generates alerts with varying severity levels (Low, Medium, High, Critical).
* **Session Recording & Reporting:** Records exam sessions and exports detailed reports in CSV format.
* **User-Friendly GUI:** Built with Tkinter, providing tabs for monitoring, student enrollment, history, and settings.

## Prerequisites

Before running the application, ensure you have the following installed:

* Python 3.8+
* Required Python Libraries:

    ```bash
    pip install opencv-python numpy tkinter mediapipe tensorflow dlib pillow pandas matplotlib scipy
    ```

* dlib Shape Predictor: Download the `shape_predictor_68_face_landmarks.dat` file from [dlib.net](http://dlib.net), extract it, and place it in the project root directory.
* SQLite: The application uses SQLite for database storage (included with Python).

## Installation

1.  Clone the Repository:

    ```bash
    git clone [https://github.com/yourusername/ai-exam-proctoring-system.git](https://github.com/yourusername/ai-exam-proctoring-system.git)
    cd ai-exam-proctoring-system
    ```

2.  Install Dependencies:

    ```bash
    pip install -r Dependencies.txt
    ```

3.  Place the dlib Shape Predictor:

    * Download and extract `shape_predictor_68_face_landmarks.dat`.
    * Place it in the project root directory.

4.  Run the Application:

    ```bash
    python modal.py
    ```

## Usage

1.  Launch the Application:

    * Run `modal.py` to start the Tkinter-based GUI.

2.  Enroll Students:

    * Navigate to the "Enroll Student" tab.
    * Enter student details (name, ID) and capture their face using the webcam.
    * Save the enrollment to store face encodings in the SQLite database.

3.  Monitor Exams:

    * Go to the "Monitor" tab.
    * Select a student from the database.
    * Start monitoring to begin real-time detection of cheating behaviors.
    * Alerts will appear in the right panel, and snapshots are saved for suspicious activities.

4.  View History:

    * Access the "History" tab to review past sessions.
    * Filter by student, view session videos, alerts, or export reports.

5.  Configure Settings:

    * Use the "Settings" tab to adjust detection thresholds, alert levels, storage paths, and export options.

## Directory Structure
```bash
ai-exam-proctoring-system/
├── logs/                     # Session logs
|   └── session_&lt;timestamp>.log
├── snapshots/                # Snapshots of suspicious activities
|   └── suspicious_&lt;timestamp>.jpg
├── sessions/                 # Recorded session videos
|   └── session_&lt;timestamp>/
|       └── recording.mp4
├── exam_proctor.db           # SQLite database for students, sessions, and alerts
├── shape_predictor_68_face_landmarks.dat  # dlib facial landmark predictor
├── facial_expression_model.h5  # Pre-trained facial expression model (optional)
├── modal.py                  # Main application script
├── Dependencies.txt          # List of dependencies
└── README.md                 # This file
```
## Database Schema

The application uses an SQLite database (`exam_proctor.db`) with the following tables:

* **students:** Stores student information (ID, name, enrollment date, face encoding).
* **sessions:** Records session details (student ID, start/end time, video path, alert count, attention score).
* **alerts:** Logs alerts with timestamps, types, messages, severity, and optional image paths.

## Configuration

Settings are saved in JSON files:

* `detection_settings.json`: Stores detection thresholds (e.g., face detection confidence, gaze sensitivity).
* `alert_settings.json`: Configures alert thresholds and notification preferences.
* `config.json`: Defines storage paths, video quality, retention period, and export options.

## Notes

* **Performance:** The system processes frames every `PROCESS_EVERY_N_FRAMES` (default: 2) to balance accuracy and performance. Adjust this in `modal.py` for your hardware.
* **Gaze Limits:** Configurable gaze thresholds (`GAZE_LIMIT_X_MIN`, `GAZE_LIMIT_X_MAX`, etc.) determine acceptable eye movement ranges.
* **Object Detection:** Currently uses MediaPipe Objectron with a placeholder model ('Shoe'). Replace with a trained model for better detection.
* **Model Training:** If `facial_expression_model.h5` is missing, a placeholder model is trained with random data. Use a proper dataset for real applications.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -m "Add your feature"`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a Pull Request.

Built with ❤️ by SUGAM SHAW, ARYABRAT SAHOO, DEBASISH TRIPATHY, AMIT KUMAR NAYAK
