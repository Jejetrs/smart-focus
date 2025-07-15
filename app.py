from flask import Flask, render_template_string, request, Response, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dis
import cv2 as cv
import os
import time
import uuid
from datetime import datetime, timedelta
import json
import threading
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)

# FIXED Configuration for Railway deployment with absolute paths
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'uploads'))
app.config['DETECTED_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'detected'))
app.config['REPORTS_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'reports'))
app.config['RECORDINGS_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'recordings'))
app.config['MAX_CONTENT_PATH'] = 10000000

# FIXED - Create necessary directories with proper error handling
for folder in [app.config['UPLOAD_FOLDER'], app.config['DETECTED_FOLDER'], 
               app.config['REPORTS_FOLDER'], app.config['RECORDINGS_FOLDER']]:
    try:
        os.makedirs(folder, exist_ok=True)
        print(f"Directory created/verified: {folder}")
    except Exception as e:
        print(f"Error creating directory {folder}: {e}")

# Global variables for live monitoring
live_monitoring_active = False
session_data = {
    'start_time': None,
    'end_time': None,
    'detections': [],
    'alerts': [],
    'focus_statistics': {
        'unfocused_time': 0,
        'yawning_time': 0,
        'sleeping_time': 0,
        'total_persons': 0,
        'total_detections': 0
    },
    'recording_path': None,
    'session_id': None
}

# Person state tracking for timer display
person_state_timers = {}  
person_current_state = {}  
last_alert_time = {}  

# Distraction thresholds (in seconds)
DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,    
    'YAWNING': 3.5,      
    'NOT FOCUSED': 10  
}

def draw_landmarks(image, landmarks, land_mark, color):
    """Draw landmarks on the image for a single face"""
    height, width = image.shape[:2]
    for face in land_mark:
        point = landmarks.landmark[face]
        point_scale = ((int)(point.x * width), (int)(point.y * height))     
        cv.circle(image, point_scale, 2, color, 1)

def euclidean_distance(image, top, bottom):
    """Calculate euclidean distance between two points"""
    height, width = image.shape[0:2]
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, landmarks, top_bottom, left_right):
    """Calculate aspect ratio based on landmarks"""
    top = landmarks.landmark[top_bottom[0]]
    bottom = landmarks.landmark[top_bottom[1]]
    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmarks.landmark[left_right[0]]
    right = landmarks.landmark[left_right[1]]
    left_right_dis = euclidean_distance(image, left, right)
    
    aspect_ratio = left_right_dis / top_bottom_dis
    return aspect_ratio

def calculate_midpoint(points):
    """Calculate the midpoint of a set of points"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    midpoint = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
    return midpoint

def check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
    """Check if iris is in the middle of the eye"""
    left_eye_midpoint = calculate_midpoint(left_eye_points)
    right_eye_midpoint = calculate_midpoint(right_eye_points)
    left_iris_midpoint = calculate_midpoint(left_iris_points)
    right_iris_midpoint = calculate_midpoint(right_iris_points)
    deviation_threshold_horizontal = 2.8
    
    return (abs(left_iris_midpoint[0] - left_eye_midpoint[0]) <= deviation_threshold_horizontal 
            and abs(right_iris_midpoint[0] - right_eye_midpoint[0]) <= deviation_threshold_horizontal)

def detect_drowsiness(frame, landmarks):
    """Detect drowsiness and attention state based on eye aspect ratio and other metrics"""
    # Landmark colors
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_MAGENTA = (255, 0, 255)

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE_TOP_BOTTOM = [386, 374]
    LEFT_EYE_LEFT_RIGHT = [263, 362]
    RIGHT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_LEFT_RIGHT = [133, 33]
    UPPER_LOWER_LIPS = [13, 14]
    LEFT_RIGHT_LIPS = [78, 308]
    FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # Draw face landmarks
    draw_landmarks(frame, landmarks, FACE, COLOR_GREEN)
    draw_landmarks(frame, landmarks, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
    draw_landmarks(frame, landmarks, LEFT_EYE_LEFT_RIGHT, COLOR_RED)
    draw_landmarks(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
    draw_landmarks(frame, landmarks, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)
    draw_landmarks(frame, landmarks, UPPER_LOWER_LIPS, COLOR_BLUE)
    draw_landmarks(frame, landmarks, LEFT_RIGHT_LIPS, COLOR_BLUE)

    # Create mesh points for iris detection
    img_h, img_w = frame.shape[:2]
    mesh_points = []    
    for p in landmarks.landmark:
        x = int(p.x * img_w)
        y = int(p.y * img_h)
        mesh_points.append((x, y))
    mesh_points = np.array(mesh_points)            
    
    left_eye_points = mesh_points[LEFT_EYE]
    right_eye_points = mesh_points[RIGHT_EYE]
    left_iris_points = mesh_points[LEFT_IRIS]
    right_iris_points = mesh_points[RIGHT_IRIS]

    # Draw iris circles
    try:
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv.circle(frame, center_left, int(l_radius), COLOR_MAGENTA, 1, cv.LINE_AA)
        cv.circle(frame, center_right, int(r_radius), COLOR_MAGENTA, 1, cv.LINE_AA)
    except:
        pass  # Skip if iris points are invalid

    # Detect closed eyes
    ratio_left_eye = get_aspect_ratio(frame, landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
    ratio_right_eye = get_aspect_ratio(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
    eye_ratio = (ratio_left_eye + ratio_right_eye) / 2
    
    # Detect yawning
    ratio_lips = get_aspect_ratio(frame, landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
    
    # Check if iris is focused
    iris_focused = check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points)
    
    # Determine state based on conditions
    eyes_closed = eye_ratio > 5.0
    yawning = ratio_lips < 1.8
    not_focused = not iris_focused
    
    # State priority: SLEEPING > YAWNING > NOT FOCUSED > FOCUSED
    if eyes_closed:
        state = "SLEEPING"
    elif yawning:
        state = "YAWNING"
    elif not_focused:
        state = "NOT FOCUSED"
    else:
        state = "FOCUSED"
    
    status = {
        "eyes_closed": eyes_closed,
        "yawning": yawning,
        "not_focused": not_focused,
        "focused": iris_focused,
        "state": state
    }
    
    return status, state

def detect_persons_with_attention(image, mode="image"):
    """Detect persons in image with detailed info display"""
    try:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=(mode == "image"),
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        detection_results = detector.process(rgb_image)
        mesh_results = face_mesh.process(rgb_image)
        
        detections = []
        ih, iw, _ = image.shape
        current_time = time.time()
        
        if detection_results.detections:
            for i, detection in enumerate(detection_results.detections):
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                confidence_score = detection.score[0]
                
                attention_status = {
                    "eyes_closed": False,
                    "yawning": False,
                    "not_focused": False,
                    "state": "FOCUSED"
                }
                
                # Match face detection with face mesh
                matched_face_idx = -1
                if mesh_results.multi_face_landmarks:
                    for face_idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                        min_x, min_y = float('inf'), float('inf')
                        max_x, max_y = 0, 0
                        
                        for landmark in face_landmarks.landmark:
                            landmark_x, landmark_y = int(landmark.x * iw), int(landmark.y * ih)
                            min_x = min(min_x, landmark_x)
                            min_y = min(min_y, landmark_y)
                            max_x = max(max_x, landmark_x)
                            max_y = max(max_y, landmark_y)
                        
                        mesh_center_x = (min_x + max_x) // 2
                        mesh_center_y = (min_y + max_y) // 2
                        det_center_x = x + w // 2
                        det_center_y = y + h // 2
                        
                        if (abs(mesh_center_x - det_center_x) < w // 2 and 
                            abs(mesh_center_y - det_center_y) < h // 2):
                            matched_face_idx = face_idx
                            break
                
                # Process landmarks if matched
                if matched_face_idx != -1:
                    attention_status, state = detect_drowsiness(
                        image, 
                        mesh_results.multi_face_landmarks[matched_face_idx]
                    )
                
                status_text = attention_status.get("state", "FOCUSED")
                person_key = f"person_{i+1}"
                
                # Draw rectangle with basic info
                status_colors = {
                    "FOCUSED": (0, 255, 0),      # Green
                    "NOT FOCUSED": (0, 165, 255), # Orange
                    "YAWNING": (0, 255, 255),    # Yellow  
                    "SLEEPING": (0, 0, 255)      # Red
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Draw status text
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text = f"Person {i+1}: {status_text}"
                
                (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
                
                text_y = y - 10
                if text_y < text_height + 10:
                    text_y = y + h + text_height + 10
                
                # Background rectangle
                overlay = image.copy()
                cv.rectangle(overlay, (x, text_y - text_height - 5), (x + text_width + 10, text_y + 5), (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                # Draw text
                cv.putText(image, text, (x + 5, text_y), font, font_scale, main_color, thickness)
                
                detections.append({
                    "id": i+1,
                    "confidence": float(confidence_score),
                    "bbox": [x, y, w, h],
                    "status": status_text,
                    "timestamp": datetime.now().isoformat(),
                    "duration": 0
                })
        
        # Add detection count
        if detections:
            cv.putText(image, f"Total persons detected: {len(detections)}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv.putText(image, "No persons detected", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return image, detections
    
    except Exception as e:
        print(f"Error in detect_persons_with_attention: {e}")
        return image, []

# FIXED - Simplified routes for basic functionality

@app.route('/')
def index():
    """Main landing page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Focus Alert - Railway Deployment</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                text-align: center;
                max-width: 600px;
            }
            h1 {
                font-size: 2.5em;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            p {
                font-size: 1.2em;
                margin-bottom: 30px;
                line-height: 1.6;
            }
            .status {
                background: rgba(46, 204, 113, 0.8);
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                font-weight: bold;
            }
            .feature {
                background: rgba(52, 152, 219, 0.8);
                padding: 10px;
                margin: 10px 0;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Smart Focus Alert</h1>
            <div class="status">
                ✅ Service is running successfully on Railway!
            </div>
            <p>Advanced real-time focus monitoring system with AI-powered facial recognition and attention detection.</p>
            
            <div class="feature">📊 Real-time Face Detection</div>
            <div class="feature">👁️ Focus & Attention Monitoring</div>
            <div class="feature">😴 Drowsiness Detection</div>
            <div class="feature">📱 Live Video Processing</div>
            <div class="feature">📄 Automated PDF Reports</div>
            
            <p style="margin-top: 30px; font-size: 1em; opacity: 0.8;">
                Deploy timestamp: {{ timestamp }}<br>
                Status: <span style="color: #2ecc71;">Healthy</span>
            </p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content, timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "Smart Focus Alert",
        "version": "1.0.0"
    })

@app.route('/webcam')
def webcam():
    """Webcam monitoring page"""
    # Read the HTML content from your file and return it
    # For Railway, we'll use a simplified version
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Focus Monitoring</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: #0f172a; 
                color: white; 
                padding: 20px; 
                text-align: center;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
            }
            video { 
                width: 100%; 
                max-width: 640px; 
                border-radius: 10px; 
                margin: 20px 0;
            }
            button {
                background: linear-gradient(135deg, #60a5fa, #a78bfa);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                margin: 10px;
            }
            button:hover { transform: translateY(-2px); }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            .status {
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Live Focus Monitoring</h1>
            <div class="status">
                <h3>Camera Feed</h3>
                <video id="video" autoplay muted></video>
                <canvas id="canvas" style="display: none;"></canvas>
            </div>
            
            <button onclick="startCamera()">Start Camera</button>
            <button onclick="stopCamera()">Stop Camera</button>
            
            <div id="results"></div>
        </div>
        
        <script>
            let video, canvas, ctx, stream;
            
            async function startCamera() {
                try {
                    video = document.getElementById('video');
                    canvas = document.getElementById('canvas');
                    ctx = canvas.getContext('2d');
                    
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 640, height: 480 } 
                    });
                    
                    video.srcObject = stream;
                    document.getElementById('results').innerHTML = '<p style="color: #10b981;">✅ Camera started successfully!</p>';
                } catch (error) {
                    document.getElementById('results').innerHTML = '<p style="color: #ef4444;">❌ Camera access failed: ' + error.message + '</p>';
                }
            }
            
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                    document.getElementById('results').innerHTML = '<p style="color: #f59e0b;">⏹️ Camera stopped</p>';
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a frame from client-side camera"""
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400
            
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400
        
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        _, buffer = cv.imencode('.jpg', processed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections
        })
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring session"""
    global live_monitoring_active, session_data
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    session_data = {
        'session_id': session_id,
        'start_time': datetime.now(),
        'end_time': None,
        'detections': [],
        'alerts': [],
        'focus_statistics': {
            'unfocused_time': 0,
            'yawning_time': 0,
            'sleeping_time': 0,
            'total_persons': 0,
            'total_detections': 0
        }
    }
    
    live_monitoring_active = True
    
    return jsonify({
        "status": "success", 
        "message": "Monitoring started",
        "session_id": session_id
    })

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop monitoring session"""
    global live_monitoring_active, session_data
    
    live_monitoring_active = False
    session_data['end_time'] = datetime.now()
    
    return jsonify({
        "status": "success", 
        "message": "Monitoring stopped"
    })

@app.route('/check_camera')
def check_camera():
    """Check camera availability - always return False for Railway"""
    return jsonify({"camera_available": False})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Smart Focus Alert on port {port}")
    print(f"Health check available at: http://localhost:{port}/health")
    app.run(host='0.0.0.0', port=port, debug=False)
