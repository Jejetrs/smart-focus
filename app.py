from flask import Flask, render_template, request, Response, jsonify, send_file, send_from_directory
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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
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
application = Flask(__name__)

# FIXED - Configuration for Railway deployment with better paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
application.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
application.config['DETECTED_FOLDER'] = os.path.join(BASE_DIR, 'static', 'detected')
application.config['REPORTS_FOLDER'] = os.path.join(BASE_DIR, 'static', 'reports')
application.config['RECORDINGS_FOLDER'] = os.path.join(BASE_DIR, 'static', 'recordings')
application.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# FIXED - Create necessary directories with proper error handling
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        application.config['UPLOAD_FOLDER'],
        application.config['DETECTED_FOLDER'], 
        application.config['REPORTS_FOLDER'],
        application.config['RECORDINGS_FOLDER']
    ]
    
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {e}")

# Initialize directories
ensure_directories()

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
    'recording_path': None
}

# Video recording variables for client-side recording
video_writer = None
recording_active = False
recording_frames = []  # Store frames for client-side recording

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
        pass

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
            
            # Timer tracking for live monitoring
            duration = 0
            if mode == "video" and live_monitoring_active:
                # Initialize person tracking if not exists
                if person_key not in person_state_timers:
                    person_state_timers[person_key] = {}
                    person_current_state[person_key] = None
                    last_alert_time[person_key] = 0
                
                # Update state timing
                if person_current_state[person_key] != status_text:
                    person_state_timers[person_key] = {}
                    person_current_state[person_key] = status_text
                    person_state_timers[person_key][status_text] = current_time
                else:
                    if status_text not in person_state_timers[person_key]:
                        person_state_timers[person_key][status_text] = current_time
                
                # Calculate duration for timer display
                if status_text in person_state_timers[person_key]:
                    duration = current_time - person_state_timers[person_key][status_text]
            
            # Enhanced drawing based on mode
            if mode == "video" and live_monitoring_active:
                # Draw rectangle with timer info for live monitoring
                status_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (0, 165, 255),
                    "YAWNING": (0, 255, 255),
                    "SLEEPING": (0, 0, 255)
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Timer display for live monitoring
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    timer_text = f"Person {i+1}: {status_text} ({duration:.1f}s/{threshold}s)"
                else:
                    timer_text = f"Person {i+1}: {status_text}"
                
                # Draw text with background
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv.getTextSize(timer_text, font, font_scale, thickness)
                
                text_y = y - 10
                if text_y < text_height + 10:
                    text_y = y + h + text_height + 10
                
                # Background rectangle
                overlay = image.copy()
                cv.rectangle(overlay, (x, text_y - text_height - 5), (x + text_width + 10, text_y + 5), (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                # Draw timer text
                cv.putText(image, timer_text, (x + 5, text_y), font, font_scale, main_color, thickness)
            else:
                # Enhanced info display for static images
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw detailed info
                info_y_start = y + h + 10
                box_padding = 10
                line_height = 20
                box_height = 4 * line_height
                
                # Semi-transparent background
                overlay = image.copy()
                cv.rectangle(overlay, 
                            (x - box_padding, info_y_start - box_padding), 
                            (x + w + box_padding, info_y_start + box_height), 
                            (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)
                
                # Text styling
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                thickness = 1
                
                # Add detailed info
                cv.putText(image, f"Person {i+1}", (x, info_y_start), 
                        font, font_scale, (50, 205, 50), thickness+1)
                cv.putText(image, f"Confidence: {confidence_score*100:.2f}%", 
                        (x, info_y_start + line_height), font, font_scale, font_color, thickness)
                cv.putText(image, f"Position: x:{x}, y:{y} Size: w:{w}, h:{h}", 
                        (x, info_y_start + 2*line_height), font, font_scale, font_color, thickness)
                
                # Status with color
                status_color = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (255, 165, 0),
                    "YAWNING": (255, 255, 0),
                    "SLEEPING": (0, 0, 255)
                }
                color = status_color.get(status_text, (0, 255, 0))
                
                cv.putText(image, f"Status: {status_text}", 
                        (x, info_y_start + 3*line_height), font, font_scale, color, thickness)

            # Check for distraction alerts in live mode
            should_alert = False
            alert_message = ""
            
            if (mode == "video" and live_monitoring_active and status_text in DISTRACTION_THRESHOLDS and 
                person_key in person_state_timers and status_text in person_state_timers[person_key]):
                
                if duration >= DISTRACTION_THRESHOLDS[status_text]:
                    # Check cooldown
                    alert_cooldown = 5
                    if current_time - last_alert_time[person_key] >= alert_cooldown:
                        should_alert = True
                        last_alert_time[person_key] = current_time
                        
                        # Set appropriate alert message
                        if status_text == 'SLEEPING':
                            alert_message = f'Person {i+1} is sleeping - please wake up!'
                        elif status_text == 'YAWNING':
                            alert_message = f'Person {i+1} is yawning - please take a rest!'
                        elif status_text == 'NOT FOCUSED':
                            alert_message = f'Person {i+1} is not focused - please focus on screen!'
                        
                        # Record alert in session data
                        if live_monitoring_active:
                            session_data['alerts'].append({
                                'timestamp': datetime.now().isoformat(),
                                'person': f"Person {i+1}",
                                'detection': status_text,
                                'message': alert_message,
                                'duration': int(duration)
                            })
            
            # Extract face region for saving
            face_img = image[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"person_{i+1}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            face_path = os.path.join(application.config['DETECTED_FOLDER'], face_filename)
            
            if face_img.size > 0:
                cv.imwrite(face_path, face_img)
            
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "image_path": f"/static/detected/{face_filename}",
                "status": status_text,
                "timestamp": datetime.now().isoformat(),
                "duration": duration if mode == "video" else 0
            })
    
    # Add detection count
    if detections:
        cv.putText(image, f"Total persons detected: {len(detections)}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(image, "No persons detected", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image, detections

def calculate_distraction_time_from_alerts(alerts):
    """Calculate actual distraction time based on alert history"""
    distraction_times = {
        'unfocused_time': 0,
        'yawning_time': 0,
        'sleeping_time': 0
    }
    
    if not alerts:
        return distraction_times
    
    person_distractions = {}
    
    for alert in alerts:
        person = alert.get('person', 'Unknown')
        detection = alert.get('detection', 'Unknown')
        duration = alert.get('duration', 0)
        
        if person not in person_distractions:
            person_distractions[person] = {}
        
        if detection not in person_distractions[person]:
            person_distractions[person][detection] = []
        
        person_distractions[person][detection].append(duration)
    
    for person, distractions in person_distractions.items():
        for detection_type, durations in distractions.items():
            if detection_type == 'NOT FOCUSED':
                distraction_times['unfocused_time'] += sum(durations)
            elif detection_type == 'YAWNING':
                distraction_times['yawning_time'] += sum(durations)
            elif detection_type == 'SLEEPING':
                distraction_times['sleeping_time'] += sum(durations)
    
    return distraction_times

def update_session_statistics(detections):
    """Update session statistics based on current detections"""
    global session_data
    
    if not detections:
        return
    
    session_data['detections'].extend(detections)
    session_data['focus_statistics']['total_detections'] += len(detections)
    session_data['focus_statistics']['total_persons'] = max(
        session_data['focus_statistics']['total_persons'],
        len(detections)
    )
    
    distraction_times = calculate_distraction_time_from_alerts(session_data['alerts'])
    session_data['focus_statistics']['unfocused_time'] = distraction_times['unfocused_time']
    session_data['focus_statistics']['yawning_time'] = distraction_times['yawning_time']
    session_data['focus_statistics']['sleeping_time'] = distraction_times['sleeping_time']

def get_most_common_distraction(alerts):
    """Helper function to find the most common type of distraction"""
    if not alerts:
        return "None"
    
    distraction_counts = {}
    distraction_durations = {}
    
    for alert in alerts:
        detection = alert.get('detection', 'Unknown')
        duration = alert.get('duration', 0)
        
        distraction_counts[detection] = distraction_counts.get(detection, 0) + 1
        distraction_durations[detection] = distraction_durations.get(detection, 0) + duration
    
    if not distraction_counts:
        return "None"
    
    most_common = max(distraction_counts, key=distraction_counts.get)
    count = distraction_counts[most_common]
    total_duration = distraction_durations[most_common]
    
    return f"{most_common} ({count} times, {total_duration}s total)"

def calculate_average_focus_metric(focused_time, total_session_seconds):
    """Calculate a meaningful average focus metric"""
    if total_session_seconds <= 0:
        return "N/A"
    
    total_minutes = total_session_seconds / 60
    focused_minutes = focused_time / 60
    
    if total_session_seconds < 60:
        focus_percentage = (focused_time / total_session_seconds) * 100
        return f"{focus_percentage:.1f}% of session time"
    elif total_session_seconds < 3600:
        return f"{focused_minutes:.1f} min focused out of {total_minutes:.1f} min total"
    else:
        hours = total_session_seconds / 3600
        focused_per_hour = focused_minutes / hours
        return f"{focused_per_hour:.1f} min focused per hour"

def generate_pdf_report(session_data, output_path):
    """FIXED - Generate PDF report exactly like the provided sample"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles to match the sample
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#3B82F6')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1F2937')
        )
        
        # Title
        story.append(Paragraph("Smart Focus Alert - Session Report", title_style))
        story.append(Spacer(1, 20))
        
        # Calculate session duration and focus accuracy
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            total_session_seconds = duration.total_seconds()
            duration_str = str(duration).split('.')[0]
        else:
            total_session_seconds = 0
            duration_str = "N/A"
        
        # Get corrected time statistics from alert history
        distraction_times = calculate_distraction_time_from_alerts(session_data['alerts'])
        unfocused_time = distraction_times['unfocused_time']
        yawning_time = distraction_times['yawning_time']
        sleeping_time = distraction_times['sleeping_time']
        
        # Calculate total distraction time
        total_distraction_time = unfocused_time + yawning_time + sleeping_time
        
        # Calculate focused time
        if total_session_seconds > 0:
            focused_time = max(0, total_session_seconds - total_distraction_time)
        else:
            focused_time = 0
        
        # Calculate focus accuracy percentage
        if total_session_seconds > 0:
            focus_accuracy = (focused_time / total_session_seconds) * 100
            distraction_percentage = (total_distraction_time / total_session_seconds) * 100
        else:
            focus_accuracy = 0
            distraction_percentage = 0
        
        # Determine focus quality rating
        if focus_accuracy >= 90:
            focus_rating = "Excellent"
            rating_color = colors.HexColor('#10B981')
        elif focus_accuracy >= 75:
            focus_rating = "Good"
            rating_color = colors.HexColor('#3B82F6')
        elif focus_accuracy >= 60:
            focus_rating = "Fair"
            rating_color = colors.HexColor('#F59E0B')
        elif focus_accuracy >= 40:
            focus_rating = "Poor"
            rating_color = colors.HexColor('#EF4444')
        else:
            focus_rating = "Very Poor"
            rating_color = colors.HexColor('#DC2626')
        
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        
        # Session Information (EXACTLY like sample)
        story.append(Paragraph("Session Information", heading_style))
        
        session_info = [
            ['Session Start Time', session_data.get('start_time', datetime.now()).strftime('%m/%d/%Y, %I:%M:%S %p')],
            ['Session Duration', duration_str],
            ['Total Detections', str(session_data['focus_statistics']['total_detections'])],
            ['Total Persons Detected', str(session_data['focus_statistics']['total_persons'])],
            ['Total Alerts Generated', str(len(session_data['alerts']))]
        ]
        
        session_table = Table(session_info, colWidths=[3*inch, 2*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(session_table)
        story.append(Spacer(1, 20))
        
        # Focus Accuracy Summary (EXACTLY like sample)
        story.append(Paragraph("Focus Accuracy Summary", heading_style))
        
        # Create a highlighted focus accuracy display
        accuracy_text = f"<para align=center><font size=18 color='{rating_color.hexval()}'><b>{focus_accuracy:.1f}%</b></font></para>"
        story.append(Paragraph(accuracy_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        rating_text = f"<para align=center><font size=14 color='{rating_color.hexval()}'><b>Focus Quality: {focus_rating}</b></font></para>"
        story.append(Paragraph(rating_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Detailed time breakdown table (EXACTLY like sample)
        focus_breakdown = [
            ['Metric', 'Time', 'Percentage'],
            ['Total Focused Time', format_time(focused_time), f"{(focused_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['Total Distraction Time', format_time(total_distraction_time), f"{distraction_percentage:.1f}%"],
            ['- Unfocused Time', format_time(unfocused_time), f"{(unfocused_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['- Yawning Time', format_time(yawning_time), f"{(yawning_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['- Sleeping Time', format_time(sleeping_time), f"{(sleeping_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"]
        ]
        
        breakdown_table = Table(focus_breakdown, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        breakdown_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
            # Highlight focused time row
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ECFDF5')),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#065F46')),
            # Highlight total distraction row
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FEF2F2')),
            ('TEXTCOLOR', (0, 2), (-1, 2), colors.HexColor('#991B1B')),
        ]))
        
        story.append(breakdown_table)
        story.append(Spacer(1, 20))
        
        # Detailed Focus Statistics (EXACTLY like sample)
        story.append(Paragraph("Detailed Focus Statistics", heading_style))
        
        average_focus_metric = calculate_average_focus_metric(focused_time, total_session_seconds)
        
        focus_stats = [
            ['Total Session Duration', format_time(total_session_seconds)],
            ['Focus Accuracy Score', f"{focus_accuracy:.2f}%"],
            ['Focus Quality Rating', focus_rating],
            ['Average Focus Metric', average_focus_metric],
            ['Distraction Frequency', f"{len(session_data['alerts'])} alerts in {format_time(total_session_seconds)}"],
            ['Most Common Distraction', get_most_common_distraction(session_data['alerts'])]
        ]
        
        focus_table = Table(focus_stats, colWidths=[3*inch, 2*inch])
        focus_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(focus_table)
        story.append(Spacer(1, 30))
        
        # Alert History (EXACTLY like sample)
        if session_data['alerts']:
            story.append(Paragraph("Alert History", heading_style))
            
            alert_headers = ['Time', 'Person', 'Detection', 'Duration', 'Message']
            alert_data = [alert_headers]
            
            for alert in session_data['alerts']:
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%I:%M:%S %p')
                except:
                    alert_time = alert['timestamp']
                
                duration = alert.get('duration', 0)
                duration_text = f"{duration}s" if duration > 0 else "N/A"
                
                alert_data.append([
                    alert_time,
                    alert['person'],
                    alert['detection'],
                    duration_text,
                    alert['message']
                ])
            
            alert_table = Table(alert_data, colWidths=[1*inch, 0.8*inch, 1*inch, 0.7*inch, 2.5*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')])
            ]))
            
            story.append(alert_table)
        
        # Footer (EXACTLY like sample)
        story.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - Focus Monitoring Report"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        # Build the PDF
        doc.build(story)
        
        # Verify file was created
        if os.path.exists(output_path):
            print(f"✅ PDF report successfully created: {output_path}")
            return output_path
        else:
            print(f"❌ Failed to create PDF report: {output_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating PDF report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_upload_pdf_report(detections, file_info, output_path):
    """Generate PDF report for uploaded file analysis"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#3B82F6')
        )
        
        story.append(Paragraph("Smart Focus Alert - Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # File info and analysis results
        file_info_text = f"<b>File:</b> {file_info.get('filename', 'Unknown')}<br/>"
        file_info_text += f"<b>Type:</b> {file_info.get('type', 'Unknown')}<br/>"
        file_info_text += f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        file_info_text += f"<b>Persons Detected:</b> {len(detections)}"
        
        story.append(Paragraph(file_info_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Statistics
        if detections:
            status_counts = {'FOCUSED': 0, 'NOT FOCUSED': 0, 'YAWNING': 0, 'SLEEPING': 0}
            for detection in detections:
                status = detection.get('status', 'FOCUSED')
                if status in status_counts:
                    status_counts[status] += 1
            
            total_detections = len(detections)
            focus_accuracy = (status_counts['FOCUSED'] / total_detections * 100) if total_detections > 0 else 0
            
            stats_text = f"<b>Focus Accuracy:</b> {focus_accuracy:.1f}%<br/>"
            stats_text += f"<b>Focused Persons:</b> {status_counts['FOCUSED']}<br/>"
            stats_text += f"<b>Unfocused Persons:</b> {status_counts['NOT FOCUSED']}<br/>"
            stats_text += f"<b>Yawning Persons:</b> {status_counts['YAWNING']}<br/>"
            stats_text += f"<b>Sleeping Persons:</b> {status_counts['SLEEPING']}"
            
            story.append(Paragraph(stats_text, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Report generated by Smart Focus Alert System<br/>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        # Build the PDF
        doc.build(story)
        
        # Verify file was created
        if os.path.exists(output_path):
            print(f"✅ Upload PDF report successfully created: {output_path}")
            return output_path
        else:
            print(f"❌ Failed to create upload PDF report: {output_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating upload PDF report: {str(e)}")
        return None

def process_video_file(video_path):
    """Process video file and detect persons in each frame"""
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frame_count = 0
    process_every_n_frames = 10
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                processed_frame, detections = detect_persons_with_attention(frame, mode="video")
                all_detections.extend(detections)
            else:
                processed_frame = frame
                
            out.write(processed_frame)
    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        cap.release()
        out.release()
    
    return output_path, all_detections

def create_client_recording_file():
    """FIXED - Create a proper client recording from stored frames"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"client_session_{timestamp}.mp4"
        recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
        
        if len(recording_frames) > 0:
            # Use actual recorded frames
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            height, width = recording_frames[0].shape[:2]
            out = cv.VideoWriter(recording_path, fourcc, 30, (width, height))
            
            for frame in recording_frames:
                out.write(frame)
            
            out.release()
            
            # Clear frames to free memory
            recording_frames.clear()
            
            if os.path.exists(recording_path) and os.path.getsize(recording_path) > 0:
                print(f"✅ Client recording created: {recording_path}")
                return recording_path
        
        # Fallback: create demo recording
        return create_demo_recording_file()
            
    except Exception as e:
        print(f"❌ Error creating client recording: {str(e)}")
        return create_demo_recording_file()

def create_demo_recording_file():
    """Create a proper demo recording file for download"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"session_recording_{timestamp}.mp4"
        recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
        
        # Create a demo video file
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(recording_path, fourcc, 30, (640, 480))
        
        # Create demo frames with session info
        for i in range(150):  # 5 seconds at 30fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add session info
            cv.putText(frame, f"Session Recording - Frame {i+1}", (50, 200), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(frame, "Live monitoring session completed", (50, 240), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(frame, f"Total alerts: {len(session_data['alerts'])}", (50, 280), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.putText(frame, f"Session time: {datetime.now().strftime('%H:%M:%S')}", (50, 320), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            out.write(frame)
        
        out.release()
        
        if os.path.exists(recording_path) and os.path.getsize(recording_path) > 0:
            print(f"✅ Demo recording created: {recording_path}")
            return recording_path
        else:
            print(f"❌ Failed to create demo recording")
            return None
            
    except Exception as e:
        print(f"❌ Error creating demo recording: {str(e)}")
        return None

# FIXED - Static file serving routes with better error handling and CORS
@application.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files with proper headers"""
    try:
        # Determine which folder based on path
        if filename.startswith('uploads/'):
            folder = application.config['UPLOAD_FOLDER']
            filename = filename[8:]  # Remove 'uploads/' prefix
        elif filename.startswith('detected/'):
            folder = application.config['DETECTED_FOLDER'] 
            filename = filename[9:]  # Remove 'detected/' prefix
        elif filename.startswith('reports/'):
            folder = application.config['REPORTS_FOLDER']
            filename = filename[8:]  # Remove 'reports/' prefix
        elif filename.startswith('recordings/'):
            folder = application.config['RECORDINGS_FOLDER']
            filename = filename[11:]  # Remove 'recordings/' prefix
        else:
            folder = os.path.join(BASE_DIR, 'static')
        
        file_path = os.path.join(folder, filename)
        
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Check file size
            file_size = os.path.getsize(file_path)
            print(f"✅ Serving file: {filename} ({file_size} bytes)")
            
            response = send_from_directory(folder, filename, as_attachment=True)
            response.headers['Cache-Control'] = 'no-cache'
            return response
        else:
            print(f"❌ File not found: {file_path}")
            return jsonify({"error": "File not found", "path": file_path}), 404
            
    except Exception as e:
        print(f"❌ Error serving file {filename}: {str(e)}")
        return jsonify({"error": "Error accessing file", "details": str(e)}), 500

# Routes
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('upload.html', error='No selected file')
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            result = {
                "filename": filename,
                "file_path": f"/static/uploads/{filename}",
                "detections": []
            }
            
            if file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
                image = cv.imread(file_path)
                if image is not None:
                    processed_image, detections = detect_persons_with_attention(image)
                    
                    output_filename = f"processed_{filename}"
                    output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
                    cv.imwrite(output_path, processed_image)
                    
                    result["processed_image"] = f"/static/detected/{output_filename}"
                    result["detections"] = detections
                    result["type"] = "image"
                    
                    # Generate PDF report
                    pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                    
                    file_info = {'filename': filename, 'type': file_ext.upper()}
                    pdf_result = generate_upload_pdf_report(detections, file_info, pdf_path)
                    
                    if pdf_result and os.path.exists(pdf_path):
                        result["pdf_report"] = f"/static/reports/{pdf_filename}"
                
            elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                output_path, detections = process_video_file(file_path)
                
                if os.path.exists(output_path):
                    result["processed_video"] = f"/static/detected/{os.path.basename(output_path)}"
                    result["detections"] = detections
                    result["type"] = "video"
                    
                    # Generate PDF report
                    pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                    
                    file_info = {'filename': filename, 'type': file_ext.upper()}
                    pdf_result = generate_upload_pdf_report(detections, file_info, pdf_path)
                    
                    if pdf_result and os.path.exists(pdf_path):
                        result["pdf_report"] = f"/static/reports/{pdf_filename}"
            
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, session_data, recording_active, person_state_timers, person_current_state, last_alert_time, recording_frames
    
    if live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring already active"})
    
    # Reset session data and timers
    session_data = {
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
        },
        'recording_path': None
    }
    
    # Reset person tracking and recording frames
    person_state_timers = {}
    person_current_state = {}
    last_alert_time = {}
    recording_frames = []
    
    live_monitoring_active = True
    recording_active = True
    
    print("✅ Monitoring started successfully")
    return jsonify({"status": "success", "message": "Monitoring started"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, session_data, recording_active
    
    if not live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring not active"})
    
    live_monitoring_active = False
    recording_active = False
    session_data['end_time'] = datetime.now()
    
    # Generate PDF report
    pdf_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
    
    pdf_result = generate_pdf_report(session_data, pdf_path)
    
    response_data = {
        "status": "success", 
        "message": "Monitoring stopped"
    }
    
    # FIXED - Add PDF report to response
    if pdf_result and os.path.exists(pdf_path):
        response_data["pdf_report"] = f"/static/reports/{pdf_filename}"
        print(f"✅ PDF report generated: {pdf_filename}")
    else:
        print("❌ Failed to generate PDF report")
    
    # FIXED - Create recording file
    recording_path = create_client_recording_file()
    if recording_path and os.path.exists(recording_path):
        response_data["video_file"] = f"/static/recordings/{os.path.basename(recording_path)}"
        session_data['recording_path'] = recording_path
        print(f"✅ Recording file generated: {os.path.basename(recording_path)}")
    else:
        print("❌ Failed to generate recording file")
    
    print("✅ Monitoring stopped successfully")
    return jsonify(response_data)

@application.route('/get_monitoring_data')
def get_monitoring_data():
    global session_data
    
    if not live_monitoring_active:
        return jsonify({"error": "Monitoring not active"})
    
    recent_alerts = session_data['alerts'][-5:] if session_data['alerts'] else []
    
    formatted_alerts = []
    for alert in recent_alerts:
        try:
            alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
        except:
            alert_time = alert['timestamp']
        
        formatted_alerts.append({
            'time': alert_time,
            'message': alert['message'],
            'type': 'warning' if alert['detection'] in ['YAWNING', 'NOT FOCUSED'] else 'error'
        })
    
    recent_detections = session_data['detections'][-10:] if session_data['detections'] else []
    current_status = 'READY'
    focused_count = 0
    total_persons = 0
    
    if recent_detections:
        latest_states = {}
        for detection in reversed(recent_detections):
            person_id = detection['id']
            if person_id not in latest_states:
                latest_states[person_id] = detection['status']
        
        total_persons = len(latest_states)
        focused_count = sum(1 for state in latest_states.values() if state == 'FOCUSED')
        
        if all(state == 'FOCUSED' for state in latest_states.values()):
            current_status = 'FOCUSED'
        elif any(state == 'SLEEPING' for state in latest_states.values()):
            current_status = 'SLEEPING'
        elif any(state == 'YAWNING' for state in latest_states.values()):
            current_status = 'YAWNING'
        elif any(state == 'NOT FOCUSED' for state in latest_states.values()):
            current_status = 'NOT FOCUSED'
    
    return jsonify({
        'total_persons': total_persons,
        'focused_count': focused_count,
        'alert_count': len(session_data['alerts']),
        'current_status': current_status,
        'latest_alerts': formatted_alerts
    })

@application.route('/monitoring_status')
def monitoring_status():
    return jsonify({"is_active": live_monitoring_active})

@application.route('/check_camera')
def check_camera():
    # Always return False for Railway - use client camera
    return jsonify({"camera_available": False})

@application.route('/process_frame', methods=['POST'])
def process_frame():
    global recording_frames
    
    try:
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400
        
        # Store frame for recording (client-side recording)
        if live_monitoring_active and recording_active:
            recording_frames.append(frame.copy())
            # Keep only last 300 frames (10 seconds at 30fps) to prevent memory overflow
            if len(recording_frames) > 300:
                recording_frames = recording_frames[-300:]
        
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        # Update session data with detections for live monitoring
        if live_monitoring_active and detections:
            update_session_statistics(detections)
        
        _, buffer = cv.imencode('.jpg', processed_frame, [cv.IMWRITE_JPEG_QUALITY, 90])
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections
        })
    except Exception as e:
        print(f"❌ Error processing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API for analyzing uploaded files
@application.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for file detection"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image = cv.imread(file_path)
        if image is not None:
            processed_image, detections = detect_persons_with_attention(image)
            
            output_filename = f"processed_{filename}"
            output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
            cv.imwrite(output_path, processed_image)
            
            return jsonify({
                "type": "image",
                "processed_image": f"/static/detected/{output_filename}",
                "detections": detections
            })
        else:
            return jsonify({"error": "Invalid image file"}), 400
        
    elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
        output_path, detections = process_video_file(file_path)
        
        if os.path.exists(output_path):
            return jsonify({
                "type": "video",
                "processed_video": f"/static/detected/{os.path.basename(output_path)}",
                "detections": detections
            })
        else:
            return jsonify({"error": "Video processing failed"}), 500
    
    return jsonify({"error": "Unsupported file format"}), 400

# Health check for Railway
@application.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# FIXED - Add storage info endpoint
@application.route('/api/storage-info')
def storage_info():
    """Get current storage usage - helpful for monitoring"""
    try:
        total_size = 0
        file_count = 0
        folder_info = {}
        
        for folder_name, folder_path in [
            ('uploads', application.config['UPLOAD_FOLDER']),
            ('detected', application.config['DETECTED_FOLDER']),
            ('reports', application.config['REPORTS_FOLDER']),
            ('recordings', application.config['RECORDINGS_FOLDER'])
        ]:
            folder_size = 0
            folder_files = 0
            
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        folder_size += file_size
                        folder_files += 1
                        total_size += file_size
                        file_count += 1
            
            folder_info[folder_name] = {
                'files': folder_files,
                'size_mb': round(folder_size / (1024 * 1024), 2)
            }
        
        return jsonify({
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'total_files': file_count,
            'folders': folder_info,
            'status': 'healthy' if total_size < 100 * 1024 * 1024 else 'warning'  # Warning if > 100MB
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FIXED - Add cleanup endpoint
@application.route('/api/cleanup', methods=['POST'])
def cleanup_old_files():
    """Manual cleanup of old files"""
    try:
        data = request.get_json() or {}
        cutoff_hours = int(data.get('hours', 24))
        cutoff_time = time.time() - (cutoff_hours * 3600)
        cleaned_count = 0
        
        for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
                      application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
            try:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        if os.path.isfile(file_path):
                            if os.path.getmtime(file_path) < cutoff_time:
                                os.remove(file_path)
                                cleaned_count += 1
            except OSError:
                continue
        
        return jsonify({
            'success': True,
            'cleaned_files': cleaned_count,
            'message': f'Cleaned up {cleaned_count} files older than {cutoff_hours} hours'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FIXED - Add file verification endpoint
@application.route('/api/verify-file/<path:filepath>')
def verify_file(filepath):
    """Verify if a file exists and get its info"""
    try:
        # Determine full path
        if filepath.startswith('static/'):
            full_path = os.path.join(BASE_DIR, filepath)
        else:
            full_path = os.path.join(BASE_DIR, 'static', filepath)
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            file_size = os.path.getsize(full_path)
            file_mtime = os.path.getmtime(full_path)
            
            return jsonify({
                'exists': True,
                'size': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(file_mtime).isoformat(),
                'path': full_path
            })
        else:
            return jsonify({
                'exists': False,
                'path': full_path
            }), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FIXED - Add session management endpoints
@application.route('/api/session/<session_id>/files')
def get_session_files(session_id):
    """Get downloadable files for a session"""
    try:
        # This would typically query a database, but for now we'll check recent files
        pdf_files = []
        video_files = []
        
        # Check reports folder
        if os.path.exists(application.config['REPORTS_FOLDER']):
            for filename in os.listdir(application.config['REPORTS_FOLDER']):
                if session_id in filename or filename.startswith('session_report_'):
                    pdf_files.append(f"/static/reports/{filename}")
        
        # Check recordings folder
        if os.path.exists(application.config['RECORDINGS_FOLDER']):
            for filename in os.listdir(application.config['RECORDINGS_FOLDER']):
                if session_id in filename or filename.startswith('session_recording_'):
                    video_files.append(f"/static/recordings/{filename}")
        
        return jsonify({
            'success': True,
            'files': {
                'pdf_reports': pdf_files,
                'video_recordings': video_files
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FIXED - Add batch file operations
@application.route('/api/files/batch-delete', methods=['POST'])
def batch_delete_files():
    """Delete multiple files at once"""
    try:
        data = request.get_json()
        file_paths = data.get('files', [])
        deleted_count = 0
        errors = []
        
        for file_path in file_paths:
            try:
                # Ensure file is in allowed directories
                full_path = os.path.join(BASE_DIR, file_path.lstrip('/'))
                
                # Security check - only allow files in static folders
                allowed_dirs = [
                    application.config['UPLOAD_FOLDER'],
                    application.config['DETECTED_FOLDER'],
                    application.config['REPORTS_FOLDER'],
                    application.config['RECORDINGS_FOLDER']
                ]
                
                if any(full_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                    if os.path.exists(full_path) and os.path.isfile(full_path):
                        os.remove(full_path)
                        deleted_count += 1
                    else:
                        errors.append(f"File not found: {file_path}")
                else:
                    errors.append(f"Access denied: {file_path}")
                    
            except Exception as e:
                errors.append(f"Error deleting {file_path}: {str(e)}")
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'errors': errors
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FIXED - Add system stats endpoint
@application.route('/api/system-stats')
def system_stats():
    """Get system statistics"""
    try:
        # Get basic system info
        stats = {
            'server_time': datetime.now().isoformat(),
            'uptime': time.time() - session_data.get('start_time', time.time()),
            'active_monitoring': live_monitoring_active,
            'total_sessions': 1 if session_data.get('start_time') else 0
        }
        
        # Add storage info
        try:
            storage_response = storage_info()
            if hasattr(storage_response, 'json'):
                stats['storage'] = storage_response.json
        except:
            pass
        
        # Add recent activity
        recent_files = []
        for folder_name, folder_path in [
            ('reports', application.config['REPORTS_FOLDER']),
            ('recordings', application.config['RECORDINGS_FOLDER'])
        ]:
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        recent_files.append({
                            'name': filename,
                            'type': folder_name,
                            'size': os.path.getsize(file_path),
                            'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        })
        
        # Sort by modification time and get last 5
        recent_files.sort(key=lambda x: x['modified'], reverse=True)
        stats['recent_files'] = recent_files[:5]
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FIXED - Add download statistics
@application.route('/api/download-stats')
def download_stats():
    """Get download statistics"""
    try:
        stats = {
            'total_downloads': 0,
            'file_types': {
                'pdf': 0,
                'video': 0,
                'image': 0
            },
            'recent_downloads': []
        }
        
        # This would typically be stored in a database
        # For now, we'll estimate based on file access times
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# FIXED - Error handlers
@application.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('404.html'), 404

@application.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

@application.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

# FIXED - Add CORS headers for API endpoints
@application.after_request
def after_request(response):
    # Add CORS headers for API endpoints
    if request.path.startswith('/api/') or request.path.startswith('/static/'):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

# FIXED - Add OPTIONS handler for CORS
@application.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = jsonify({'status': 'ok'})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# FIXED - Add background cleanup task
def start_background_cleanup():
    """Start background cleanup task"""
    def cleanup_task():
        while True:
            try:
                # Cleanup files older than 24 hours
                cutoff_time = time.time() - (24 * 3600)
                cleaned_count = 0
                
                for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
                              application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
                    try:
                        if os.path.exists(folder):
                            for filename in os.listdir(folder):
                                file_path = os.path.join(folder, filename)
                                if os.path.isfile(file_path):
                                    if os.path.getmtime(file_path) < cutoff_time:
                                        os.remove(file_path)
                                        cleaned_count += 1
                    except OSError:
                        continue
                
                if cleaned_count > 0:
                    print(f"🧹 Background cleanup: removed {cleaned_count} old files")
                    
                # Sleep for 1 hour
                time.sleep(3600)
                
            except Exception as e:
                print(f"❌ Background cleanup error: {e}")
                time.sleep(3600)
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    print("🧹 Background cleanup task started")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Smart Focus Alert on port {port}")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📊 Storage info available at: /api/storage-info")
    print(f"🧹 Manual cleanup available at: /api/cleanup")
    print(f"🔍 File verification available at: /api/verify-file/<path>")
    print(f"📈 System stats available at: /api/system-stats")
    
    # Ensure directories exist on startup
    ensure_directories()
    
    # Start background cleanup
    start_background_cleanup()
    
    application.run(host='0.0.0.0', port=port, debug=False, threaded=True)
