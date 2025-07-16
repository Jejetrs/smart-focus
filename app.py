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
import tempfile
import shutil

# Initialize Flask app
application = Flask(__name__)

# FIXED - Configuration for Railway deployment with proper directory handling
application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['MAX_CONTENT_PATH'] = 10000000

# FIXED - Create necessary directories with proper permissions
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        os.chmod(folder, 0o755)

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
    'recording_frames': []  # Store frames for video creation
}

# Video recording variables
video_writer = None
recording_active = False
recording_frames = []  # Store frames for client-side recording

# Person state tracking for timer display
person_state_timers = {}  # Track state duration for each person
person_current_state = {}  # Track current state for each person
last_alert_time = {}  # Track last alert time for cooldown

# Distraction thresholds (in seconds)
DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,    # 10 seconds
    'YAWNING': 3.5,      # 3.5 seconds  
    'NOT FOCUSED': 10  # 10 seconds
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
            
            # TIMER TRACKING for live monitoring only
            duration = 0
            if mode == "video" and live_monitoring_active:
                # Initialize person tracking if not exists
                if person_key not in person_state_timers:
                    person_state_timers[person_key] = {}
                    person_current_state[person_key] = None
                    last_alert_time[person_key] = 0
                
                # Update state timing
                if person_current_state[person_key] != status_text:
                    # State changed, reset all timers and set new state
                    person_state_timers[person_key] = {}
                    person_current_state[person_key] = status_text
                    person_state_timers[person_key][status_text] = current_time
                else:
                    # Same state continues, update timer if not exists
                    if status_text not in person_state_timers[person_key]:
                        person_state_timers[person_key][status_text] = current_time
                
                # Calculate duration for timer display
                if status_text in person_state_timers[person_key]:
                    duration = current_time - person_state_timers[person_key][status_text]
            
            # ENHANCED DRAWING BASED ON REFERENCE CODE
            if mode == "video" and live_monitoring_active:
                # Draw rectangle with timer info for live monitoring
                status_colors = {
                    "FOCUSED": (0, 255, 0),      # Green
                    "NOT FOCUSED": (0, 165, 255), # Orange
                    "YAWNING": (0, 255, 255),    # Yellow  
                    "SLEEPING": (0, 0, 255)      # Red
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Timer display for live monitoring
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    timer_text = f"Person {i+1}: {status_text} ({duration:.1f}s/{threshold}s)"
                else:
                    timer_text = f"Person {i+1}: {status_text}"
                
                # Draw text background and timer
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
                    alert_cooldown = 5  # 5 seconds cooldown
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
    """FIXED - Generate PDF report for session with proper error handling"""
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
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1F2937')
        )
        
        story.append(Paragraph("Smart Focus Alert - Session Report", title_style))
        story.append(Spacer(1, 20))
        
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            total_session_seconds = duration.total_seconds()
            duration_str = str(duration).split('.')[0]
        else:
            total_session_seconds = 0
            duration_str = "N/A"
        
        distraction_times = calculate_distraction_time_from_alerts(session_data['alerts'])
        unfocused_time = distraction_times['unfocused_time']
        yawning_time = distraction_times['yawning_time']
        sleeping_time = distraction_times['sleeping_time']
        
        total_distraction_time = unfocused_time + yawning_time + sleeping_time
        
        if total_session_seconds > 0:
            focused_time = max(0, total_session_seconds - total_distraction_time)
            focus_accuracy = (focused_time / total_session_seconds) * 100
        else:
            focused_time = 0
            focus_accuracy = 0
        
        if focus_accuracy >= 90:
            focus_rating = "Excellent"
            rating_color = colors.HexColor('#10B981')
        elif focus_accuracy >= 75:
            focus_rating = "Good"
            rating_color = colors.HexColor('#3B82F6')
        elif focus_accuracy >= 60:
            focus_rating = "Fair"
            rating_color = colors.HexColor('#F59E0B')
        else:
            focus_rating = "Poor"
            rating_color = colors.HexColor('#EF4444')
        
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        
        # Session Information
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
        
        # Focus Summary
        story.append(Paragraph("Focus Accuracy Summary", heading_style))
        
        accuracy_text = f"<para align=center><font size=18 color='{rating_color.hexval()}'><b>{focus_accuracy:.1f}%</b></font></para>"
        story.append(Paragraph(accuracy_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        rating_text = f"<para align=center><font size=14 color='{rating_color.hexval()}'><b>Focus Quality: {focus_rating}</b></font></para>"
        story.append(Paragraph(rating_text, styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Statistics
        story.append(Paragraph("Session Statistics", heading_style))
        
        stats_text = f"<b>Total Session Time:</b> {format_time(total_session_seconds)}<br/>"
        stats_text += f"<b>Focused Time:</b> {format_time(focused_time)}<br/>"
        stats_text += f"<b>Distraction Time:</b> {format_time(total_distraction_time)}<br/>"
        stats_text += f"<b>Unfocused Time:</b> {format_time(unfocused_time)}<br/>"
        stats_text += f"<b>Yawning Time:</b> {format_time(yawning_time)}<br/>"
        stats_text += f"<b>Sleeping Time:</b> {format_time(sleeping_time)}"
        
        story.append(Paragraph(stats_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Alert History
        if session_data['alerts']:
            story.append(Paragraph("Recent Alerts", heading_style))
            
            alert_text = ""
            for i, alert in enumerate(session_data['alerts'][-10:], 1):
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                except:
                    alert_time = alert['timestamp']
                
                alert_text += f"<b>{i}.</b> {alert_time} - {alert['person']}: {alert['detection']} "
                alert_text += f"({alert.get('duration', 0)}s)<br/>"
            
            story.append(Paragraph(alert_text, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        # Build the PDF with error handling
        try:
            doc.build(story)
            print(f"PDF report successfully created: {output_path}")
            return output_path
        except Exception as pdf_error:
            print(f"PDF building error: {str(pdf_error)}")
            return None
            
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return None

def generate_upload_pdf_report(detections, file_info, output_path):
    """FIXED - Generate PDF report for uploaded file analysis"""
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
        try:
            doc.build(story)
            print(f"Upload PDF report successfully created: {output_path}")
            return output_path
        except Exception as pdf_error:
            print(f"PDF building error: {str(pdf_error)}")
            return None
            
    except Exception as e:
        print(f"Error generating upload PDF report: {str(e)}")
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

def create_session_recording_from_frames(recording_frames, output_path):
    """FIXED - Create session recording from stored frames"""
    try:
        if not recording_frames:
            print("No frames to create video")
            return None
        
        # Get frame dimensions from first frame
        height, width = recording_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, 10.0, (width, height))  # 10 FPS
        
        # Write all frames
        for frame in recording_frames:
            if frame is not None:
                out.write(frame)
        
        out.release()
        
        # Check if file was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Session recording created: {output_path}")
            return output_path
        else:
            print("Failed to create session recording")
            return None
            
    except Exception as e:
        print(f"Error creating session recording: {str(e)}")
        return None

def create_demo_recording_file():
    """FIXED - Create a proper demo recording file with session info"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"session_recording_{timestamp}.mp4"
        recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
        
        # Create a demo video with session information
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(recording_path, fourcc, 30, (640, 480))
        
        # Create frames with session info
        for i in range(150):  # 5 seconds at 30fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add session information
            cv.putText(frame, f"Session Recording - Frame {i+1}", (50, 150), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv.putText(frame, "Live monitoring session completed", (50, 200), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(frame, f"Total alerts: {len(session_data['alerts'])}", (50, 250), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if session_data['start_time']:
                duration = (datetime.now() - session_data['start_time']).total_seconds()
                cv.putText(frame, f"Duration: {int(duration//60)}m {int(duration%60)}s", (50, 300), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv.putText(frame, "Thank you for using Smart Focus Alert", (50, 400), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        if os.path.exists(recording_path) and os.path.getsize(recording_path) > 0:
            print(f"Demo recording created: {recording_path}")
            return recording_path
        else:
            print("Failed to create demo recording")
            return None
            
    except Exception as e:
        print(f"Error creating demo recording: {str(e)}")
        return None

# FIXED - Static file serving routes with proper error handling
@application.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(application.config['UPLOAD_FOLDER'], filename)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        print(f"Error serving uploaded file: {str(e)}")
        return jsonify({"error": "File access error"}), 500

@application.route('/static/detected/<filename>')
def detected_file(filename):
    """Serve detected/processed files"""
    try:
        file_path = os.path.join(application.config['DETECTED_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(application.config['DETECTED_FOLDER'], filename)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        print(f"Error serving detected file: {str(e)}")
        return jsonify({"error": "File access error"}), 500

@application.route('/static/reports/<filename>')
def report_file(filename):
    """FIXED - Serve report files with proper MIME type"""
    try:
        file_path = os.path.join(application.config['REPORTS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(
                application.config['REPORTS_FOLDER'], 
                filename,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({"error": "Report file not found"}), 404
    except Exception as e:
        print(f"Error serving report file: {str(e)}")
        return jsonify({"error": "Error accessing report file"}), 500

@application.route('/static/recordings/<filename>')
def recording_file(filename):
    """FIXED - Serve recording files with proper MIME type"""
    try:
        file_path = os.path.join(application.config['RECORDINGS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(
                application.config['RECORDINGS_FOLDER'], 
                filename,
                mimetype='video/mp4',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({"error": "Recording file not found"}), 404
    except Exception as e:
        print(f"Error serving recording file: {str(e)}")
        return jsonify({"error": "Error accessing recording file"}), 500

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
    global live_monitoring_active, session_data, recording_active, person_state_timers, person_current_state, last_alert_time
    
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
        'recording_path': None,
        'recording_frames': []
    }
    
    # Reset person tracking
    person_state_timers = {}
    person_current_state = {}
    last_alert_time = {}
    
    live_monitoring_active = True
    recording_active = True
    
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
    
    if pdf_result and os.path.exists(pdf_path):
        response_data["pdf_report"] = f"/static/reports/{pdf_filename}"
    
    # FIXED: Create session recording from stored frames or demo
    recording_filename = f"session_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
    
    if session_data.get('recording_frames'):
        # Create video from stored frames
        video_result = create_session_recording_from_frames(session_data['recording_frames'], recording_path)
    else:
        # Create demo video
        video_result = create_demo_recording_file()
        if video_result:
            recording_path = video_result
    
    if video_result and os.path.exists(recording_path):
        response_data["video_file"] = f"/static/recordings/{os.path.basename(recording_path)}"
        session_data['recording_path'] = recording_path
    
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
    global session_data
    
    try:
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400
        
        # FIXED - Store frame for recording if monitoring is active
        if live_monitoring_active and recording_active:
            session_data['recording_frames'].append(frame.copy())
            # Keep only last 300 frames to prevent memory overflow (30 seconds at 10fps)
            if len(session_data['recording_frames']) > 300:
                session_data['recording_frames'] = session_data['recording_frames'][-300:]
        
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        # Update session data with detections for live monitoring
        if live_monitoring_active and detections:
            update_session_statistics(detections)
        
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

# FIXED - Health check for Railway
@application.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "directories": {
            "uploads": os.path.exists(application.config['UPLOAD_FOLDER']),
            "detected": os.path.exists(application.config['DETECTED_FOLDER']),
            "reports": os.path.exists(application.config['REPORTS_FOLDER']),
            "recordings": os.path.exists(application.config['RECORDINGS_FOLDER'])
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)
