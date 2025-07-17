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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import tempfile
import shutil
import traceback

application = Flask(__name__)

application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['MAX_CONTENT_PATH'] = 10000000

# Ensure all directories exist
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            os.chmod(folder, 0o755)
        print(f"Directory ready: {folder}")
    except Exception as e:
        print(f"Error creating directory {folder}: {str(e)}")

# CRITICAL FIX: Enhanced global variables with thread-safe management
monitoring_lock = threading.RLock()
monitoring_state = {
    'active': False,
    'recording': False,
    'session_id': None,
    'last_activity': None
}

# Enhanced session data structure
session_data = {
    'session_id': None,
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
    'recording_frames': []
}

video_writer = None
person_state_timers = {}
person_current_state = {}
last_alert_time = {}

DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,
    'YAWNING': 3.5,
    'NOT FOCUSED': 10
}

def draw_landmarks(image, landmarks, land_mark, color):
    height, width = image.shape[:2]
    for face in land_mark:
        point = landmarks.landmark[face]
        point_scale = ((int)(point.x * width), (int)(point.y * height))     
        cv.circle(image, point_scale, 2, color, 1)

def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, landmarks, top_bottom, left_right):
    top = landmarks.landmark[top_bottom[0]]
    bottom = landmarks.landmark[top_bottom[1]]
    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmarks.landmark[left_right[0]]
    right = landmarks.landmark[left_right[1]]
    left_right_dis = euclidean_distance(image, left, right)
    
    aspect_ratio = left_right_dis / top_bottom_dis
    return aspect_ratio

def calculate_midpoint(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    midpoint = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
    return midpoint

def check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
    left_eye_midpoint = calculate_midpoint(left_eye_points)
    right_eye_midpoint = calculate_midpoint(right_eye_points)
    left_iris_midpoint = calculate_midpoint(left_iris_points)
    right_iris_midpoint = calculate_midpoint(right_iris_points)
    deviation_threshold_horizontal = 2.8
    
    return (abs(left_iris_midpoint[0] - left_eye_midpoint[0]) <= deviation_threshold_horizontal 
            and abs(right_iris_midpoint[0] - right_eye_midpoint[0]) <= deviation_threshold_horizontal)

def detect_drowsiness(frame, landmarks):
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

    draw_landmarks(frame, landmarks, FACE, COLOR_GREEN)
    draw_landmarks(frame, landmarks, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
    draw_landmarks(frame, landmarks, LEFT_EYE_LEFT_RIGHT, COLOR_RED)
    draw_landmarks(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
    draw_landmarks(frame, landmarks, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)
    draw_landmarks(frame, landmarks, UPPER_LOWER_LIPS, COLOR_BLUE)
    draw_landmarks(frame, landmarks, LEFT_RIGHT_LIPS, COLOR_BLUE)

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

    try:
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv.circle(frame, center_left, int(l_radius), COLOR_MAGENTA, 1, cv.LINE_AA)
        cv.circle(frame, center_right, int(r_radius), COLOR_MAGENTA, 1, cv.LINE_AA)
    except:
        pass

    ratio_left_eye = get_aspect_ratio(frame, landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
    ratio_right_eye = get_aspect_ratio(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
    eye_ratio = (ratio_left_eye + ratio_right_eye) / 2
    
    ratio_lips = get_aspect_ratio(frame, landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
    
    iris_focused = check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points)
    
    eyes_closed = eye_ratio > 5.0
    yawning = ratio_lips < 1.8
    not_focused = not iris_focused
    
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

# CRITICAL FIX: Enhanced monitoring state management
def is_monitoring_active():
    """Thread-safe check for monitoring status"""
    with monitoring_lock:
        return monitoring_state['active']

def set_monitoring_active(active, session_id=None):
    """Thread-safe setter for monitoring status"""
    with monitoring_lock:
        monitoring_state['active'] = active
        monitoring_state['last_activity'] = time.time()
        if session_id:
            monitoring_state['session_id'] = session_id
        print(f"Monitoring state updated: active={active}, session_id={monitoring_state['session_id']}")

def get_current_session_id():
    """Get current session ID thread-safely"""
    with monitoring_lock:
        return monitoring_state.get('session_id')

def detect_persons_with_attention(image, mode="image"):
    global person_state_timers, person_current_state, last_alert_time
    
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
    
    # CRITICAL FIX: More robust monitoring check
    is_monitoring = is_monitoring_active()
    current_session_id = get_current_session_id()
    
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
            
            if matched_face_idx != -1:
                attention_status, state = detect_drowsiness(
                    image, 
                    mesh_results.multi_face_landmarks[matched_face_idx]
                )
            
            status_text = attention_status.get("state", "FOCUSED")
            person_key = f"person_{i+1}"
            
            duration = 0
            if mode == "video" and is_monitoring and current_session_id:
                with monitoring_lock:
                    if person_key not in person_state_timers:
                        person_state_timers[person_key] = {}
                        person_current_state[person_key] = None
                        last_alert_time[person_key] = 0
                    
                    if person_current_state[person_key] != status_text:
                        person_state_timers[person_key] = {}
                        person_current_state[person_key] = status_text
                        if status_text in DISTRACTION_THRESHOLDS:
                            person_state_timers[person_key][status_text] = current_time
                    else:
                        if status_text in DISTRACTION_THRESHOLDS:
                            if status_text not in person_state_timers[person_key]:
                                person_state_timers[person_key][status_text] = current_time
                    
                    if status_text in person_state_timers[person_key]:
                        duration = current_time - person_state_timers[person_key][status_text]
            
            if mode == "video" and is_monitoring:
                status_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (0, 165, 255),
                    "YAWNING": (0, 255, 255),
                    "SLEEPING": (0, 0, 255)
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    timer_text = f"Person {i+1}: {status_text} ({duration:.1f}s/{threshold}s)"
                else:
                    timer_text = f"Person {i+1}: {status_text}"
                
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv.getTextSize(timer_text, font, font_scale, thickness)
                
                text_y = y - 10
                if text_y < text_height + 10:
                    text_y = y + h + text_height + 10
                
                overlay = image.copy()
                cv.rectangle(overlay, (x, text_y - text_height - 5), (x + text_width + 10, text_y + 5), (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                cv.putText(image, timer_text, (x + 5, text_y), font, font_scale, main_color, thickness)
            else:
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                info_y_start = y + h + 10
                box_padding = 10
                line_height = 20
                box_height = 4 * line_height
                
                overlay = image.copy()
                cv.rectangle(overlay, 
                            (x - box_padding, info_y_start - box_padding), 
                            (x + w + box_padding, info_y_start + box_height), 
                            (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)
                
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                thickness = 1
                
                cv.putText(image, f"Person {i+1}", (x, info_y_start), 
                        font, font_scale, (50, 205, 50), thickness+1)
                cv.putText(image, f"Confidence: {confidence_score*100:.2f}%", 
                        (x, info_y_start + line_height), font, font_scale, font_color, thickness)
                cv.putText(image, f"Position: x:{x}, y:{y} Size: w:{w}, h:{h}", 
                        (x, info_y_start + 2*line_height), font, font_scale, font_color, thickness)
                
                status_color = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (255, 165, 0),
                    "YAWNING": (255, 255, 0),
                    "SLEEPING": (0, 0, 255)
                }
                color = status_color.get(status_text, (0, 255, 0))
                
                cv.putText(image, f"Status: {status_text}", 
                        (x, info_y_start + 3*line_height), font, font_scale, color, thickness)

            # CRITICAL FIX: Enhanced alert handling with session validation
            should_alert = False
            alert_message = ""
            
            if (mode == "video" and is_monitoring and current_session_id and 
                status_text in DISTRACTION_THRESHOLDS and 
                person_key in person_state_timers and 
                status_text in person_state_timers[person_key]):
                
                if duration >= DISTRACTION_THRESHOLDS[status_text]:
                    alert_cooldown = 5
                    with monitoring_lock:
                        # Double-check monitoring is still active
                        if monitoring_state['active'] and current_time - last_alert_time.get(person_key, 0) >= alert_cooldown:
                            should_alert = True
                            last_alert_time[person_key] = current_time
                            
                            if status_text == 'SLEEPING':
                                alert_message = f'Person {i+1} is sleeping - please wake up!'
                            elif status_text == 'YAWNING':
                                alert_message = f'Person {i+1} is yawning - please take a rest!'
                            elif status_text == 'NOT FOCUSED':
                                alert_message = f'Person {i+1} is not focused - please focus on screen!'
                            
                            # Add alert to session data with validation
                            if session_data and session_data.get('session_id') == current_session_id:
                                session_data['alerts'].append({
                                    'timestamp': datetime.now().isoformat(),
                                    'person': f"Person {i+1}",
                                    'detection': status_text,
                                    'message': alert_message,
                                    'duration': int(duration),
                                    'session_id': current_session_id
                                })
                                print(f"Alert added to session {current_session_id}: {alert_message} (Total: {len(session_data['alerts'])})")
                            else:
                                print(f"WARNING: Alert skipped - session mismatch or invalid session data")
            
            face_img = image[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"person_{i+1}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            face_path = os.path.join(application.config['DETECTED_FOLDER'], face_filename)
            
            if face_img.size > 0:
                try:
                    cv.imwrite(face_path, face_img)
                except Exception as e:
                    print(f"Error saving face image: {str(e)}")
            
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "image_path": f"/static/detected/{face_filename}",
                "status": status_text,
                "timestamp": datetime.now().isoformat(),
                "duration": duration if mode == "video" else 0,
                "session_id": current_session_id
            })
    
    if detections:
        cv.putText(image, f"Persons detected: {len(detections)}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(image, "No persons detected", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image, detections

def calculate_distraction_time_from_alerts(alerts):
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
    """Update session statistics with thread safety"""
    global session_data
    
    if not detections:
        return
    
    with monitoring_lock:
        # Only update if monitoring is active and session is valid
        if monitoring_state['active'] and session_data and session_data.get('session_id'):
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
    """Helper function to find the most common type of distraction with total duration"""
    if not alerts:
        return "None"
    
    distraction_counts = {}
    distraction_durations = {}
    
    for alert in alerts:
        detection = alert.get('detection', 'Unknown')
        duration = alert.get('duration', 0)
        
        # Count occurrences
        distraction_counts[detection] = distraction_counts.get(detection, 0) + 1
        
        # Sum durations
        distraction_durations[detection] = distraction_durations.get(detection, 0) + duration
    
    if not distraction_counts:
        return "None"
    
    # Find most common by count
    most_common = max(distraction_counts, key=distraction_counts.get)
    count = distraction_counts[most_common]
    total_duration = distraction_durations[most_common]
    
    return f"{most_common} ({count} times, {total_duration}s total)"

def calculate_average_focus_metric(focused_time, total_session_seconds):
    """Calculate a meaningful average focus metric based on session duration"""
    if total_session_seconds <= 0:
        return "N/A"
    
    # Convert to minutes for easier reading
    total_minutes = total_session_seconds / 60
    focused_minutes = focused_time / 60
    
    # Different metrics based on session duration
    if total_session_seconds < 60:  # Less than 1 minute
        # Show focus percentage of session time
        focus_percentage = (focused_time / total_session_seconds) * 100
        return f"{focus_percentage:.1f}% of session time"
    
    elif total_session_seconds < 3600:  # Less than 1 hour
        # Show focused minutes per session
        return f"{focused_minutes:.1f} min focused out of {total_minutes:.1f} min total"
    
    else:  # 1 hour or more
        # Show focused minutes per hour (extrapolated)
        hours = total_session_seconds / 3600
        focused_per_hour = focused_minutes / hours
        return f"{focused_per_hour:.1f} min focused per hour"

def generate_pdf_report(session_data, output_path):
    """Generate PDF report for session with corrected focus accuracy calculation"""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
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
            duration_str = str(duration).split('.')[0]  # Remove microseconds
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
        
        # Calculate focused time (session time minus distraction time)
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
        
        # Session Information
        story.append(Paragraph("Session Information", heading_style))
        
        session_info = [
            ['Session ID', session_data.get('session_id', 'Unknown')],
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
        
        # Focus Accuracy Summary
        story.append(Paragraph("Focus Accuracy Summary", heading_style))
        
        # Create a highlighted focus accuracy display
        accuracy_text = f"<para align=center><font size=18 color='{rating_color.hexval()}'><b>{focus_accuracy:.1f}%</b></font></para>"
        story.append(Paragraph(accuracy_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        rating_text = f"<para align=center><font size=14 color='{rating_color.hexval()}'><b>Focus Quality: {focus_rating}</b></font></para>"
        story.append(Paragraph(rating_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Detailed time breakdown
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
        
        # Focus Statistics - FIXED AVERAGE CALCULATION
        story.append(Paragraph("Detailed Focus Statistics", heading_style))
        
        # Calculate corrected average focus metric
        average_focus_metric = calculate_average_focus_metric(focused_time, total_session_seconds)
        
        focus_stats = [
            ['Total Session Duration', format_time(total_session_seconds)],
            ['Focus Accuracy Score', f"{focus_accuracy:.2f}%"],
            ['Focus Quality Rating', focus_rating],
            ['Average Focus Metric', average_focus_metric],  # FIXED: More meaningful metric
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
        
        # Alert History
        if session_data['alerts']:
            story.append(Paragraph("Alert History", heading_style))
            
            alert_headers = ['Time', 'Person', 'Detection', 'Duration', 'Message']
            alert_data = [alert_headers]
            
            for alert in session_data['alerts'][-10:]:  # Show last 10 alerts
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
        
        # Footer
        story.append(Spacer(1, 20))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - Focus Monitoring Report"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        print(f"PDF report generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        traceback.print_exc()
        return None

def generate_upload_pdf_report(detections, file_info, output_path):
    """Generate PDF report for uploaded file analysis"""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
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
        story.append(Paragraph("Smart Focus Alert - Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # File Information
        story.append(Paragraph("File Information", heading_style))
        
        file_info_data = [
            ['File Name', file_info.get('filename', 'Unknown')],
            ['File Type', file_info.get('type', 'Unknown')],
            ['Analysis Date', datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')],
            ['Total Persons Detected', str(len(detections))]
        ]
        
        file_table = Table(file_info_data, colWidths=[3*inch, 2*inch])
        file_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(file_table)
        story.append(Spacer(1, 20))
        
        # Analysis Statistics
        story.append(Paragraph("Analysis Statistics", heading_style))
        
        # Count statuses
        status_counts = {'FOCUSED': 0, 'NOT FOCUSED': 0, 'YAWNING': 0, 'SLEEPING': 0}
        for detection in detections:
            status = detection.get('status', 'FOCUSED')
            if status in status_counts:
                status_counts[status] += 1
        
        total_detections = len(detections)
        focus_accuracy = 0
        if total_detections > 0:
            focus_accuracy = (status_counts['FOCUSED'] / total_detections) * 100
        
        analysis_stats = [
            ['Focus Accuracy', f"{focus_accuracy:.1f}%"],
            ['Focused Persons', str(status_counts['FOCUSED'])],
            ['Unfocused Persons', str(status_counts['NOT FOCUSED'])],
            ['Yawning Persons', str(status_counts['YAWNING'])],
            ['Sleeping Persons', str(status_counts['SLEEPING'])]
        ]
        
        analysis_table = Table(analysis_stats, colWidths=[3*inch, 2*inch])
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(analysis_table)
        story.append(Spacer(1, 20))
        
        # Individual Results
        if detections:
            story.append(Paragraph("Individual Detection Results", heading_style))
            
            detection_headers = ['Person ID', 'Status', 'Confidence', 'Position (X,Y)', 'Size (W,H)']
            detection_data = [detection_headers]
            
            for detection in detections:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                detection_data.append([
                    f"Person {detection.get('id', 'N/A')}",
                    detection.get('status', 'Unknown'),
                    f"{detection.get('confidence', 0)*100:.1f}%",
                    f"({bbox[0]}, {bbox[1]})",
                    f"({bbox[2]}, {bbox[3]})"
                ])
            
            detection_table = Table(detection_data, colWidths=[1*inch, 1.5*inch, 1*inch, 1.2*inch, 1.3*inch])
            detection_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')])
            ]))
            
            story.append(detection_table)
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - File Analysis Report"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        print(f"Upload PDF report generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating upload PDF report: {str(e)}")
        traceback.print_exc()
        return None

def process_video_file(video_path):
    try:
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, []
            
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            cap.release()
            return None, []
        
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
                    try:
                        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
                        all_detections.extend(detections)
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {str(e)}")
                        processed_frame = frame
                else:
                    processed_frame = frame
                    
                out.write(processed_frame)
                
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
        finally:
            cap.release()
            out.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Video processing completed: {output_path}")
            return output_path, all_detections
        else:
            print(f"Video processing failed: output file not created or empty")
            return None, []
            
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        traceback.print_exc()
        return None, []

def create_session_recording_from_frames(recording_frames, output_path):
    try:
        if not recording_frames:
            print("No frames to create video")
            return None
        
        print(f"Creating session recording from {len(recording_frames)} processed frames")
        height, width = recording_frames[0].shape[:2]
        
        # Use frame rate of 10 fps for smooth playback of recorded frames
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, 10.0, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return None
        
        frames_written = 0
        for frame in recording_frames:
            if frame is not None and frame.size > 0:
                # Ensure frame has correct dimensions
                if frame.shape[:2] == (height, width):
                    out.write(frame)
                    frames_written += 1
                else:
                    # Resize frame if dimensions don't match
                    resized_frame = cv.resize(frame, (width, height))
                    out.write(resized_frame)
                    frames_written += 1
        
        out.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Session recording created successfully: {output_path}")
            print(f"Total frames written: {frames_written}")
            return output_path
        else:
            print("Failed to create session recording - file not created or empty")
            return None
            
    except Exception as e:
        print(f"Error creating session recording: {str(e)}")
        traceback.print_exc()
        return None

def create_demo_recording_file():
    """Create a proper demo recording file using actual recorded frames with face landmarks"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"session_recording_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
        recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
        
        # Use recorded frames if available, otherwise create a meaningful demo
        with monitoring_lock:
            current_frames = session_data.get('recording_frames', []).copy() if session_data else []
        
        if current_frames and len(current_frames) > 0:
            print("Using actual recorded frames for demo video")
            return create_session_recording_from_frames(current_frames, recording_path)
        
        # If no recorded frames, create a demo with simulated detection
        print("Creating demo recording with simulated detection")
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(recording_path, fourcc, 15, (640, 480))
        
        if not out.isOpened():
            print(f"Error: Could not open demo video writer for {recording_path}")
            return None
        
        # Create frames with session summary and visual elements
        for i in range(90):  # 6 seconds at 15 fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add background gradient
            for y in range(480):
                intensity = int(30 + (y / 480) * 50)
                frame[y, :] = [intensity//3, intensity//2, intensity]
            
            # Main title
            cv.putText(frame, "Smart Focus Alert - Session Complete", (50, 80), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Session stats
            y_offset = 140
            stats = [
                f"Session Duration: {format_session_duration()}",
                f"Total Alerts Generated: {len(session_data.get('alerts', []))}",
                f"Persons Detected: {session_data.get('focus_statistics', {}).get('total_persons', 0)}",
                f"Total Detections: {session_data.get('focus_statistics', {}).get('total_detections', 0)}"
            ]
            
            for j, stat in enumerate(stats):
                color = (100, 255, 100) if j == 0 else (200, 200, 255)
                cv.putText(frame, stat, (50, y_offset + j*30), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add alert summary if any alerts exist
            current_alerts = session_data.get('alerts', []) if session_data else []
            if current_alerts:
                cv.putText(frame, "Recent Alerts:", (50, 300), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
                
                recent_alerts = current_alerts[-3:]  # Show last 3 alerts
                for k, alert in enumerate(recent_alerts):
                    try:
                        alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                        alert_text = f"{alert_time} - {alert['detection']} ({alert['person']})"
                    except:
                        alert_text = f"{alert.get('detection', 'Alert')} - {alert.get('person', 'Unknown')}"
                    
                    alert_color = (100, 100, 255) if alert.get('detection') == 'SLEEPING' else (100, 255, 255)
                    cv.putText(frame, alert_text, (70, 330 + k*25), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
            
            # Add animated elements
            pulse = int(50 + 30 * np.sin(i * 0.3))
            cv.circle(frame, (580, 50), 20, (pulse, pulse, 255), -1)
            cv.putText(frame, "REC", (560, 58), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Thank you message
            cv.putText(frame, "Thank you for using Smart Focus Alert!", (80, 430), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Progress bar
            progress_width = int((i / 89) * 540)
            cv.rectangle(frame, (50, 450), (50 + progress_width, 460), (100, 255, 100), -1)
            cv.rectangle(frame, (50, 450), (590, 460), (100, 100, 100), 2)
            
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
        traceback.print_exc()
        return None

def format_session_duration():
    """Format session duration for display"""
    try:
        with monitoring_lock:
            if session_data and session_data.get('start_time') and session_data.get('end_time'):
                duration = session_data['end_time'] - session_data['start_time']
                total_seconds = int(duration.total_seconds())
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                return f"{minutes}m {seconds}s"
            elif session_data and session_data.get('start_time'):
                duration = datetime.now() - session_data['start_time']
                total_seconds = int(duration.total_seconds())
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                return f"{minutes}m {seconds}s"
            else:
                return "Unknown"
    except:
        return "N/A"

@application.route('/static/uploads/<filename>')
def uploaded_file(filename):
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

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
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
                        
                        pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                        
                        file_info = {'filename': filename, 'type': file_ext.upper()}
                        pdf_result = generate_upload_pdf_report(detections, file_info, pdf_path)
                        
                        if pdf_result and os.path.exists(pdf_path):
                            result["pdf_report"] = f"/static/reports/{pdf_filename}"
                    
                elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                    output_path, detections = process_video_file(file_path)
                    
                    if output_path and os.path.exists(output_path):
                        result["processed_video"] = f"/static/detected/{os.path.basename(output_path)}"
                        result["detections"] = detections
                        result["type"] = "video"
                        
                        pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                        
                        file_info = {'filename': filename, 'type': file_ext.upper()}
                        pdf_result = generate_upload_pdf_report(detections, file_info, pdf_path)
                        
                        if pdf_result and os.path.exists(pdf_path):
                            result["pdf_report"] = f"/static/reports/{pdf_filename}"
                
                return render_template('result.html', result=result)
                
        except Exception as e:
            print(f"Upload error: {str(e)}")
            traceback.print_exc()
            return render_template('upload.html', error=f'Upload failed: {str(e)}')
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

# CRITICAL FIX: Enhanced start_monitoring with better session management
@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global session_data, person_state_timers, person_current_state, last_alert_time
    
    try:
        with monitoring_lock:
            print(f"=== START MONITORING REQUEST ===")
            print(f"Current monitoring state: {monitoring_state}")
            
            # Check if already monitoring
            if monitoring_state['active']:
                print("WARNING: Monitoring already active")
                return jsonify({"status": "error", "message": "Monitoring already active"})
            
            # Generate unique session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # CRITICAL FIX: Complete session data reset
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
                },
                'recording_path': None,
                'recording_frames': []
            }
            
            # Reset all tracking variables
            person_state_timers = {}
            person_current_state = {}
            last_alert_time = {}
            
            # Set monitoring active with session ID
            set_monitoring_active(True, session_id)
            monitoring_state['recording'] = True
            
            print(f"NEW SESSION STARTED: {session_id}")
            print(f"Session start time: {session_data['start_time']}")
            print(f"Monitoring state: {monitoring_state}")
            print(f"=== START MONITORING SUCCESS ===")
            
            return jsonify({
                "status": "success", 
                "message": "Monitoring started",
                "session_id": session_id
            })
        
    except Exception as e:
        print(f"CRITICAL ERROR starting monitoring: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to start monitoring: {str(e)}"})

# CRITICAL FIX: Enhanced stop_monitoring with better validation
@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global session_data
    
    try:
        with monitoring_lock:
            print(f"=== STOP MONITORING REQUEST ===")
            print(f"Current monitoring state: {monitoring_state}")
            print(f"Session data exists: {session_data is not None}")
            
            if session_data:
                print(f"Session ID: {session_data.get('session_id')}")
                print(f"Session start: {session_data.get('start_time')}")
                print(f"Total alerts: {len(session_data.get('alerts', []))}")
                print(f"Total frames: {len(session_data.get('recording_frames', []))}")
            
            # CRITICAL FIX: More flexible validation
            if not monitoring_state['active']:
                # Check if we have valid session data even if monitoring flag is off
                if not session_data or not session_data.get('session_id') or not session_data.get('start_time'):
                    print("ERROR: No valid session found")
                    return jsonify({"status": "error", "message": "Monitoring not active"})
                else:
                    print("WARNING: Monitoring flag off but valid session exists - proceeding")
            
            # Ensure session_data exists
            if not session_data:
                print("WARNING: Creating minimal session data")
                session_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                session_data = {
                    'session_id': session_id,
                    'start_time': datetime.now() - timedelta(minutes=1),
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
            
            # Ensure required fields exist
            if not session_data.get('session_id'):
                session_data['session_id'] = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if not session_data.get('start_time'):
                session_data['start_time'] = datetime.now() - timedelta(minutes=1)
            
            # Stop monitoring
            set_monitoring_active(False)
            monitoring_state['recording'] = False
            session_data['end_time'] = datetime.now()
            
            current_session_id = session_data['session_id']
            print(f"SESSION STOPPED: {current_session_id}")
            print(f"Session duration: {session_data['end_time'] - session_data['start_time']}")
            
            response_data = {
                "status": "success", 
                "message": "Monitoring stopped",
                "session_id": current_session_id
            }
            
            # Generate PDF report with enhanced error handling
            print("=== GENERATING PDF REPORT ===")
            try:
                pdf_filename = f"session_report_{current_session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                print(f"PDF generation for session: {current_session_id}")
                print(f"PDF path: {pdf_path}")
                print(f"Session alerts: {len(session_data.get('alerts', []))}")
                
                pdf_result = generate_pdf_report(session_data, pdf_path)
                
                if pdf_result and os.path.exists(pdf_path):
                    response_data["pdf_report"] = f"/static/reports/{pdf_filename}"
                    print(f"PDF SUCCESS: {pdf_filename}")
                else:
                    print("PDF FAILED: File not created")
                    
            except Exception as pdf_error:
                print(f"PDF ERROR: {str(pdf_error)}")
                traceback.print_exc()
            
            # Generate video recording with enhanced error handling
            print("=== GENERATING VIDEO RECORDING ===")
            try:
                recording_filename = f"session_recording_{current_session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
                
                print(f"Video generation for session: {current_session_id}")
                print(f"Video path: {recording_path}")
                
                frame_count = len(session_data.get('recording_frames', []))
                print(f"Available frames: {frame_count}")
                
                video_result = None
                if frame_count > 0:
                    print(f"Creating video from {frame_count} recorded frames")
                    video_result = create_session_recording_from_frames(session_data['recording_frames'], recording_path)
                else:
                    print("No recorded frames, creating demo video")
                    video_result = create_demo_recording_file()
                    if video_result:
                        recording_path = video_result
                
                if video_result and os.path.exists(recording_path):
                    response_data["video_file"] = f"/static/recordings/{os.path.basename(recording_path)}"
                    session_data['recording_path'] = recording_path
                    print(f"VIDEO SUCCESS: {os.path.basename(recording_path)}")
                else:
                    print("VIDEO FAILED: File not created")
                    
            except Exception as video_error:
                print(f"VIDEO ERROR: {str(video_error)}")
                traceback.print_exc()
            
            print(f"=== STOP MONITORING COMPLETE ===")
            print(f"Final response: {response_data}")
            return jsonify(response_data)
        
    except Exception as e:
        print(f"FATAL ERROR stopping monitoring: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to stop monitoring: {str(e)}"})

# CRITICAL FIX: Enhanced get_monitoring_data with better validation
@application.route('/get_monitoring_data')
def get_monitoring_data():
    try:
        with monitoring_lock:
            # More robust checking
            if not monitoring_state['active']:
                return jsonify({"error": "Monitoring not active"})
            
            # Validate session data
            if not session_data or not session_data.get('session_id'):
                return jsonify({"error": "No valid session data"})
            
            current_alerts = session_data.get('alerts', [])
            recent_alerts = current_alerts[-5:] if current_alerts else []
            
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
            
            current_detections = session_data.get('detections', [])
            recent_detections = current_detections[-10:] if current_detections else []
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
                'alert_count': len(current_alerts),
                'current_status': current_status,
                'latest_alerts': formatted_alerts,
                'session_id': session_data.get('session_id'),
                'session_active': monitoring_state['active']
            })
        
    except Exception as e:
        print(f"Error getting monitoring data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get monitoring data: {str(e)}"})

@application.route('/monitoring_status')
def monitoring_status():
    try:
        with monitoring_lock:
            return jsonify({
                "is_active": monitoring_state['active'],
                "session_id": monitoring_state.get('session_id'),
                "recording": monitoring_state.get('recording', False)
            })
    except Exception as e:
        print(f"Error getting monitoring status: {str(e)}")
        return jsonify({"is_active": False})

@application.route('/check_camera')
def check_camera():
    try:
        return jsonify({"camera_available": False})
    except Exception as e:
        print(f"Error checking camera: {str(e)}")
        return jsonify({"camera_available": False})

# CRITICAL FIX: Enhanced process_frame with better session validation
@application.route('/process_frame', methods=['POST'])
def process_frame():
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
        
        # Process frame for detection FIRST to get face landmarks and overlays
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        # CRITICAL FIX: Enhanced frame storage validation
        with monitoring_lock:
            if (monitoring_state['active'] and monitoring_state.get('recording', False) and 
                session_data and session_data.get('session_id')):
                
                # Store PROCESSED frame with face landmarks for recording
                session_data['recording_frames'].append(processed_frame.copy())
                
                # Keep only last 300 frames to prevent memory issues
                if len(session_data['recording_frames']) > 300:
                    session_data['recording_frames'] = session_data['recording_frames'][-300:]
                
                # Debug log every 10th frame
                frame_count = len(session_data['recording_frames'])
                if frame_count % 10 == 0:
                    print(f"FRAME STORAGE [Session: {session_data.get('session_id')}]: {frame_count} frames, {len(detections)} detections")
            elif not monitoring_state['active']:
                print("WARNING: Frame received but monitoring not active")
            elif not session_data or not session_data.get('session_id'):
                print("WARNING: Frame received but no valid session")
        
        # Update session statistics if monitoring is active
        if monitoring_state['active'] and detections:
            update_session_statistics(detections)
        
        # Encode processed frame back to base64
        _, buffer = cv.imencode('.jpg', processed_frame, [cv.IMWRITE_JPEG_QUALITY, 90])
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections,
            "session_id": session_data.get('session_id') if session_data else None
        })
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Frame processing failed: {str(e)}"}), 500

@application.route('/debug_status')
def debug_status():
    """Debug endpoint to check detailed system status"""
    try:
        with monitoring_lock:
            status = {
                "monitoring_state": monitoring_state.copy(),
                "session_data_exists": session_data is not None,
                "session_id": session_data.get('session_id') if session_data else None,
                "session_start_time": session_data.get('start_time').isoformat() if session_data and session_data.get('start_time') else None,
                "session_end_time": session_data.get('end_time').isoformat() if session_data and session_data.get('end_time') else None,
                "alerts_count": len(session_data.get('alerts', [])) if session_data else 0,
                "recording_frames_count": len(session_data.get('recording_frames', [])) if session_data else 0,
                "detections_count": len(session_data.get('detections', [])) if session_data else 0,
                "person_state_timers": len(person_state_timers),
                "person_current_state": len(person_current_state),
                "last_alert_time": len(last_alert_time),
                "directories": {
                    "uploads": os.path.exists(application.config['UPLOAD_FOLDER']),
                    "detected": os.path.exists(application.config['DETECTED_FOLDER']),
                    "reports": os.path.exists(application.config['REPORTS_FOLDER']),
                    "recordings": os.path.exists(application.config['RECORDINGS_FOLDER'])
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add session duration if active
            if session_data and session_data.get('start_time'):
                current_time = session_data.get('end_time', datetime.now())
                duration = current_time - session_data['start_time']
                status["session_duration_seconds"] = duration.total_seconds()
                status["session_duration_formatted"] = str(duration).split('.')[0]
            
            return jsonify(status)
    except Exception as e:
        print(f"Debug status error: {str(e)}")
        return jsonify({"error": str(e), "timestamp": datetime.now().isoformat()}), 500

@application.route('/health')
def health_check():
    try:
        with monitoring_lock:
            return jsonify({
                "status": "healthy", 
                "timestamp": datetime.now().isoformat(),
                "monitoring_state": monitoring_state.copy(),
                "directories": {
                    "uploads": os.path.exists(application.config['UPLOAD_FOLDER']),
                    "detected": os.path.exists(application.config['DETECTED_FOLDER']),
                    "reports": os.path.exists(application.config['REPORTS_FOLDER']),
                    "recordings": os.path.exists(application.config['RECORDINGS_FOLDER'])
                },
                "session_alerts": len(session_data.get('alerts', [])) if session_data else 0,
                "recording_frames": len(session_data.get('recording_frames', [])) if session_data else 0,
                "session_id": session_data.get('session_id') if session_data else None
            })
    except Exception as e:
        print(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == "__main__":
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting Smart Focus Alert application on port {port}")
        print("Directories:")
        for name, path in [
            ("UPLOAD", application.config['UPLOAD_FOLDER']),
            ("DETECTED", application.config['DETECTED_FOLDER']),
            ("REPORTS", application.config['REPORTS_FOLDER']),
            ("RECORDINGS", application.config['RECORDINGS_FOLDER'])
        ]:
            print(f"  {name}: {path} (exists: {os.path.exists(path)})")
        
        print("\nSystem Status:")
        print(f"  Monitoring State: {monitoring_state}")
        print(f"  Session Data: {session_data is not None}")
        
        application.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Application startup error: {str(e)}")
        traceback.print_exc()
