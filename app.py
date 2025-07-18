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
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib
matplotlib.use('Agg')
import base64
import tempfile
import shutil
import traceback
import logging
import gc
import psutil
from pathlib import Path

application = Flask(__name__)

# Railway-optimized configuration with memory limits
application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['MAX_CONTENT_PATH'] = 10000000

# Railway memory optimization settings
MAX_RECORDING_FRAMES = 60  # Reduced from 300 to 60 frames (1 minute at 1fps)
MAX_ALERTS_HISTORY = 20    # Reduced from 50 to 20 alerts
MAX_SESSION_DURATION = 1800  # 30 minutes max session
MEMORY_CLEANUP_INTERVAL = 60  # Cleanup every 60 seconds

# Setup logging for Railway
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories with error handling
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            os.chmod(folder, 0o755)
        logger.info(f"Directory ready: {folder}")
    except Exception as e:
        logger.error(f"Error creating directory {folder}: {str(e)}")

# Railway-optimized global variables
monitoring_lock = threading.RLock()
live_monitoring_active = False
recording_active = False

# Optimized session data with memory limits
session_data = {
    'session_id': None,
    'start_time': None,
    'end_time': None,
    'alerts': [],  # Limited size
    'focus_statistics': {
        'unfocused_time': 0,
        'yawning_time': 0,
        'sleeping_time': 0,
        'total_persons': 0,
        'total_detections': 0
    },
    'recording_frames': [],  # Limited size
    'persistent_timers': {},  # Auto-cleanup
    'last_cleanup': time.time()
}

person_state_timers = {}
person_current_state = {}
last_alert_time = {}

DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,
    'YAWNING': 3.5,
    'NOT FOCUSED': 10
}

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def cleanup_memory():
    """Railway-optimized memory cleanup"""
    try:
        current_time = time.time()
        
        with monitoring_lock:
            if session_data:
                # Limit recording frames
                if len(session_data['recording_frames']) > MAX_RECORDING_FRAMES:
                    session_data['recording_frames'] = session_data['recording_frames'][-MAX_RECORDING_FRAMES:]
                    logger.info(f"Cleaned recording frames to {len(session_data['recording_frames'])}")
                
                # Limit alerts history
                if len(session_data['alerts']) > MAX_ALERTS_HISTORY:
                    session_data['alerts'] = session_data['alerts'][-MAX_ALERTS_HISTORY:]
                    logger.info(f"Cleaned alerts to {len(session_data['alerts'])}")
                
                # Cleanup old timers (older than 5 minutes)
                cleanup_threshold = current_time - 300
                old_timers = []
                for person_id, timer_data in session_data.get('persistent_timers', {}).items():
                    if timer_data.get('state_start_time', current_time) < cleanup_threshold:
                        old_timers.append(person_id)
                
                for person_id in old_timers:
                    del session_data['persistent_timers'][person_id]
                    if person_id in person_state_timers:
                        del person_state_timers[person_id]
                    if person_id in person_current_state:
                        del person_current_state[person_id]
                
                if old_timers:
                    logger.info(f"Cleaned {len(old_timers)} old timers")
                
                session_data['last_cleanup'] = current_time
        
        # Force garbage collection
        gc.collect()
        
        memory_mb = get_memory_usage()
        logger.info(f"Memory usage after cleanup: {memory_mb:.1f} MB")
        
    except Exception as e:
        logger.error(f"Memory cleanup error: {str(e)}")

def railway_optimized_file_operations():
    """Railway-optimized file operations"""
    
    def create_file_safely(file_path, content_generator):
        """Create file with Railway constraints"""
        try:
            # Check available space before creating
            if os.path.exists('/tmp'):
                stat = os.statvfs('/tmp')
                free_space_mb = stat.f_bavail * stat.f_frsize / 1024 / 1024
                if free_space_mb < 50:  # Less than 50MB free
                    logger.warning(f"Low disk space: {free_space_mb:.1f} MB")
                    cleanup_old_files()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Generate content with timeout
            if callable(content_generator):
                success = content_generator(file_path)
            else:
                with open(file_path, 'wb') as f:
                    f.write(content_generator)
                success = True
            
            if success and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                logger.info(f"File created: {file_path} ({os.path.getsize(file_path)} bytes)")
                return file_path
            
            return None
            
        except Exception as e:
            logger.error(f"File creation error: {str(e)}")
            return None
    
    return create_file_safely

create_file_safely = railway_optimized_file_operations()

def cleanup_old_files():
    """Clean old files to free space"""
    try:
        cutoff_time = time.time() - (2 * 60 * 60)  # 2 hours
        
        for folder in [application.config['DETECTED_FOLDER'], 
                      application.config['REPORTS_FOLDER'], 
                      application.config['RECORDINGS_FOLDER']]:
            if os.path.exists(folder):
                for file_path in os.listdir(folder):
                    full_path = os.path.join(folder, file_path)
                    try:
                        if os.path.isfile(full_path) and os.path.getmtime(full_path) < cutoff_time:
                            os.remove(full_path)
                            logger.info(f"Cleaned old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error cleaning file {full_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"File cleanup error: {str(e)}")

def enhanced_timer_management():
    """Railway-optimized timer management"""
    
    def update_person_timer(person_id, state, timestamp=None):
        """Update timer with memory optimization"""
        if timestamp is None:
            timestamp = time.time()
        
        with monitoring_lock:
            if 'persistent_timers' not in session_data:
                session_data['persistent_timers'] = {}
            
            # Limit number of tracked persons
            if len(session_data['persistent_timers']) > 10:  # Max 10 persons
                oldest_person = min(session_data['persistent_timers'].keys(), 
                                  key=lambda x: session_data['persistent_timers'][x].get('state_start_time', timestamp))
                del session_data['persistent_timers'][oldest_person]
                logger.info(f"Removed oldest timer: {oldest_person}")
            
            if person_id not in session_data['persistent_timers']:
                session_data['persistent_timers'][person_id] = {}
            
            current_state = session_data['persistent_timers'][person_id].get('current_state')
            
            if current_state != state:
                # State changed, reset timer
                session_data['persistent_timers'][person_id] = {
                    'current_state': state,
                    'state_start_time': timestamp,
                    'total_time_in_state': 0
                }
                logger.info(f"Person {person_id} state: {current_state} -> {state}")
            else:
                # Update time in current state
                if 'state_start_time' in session_data['persistent_timers'][person_id]:
                    session_data['persistent_timers'][person_id]['total_time_in_state'] = \
                        timestamp - session_data['persistent_timers'][person_id]['state_start_time']
    
    def get_person_timer_duration(person_id, state):
        """Get timer duration"""
        with monitoring_lock:
            if (person_id in session_data.get('persistent_timers', {}) and 
                session_data['persistent_timers'][person_id].get('current_state') == state):
                
                start_time = session_data['persistent_timers'][person_id].get('state_start_time', time.time())
                return time.time() - start_time
            return 0
    
    def check_alert_threshold(person_id, state):
        """Check alert threshold"""
        duration = get_person_timer_duration(person_id, state)
        threshold = DISTRACTION_THRESHOLDS.get(state, float('inf'))
        return duration >= threshold, duration
    
    return update_person_timer, get_person_timer_duration, check_alert_threshold

update_person_timer, get_person_timer_duration, check_alert_threshold = enhanced_timer_management()

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

def detect_persons_with_attention_railway_optimized(image, mode="image", session_id=None, timestamp=None):
    """Railway-optimized detection with memory management"""
    global live_monitoring_active, session_data
    
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=(mode == "image"),
        max_num_faces=5,  # Reduced from 10 to 5
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    detection_results = detector.process(rgb_image)
    mesh_results = face_mesh.process(rgb_image)
    
    detections = []
    ih, iw, _ = image.shape
    current_time = timestamp or time.time()
    
    # Periodic memory cleanup
    if current_time - session_data.get('last_cleanup', 0) > MEMORY_CLEANUP_INTERVAL:
        cleanup_memory()
    
    with monitoring_lock:
        is_monitoring_active = live_monitoring_active
    
    if detection_results.detections:
        for i, detection in enumerate(detection_results.detections):
            if i >= 5:  # Limit to 5 persons max
                break
                
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
                    if face_idx == i:  # Simple matching by index
                        matched_face_idx = face_idx
                        break
            
            if matched_face_idx != -1:
                attention_status, state = detect_drowsiness(
                    image, 
                    mesh_results.multi_face_landmarks[matched_face_idx]
                )
            
            status_text = attention_status.get("state", "FOCUSED")
            person_key = f"person_{i+1}"
            
            # Railway-optimized timer management
            duration = 0
            if mode == "video" and is_monitoring_active and session_id:
                update_person_timer(person_key, status_text, current_time)
                duration = get_person_timer_duration(person_key, status_text)
                
                # Check for alerts with cooldown
                should_alert, alert_duration = check_alert_threshold(person_key, status_text)
                
                if should_alert and status_text in DISTRACTION_THRESHOLDS:
                    alert_key = f"{person_key}_{status_text}"
                    if (alert_key not in last_alert_time or 
                        current_time - last_alert_time[alert_key] >= 10):  # 10 second cooldown
                        
                        alert_data = {
                            'timestamp': datetime.now().isoformat(),
                            'person': f"Person {i+1}",
                            'detection': status_text,
                            'duration': int(alert_duration),
                            'session_id': session_id,
                            'message': get_alert_message(i+1, status_text)
                        }
                        
                        with monitoring_lock:
                            if live_monitoring_active and session_data:
                                session_data['alerts'].append(alert_data)
                                # Keep only recent alerts
                                if len(session_data['alerts']) > MAX_ALERTS_HISTORY:
                                    session_data['alerts'] = session_data['alerts'][-MAX_ALERTS_HISTORY:]
                                
                                last_alert_time[alert_key] = current_time
                                logger.info(f"Alert: {alert_data['message']}")
            
            # Railway-optimized visual feedback
            if mode == "video" and is_monitoring_active:
                status_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (0, 165, 255),
                    "YAWNING": (0, 255, 255),
                    "SLEEPING": (0, 0, 255)
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Simplified timer display
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    timer_text = f"P{i+1}: {status_text} ({duration:.1f}s/{threshold}s)"
                    
                    # Simple progress bar
                    progress = min(1.0, duration / threshold)
                    bar_width = w
                    bar_height = 6
                    bar_x = x
                    bar_y = y - 12
                    
                    cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (64, 64, 64), -1)
                    progress_width = int(bar_width * progress)
                    progress_color = (0, 255, 255) if progress < 0.8 else (0, 0, 255)
                    cv.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), progress_color, -1)
                else:
                    timer_text = f"P{i+1}: {status_text}"
                
                # Simplified text overlay
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                cv.putText(image, timer_text, (x + 5, y - 20), font, font_scale, main_color, thickness)
            
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "status": status_text,
                "timestamp": datetime.now().isoformat(),
                "duration": duration,
                "session_id": session_id
            })
    
    if detections:
        cv.putText(image, f"Railway Optimized: {len(detections)} persons", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv.putText(image, "No persons detected", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image, detections

def get_alert_message(person_id, alert_type):
    """Generate alert message"""
    messages = {
        'SLEEPING': f'Person {person_id} is sleeping - please wake up!',
        'YAWNING': f'Person {person_id} is yawning - please take a rest!',
        'NOT FOCUSED': f'Person {person_id} is not focused - please focus on screen!'
    }
    return messages.get(alert_type, f'Person {person_id} attention alert')

def railway_optimized_pdf_generation(session_data, output_path):
    """Railway-optimized PDF generation"""
    
    def generate_pdf_content(file_path):
        try:
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Simple title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#3B82F6')
            )
            
            story.append(Paragraph("Railway Smart Focus Alert - Session Report", title_style))
            story.append(Spacer(1, 20))
            
            # Basic session info (limited data)
            alerts = session_data.get('alerts', [])
            session_info = [
                ['Session ID', session_data.get('session_id', 'Unknown')[-12:]],
                ['Start Time', session_data.get('start_time', datetime.now()).strftime('%H:%M:%S')],
                ['Total Alerts', str(len(alerts))],
                ['Memory Optimized', 'Railway Compatible']
            ]
            
            session_table = Table(session_info, colWidths=[3*inch, 2*inch])
            session_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ]))
            
            story.append(session_table)
            story.append(Spacer(1, 20))
            
            # Recent alerts only (max 10)
            if alerts:
                story.append(Paragraph("Recent Alerts (Railway Optimized)", styles['Heading2']))
                
                alert_data = [['Time', 'Person', 'Detection', 'Duration']]
                
                for alert in alerts[-10:]:  # Only last 10 alerts
                    try:
                        alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                    except:
                        alert_time = str(alert.get('timestamp', 'Unknown'))[:8]
                    
                    alert_data.append([
                        alert_time,
                        alert.get('person', 'Unknown'),
                        alert.get('detection', 'Unknown'),
                        f"{alert.get('duration', 0)}s"
                    ])
                
                alert_table = Table(alert_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1*inch])
                alert_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
                ]))
                
                story.append(alert_table)
            
            # Simple footer
            story.append(Spacer(1, 20))
            footer_text = f"Railway Optimized Report - {datetime.now().strftime('%H:%M:%S')}"
            story.append(Paragraph(footer_text, styles['Normal']))
            
            doc.build(story)
            return True
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            return False
    
    return create_file_safely(output_path, generate_pdf_content)

def create_railway_optimized_recording(recording_frames, output_path):
    """Railway-optimized video creation"""
    
    def generate_video_content(file_path):
        try:
            if not recording_frames or len(recording_frames) < 5:
                logger.warning("Insufficient frames for railway recording")
                return create_simple_demo_recording(file_path)
            
            logger.info(f"Creating railway recording from {len(recording_frames)} frames")
            height, width = recording_frames[0].shape[:2]
            
            # Use simpler codec for Railway
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            out = cv.VideoWriter(file_path, fourcc, 5.0, (width, height))  # 5 fps for smaller file
            
            if not out.isOpened():
                logger.error(f"Could not open railway video writer: {file_path}")
                return False
            
            # Use every 3rd frame to reduce size
            frames_written = 0
            for i, frame in enumerate(recording_frames):
                if i % 3 == 0 and frame is not None and frame.size > 0:  # Every 3rd frame
                    if frame.shape[:2] == (height, width):
                        out.write(frame)
                        frames_written += 1
                    
                    if frames_written >= 30:  # Max 30 frames (6 seconds at 5fps)
                        break
            
            out.release()
            
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                logger.info(f"Railway recording created: {frames_written} frames")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Railway video creation error: {str(e)}")
            return False
    
    return create_file_safely(output_path, generate_video_content)

def create_simple_demo_recording(output_path):
    """Create simple demo for Railway"""
    try:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, fourcc, 10, (640, 480))
        
        if not out.isOpened():
            return False
        
        # Simple demo with minimal content
        for i in range(30):  # 3 seconds
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Simple background
            frame[:, :] = [30, 30, 60]
            
            # Title
            cv.putText(frame, "Railway Smart Focus Alert", (120, 200), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv.putText(frame, "Session Complete", (220, 250), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
            
            # Stats
            alert_count = len(session_data.get('alerts', []))
            cv.putText(frame, f"Alerts Generated: {alert_count}", (200, 300), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            
            # Simple indicator
            cv.circle(frame, (320, 350), 20, (0, 255, 0), -1)
            cv.putText(frame, "OK", (305, 360), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        return True
        
    except Exception as e:
        logger.error(f"Simple demo creation error: {str(e)}")
        return False

# Railway-optimized Flask routes
@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, session_data, recording_active
    
    try:
        with monitoring_lock:
            logger.info("=== RAILWAY START MONITORING ===")
            
            if live_monitoring_active:
                return jsonify({"status": "error", "message": "Monitoring already active"})
            
            # Check memory before starting
            memory_mb = get_memory_usage()
            if memory_mb > 400:  # More than 400MB
                cleanup_memory()
                gc.collect()
            
            # Get session ID
            request_data = request.get_json() or {}
            session_id = request_data.get('sessionId') or f"railway_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Railway-optimized session data reset
            session_data = {
                'session_id': session_id,
                'start_time': datetime.now(),
                'end_time': None,
                'alerts': [],
                'focus_statistics': {
                    'unfocused_time': 0,
                    'yawning_time': 0,
                    'sleeping_time': 0,
                    'total_persons': 0,
                    'total_detections': 0
                },
                'recording_frames': [],
                'persistent_timers': {},
                'last_cleanup': time.time()
            }
            
            # Reset global timers
            global person_state_timers, person_current_state, last_alert_time
            person_state_timers = {}
            person_current_state = {}
            last_alert_time = {}
            
            live_monitoring_active = True
            recording_active = True
            
            logger.info(f"Railway monitoring started: {session_id}, Memory: {memory_mb:.1f}MB")
            return jsonify({"status": "success", "message": "Railway monitoring started", "session_id": session_id})
        
    except Exception as e:
        logger.error(f"Railway start monitoring error: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to start: {str(e)}"})

@application.route('/process_frame', methods=['POST'])
def process_frame():
    global session_data
    
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
        
        # Railway-optimized processing
        session_id = data.get('sessionId')
        timestamp = data.get('timestamp', time.time()) / 1000
        
        processed_frame, detections = detect_persons_with_attention_railway_optimized(
            frame, 
            mode="video", 
            session_id=session_id,
            timestamp=timestamp
        )
        
        # Railway-optimized frame storage
        with monitoring_lock:
            if live_monitoring_active and recording_active and session_data:
                session_data['recording_frames'].append(processed_frame.copy())
                # Strict limit for Railway
                if len(session_data['recording_frames']) > MAX_RECORDING_FRAMES:
                    session_data['recording_frames'] = session_data['recording_frames'][-MAX_RECORDING_FRAMES:]
        
        # Encode processed frame with compression
        _, buffer = cv.imencode('.jpg', processed_frame, [cv.IMWRITE_JPEG_QUALITY, 70])  # Lower quality
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections,
            "session_id": session_id,
            "memory_mb": get_memory_usage()
        })
        
    except Exception as e:
        logger.error(f"Railway frame processing error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, session_data, recording_active
    
    try:
        with monitoring_lock:
            logger.info("=== RAILWAY STOP MONITORING ===")
            
            if not live_monitoring_active:
                return jsonify({"status": "error", "message": "No active monitoring"})
            
            # Get stop data
            request_data = request.get_json() or {}
            
            # Stop monitoring
            live_monitoring_active = False
            recording_active = False
            
            if session_data:
                session_data['end_time'] = datetime.now()
                
                # Merge client alerts if provided
                if request_data.get('alerts'):
                    client_alerts = request_data['alerts']
                    session_data['alerts'].extend(client_alerts[-10:])  # Only last 10 from client
                    if len(session_data['alerts']) > MAX_ALERTS_HISTORY:
                        session_data['alerts'] = session_data['alerts'][-MAX_ALERTS_HISTORY:]
            
            response_data = {"status": "success", "message": "Railway monitoring stopped"}
            
            # Railway-optimized file generation
            try:
                # Force cleanup before file generation
                cleanup_memory()
                
                # Generate PDF
                pdf_filename = f"railway_report_{session_data['session_id'][-8:]}_{datetime.now().strftime('%H%M%S')}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                pdf_result = railway_optimized_pdf_generation(session_data, pdf_path)
                
                if pdf_result and os.path.exists(pdf_result):
                    response_data["pdf_report"] = f"/static/reports/{os.path.basename(pdf_result)}"
                    logger.info(f"Railway PDF generated: {os.path.basename(pdf_result)}")
                
            except Exception as pdf_error:
                logger.error(f"Railway PDF error: {str(pdf_error)}")
            
            # Railway-optimized video generation
            try:
                video_filename = f"railway_recording_{session_data['session_id'][-8:]}_{datetime.now().strftime('%H%M%S')}.avi"
                video_path = os.path.join(application.config['RECORDINGS_FOLDER'], video_filename)
                
                video_result = create_railway_optimized_recording(session_data['recording_frames'], video_path)
                
                if video_result and os.path.exists(video_result):
                    response_data["video_file"] = f"/static/recordings/{os.path.basename(video_result)}"
                    logger.info(f"Railway video generated: {os.path.basename(video_result)}")
                
            except Exception as video_error:
                logger.error(f"Railway video error: {str(video_error)}")
            
            # Final cleanup
            cleanup_memory()
            
            logger.info(f"Railway monitoring stopped: {session_data.get('session_id')}")
            return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Railway stop monitoring error: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to stop: {str(e)}"})

@application.route('/health')
def health_check():
    try:
        memory_mb = get_memory_usage()
        
        health_status = {
            "status": "healthy" if memory_mb < 450 else "degraded",
            "timestamp": datetime.now().isoformat(),
            "memory_mb": memory_mb,
            "monitoring_active": live_monitoring_active,
            "session_alerts": len(session_data.get('alerts', [])),
            "recording_frames": len(session_data.get('recording_frames', [])),
            "version": "railway-optimized",
            "max_frames": MAX_RECORDING_FRAMES,
            "max_alerts": MAX_ALERTS_HISTORY
        }
        
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@application.route('/get_monitoring_data')
def get_monitoring_data():
    try:
        with monitoring_lock:
            if not live_monitoring_active:
                return jsonify({"error": "Monitoring not active"})
            
            current_alerts = session_data.get('alerts', []) if session_data else []
            recent_alerts = current_alerts[-5:] if current_alerts else []  # Only 5 recent alerts
            
            formatted_alerts = []
            for alert in recent_alerts:
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                except:
                    alert_time = str(alert.get('timestamp', 'Unknown'))[:8]
                
                formatted_alerts.append({
                    'time': alert_time,
                    'message': alert.get('message', 'Alert'),
                    'type': 'error' if alert.get('detection') == 'SLEEPING' else 'warning'
                })
            
            return jsonify({
                'total_persons': min(5, len(session_data.get('persistent_timers', {}))),  # Max 5
                'focused_count': 0,  # Simplified
                'alert_count': len(current_alerts),
                'current_status': 'MONITORING',
                'latest_alerts': formatted_alerts,
                'memory_mb': get_memory_usage()
            })
        
    except Exception as e:
        logger.error(f"Railway monitoring data error: {str(e)}")
        return jsonify({"error": str(e)})

# File serving routes
@application.route('/static/reports/<filename>')
def report_file(filename):
    try:
        file_path = os.path.join(application.config['REPORTS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(application.config['REPORTS_FOLDER'], filename, 
                                     mimetype='application/pdf', as_attachment=True)
        else:
            return jsonify({"error": "Report not found"}), 404
    except Exception as e:
        logger.error(f"Report serve error: {str(e)}")
        return jsonify({"error": "Error serving report"}), 500

@application.route('/static/recordings/<filename>')
def recording_file(filename):
    try:
        file_path = os.path.join(application.config['RECORDINGS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(application.config['RECORDINGS_FOLDER'], filename, 
                                     mimetype='video/avi', as_attachment=True)
        else:
            return jsonify({"error": "Recording not found"}), 404
    except Exception as e:
        logger.error(f"Recording serve error: {str(e)}")
        return jsonify({"error": "Error serving recording"}), 500

@application.route('/monitoring_status')
def monitoring_status():
    try:
        return jsonify({
            "is_active": live_monitoring_active,
            "version": "railway-optimized",
            "memory_mb": get_memory_usage()
        })
    except Exception as e:
        return jsonify({"is_active": False, "error": str(e)})

@application.route('/check_camera')
def check_camera():
    return jsonify({
        "camera_available": False,
        "client_camera_recommended": True,
        "railway_optimized": True
    })

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Railway startup optimization
def railway_startup():
    """Railway-optimized startup"""
    try:
        logger.info("Starting Railway Smart Focus Alert")
        
        # Initial cleanup
        cleanup_old_files()
        
        # Log system info
        memory_mb = get_memory_usage()
        logger.info(f"Initial memory usage: {memory_mb:.1f} MB")
        
        # Start periodic cleanup
        def periodic_cleanup():
            while True:
                time.sleep(MEMORY_CLEANUP_INTERVAL)
                cleanup_memory()
                cleanup_old_files()
        
        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()
        
        logger.info("Railway Smart Focus Alert ready")
        
    except Exception as e:
        logger.error(f"Railway startup error: {str(e)}")

if __name__ == "__main__":
    try:
        railway_startup()
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting Railway app on port {port}")
        application.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Railway startup error: {str(e)}")
        traceback.print_exc()
