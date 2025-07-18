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
import logging
from pathlib import Path

application = Flask(__name__)

# Enhanced configuration with multiple storage paths
application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['BACKUP_FOLDER'] = '/tmp/backup'
application.config['MAX_CONTENT_PATH'] = 10000000

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced directory creation with backup paths
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER'],
               application.config['BACKUP_FOLDER']]:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            os.chmod(folder, 0o755)
        logger.info(f"Directory ready: {folder}")
    except Exception as e:
        logger.error(f"Error creating directory {folder}: {str(e)}")

# Enhanced global variables with better synchronization
monitoring_lock = threading.RLock()
live_monitoring_active = False
recording_active = False

# Enhanced session data structure with backup tracking
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
    'recording_frames': [],
    'persistent_timers': {},  # Server-side timer tracking
    'state_history': {},      # Track state changes
    'backup_attempts': {}     # Track file generation attempts
}

video_writer = None
person_state_timers = {}
person_current_state = {}
last_alert_time = {}
alert_queue = []  # Queue for processing alerts

DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,
    'YAWNING': 3.5,
    'NOT FOCUSED': 10
}

def enhanced_file_operations():
    """Enhanced file operations with backup mechanism"""
    
    def create_file_with_backup(file_path, content_generator, backup_paths=None):
        """Create file with multiple backup attempts"""
        if backup_paths is None:
            backup_paths = []
        
        all_paths = [file_path] + backup_paths
        
        for attempt, path in enumerate(all_paths):
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                # Generate content
                if callable(content_generator):
                    success = content_generator(path)
                else:
                    with open(path, 'wb') as f:
                        f.write(content_generator)
                    success = True
                
                if success and os.path.exists(path) and os.path.getsize(path) > 0:
                    logger.info(f"File created successfully: {path}")
                    return path
                    
            except Exception as e:
                logger.error(f"File creation attempt {attempt + 1} failed for {path}: {str(e)}")
                continue
        
        logger.error(f"All file creation attempts failed for {file_path}")
        return None
    
    return create_file_with_backup

create_file_with_backup = enhanced_file_operations()

def enhanced_timer_management():
    """Enhanced timer management with server-side persistence"""
    
    def update_person_timer(person_id, state, timestamp=None):
        """Update persistent timer for a person's state"""
        if timestamp is None:
            timestamp = time.time()
        
        with monitoring_lock:
            if 'persistent_timers' not in session_data:
                session_data['persistent_timers'] = {}
            
            if person_id not in session_data['persistent_timers']:
                session_data['persistent_timers'][person_id] = {}
                session_data['state_history'][person_id] = []
            
            # Track state changes
            current_state = session_data['persistent_timers'][person_id].get('current_state')
            
            if current_state != state:
                # State changed, record history
                session_data['state_history'][person_id].append({
                    'from_state': current_state,
                    'to_state': state,
                    'timestamp': timestamp
                })
                
                # Reset timer for new state
                session_data['persistent_timers'][person_id] = {
                    'current_state': state,
                    'state_start_time': timestamp,
                    'total_time_in_state': 0
                }
                
                logger.info(f"Person {person_id} state changed: {current_state} -> {state}")
            else:
                # Update time in current state
                if 'state_start_time' in session_data['persistent_timers'][person_id]:
                    session_data['persistent_timers'][person_id]['total_time_in_state'] = \
                        timestamp - session_data['persistent_timers'][person_id]['state_start_time']
    
    def get_person_timer_duration(person_id, state):
        """Get current duration for person in specific state"""
        with monitoring_lock:
            if (person_id in session_data.get('persistent_timers', {}) and 
                session_data['persistent_timers'][person_id].get('current_state') == state):
                
                start_time = session_data['persistent_timers'][person_id].get('state_start_time', time.time())
                return time.time() - start_time
            return 0
    
    def check_alert_threshold(person_id, state):
        """Check if alert threshold is reached"""
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

def detect_persons_with_attention_enhanced(image, mode="image", session_id=None, timestamp=None):
    """Enhanced detection with persistent timer management"""
    global live_monitoring_active, session_data
    
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
    current_time = timestamp or time.time()
    
    with monitoring_lock:
        is_monitoring_active = live_monitoring_active
        current_session_data = session_data.copy() if session_data else None
    
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
            
            # Enhanced timer management
            duration = 0
            if mode == "video" and is_monitoring_active and session_id:
                # Update persistent timer
                update_person_timer(person_key, status_text, current_time)
                duration = get_person_timer_duration(person_key, status_text)
                
                # Check for alerts
                should_alert, alert_duration = check_alert_threshold(person_key, status_text)
                
                if should_alert and status_text in DISTRACTION_THRESHOLDS:
                    # Add to alert queue for processing
                    alert_data = {
                        'timestamp': datetime.now().isoformat(),
                        'person': f"Person {i+1}",
                        'detection': status_text,
                        'duration': int(alert_duration),
                        'session_id': session_id,
                        'person_key': person_key
                    }
                    
                    with monitoring_lock:
                        if live_monitoring_active and session_data:
                            # Check cooldown to prevent spam
                            last_alert_key = f"{person_key}_{status_text}"
                            if (last_alert_key not in last_alert_time or 
                                current_time - last_alert_time[last_alert_key] >= 5):
                                
                                alert_data['message'] = get_alert_message(i+1, status_text)
                                session_data['alerts'].append(alert_data)
                                alert_queue.append(alert_data)
                                last_alert_time[last_alert_key] = current_time
                                
                                logger.info(f"Enhanced alert triggered: {alert_data['message']}")
            
            # Enhanced visual feedback
            if mode == "video" and is_monitoring_active:
                status_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (0, 165, 255),
                    "YAWNING": (0, 255, 255),
                    "SLEEPING": (0, 0, 255)
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Enhanced timer display
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    timer_text = f"Person {i+1}: {status_text} ({duration:.1f}s/{threshold}s)"
                    
                    # Progress bar
                    progress = min(1.0, duration / threshold)
                    bar_width = w
                    bar_height = 8
                    bar_x = x
                    bar_y = y - 15
                    
                    # Background bar
                    cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (64, 64, 64), -1)
                    # Progress bar
                    progress_width = int(bar_width * progress)
                    progress_color = (0, 255, 255) if progress < 0.8 else (0, 0, 255)
                    cv.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), progress_color, -1)
                    
                else:
                    timer_text = f"Person {i+1}: {status_text}"
                
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv.getTextSize(timer_text, font, font_scale, thickness)
                
                text_y = y - 25
                if text_y < text_height + 10:
                    text_y = y + h + text_height + 10
                
                overlay = image.copy()
                cv.rectangle(overlay, (x, text_y - text_height - 5), (x + text_width + 10, text_y + 5), (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                cv.putText(image, timer_text, (x + 5, text_y), font, font_scale, main_color, thickness)
            
            # Save face image
            face_img = image[y:y+h, x:x+w]
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"person_{i+1}_{timestamp_str}_{uuid.uuid4().hex[:8]}.jpg"
            face_path = os.path.join(application.config['DETECTED_FOLDER'], face_filename)
            
            if face_img.size > 0:
                try:
                    cv.imwrite(face_path, face_img)
                except Exception as e:
                    logger.error(f"Error saving face image: {str(e)}")
            
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "image_path": f"/static/detected/{face_filename}",
                "status": status_text,
                "timestamp": datetime.now().isoformat(),
                "duration": duration,
                "session_id": session_id
            })
    
    if detections:
        cv.putText(image, f"Enhanced Detection: {len(detections)} persons", 
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

def enhanced_pdf_generation(session_data, output_path):
    """Enhanced PDF generation with backup paths"""
    
    def generate_pdf_content(file_path):
        try:
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Enhanced title with session info
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#3B82F6')
            )
            
            story.append(Paragraph("Enhanced Smart Focus Alert - Session Report", title_style))
            story.append(Spacer(1, 20))
            
            # Session information with enhanced details
            session_info = [
                ['Session ID', session_data.get('session_id', 'Unknown')[:16] + '...'],
                ['Start Time', session_data.get('start_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')],
                ['Duration', str(session_data.get('end_time', datetime.now()) - session_data.get('start_time', datetime.now())).split('.')[0]],
                ['Total Alerts', str(len(session_data.get('alerts', [])))],
                ['Alert Types', ', '.join(set([alert.get('detection', 'Unknown') for alert in session_data.get('alerts', [])]))],
                ['Persons Tracked', str(len(session_data.get('persistent_timers', {})))]
            ]
            
            session_table = Table(session_info, colWidths=[3*inch, 3*inch])
            session_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(session_table)
            story.append(Spacer(1, 30))
            
            # Enhanced alert history
            if session_data.get('alerts'):
                story.append(Paragraph("Enhanced Alert History", styles['Heading2']))
                
                alert_data = [['Time', 'Person', 'Detection', 'Duration', 'Session']]
                
                for alert in session_data['alerts'][-20:]:  # Last 20 alerts
                    try:
                        alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                    except:
                        alert_time = str(alert.get('timestamp', 'Unknown'))[:8]
                    
                    alert_data.append([
                        alert_time,
                        alert.get('person', 'Unknown'),
                        alert.get('detection', 'Unknown'),
                        f"{alert.get('duration', 0)}s",
                        alert.get('session_id', 'Unknown')[-8:]
                    ])
                
                alert_table = Table(alert_data, colWidths=[1*inch, 1*inch, 1.5*inch, 0.8*inch, 1.2*inch])
                alert_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                story.append(alert_table)
            
            # Enhanced footer
            story.append(Spacer(1, 30))
            footer_text = f"Enhanced Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System v2.0 - Persistent Timer Edition"
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#6B7280')
            )
            story.append(Paragraph(footer_text, footer_style))
            
            doc.build(story)
            return True
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            return False
    
    # Enhanced backup paths
    backup_paths = [
        os.path.join(application.config['BACKUP_FOLDER'], f"backup_{os.path.basename(output_path)}"),
        os.path.join(application.config['REPORTS_FOLDER'], f"fallback_{os.path.basename(output_path)}")
    ]
    
    return create_file_with_backup(output_path, generate_pdf_content, backup_paths)

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, session_data, recording_active
    
    try:
        with monitoring_lock:
            logger.info("=== ENHANCED START MONITORING ===")
            
            if live_monitoring_active:
                return jsonify({"status": "error", "message": "Monitoring already active"})
            
            # Get session ID from request
            request_data = request.get_json() or {}
            session_id = request_data.get('sessionId') or f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Enhanced session data reset
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
                'recording_frames': [],
                'persistent_timers': {},
                'state_history': {},
                'backup_attempts': {}
            }
            
            # Reset global timers
            global person_state_timers, person_current_state, last_alert_time, alert_queue
            person_state_timers = {}
            person_current_state = {}
            last_alert_time = {}
            alert_queue = []
            
            live_monitoring_active = True
            recording_active = True
            
            logger.info(f"Enhanced monitoring started with session ID: {session_id}")
            return jsonify({"status": "success", "message": "Enhanced monitoring started", "session_id": session_id})
        
    except Exception as e:
        logger.error(f"Error starting enhanced monitoring: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to start monitoring: {str(e)}"})

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
        
        # Enhanced processing with session tracking
        session_id = data.get('sessionId')
        timestamp = data.get('timestamp', time.time()) / 1000  # Convert from JS timestamp
        
        processed_frame, detections = detect_persons_with_attention_enhanced(
            frame, 
            mode="video", 
            session_id=session_id,
            timestamp=timestamp
        )
        
        # Store enhanced frame for recording
        with monitoring_lock:
            if live_monitoring_active and recording_active and session_data:
                session_data['recording_frames'].append(processed_frame.copy())
                # Enhanced frame management - keep last 5 minutes at 1fps
                if len(session_data['recording_frames']) > 300:
                    session_data['recording_frames'] = session_data['recording_frames'][-300:]
                
                # Update session statistics
                if detections:
                    session_data['detections'].extend(detections)
                    session_data['focus_statistics']['total_detections'] += len(detections)
                    session_data['focus_statistics']['total_persons'] = max(
                        session_data['focus_statistics']['total_persons'],
                        len(detections)
                    )
        
        # Encode processed frame
        _, buffer = cv.imencode('.jpg', processed_frame, [cv.IMWRITE_JPEG_QUALITY, 85])
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections,
            "session_id": session_id,
            "timestamp": timestamp
        })
        
    except Exception as e:
        logger.error(f"Enhanced frame processing error: {str(e)}")
        return jsonify({"error": f"Frame processing failed: {str(e)}"}), 500

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, session_data, recording_active
    
    try:
        with monitoring_lock:
            logger.info("=== ENHANCED STOP MONITORING ===")
            
            if not live_monitoring_active and (not session_data or not session_data.get('start_time')):
                return jsonify({"status": "error", "message": "No active monitoring session"})
            
            # Get enhanced stop data
            request_data = request.get_json() or {}
            
            # Ensure session_data exists
            if not session_data:
                session_data = {
                    'session_id': request_data.get('sessionId', f"recovery_{int(time.time())}"),
                    'start_time': datetime.now() - timedelta(minutes=1),
                    'end_time': None,
                    'detections': [],
                    'alerts': request_data.get('alerts', []),
                    'focus_statistics': {'unfocused_time': 0, 'yawning_time': 0, 'sleeping_time': 0, 'total_persons': 0, 'total_detections': 0},
                    'recording_path': None,
                    'recording_frames': [],
                    'persistent_timers': request_data.get('personTimers', {}),
                    'state_history': {},
                    'backup_attempts': {}
                }
            
            # Stop monitoring
            live_monitoring_active = False
            recording_active = False
            session_data['end_time'] = datetime.now()
            
            # Merge client data
            if request_data.get('alerts'):
                session_data['alerts'].extend(request_data['alerts'])
            
            response_data = {"status": "success", "message": "Enhanced monitoring stopped"}
            
            # Enhanced PDF generation with backup
            try:
                pdf_filename = f"enhanced_report_{session_data['session_id'][-8:]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                pdf_result = enhanced_pdf_generation(session_data, pdf_path)
                
                if pdf_result and os.path.exists(pdf_result):
                    response_data["pdf_report"] = f"/static/reports/{os.path.basename(pdf_result)}"
                    logger.info(f"Enhanced PDF generated: {pdf_result}")
                else:
                    logger.warning("Enhanced PDF generation failed")
                    
            except Exception as pdf_error:
                logger.error(f"Enhanced PDF error: {str(pdf_error)}")
            
            # Enhanced video generation
            try:
                video_filename = f"enhanced_recording_{session_data['session_id'][-8:]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                video_path = os.path.join(application.config['RECORDINGS_FOLDER'], video_filename)
                
                video_result = create_enhanced_recording(session_data['recording_frames'], video_path)
                
                if video_result and os.path.exists(video_result):
                    response_data["video_file"] = f"/static/recordings/{os.path.basename(video_result)}"
                    logger.info(f"Enhanced video generated: {video_result}")
                else:
                    logger.warning("Enhanced video generation failed")
                    
            except Exception as video_error:
                logger.error(f"Enhanced video error: {str(video_error)}")
            
            logger.info(f"Enhanced monitoring stopped for session: {session_data.get('session_id')}")
            return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Enhanced stop monitoring error: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to stop monitoring: {str(e)}"})

def create_enhanced_recording(recording_frames, output_path):
    """Enhanced recording creation with backup"""
    
    def generate_video_content(file_path):
        try:
            if not recording_frames:
                logger.warning("No frames available for enhanced recording")
                return create_demo_recording_enhanced(file_path)
            
            logger.info(f"Creating enhanced recording from {len(recording_frames)} frames")
            height, width = recording_frames[0].shape[:2]
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(file_path, fourcc, 10.0, (width, height))
            
            if not out.isOpened():
                logger.error(f"Could not open enhanced video writer: {file_path}")
                return False
            
            frames_written = 0
            for frame in recording_frames:
                if frame is not None and frame.size > 0:
                    if frame.shape[:2] == (height, width):
                        out.write(frame)
                        frames_written += 1
                    else:
                        resized_frame = cv.resize(frame, (width, height))
                        out.write(resized_frame)
                        frames_written += 1
            
            out.release()
            
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                logger.info(f"Enhanced recording created: {frames_written} frames")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Enhanced video creation error: {str(e)}")
            return False
    
    backup_paths = [
        os.path.join(application.config['BACKUP_FOLDER'], f"backup_{os.path.basename(output_path)}"),
        os.path.join(application.config['RECORDINGS_FOLDER'], f"fallback_{os.path.basename(output_path)}")
    ]
    
    return create_file_with_backup(output_path, generate_video_content, backup_paths)

def create_demo_recording_enhanced(output_path):
    """Create enhanced demo recording"""
    try:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, 15, (640, 480))
        
        if not out.isOpened():
            return False
        
        for i in range(90):  # 6 seconds
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Enhanced gradient background
            for y in range(480):
                intensity = int(30 + (y / 480) * 50)
                frame[y, :] = [intensity//3, intensity//2, intensity]
            
            # Enhanced title
            cv.putText(frame, "Enhanced Smart Focus Alert", (80, 80), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv.putText(frame, "Session Complete - Persistent Timers Active", (50, 120), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            
            # Enhanced session stats
            stats = [
                f"Session ID: {session_data.get('session_id', 'Demo')[-12:]}",
                f"Enhanced Alerts: {len(session_data.get('alerts', []))}",
                f"Persistent Timers: {len(session_data.get('persistent_timers', {}))}",
                f"Backup System: Active"
            ]
            
            for j, stat in enumerate(stats):
                cv.putText(frame, stat, (50, 180 + j*30), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            
            # Enhanced progress indicator
            pulse = int(50 + 30 * np.sin(i * 0.3))
            cv.circle(frame, (580, 50), 25, (pulse, pulse, 255), -1)
            cv.putText(frame, "V2.0", (558, 58), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
        
        out.release()
        return True
        
    except Exception as e:
        logger.error(f"Enhanced demo creation error: {str(e)}")
        return False

@application.route('/health')
def health_check():
    try:
        with monitoring_lock:
            health_status = {
                "status": "healthy", 
                "timestamp": datetime.now().isoformat(),
                "directories": {
                    "uploads": os.path.exists(application.config['UPLOAD_FOLDER']),
                    "detected": os.path.exists(application.config['DETECTED_FOLDER']),
                    "reports": os.path.exists(application.config['REPORTS_FOLDER']),
                    "recordings": os.path.exists(application.config['RECORDINGS_FOLDER']),
                    "backup": os.path.exists(application.config['BACKUP_FOLDER'])
                },
                "monitoring_active": live_monitoring_active,
                "session_alerts": len(session_data.get('alerts', [])),
                "recording_frames": len(session_data.get('recording_frames', [])),
                "persistent_timers": len(session_data.get('persistent_timers', {})),
                "version": "2.0-enhanced"
            }
            
            # Check directory permissions
            for name, exists in health_status["directories"].items():
                if not exists:
                    health_status["status"] = "degraded"
                    break
            
            return jsonify(health_status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@application.route('/queue_alert', methods=['POST'])
def queue_alert():
    """Enhanced alert queuing endpoint"""
    try:
        alert_data = request.get_json()
        if not alert_data:
            return jsonify({"status": "error", "message": "No alert data"}), 400
        
        # Add to processing queue
        with monitoring_lock:
            if session_data:
                session_data['alerts'].append(alert_data)
                logger.info(f"Alert queued: {alert_data.get('message', 'Unknown')}")
        
        return jsonify({"status": "success", "message": "Alert queued for processing"})
        
    except Exception as e:
        logger.error(f"Alert queue error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Enhanced file serving routes with better error handling
@application.route('/static/reports/<filename>')
def report_file(filename):
    try:
        file_path = os.path.join(application.config['REPORTS_FOLDER'], filename)
        backup_path = os.path.join(application.config['BACKUP_FOLDER'], filename)
        
        if os.path.exists(file_path):
            return send_from_directory(application.config['REPORTS_FOLDER'], filename, mimetype='application/pdf', as_attachment=True)
        elif os.path.exists(backup_path):
            return send_from_directory(application.config['BACKUP_FOLDER'], filename, mimetype='application/pdf', as_attachment=True)
        else:
            return jsonify({"error": "Report file not found"}), 404
    except Exception as e:
        logger.error(f"Error serving report file {filename}: {str(e)}")
        return jsonify({"error": "Error accessing report file"}), 500

@application.route('/static/recordings/<filename>')
def recording_file(filename):
    try:
        file_path = os.path.join(application.config['RECORDINGS_FOLDER'], filename)
        backup_path = os.path.join(application.config['BACKUP_FOLDER'], filename)
        
        if os.path.exists(file_path):
            return send_from_directory(application.config['RECORDINGS_FOLDER'], filename, mimetype='video/mp4', as_attachment=True)
        elif os.path.exists(backup_path):
            return send_from_directory(application.config['BACKUP_FOLDER'], filename, mimetype='video/mp4', as_attachment=True)
        else:
            return jsonify({"error": "Recording file not found"}), 404
    except Exception as e:
        logger.error(f"Error serving recording file {filename}: {str(e)}")
        return jsonify({"error": "Error accessing recording file"}), 500

@application.route('/static/uploads/<filename>')
def uploaded_file(filename):
    try:
        file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(application.config['UPLOAD_FOLDER'], filename)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error serving uploaded file: {str(e)}")
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
        logger.error(f"Error serving detected file: {str(e)}")
        return jsonify({"error": "File access error"}), 500

@application.route('/download/report/<session_id>')
def download_report_by_session(session_id):
    """Enhanced download endpoint with session ID lookup"""
    try:
        # Try to find report by session ID
        report_patterns = [
            f"enhanced_report_{session_id}*.pdf",
            f"session_report_{session_id}*.pdf",
            f"*{session_id}*.pdf"
        ]
        
        import glob
        for pattern in report_patterns:
            files = glob.glob(os.path.join(application.config['REPORTS_FOLDER'], pattern))
            if files:
                filename = os.path.basename(files[0])
                return send_from_directory(application.config['REPORTS_FOLDER'], filename, 
                                        mimetype='application/pdf', as_attachment=True)
            
            # Check backup folder
            files = glob.glob(os.path.join(application.config['BACKUP_FOLDER'], pattern))
            if files:
                filename = os.path.basename(files[0])
                return send_from_directory(application.config['BACKUP_FOLDER'], filename, 
                                        mimetype='application/pdf', as_attachment=True)
        
        return jsonify({"error": f"No report found for session {session_id}"}), 404
        
    except Exception as e:
        logger.error(f"Error downloading report for session {session_id}: {str(e)}")
        return jsonify({"error": "Download failed"}), 500

@application.route('/download/recording/<session_id>')
def download_recording_by_session(session_id):
    """Enhanced download endpoint with session ID lookup"""
    try:
        # Try to find recording by session ID
        recording_patterns = [
            f"enhanced_recording_{session_id}*.mp4",
            f"session_recording_{session_id}*.mp4",
            f"*{session_id}*.mp4"
        ]
        
        import glob
        for pattern in recording_patterns:
            files = glob.glob(os.path.join(application.config['RECORDINGS_FOLDER'], pattern))
            if files:
                filename = os.path.basename(files[0])
                return send_from_directory(application.config['RECORDINGS_FOLDER'], filename, 
                                        mimetype='video/mp4', as_attachment=True)
            
            # Check backup folder
            files = glob.glob(os.path.join(application.config['BACKUP_FOLDER'], pattern))
            if files:
                filename = os.path.basename(files[0])
                return send_from_directory(application.config['BACKUP_FOLDER'], filename, 
                                        mimetype='video/mp4', as_attachment=True)
        
        return jsonify({"error": f"No recording found for session {session_id}"}), 404
        
    except Exception as e:
        logger.error(f"Error downloading recording for session {session_id}: {str(e)}")
        return jsonify({"error": "Download failed"}), 500

@application.route('/get_monitoring_data')
def get_monitoring_data():
    """Enhanced monitoring data endpoint"""
    global session_data
    
    try:
        with monitoring_lock:
            if not live_monitoring_active:
                return jsonify({"error": "Monitoring not active"})
            
            current_alerts = session_data.get('alerts', []) if session_data else []
            recent_alerts = current_alerts[-10:] if current_alerts else []
            
            formatted_alerts = []
            for alert in recent_alerts:
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                except:
                    alert_time = str(alert.get('timestamp', 'Unknown'))[:8]
                
                formatted_alerts.append({
                    'time': alert_time,
                    'message': alert.get('message', 'Unknown alert'),
                    'type': 'error' if alert.get('detection') == 'SLEEPING' else 'warning',
                    'session_id': alert.get('session_id', 'Unknown')[-8:]
                })
            
            current_detections = session_data.get('detections', []) if session_data else []
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
            
            # Enhanced response with timer information
            response_data = {
                'total_persons': total_persons,
                'focused_count': focused_count,
                'alert_count': len(current_alerts),
                'current_status': current_status,
                'latest_alerts': formatted_alerts,
                'session_id': session_data.get('session_id', 'Unknown')[-8:] if session_data else 'Unknown',
                'persistent_timers_count': len(session_data.get('persistent_timers', {})) if session_data else 0,
                'recording_frames_count': len(session_data.get('recording_frames', [])) if session_data else 0
            }
            
            return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting enhanced monitoring data: {str(e)}")
        return jsonify({"error": f"Failed to get monitoring data: {str(e)}"})

@application.route('/monitoring_status')
def monitoring_status():
    """Enhanced monitoring status endpoint"""
    try:
        with monitoring_lock:
            return jsonify({
                "is_active": live_monitoring_active,
                "session_id": session_data.get('session_id', None) if session_data else None,
                "version": "2.0-enhanced"
            })
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        return jsonify({"is_active": False, "error": str(e)})

@application.route('/check_camera')
def check_camera():
    """Enhanced camera check endpoint"""
    try:
        # Enhanced camera availability check
        return jsonify({
            "camera_available": False,  # Railway/ngrok compatibility
            "client_camera_recommended": True,
            "enhanced_features": True
        })
    except Exception as e:
        logger.error(f"Error checking camera: {str(e)}")
        return jsonify({"camera_available": False, "error": str(e)})

@application.route('/debug_status')
def debug_status():
    """Enhanced debug endpoint"""
    try:
        with monitoring_lock:
            status = {
                "live_monitoring_active": live_monitoring_active,
                "recording_active": recording_active,
                "session_data_exists": session_data is not None,
                "session_id": session_data.get('session_id') if session_data else None,
                "session_start_time": session_data.get('start_time').isoformat() if session_data and session_data.get('start_time') else None,
                "session_end_time": session_data.get('end_time').isoformat() if session_data and session_data.get('end_time') else None,
                "alerts_count": len(session_data.get('alerts', [])) if session_data else 0,
                "recording_frames_count": len(session_data.get('recording_frames', [])) if session_data else 0,
                "detections_count": len(session_data.get('detections', [])) if session_data else 0,
                "persistent_timers_count": len(session_data.get('persistent_timers', {})) if session_data else 0,
                "state_history_count": len(session_data.get('state_history', {})) if session_data else 0,
                "backup_attempts_count": len(session_data.get('backup_attempts', {})) if session_data else 0,
                "person_state_timers": len(person_state_timers),
                "person_current_state": len(person_current_state),
                "last_alert_time": len(last_alert_time),
                "alert_queue_size": len(alert_queue),
                "directories": {
                    "uploads": os.path.exists(application.config['UPLOAD_FOLDER']),
                    "detected": os.path.exists(application.config['DETECTED_FOLDER']),
                    "reports": os.path.exists(application.config['REPORTS_FOLDER']),
                    "recordings": os.path.exists(application.config['RECORDINGS_FOLDER']),
                    "backup": os.path.exists(application.config['BACKUP_FOLDER'])
                },
                "timestamp": datetime.now().isoformat(),
                "version": "2.0-enhanced-debug"
            }
            
            # Add session duration if active
            if session_data and session_data.get('start_time'):
                duration = datetime.now() - session_data['start_time']
                status["session_duration_seconds"] = duration.total_seconds()
                status["session_duration_formatted"] = str(duration).split('.')[0]
            
            return jsonify(status)
    except Exception as e:
        logger.error(f"Enhanced debug status error: {str(e)}")
        return jsonify({"error": str(e), "timestamp": datetime.now().isoformat()}), 500

# Additional upload and other routes remain the same but with enhanced error handling
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
                        # Use enhanced detection for uploaded images
                        processed_image, detections = detect_persons_with_attention_enhanced(image, mode="image")
                        
                        output_filename = f"enhanced_processed_{filename}"
                        output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
                        cv.imwrite(output_path, processed_image)
                        
                        result["processed_image"] = f"/static/detected/{output_filename}"
                        result["detections"] = detections
                        result["type"] = "image"
                        result["enhanced"] = True
                
                return render_template('result.html', result=result)
                
        except Exception as e:
            logger.error(f"Enhanced upload error: {str(e)}")
            return render_template('upload.html', error=f'Upload failed: {str(e)}')
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Enhanced error handlers
@application.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found", "version": "2.0-enhanced"}), 404

@application.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error", "version": "2.0-enhanced"}), 500

@application.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({"error": "An unexpected error occurred", "version": "2.0-enhanced"}), 500

# Enhanced cleanup function
def cleanup_old_files():
    """Enhanced cleanup function for old files"""
    try:
        import glob
        import time
        
        # Clean files older than 24 hours
        cutoff_time = time.time() - (24 * 60 * 60)
        
        for folder in [application.config['DETECTED_FOLDER'], 
                      application.config['REPORTS_FOLDER'], 
                      application.config['RECORDINGS_FOLDER'],
                      application.config['BACKUP_FOLDER']]:
            if os.path.exists(folder):
                for file_path in glob.glob(os.path.join(folder, '*')):
                    try:
                        if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                            os.remove(file_path)
                            logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error cleaning file {file_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

# Enhanced startup with cleanup
def enhanced_startup():
    """Enhanced startup procedures"""
    try:
        logger.info("Starting Enhanced Smart Focus Alert v2.0")
        
        # Run initial cleanup
        cleanup_old_files()
        
        # Log configuration
        logger.info("Enhanced Configuration:")
        for name, path in [
            ("UPLOAD", application.config['UPLOAD_FOLDER']),
            ("DETECTED", application.config['DETECTED_FOLDER']),
            ("REPORTS", application.config['REPORTS_FOLDER']),
            ("RECORDINGS", application.config['RECORDINGS_FOLDER']),
            ("BACKUP", application.config['BACKUP_FOLDER'])
        ]:
            logger.info(f"  {name}: {path} (exists: {os.path.exists(path)})")
        
        # Schedule periodic cleanup
        import threading
        def periodic_cleanup():
            while True:
                time.sleep(3600)  # Clean every hour
                cleanup_old_files()
        
        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()
        
        logger.info("Enhanced Smart Focus Alert ready")
        
    except Exception as e:
        logger.error(f"Enhanced startup error: {str(e)}")

if __name__ == "__main__":
    try:
        enhanced_startup()
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting Enhanced Smart Focus Alert on port {port}")
        application.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Enhanced application startup error: {str(e)}")
        traceback.print_exc()
