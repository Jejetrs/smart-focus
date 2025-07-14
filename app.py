from flask import Flask, render_template, request, Response, jsonify, send_file
from werkzeug.utils import secure_filename
import mediapipe as mp
import numpy as np
import pyttsx3
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
from pathlib import Path
import sys

# ===== ENVIRONMENT DETECTION & CONFIGURATION =====
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') == 'production' or 'railway' in os.environ.get('PYTHONPATH', '').lower()
IS_LOCAL = not IS_RAILWAY

print(f"🚀 Smart Focus Alert - Environment: {'Railway' if IS_RAILWAY else 'Local'}")

# Configure environment-specific settings
if IS_RAILWAY:
    print("⚙️  Configuring for Railway deployment...")
    # Disable GUI-related warnings for headless environment
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['DISPLAY'] = ''
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    # Disable matplotlib GUI backend
    import matplotlib
    matplotlib.use('Agg')
else:
    print("💻 Running in Local Environment")

# Environment-specific configurations
DETECTION_CONFIG = {
    'show_landmarks': True,         # Always show landmarks for consistency
    'enable_speech': IS_LOCAL,      # Only enable TTS locally (browser handles Railway)
    'debug_mode': IS_LOCAL,         # Debug output only locally
    'use_detailed_overlay': True,   # Always use detailed overlay
    'force_timer_display': True,    # Always show timer like local
    'alert_cooldown': 5,           # 5 seconds between alerts
    'thresholds': {
        'SLEEPING': 10,            # 10 seconds
        'YAWNING': 3.5,           # 3.5 seconds  
        'NOT FOCUSED': 10         # 10 seconds
    }
}

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = os.path.join(os.path.realpath('.'), 'static', 'uploads')
application.config['DETECTED_FOLDER'] = os.path.join(os.path.realpath('.'), 'static', 'detected')
application.config['REPORTS_FOLDER'] = os.path.join(os.path.realpath('.'), 'static', 'reports')
application.config['RECORDINGS_FOLDER'] = os.path.join(os.path.realpath('.'), 'static', 'recordings')
application.config['MAX_CONTENT_PATH'] = 10000000

# Create necessary directories
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Created directory: {folder}")

# Create persistent data directory for session history
DATA_DIR = Path(os.path.realpath('.')) / 'data'
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / 'session_history.json'

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

# Video recording variables
video_writer = None
recording_active = False

# ===== UTILITY FUNCTIONS =====
def save_session_data(session_data):
    """Save session data to persistent storage"""
    try:
        # Load existing history
        history = load_session_history()
        
        # Calculate session duration and stats
        duration_minutes = 0
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            duration_minutes = duration.total_seconds() / 60
        
        # Add current session
        session_record = {
            'session_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'start_time': session_data['start_time'].isoformat() if session_data['start_time'] else None,
            'end_time': session_data['end_time'].isoformat() if session_data['end_time'] else None,
            'duration_minutes': round(duration_minutes, 2),
            'total_detections': session_data['focus_statistics']['total_detections'],
            'total_alerts': len(session_data['alerts']),
            'total_persons': session_data['focus_statistics']['total_persons'],
            'focus_stats': session_data['focus_statistics'],
            'alerts_summary': {
                'sleeping': len([a for a in session_data['alerts'] if a.get('detection') == 'SLEEPING']),
                'yawning': len([a for a in session_data['alerts'] if a.get('detection') == 'YAWNING']),
                'unfocused': len([a for a in session_data['alerts'] if a.get('detection') == 'NOT FOCUSED'])
            }
        }
        
        history.append(session_record)
        
        # Keep only last 50 sessions
        if len(history) > 50:
            history = history[-50:]
        
        # Save to file
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
            
        if DETECTION_CONFIG['debug_mode']:
            print(f"💾 Session data saved: {len(session_data['alerts'])} alerts, {duration_minutes:.1f} min")
        
    except Exception as e:
        print(f"❌ Error saving session data: {e}")

def load_session_history():
    """Load session history from persistent storage"""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"❌ Error loading session history: {e}")
        return []

def draw_landmarks(image, landmarks, land_mark, color):
    """Draw landmarks on the image for a single face"""
    if not DETECTION_CONFIG['show_landmarks']:
        return
        
    height, width = image.shape[:2]
    for face in land_mark:
        try:
            point = landmarks.landmark[face]
            point_scale = ((int)(point.x * width), (int)(point.y * height))     
            cv.circle(image, point_scale, 2, color, 1)
        except Exception as e:
            if DETECTION_CONFIG['debug_mode']:
                print(f"⚠️  Landmark drawing error: {e}")

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

def detect_drowsiness(frame, landmarks, speech_engine=None):
    """Detect drowsiness and attention state based on eye aspect ratio and other metrics"""
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

    # Detect closed eyes
    ratio_left_eye = get_aspect_ratio(frame, landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
    ratio_right_eye = get_aspect_ratio(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
    eye_ratio = (ratio_left_eye + ratio_right_eye) / 2
    
    # Detect yawning
    ratio_lips = get_aspect_ratio(frame, landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
    
    # Check if iris is focused (looking at center/screen)
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
    """Detect persons in image or video frame with attention status"""
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
    
    if detection_results.detections:
        for i, detection in enumerate(detection_results.detections):
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
                    mesh_results.multi_face_landmarks[matched_face_idx],
                    None
                )
            
            status_text = attention_status.get("state", "FOCUSED")
            
            # ===== FORCE CONSISTENT DISPLAY FORMAT =====
            # Show detailed info like local version, not just coordinates
            if DETECTION_CONFIG['use_detailed_overlay']:
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
            
            # Extract face region with error handling
            try:
                face_img = image[y:y+h, x:x+w]
                if face_img.size > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    face_filename = f"person_{i+1}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
                    face_path = os.path.join(application.config['DETECTED_FOLDER'], face_filename)
                    cv.imwrite(face_path, face_img)
                    image_path = f"/static/detected/{face_filename}"
                else:
                    image_path = None
            except Exception as e:
                if DETECTION_CONFIG['debug_mode']:
                    print(f"⚠️  Face extraction error: {e}")
                image_path = None
            
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "image_path": image_path,
                "status": status_text,
                "timestamp": datetime.now().isoformat()
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
    
    # Group alerts by person and distraction type for proper accumulation
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
    
    # Calculate total time for each distraction type by SUMMING all durations
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
    
    # Update distraction times based on actual alert history
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
        story.append(Spacer(1, 20))
        
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
        
        # Focus Statistics
        story.append(Paragraph("Detailed Focus Statistics", heading_style))
        
        # Calculate corrected average focus metric
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
        
        # Alert History
        if session_data['alerts']:
            story.append(Paragraph("Alert History", heading_style))
            
            alert_headers = ['Time', 'Person', 'Detection', 'Duration', 'Message']
            alert_data = [alert_headers]
            
            for alert in session_data['alerts'][-20:]:  # Show last 20 alerts
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
                    alert['message'][:50] + '...' if len(alert['message']) > 50 else alert['message']
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
        story.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - Enhanced Railway Compatible Version<br/>Environment: {'Railway Cloud' if IS_RAILWAY else 'Local Development'}"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        print(f"📊 PDF report generated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")
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
            ['Total Persons Detected', str(len(detections))],
            ['Processing Environment', 'Railway Cloud' if IS_RAILWAY else 'Local Development']
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
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - File Analysis Report<br/>Environment: {'Railway Cloud' if IS_RAILWAY else 'Local Development'}"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        print(f"📊 Upload PDF report generated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error generating upload PDF report: {e}")
        return None

def process_video_file(video_path):
    """Process video file and detect persons in each frame"""
    try:
        cap = cv.VideoCapture(video_path)
        fps = cap.get(cv.CAP_PROP_FPS)
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
        
        print(f"🎬 Processing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                processed_frame, detections = detect_persons_with_attention(frame, mode="video")
                all_detections.extend(detections)
                if DETECTION_CONFIG['debug_mode'] and detections:
                    print(f"Frame {frame_count}: {len(detections)} detections")
            else:
                processed_frame = frame
                
            out.write(processed_frame)
        
        cap.release()
        out.release()
        
        print(f"✅ Video processing complete: {len(all_detections)} total detections")
        return output_path, all_detections
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None, []

def gen_frames():  
    """Generate frames for webcam streaming with person detection and state analysis - ENHANCED VERSION"""
    global live_monitoring_active, session_data, video_writer, recording_active
    
    # Face mesh parameters
    STATIC_IMAGE = False
    REFINE_LANDMARKS = True
    MAX_NO_FACES = 10
    DETECTION_CONFIDENCE = 0.6
    TRACKING_CONFIDENCE = 0.6

    # Landmark colors
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_MAGENTA = (255, 0, 255)

    # Landmark indices
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
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=STATIC_IMAGE,
        max_num_faces=MAX_NO_FACES,
        refine_landmarks=REFINE_LANDMARKS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    )
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    # Initialize text-to-speech dengan error handling untuk server environment
    speech = None
    if DETECTION_CONFIG['enable_speech']:
        try:
            speech = pyttsx3.init()
            speech.setProperty('rate', 150)
            if DETECTION_CONFIG['debug_mode']:
                print("🔊 TTS initialized successfully")
        except Exception as e:
            if DETECTION_CONFIG['debug_mode']:
                print(f"⚠️  TTS initialization failed (normal in server environment): {e}")
            speech = None
    
    # Initialize webcam
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Failed to open camera")
        return
    
    print("📹 Camera initialized for live monitoring")
    
    # Variables for time-based distraction detection
    person_state_timers = {}  # Track state duration for each person
    person_current_state = {}  # Track current state for each person
    last_alert_time = {}  # Track last alert time for cooldown
    alert_cooldown = DETECTION_CONFIG['alert_cooldown']
    
    # Distraction thresholds (in seconds)
    DISTRACTION_THRESHOLDS = DETECTION_CONFIG['thresholds']
    
    # Setup video recording if monitoring is active
    if live_monitoring_active and recording_active:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"session_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
        recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
        session_data['recording_path'] = recording_path
        
        # Get camera properties
        fps = camera.get(cv.CAP_PROP_FPS) or 30
        width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(recording_path, fourcc, fps, (width, height))
        print(f"🎬 Recording started: {recording_path}")
    
    frame_count = 0
    
    while live_monitoring_active:
        success, frame = camera.read()
        
        if not success:
            print("⚠️  Failed to read camera frame")
            break
        
        frame_count += 1
        
        # Convert to RGB for MediaPipe
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        
        # Detect persons
        detection_results = face_detection.process(image_rgb)
        mesh_results = face_mesh.process(image_rgb)
        
        # Process detected persons
        current_time = time.time()
        detected_persons = []
        frame_detections = []
        
        if detection_results.detections and mesh_results.multi_face_landmarks:
            num_persons = min(len(detection_results.detections), len(mesh_results.multi_face_landmarks))
            
            for i in range(num_persons):
                detection = detection_results.detections[i]
                face_landmarks = mesh_results.multi_face_landmarks[i]
                
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * img_w), int(bboxC.ymin * img_h), \
                            int(bboxC.width * img_w), int(bboxC.height * img_h)
                
                x = max(0, x)
                y = max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                
                # ===== ENHANCED LANDMARK DRAWING =====
                try:
                    if DETECTION_CONFIG['show_landmarks']:
                        draw_landmarks(frame, face_landmarks, FACE, COLOR_GREEN)
                        draw_landmarks(frame, face_landmarks, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
                        draw_landmarks(frame, face_landmarks, LEFT_EYE_LEFT_RIGHT, COLOR_RED)
                        draw_landmarks(frame, face_landmarks, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
                        draw_landmarks(frame, face_landmarks, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)
                        draw_landmarks(frame, face_landmarks, UPPER_LOWER_LIPS, COLOR_BLUE)
                        draw_landmarks(frame, face_landmarks, LEFT_RIGHT_LIPS, COLOR_BLUE)
                except Exception as e:
                    if DETECTION_CONFIG['debug_mode']:
                        print(f"⚠️  Landmark drawing error: {e}")
                
                # Create mesh points for iris detection
                mesh_points = []    
                for p in face_landmarks.landmark:
                    px = int(p.x * img_w)
                    py = int(p.y * img_h)
                    mesh_points.append((px, py))
                mesh_points = np.array(mesh_points)            
                
                left_eye_points = mesh_points[LEFT_EYE]
                right_eye_points = mesh_points[RIGHT_EYE]
                left_iris_points = mesh_points[LEFT_IRIS]
                right_iris_points = mesh_points[RIGHT_IRIS]

                # Draw iris circles dengan error handling
                try:
                    if DETECTION_CONFIG['show_landmarks']:
                        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
                        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
                        center_left = np.array([l_cx, l_cy], dtype=np.int32)
                        center_right = np.array([r_cx, r_cy], dtype=np.int32)
                        cv.circle(frame, center_left, int(l_radius), COLOR_MAGENTA, 1, cv.LINE_AA)
                        cv.circle(frame, center_right, int(r_radius), COLOR_MAGENTA, 1, cv.LINE_AA)
                except Exception as e:
                    if DETECTION_CONFIG['debug_mode']:
                        print(f"⚠️  Iris circle drawing error: {e}")
                
                # Analyze facial state
                status, state = detect_drowsiness(frame, face_landmarks)
                
                person_key = f"person_{i+1}"
                
                # Initialize person tracking if not exists
                if person_key not in person_state_timers:
                    person_state_timers[person_key] = {}
                    person_current_state[person_key] = None
                    last_alert_time[person_key] = 0
                
                # Update state timing
                if person_current_state[person_key] != state:
                    # State changed, reset all timers and set new state
                    person_state_timers[person_key] = {}
                    person_current_state[person_key] = state
                    person_state_timers[person_key][state] = current_time
                else:
                    # Same state continues, update timer if not exists
                    if state not in person_state_timers[person_key]:
                        person_state_timers[person_key][state] = current_time
                
                # Check for distraction alerts
                should_alert = False
                alert_message = ""
                distraction_duration = 0
                
                if state in DISTRACTION_THRESHOLDS:
                    if state in person_state_timers[person_key]:
                        distraction_duration = current_time - person_state_timers[person_key][state]
                        
                        # Check if distraction duration exceeds threshold
                        if distraction_duration >= DISTRACTION_THRESHOLDS[state]:
                            # Check cooldown
                            if current_time - last_alert_time[person_key] >= alert_cooldown:
                                should_alert = True
                                last_alert_time[person_key] = current_time
                                
                                # Set appropriate alert message
                                if state == 'SLEEPING':
                                    alert_message = f'Person {i+1} is sleeping - please wake up!'
                                elif state == 'YAWNING':
                                    alert_message = f'Person {i+1} is yawning - please take a rest!'
                                elif state == 'NOT FOCUSED':
                                    alert_message = f'Person {i+1} is not focused - please focus on screen!'
                                
                                # Record alert in session data
                                session_data['alerts'].append({
                                    'timestamp': datetime.now().isoformat(),
                                    'person': f"Person {i+1}",
                                    'detection': state,
                                    'message': alert_message,
                                    'duration': int(distraction_duration)
                                })
                                
                                if DETECTION_CONFIG['debug_mode']:
                                    print(f"🚨 Alert: {alert_message} (Duration: {int(distraction_duration)}s)")
                
                # Draw rectangle with color based on state
                state_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (255, 165, 0),
                    "YAWNING": (255, 255, 0),
                    "SLEEPING": (0, 0, 255)
                }
                
                color = state_colors.get(state, (0, 255, 0))
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # ===== ENHANCED TEXT DISPLAY WITH TIMER =====
                state_text = f"Person {i+1}: {state}"
                
                # ALWAYS show timer when in distraction state (force consistency)
                if DETECTION_CONFIG['force_timer_display'] and state in DISTRACTION_THRESHOLDS and state in person_state_timers[person_key]:
                    duration = current_time - person_state_timers[person_key][state]
                    threshold = DISTRACTION_THRESHOLDS[state]
                    state_text += f" ({int(duration)}s/{threshold}s)"
                
                # Display text dengan font yang konsisten
                cv.putText(frame, state_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Store detection for session data (only store focused states and confirmed distractions)
                if state == 'FOCUSED' or (state in DISTRACTION_THRESHOLDS and 
                                        state in person_state_timers[person_key] and 
                                        current_time - person_state_timers[person_key][state] >= DISTRACTION_THRESHOLDS[state]):
                    frame_detections.append({
                        "id": i+1,
                        "confidence": float(detection.score[0]),
                        "bbox": [x, y, w, h],
                        "status": state,
                        "timestamp": datetime.now().isoformat()
                    })
                
                detected_persons.append({
                    "id": i+1,
                    "state": state,
                    "status": status,
                    "duration": distraction_duration if state in DISTRACTION_THRESHOLDS else 0
                })
                
                # Give voice alert only (dengan error handling untuk server)
                if should_alert and speech:
                    try:
                        speech.say(alert_message)
                        speech.runAndWait()
                    except Exception as e:
                        if DETECTION_CONFIG['debug_mode']:
                            print(f"⚠️  Alert error (normal in server environment): {e}")
        
        # Update session statistics
        if frame_detections:
            update_session_statistics(frame_detections)
        
        # Display summary
        cv.putText(frame, f"Persons detected: {len(detected_persons)}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if detected_persons:
            y_offset = 60
            for person in detected_persons:
                state_color = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (255, 165, 0),
                    "YAWNING": (255, 255, 0),
                    "SLEEPING": (0, 0, 255)
                }.get(person["state"], (255, 255, 255))
                
                duration_text = ""
                if person["duration"] > 0:
                    duration_text = f" ({int(person['duration'])}s)"
                
                cv.putText(frame, f"P{person['id']}: {person['state']}{duration_text}", 
                          (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
                y_offset += 25
        
        # Record frame if recording is active
        if recording_active and video_writer:
            video_writer.write(frame)
        
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Cleanup
    print("🛑 Stopping camera and cleaning up...")
    camera.release()
    if video_writer:
        video_writer.release()
        video_writer = None

# ===== ROUTES =====
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
                # Process image
                print(f"📸 Processing image: {filename}")
                image = cv.imread(file_path)
                processed_image, detections = detect_persons_with_attention(image)
                
                # Save processed image
                output_filename = f"processed_{filename}"
                output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
                cv.imwrite(output_path, processed_image)
                
                result["processed_image"] = f"/static/detected/{output_filename}"
                result["detections"] = detections
                result["type"] = "image"
                
                # Generate PDF report
                pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                file_info = {
                    'filename': filename,
                    'type': file_ext.upper()
                }
                
                generate_upload_pdf_report(detections, file_info, pdf_path)
                result["pdf_report"] = f"/static/reports/{pdf_filename}"
                
                print(f"✅ Image processed: {len(detections)} detections found")
                
            elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                # Process video
                print(f"🎬 Processing video: {filename}")
                output_path, detections = process_video_file(file_path)
                
                if output_path:
                    result["processed_video"] = f"/static/detected/{os.path.basename(output_path)}"
                    result["detections"] = detections
                    result["type"] = "video"
                    
                    # Generate PDF report
                    pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                    
                    file_info = {
                        'filename': filename,
                        'type': file_ext.upper()
                    }
                    
                    generate_upload_pdf_report(detections, file_info, pdf_path)
                    result["pdf_report"] = f"/static/reports/{pdf_filename}"
                    
                    print(f"✅ Video processed: {len(detections)} detections found")
                else:
                    result["error"] = "Failed to process video file"
            
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

@application.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, session_data, recording_active
    
    if live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring already active"})
    
    # Reset session data
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
    
    live_monitoring_active = True
    recording_active = True
    
    print("🚀 Live monitoring started")
    return jsonify({"status": "success", "message": "Monitoring started"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, session_data, recording_active, video_writer
    
    if not live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring not active"})
    
    live_monitoring_active = False
    recording_active = False
    session_data['end_time'] = datetime.now()
    
    # Stop video recording
    if video_writer:
        video_writer.release()
        video_writer = None
    
    # Save session data to persistent storage
    save_session_data(session_data)
    
    # Generate PDF report
    pdf_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
    generate_pdf_report(session_data, pdf_path)
    
    response_data = {
        "status": "success", 
        "message": "Monitoring stopped",
        "pdf_report": f"/static/reports/{pdf_filename}",
        "total_alerts": len(session_data['alerts']),
        "session_duration": str(session_data['end_time'] - session_data['start_time']).split('.')[0]
    }
    
    if session_data['recording_path']:
        response_data["video_file"] = f"/static/recordings/{os.path.basename(session_data['recording_path'])}"
    
    print(f"🛑 Monitoring stopped. {len(session_data['alerts'])} alerts generated.")
    return jsonify(response_data)

# ===== ENHANCED ENDPOINT - Get Monitoring Data dengan Client Support =====
@application.route('/get_monitoring_data')
def get_monitoring_data():
    global session_data
    
    if not live_monitoring_active:
        return jsonify({"error": "Monitoring not active"})
    
    # Get recent alerts (last 5)
    recent_alerts = session_data['alerts'][-5:] if session_data['alerts'] else []
    
    # Format alerts for frontend
    formatted_alerts = []
    for alert in recent_alerts:
        try:
            alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
        except:
            alert_time = alert['timestamp']
        
        formatted_alerts.append({
            'time': alert_time,
            'message': alert['message'],
            'type': 'warning' if alert['detection'] in ['YAWNING', 'NOT FOCUSED'] else 'error',
            'duration': alert.get('duration', 0),
            'person': alert.get('person', 'Unknown'),
            'detection': alert.get('detection', 'Unknown')
        })
    
    # Calculate current focus state from recent detections
    recent_detections = session_data['detections'][-10:] if session_data['detections'] else []
    current_status = 'READY'
    focused_count = 0
    total_persons = 0
    
    if recent_detections:
        # Get latest detection states
        latest_states = {}
        for detection in reversed(recent_detections):
            person_id = detection['id']
            if person_id not in latest_states:
                latest_states[person_id] = detection['status']
        
        total_persons = len(latest_states)
        focused_count = sum(1 for state in latest_states.values() if state == 'FOCUSED')
        
        # Determine overall status
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
        'latest_alerts': formatted_alerts,
        'session_stats': session_data['focus_statistics'],
        'environment': 'Railway' if IS_RAILWAY else 'Local',
        'is_active': live_monitoring_active
    })

@application.route('/monitoring_status')
def monitoring_status():
    return jsonify({
        "is_active": live_monitoring_active,
        "environment": "Railway" if IS_RAILWAY else "Local",
        "total_alerts": len(session_data['alerts']) if session_data else 0
    })

@application.route('/get_session_history')
def get_session_history():
    """Get session history for display"""
    history = load_session_history()
    return jsonify({
        "history": history,
        "total_sessions": len(history),
        "environment": "Railway" if IS_RAILWAY else "Local"
    })

# ===== NEW ENDPOINT - Save Client Alert =====
@application.route('/save_client_alert', methods=['POST'])
def save_client_alert():
    """Save alert generated from client-side processing"""
    global session_data
    
    try:
        if not live_monitoring_active:
            return jsonify({"error": "Monitoring not active"}), 400
            
        alert_data = request.get_json()
        
        # Validate required fields
        required_fields = ['person', 'detection', 'message', 'duration']
        for field in required_fields:
            if field not in alert_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create alert object
        alert = {
            'timestamp': alert_data.get('timestamp', datetime.now().isoformat()),
            'person': alert_data['person'],
            'detection': alert_data['detection'],
            'message': alert_data['message'],
            'duration': int(alert_data['duration'])
        }
        
        # Add to session alerts
        session_data['alerts'].append(alert)
        
        # Update session statistics
        update_session_statistics([])  # Trigger stats update
        
        if DETECTION_CONFIG['debug_mode']:
            print(f"💾 Client alert saved: {alert['person']} - {alert['detection']} ({alert['duration']}s)")
        
        return jsonify({
            "status": "success",
            "message": "Alert saved successfully",
            "total_alerts": len(session_data['alerts']),
            "alert_id": alert.get('timestamp', 'unknown')
        })
        
    except Exception as e:
        print(f"❌ Error saving client alert: {e}")
        return jsonify({"error": str(e)}), 500

# ===== ENHANCED ENDPOINT - Process Frame dengan Alert Integration =====
@application.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame from browser user with Alert Integration"""
    global session_data
    
    try:
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        # Process detection like usual
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        # ===== INTEGRATE WITH SESSION DATA FOR RAILWAY =====
        if detections and live_monitoring_active:
            # Update session data with current detections
            current_time = datetime.now()
            
            for detection in detections:
                # Add detection to session for consistency with server monitoring
                session_detection = {
                    "id": detection["id"],
                    "confidence": detection.get("confidence", 0.0),
                    "bbox": detection.get("bbox", [0, 0, 0, 0]),
                    "status": detection["status"],
                    "timestamp": current_time.isoformat()
                }
                
                # Only add focused states and distractions for session tracking
                if detection["status"] == 'FOCUSED' or detection["status"] in ['SLEEPING', 'YAWNING', 'NOT FOCUSED']:
                    session_data['detections'].append(session_detection)
            
            # Update session statistics
            update_session_statistics(detections)
        
        # Send back to browser
        _, buffer = cv.imencode('.jpg', processed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections,
            "environment": "Railway" if IS_RAILWAY else "Local",
            "session_active": live_monitoring_active,
            "frame_count": len(session_data['detections']) if session_data else 0
        })
        
    except Exception as e:
        print(f"❌ Frame processing error: {e}")
        return jsonify({
            "error": str(e),
            "environment": "Railway" if IS_RAILWAY else "Local"
        }), 500

@application.route('/api/detect', methods=['POST'])
def api_detect():
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
        processed_image, detections = detect_persons_with_attention(image)
        
        output_filename = f"processed_{filename}"
        output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
        cv.imwrite(output_path, processed_image)
        
        return jsonify({
            "type": "image",
            "processed_image": f"/static/detected/{output_filename}",
            "detections": detections,
            "environment": "Railway" if IS_RAILWAY else "Local"
        })
        
    elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
        output_path, detections = process_video_file(file_path)
        
        if output_path:
            return jsonify({
                "type": "video",
                "processed_video": f"/static/detected/{os.path.basename(output_path)}",
                "detections": detections,
                "environment": "Railway" if IS_RAILWAY else "Local"
            })
        else:
            return jsonify({"error": "Failed to process video"}), 500
    
    return jsonify({"error": "Unsupported file format"}), 400

@application.route('/check_camera')
def check_camera():
    """Check if camera is available in server environment"""
    try:
        camera = cv.VideoCapture(0)
        is_available = camera.isOpened()
        camera.release()
        
        if DETECTION_CONFIG['debug_mode']:
            print(f"📹 Camera availability check: {is_available}")
            
        return jsonify({
            "camera_available": is_available,
            "environment": "Railway" if IS_RAILWAY else "Local",
            "message": "Server camera ready" if is_available else "Server camera not available - client camera will be used"
        })
    except Exception as e:
        if DETECTION_CONFIG['debug_mode']:
            print(f"❌ Camera check error: {e}")
        return jsonify({
            "camera_available": False,
            "error": str(e),
            "environment": "Railway" if IS_RAILWAY else "Local",
            "message": "Camera check failed - client camera will be used"
        })

@application.route('/health')
def health_check():
    """Health check endpoint untuk Railway"""
    return jsonify({
        "status": "healthy",
        "environment": "Railway" if IS_RAILWAY else "Local",
        "mediapipe_available": True,
        "opencv_version": cv.__version__ if hasattr(cv, '__version__') else "unknown",
        "mediapipe_version": mp.__version__ if hasattr(mp, '__version__') else "unknown",
        "monitoring_active": live_monitoring_active,
        "total_alerts": len(session_data['alerts']) if session_data else 0,
        "config": DETECTION_CONFIG
    })

@application.route('/environment')
def environment_info():
    """Get environment information for debugging"""
    return jsonify({
        "is_railway": IS_RAILWAY,
        "is_local": IS_LOCAL,
        "detection_config": DETECTION_CONFIG,
        "opencv_version": cv.__version__ if hasattr(cv, '__version__') else "unknown",
        "mediapipe_version": mp.__version__ if hasattr(mp, '__version__') else "unknown",
        "python_version": sys.version,
        "monitoring_active": live_monitoring_active,
        "session_alerts": len(session_data['alerts']) if session_data else 0,
        "environment_variables": {
            "RAILWAY_ENVIRONMENT": os.environ.get('RAILWAY_ENVIRONMENT'),
            "MEDIAPIPE_DISABLE_GPU": os.environ.get('MEDIAPIPE_DISABLE_GPU'),
            "OPENCV_LOG_LEVEL": os.environ.get('OPENCV_LOG_LEVEL'),
            "PORT": os.environ.get('PORT')
        }
    })

# Error handlers untuk production
@application.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "environment": "Railway" if IS_RAILWAY else "Local"}), 404

@application.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error", 
        "environment": "Railway" if IS_RAILWAY else "Local",
        "message": "Check logs for details"
    }), 500

if __name__ == "__main__":
    # Environment specific configuration
    port = int(os.environ.get('PORT', 5000))
    debug = IS_LOCAL  # Only enable debug in local environment
    
    print(f"🌍 Environment: {'Railway Cloud' if IS_RAILWAY else 'Local Development'}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug mode: {debug}")
    print(f"⚙️  Detection config: {DETECTION_CONFIG}")
    print(f"📹 Camera support: {'Client-side' if IS_RAILWAY else 'Server + Client'}")
    print(f"🔊 Audio support: {'Browser API' if IS_RAILWAY else 'Server + Browser'}")
    print(f"💾 Data directory: {DATA_DIR}")
    print("=" * 60)
    
    application.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug,
        threaded=True
    )
