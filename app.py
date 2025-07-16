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
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import tempfile
import shutil

# Initialize Flask app
application = Flask(__name__)

# FIXED Configuration for Railway deployment with ephemeral storage
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Use /tmp for ephemeral storage on Railway
BASE_DIR = '/tmp/smart_focus_app'
application.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
application.config['DETECTED_FOLDER'] = os.path.join(BASE_DIR, 'detected')
application.config['REPORTS_FOLDER'] = os.path.join(BASE_DIR, 'reports')
application.config['RECORDINGS_FOLDER'] = os.path.join(BASE_DIR, 'recordings')

# Create necessary directories in ephemeral storage
def ensure_directories():
    """Ensure all required directories exist"""
    folders = [
        application.config['UPLOAD_FOLDER'], 
        application.config['DETECTED_FOLDER'],
        application.config['REPORTS_FOLDER'], 
        application.config['RECORDINGS_FOLDER']
    ]
    
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            # Test write access
            test_file = os.path.join(folder, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"Directory ready: {folder}")
        except Exception as e:
            print(f"Error creating directory {folder}: {e}")

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
    """Enhanced person detection with proper overlay information"""
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
            
            # ENHANCED DRAWING FOR LIVE RECORDING
            duration = 0
            if mode == "video" and live_monitoring_active:
                # Timer tracking logic
                if person_key not in person_state_timers:
                    person_state_timers[person_key] = {}
                    person_current_state[person_key] = None
                    last_alert_time[person_key] = 0
                
                if person_current_state[person_key] != status_text:
                    person_state_timers[person_key] = {}
                    person_current_state[person_key] = status_text
                    person_state_timers[person_key][status_text] = current_time
                else:
                    if status_text not in person_state_timers[person_key]:
                        person_state_timers[person_key][status_text] = current_time
                
                if status_text in person_state_timers[person_key]:
                    duration = current_time - person_state_timers[person_key][status_text]
            
            # ENHANCED OVERLAY DRAWING FOR RECORDING
            status_colors = {
                "FOCUSED": (0, 255, 0),      # Green
                "NOT FOCUSED": (0, 165, 255), # Orange
                "YAWNING": (0, 255, 255),    # Yellow  
                "SLEEPING": (0, 0, 255)      # Red
            }
            
            main_color = status_colors.get(status_text, (0, 255, 0))
            
            # Draw main bounding box
            cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
            
            # ENHANCED INFO OVERLAY for recording
            if mode == "video":
                # Create info panel
                panel_height = 120
                panel_width = max(300, w + 50)
                panel_x = max(10, min(x, iw - panel_width - 10))
                panel_y = max(10, y - panel_height - 10) if y > panel_height + 20 else y + h + 10
                
                # Semi-transparent background
                overlay = image.copy()
                cv.rectangle(overlay, 
                            (panel_x, panel_y), 
                            (panel_x + panel_width, panel_y + panel_height), 
                            (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.8, image, 0.2, 0, image)
                
                # Border
                cv.rectangle(image, 
                            (panel_x, panel_y), 
                            (panel_x + panel_width, panel_y + panel_height), 
                            main_color, 2)
                
                # Text information
                font = cv.FONT_HERSHEY_SIMPLEX
                text_y = panel_y + 25
                line_height = 20
                
                # Person ID
                cv.putText(image, f"Person {i+1}", 
                          (panel_x + 10, text_y), font, 0.7, main_color, 2)
                
                # Status with duration
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    status_info = f"Status: {status_text} ({duration:.1f}s/{threshold}s)"
                else:
                    status_info = f"Status: {status_text}"
                
                cv.putText(image, status_info, 
                          (panel_x + 10, text_y + line_height), font, 0.5, (255, 255, 255), 1)
                
                # Confidence
                cv.putText(image, f"Confidence: {confidence_score*100:.1f}%", 
                          (panel_x + 10, text_y + 2*line_height), font, 0.5, (255, 255, 255), 1)
                
                # Position
                cv.putText(image, f"Position: ({x},{y}) Size: {w}x{h}", 
                          (panel_x + 10, text_y + 3*line_height), font, 0.5, (255, 255, 255), 1)
                
                # Timestamp
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                cv.putText(image, f"Time: {timestamp_str}", 
                          (panel_x + 10, text_y + 4*line_height), font, 0.5, (200, 200, 200), 1)
            
            # Alert checking
            should_alert = False
            alert_message = ""
            
            if (mode == "video" and live_monitoring_active and status_text in DISTRACTION_THRESHOLDS and 
                person_key in person_state_timers and status_text in person_state_timers[person_key]):
                
                if duration >= DISTRACTION_THRESHOLDS[status_text]:
                    alert_cooldown = 5
                    if current_time - last_alert_time[person_key] >= alert_cooldown:
                        should_alert = True
                        last_alert_time[person_key] = current_time
                        
                        if status_text == 'SLEEPING':
                            alert_message = f'Person {i+1} is sleeping - please wake up!'
                        elif status_text == 'YAWNING':
                            alert_message = f'Person {i+1} is yawning - please take a rest!'
                        elif status_text == 'NOT FOCUSED':
                            alert_message = f'Person {i+1} is not focused - please focus on screen!'
                        
                        if live_monitoring_active:
                            session_data['alerts'].append({
                                'timestamp': datetime.now().isoformat(),
                                'person': f"Person {i+1}",
                                'detection': status_text,
                                'message': alert_message,
                                'duration': int(duration)
                            })
            
            # Save face region
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
    
    # Add session info overlay for recording
    if mode == "video" and live_monitoring_active:
        # Top info bar
        info_height = 60
        overlay = image.copy()
        cv.rectangle(overlay, (0, 0), (iw, info_height), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Session information
        session_duration = int(time.time() - session_data['start_time'].timestamp()) if session_data['start_time'] else 0
        hours = session_duration // 3600
        minutes = (session_duration % 3600) // 60
        seconds = session_duration % 60
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        cv.putText(image, f"LIVE RECORDING - Duration: {duration_str}", 
                  (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(image, f"Persons: {len(detections)} | Alerts: {len(session_data['alerts'])}", 
                  (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Detection count overlay
    detection_text = f"Persons detected: {len(detections)}" if detections else "No persons detected"
    cv.putText(image, detection_text, (10, ih - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return image, detections

def generate_enhanced_pdf_report(session_data, output_path):
    """FIXED - Generate comprehensive PDF report"""
    try:
        ensure_directories()
        
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                              leftMargin=inch, rightMargin=inch,
                              topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2563EB'),
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#64748B')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=15,
            spaceBefore=25,
            textColor=colors.HexColor('#1F2937'),
            fontName='Helvetica-Bold'
        )
        
        # Title and header
        story.append(Paragraph("🎯 Smart Focus Alert", title_style))
        story.append(Paragraph("Comprehensive Session Analysis Report", subtitle_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary Box
        summary_data = []
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            total_session_seconds = duration.total_seconds()
            duration_str = str(duration).split('.')[0]
        else:
            total_session_seconds = 0
            duration_str = "N/A"
        
        # Calculate focus accuracy
        distraction_times = calculate_distraction_time_from_alerts(session_data['alerts'])
        total_distraction_time = sum(distraction_times.values())
        
        if total_session_seconds > 0:
            focused_time = max(0, total_session_seconds - total_distraction_time)
            focus_accuracy = (focused_time / total_session_seconds) * 100
        else:
            focused_time = 0
            focus_accuracy = 0
        
        # Focus rating
        if focus_accuracy >= 90:
            focus_rating = "🌟 Excellent"
            rating_color = colors.HexColor('#10B981')
        elif focus_accuracy >= 75:
            focus_rating = "✅ Good"
            rating_color = colors.HexColor('#3B82F6')
        elif focus_accuracy >= 60:
            focus_rating = "⚠️ Fair"
            rating_color = colors.HexColor('#F59E0B')
        else:
            focus_rating = "❌ Poor"
            rating_color = colors.HexColor('#EF4444')
        
        # Executive Summary
        story.append(Paragraph("📊 Executive Summary", heading_style))
        
        summary_table_data = [
            ['📅 Session Date', session_data.get('start_time', datetime.now()).strftime('%B %d, %Y')],
            ['⏰ Session Time', f"{session_data.get('start_time', datetime.now()).strftime('%I:%M %p')} - {session_data.get('end_time', datetime.now()).strftime('%I:%M %p')}"],
            ['⌛ Total Duration', duration_str],
            ['🎯 Focus Accuracy', f"{focus_accuracy:.1f}%"],
            ['⭐ Focus Rating', focus_rating],
            ['👥 Max Persons Detected', str(session_data['focus_statistics']['total_persons'])],
            ['🚨 Total Alerts Generated', str(len(session_data['alerts']))],
        ]
        
        summary_table = Table(summary_table_data, colWidths=[2.5*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            # Highlight focus rating row
            ('BACKGROUND', (0, 4), (-1, 4), rating_color),
            ('TEXTCOLOR', (0, 4), (-1, 4), colors.white),
            ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 30))
        
        # Detailed Statistics
        story.append(Paragraph("📈 Detailed Performance Metrics", heading_style))
        
        def format_time(seconds):
            if seconds < 60:
                return f"{int(seconds)}s"
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        
        stats_table_data = [
            ['Metric', 'Value', 'Percentage'],
            ['🕐 Total Session Time', format_time(total_session_seconds), '100%'],
            ['✅ Focused Time', format_time(focused_time), f'{(focused_time/total_session_seconds*100):.1f}%' if total_session_seconds > 0 else '0%'],
            ['😴 Sleeping Time', format_time(distraction_times.get('sleeping_time', 0)), f'{(distraction_times.get("sleeping_time", 0)/total_session_seconds*100):.1f}%' if total_session_seconds > 0 else '0%'],
            ['🥱 Yawning Time', format_time(distraction_times.get('yawning_time', 0)), f'{(distraction_times.get("yawning_time", 0)/total_session_seconds*100):.1f}%' if total_session_seconds > 0 else '0%'],
            ['👀 Unfocused Time', format_time(distraction_times.get('unfocused_time', 0)), f'{(distraction_times.get("unfocused_time", 0)/total_session_seconds*100):.1f}%' if total_session_seconds > 0 else '0%'],
        ]
        
        stats_table = Table(stats_table_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 30))
        
        # Alert Analysis
        if session_data['alerts']:
            story.append(Paragraph("🚨 Alert Analysis", heading_style))
            
            # Alert summary
            alert_types = {}
            for alert in session_data['alerts']:
                alert_type = alert.get('detection', 'Unknown')
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            alert_summary = "Alert Distribution: "
            alert_parts = []
            for alert_type, count in alert_types.items():
                emoji = {'SLEEPING': '😴', 'YAWNING': '🥱', 'NOT FOCUSED': '👀'}.get(alert_type, '⚠️')
                alert_parts.append(f"{emoji} {alert_type}: {count}")
            alert_summary += " | ".join(alert_parts)
            
            story.append(Paragraph(alert_summary, styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Recent alerts table
            story.append(Paragraph("Recent Alerts (Last 10)", ParagraphStyle(
                'SubHeading', parent=styles['Normal'], fontSize=14, 
                fontName='Helvetica-Bold', spaceAfter=10
            )))
            
            recent_alerts = session_data['alerts'][-10:]
            alert_table_data = [['Time', 'Person', 'Alert Type', 'Duration']]
            
            for alert in recent_alerts:
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                except:
                    alert_time = alert['timestamp']
                
                alert_table_data.append([
                    alert_time,
                    alert.get('person', 'Unknown'),
                    alert.get('detection', 'Unknown'),
                    f"{alert.get('duration', 0)}s"
                ])
            
            alert_table = Table(alert_table_data, colWidths=[1.2*inch, 1.5*inch, 1.8*inch, 1*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DC2626')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FEF2F2')]),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(alert_table)
        else:
            story.append(Paragraph("🎉 Excellent! No alerts were generated during this session.", styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Recommendations
        story.append(Paragraph("💡 Recommendations", heading_style))
        
        recommendations = []
        if focus_accuracy >= 90:
            recommendations.append("✅ Excellent focus! Keep maintaining this high level of attention.")
        elif focus_accuracy >= 75:
            recommendations.append("👍 Good focus overall. Try to minimize brief distractions.")
        else:
            recommendations.append("⚠️ Focus needs improvement. Consider breaks and environmental adjustments.")
        
        if distraction_times.get('sleeping_time', 0) > 60:
            recommendations.append("😴 Consider taking regular breaks to avoid fatigue.")
        
        if distraction_times.get('yawning_time', 0) > 30:
            recommendations.append("🥱 Monitor for signs of tiredness and ensure adequate rest.")
        
        if len(session_data['alerts']) > 10:
            recommendations.append("🔄 High alert frequency suggests need for environment optimization.")
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Footer
        footer_text = f"""
        <para align="center" fontSize="10" textColor="#64748B">
        Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
        Smart Focus Alert System - Advanced AI-Powered Focus Monitoring<br/>
        🔒 Confidential Session Data
        </para>
        """
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        if os.path.exists(output_path):
            print(f"Enhanced PDF report created: {output_path}")
            return output_path
        else:
            print(f"Failed to create PDF report: {output_path}")
            return None
            
    except Exception as e:
        print(f"Error generating enhanced PDF report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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

# FIXED - File serving routes with proper Railway ephemeral storage handling
@application.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from ephemeral storage"""
    try:
        file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(application.config['UPLOAD_FOLDER'], filename)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        print(f"Error serving upload file: {str(e)}")
        return jsonify({"error": "File access error"}), 500

@application.route('/static/detected/<filename>')
def detected_file(filename):
    """Serve detected/processed files from ephemeral storage"""
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
    """FIXED - Serve report files from ephemeral storage"""
    try:
        file_path = os.path.join(application.config['REPORTS_FOLDER'], filename)
        print(f"Looking for report file: {file_path}")
        
        if os.path.exists(file_path):
            print(f"Report file found, serving: {file_path}")
            return send_file(file_path, 
                           as_attachment=True, 
                           download_name=filename,
                           mimetype='application/pdf')
        else:
            print(f"Report file not found: {file_path}")
            # List available files for debugging
            if os.path.exists(application.config['REPORTS_FOLDER']):
                available_files = os.listdir(application.config['REPORTS_FOLDER'])
                print(f"Available files in reports folder: {available_files}")
            return jsonify({"error": "Report file not found"}), 404
    except Exception as e:
        print(f"Error serving report file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error accessing report file"}), 500

@application.route('/static/recordings/<filename>')
def recording_file(filename):
    """FIXED - Serve recording files from ephemeral storage"""
    try:
        file_path = os.path.join(application.config['RECORDINGS_FOLDER'], filename)
        print(f"Looking for recording file: {file_path}")
        
        if os.path.exists(file_path):
            print(f"Recording file found, serving: {file_path}")
            return send_file(file_path, 
                           as_attachment=True, 
                           download_name=filename,
                           mimetype='video/mp4')
        else:
            print(f"Recording file not found: {file_path}")
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
                    
                    # Generate enhanced PDF report
                    pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                    
                    # Create mock session data for upload report
                    mock_session = {
                        'start_time': datetime.now(),
                        'end_time': datetime.now(),
                        'detections': detections,
                        'alerts': [],
                        'focus_statistics': {
                            'unfocused_time': 0,
                            'yawning_time': 0,
                            'sleeping_time': 0,
                            'total_persons': len(detections),
                            'total_detections': len(detections)
                        }
                    }
                    
                    pdf_result = generate_enhanced_pdf_report(mock_session, pdf_path)
                    
                    if pdf_result and os.path.exists(pdf_path):
                        result["pdf_report"] = f"/static/reports/{pdf_filename}"
            
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, session_data, person_state_timers, person_current_state, last_alert_time
    
    if live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring already active"})
    
    ensure_directories()
    
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
    
    # Reset person tracking
    person_state_timers = {}
    person_current_state = {}
    last_alert_time = {}
    
    live_monitoring_active = True
    
    return jsonify({"status": "success", "message": "Monitoring started"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, session_data
    
    if not live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring not active"})
    
    live_monitoring_active = False
    session_data['end_time'] = datetime.now()
    
    # Generate enhanced PDF report
    pdf_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
    
    print(f"Generating PDF report at: {pdf_path}")
    pdf_result = generate_enhanced_pdf_report(session_data, pdf_path)
    
    response_data = {
        "status": "success", 
        "message": "Monitoring stopped"
    }
    
    if pdf_result and os.path.exists(pdf_path):
        response_data["pdf_report"] = f"/static/reports/{pdf_filename}"
        print(f"PDF report available at: {response_data['pdf_report']}")
    else:
        print("PDF report generation failed")
    
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
    try:
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400
        
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

# FIXED - Add video blob upload endpoint for client-side recording
@application.route('/upload_recording', methods=['POST'])
def upload_recording():
    """Handle client-side recording upload"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_recording_{timestamp}.webm"
        file_path = os.path.join(application.config['RECORDINGS_FOLDER'], filename)
        
        # Save the video file
        video_file.save(file_path)
        
        if os.path.exists(file_path):
            session_data['recording_path'] = file_path
            return jsonify({
                "success": True,
                "message": "Recording saved successfully",
                "file_path": f"/static/recordings/{filename}"
            })
        else:
            return jsonify({"error": "Failed to save recording"}), 500
            
    except Exception as e:
        print(f"Error uploading recording: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Health check for Railway
@application.route('/health')
def health_check():
    ensure_directories()
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "storage_ready": all(os.path.exists(folder) for folder in [
            application.config['UPLOAD_FOLDER'],
            application.config['DETECTED_FOLDER'],
            application.config['REPORTS_FOLDER'],
            application.config['RECORDINGS_FOLDER']
        ])
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)
