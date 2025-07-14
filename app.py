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

# Configuration for Railway deployment
application.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
application.config['UPLOAD_FOLDER'] = os.path.join(os.path.realpath('.'), 'static', 'uploads')
application.config['DETECTED_FOLDER'] = os.path.join(os.path.realpath('.'), 'static', 'detected')
application.config['REPORTS_FOLDER'] = os.path.join(os.path.realpath('.'), 'static', 'reports')
application.config['RECORDINGS_FOLDER'] = os.path.join(os.path.realpath('.'), 'static', 'recordings')
application.config['MAX_CONTENT_PATH'] = 10000000

# Create necessary directories
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}")

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
    """Detect persons in image or video frame with attention status and timer display"""
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
            
            # TIMER TRACKING: Update state timing for each person
            if mode == "video" and live_monitoring_active:  # Only for live monitoring
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
            duration = 0
            if mode == "video" and live_monitoring_active and person_key in person_state_timers:
                if status_text in person_state_timers[person_key]:
                    duration = current_time - person_state_timers[person_key][status_text]
            
            # ENHANCED BOUNDING BOX with timer display
            status_colors = {
                "FOCUSED": (0, 255, 0),      # Green
                "NOT FOCUSED": (0, 165, 255), # Orange
                "YAWNING": (0, 255, 255),    # Yellow  
                "SLEEPING": (0, 0, 255)      # Red
            }
            
            main_color = status_colors.get(status_text, (0, 255, 0))
            
            # Draw thick main rectangle
            cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
            
            # TIMER DISPLAY: Create status text with timer
            if status_text in DISTRACTION_THRESHOLDS:
                threshold = DISTRACTION_THRESHOLDS[status_text]
                timer_text = f"Person {i+1}: {status_text} ({duration:.1f}s/{threshold}s)"
            else:
                timer_text = f"Person {i+1}: {status_text}"
            
            # Calculate text size for background
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv.getTextSize(timer_text, font, font_scale, thickness)
            
            # Draw semi-transparent background for text
            text_y = y - 10
            if text_y < text_height + 10:
                text_y = y + h + text_height + 10
            
            # Background rectangle
            bg_start_x = x
            bg_start_y = text_y - text_height - 5
            bg_end_x = x + text_width + 10
            bg_end_y = text_y + 5
            
            # Create overlay for semi-transparent background
            overlay = image.copy()
            cv.rectangle(overlay, (bg_start_x, bg_start_y), (bg_end_x, bg_end_y), (0, 0, 0), -1)
            cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Draw timer text
            cv.putText(image, timer_text, (x + 5, text_y), font, font_scale, main_color, thickness)
            
            # Check for distraction alerts
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

def generate_pdf_report(session_data, output_path):
    """Generate PDF report for session"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Generating PDF report at: {output_path}")
        
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
        
        story.append(Paragraph("Smart Focus Alert - Session Report", title_style))
        story.append(Spacer(1, 20))
        
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            total_session_seconds = duration.total_seconds()
            duration_str = str(duration).split('.')[0]
        else:
            total_session_seconds = 0
            duration_str = "N/A"
        
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        
        # Session Information
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
        story.append(Spacer(1, 30))
        
        # Footer
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"PDF report successfully created: {output_path} (Size: {file_size} bytes)")
            return output_path
        else:
            print(f"Failed to create PDF report: {output_path}")
            return None
            
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return None

def generate_upload_pdf_report(detections, file_info, output_path):
    """Generate PDF report for uploaded file analysis"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Generating upload PDF report at: {output_path}")
        
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
        
        file_info_text = f"<b>File:</b> {file_info.get('filename', 'Unknown')}<br/>"
        file_info_text += f"<b>Type:</b> {file_info.get('type', 'Unknown')}<br/>"
        file_info_text += f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
        file_info_text += f"<b>Persons Detected:</b> {len(detections)}"
        
        story.append(Paragraph(file_info_text, styles['Normal']))
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
        
        doc.build(story)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Upload PDF report successfully created: {output_path} (Size: {file_size} bytes)")
            return output_path
        else:
            print(f"Failed to create upload PDF report: {output_path}")
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

def create_simple_recording_file():
    """Create a simple recording file like the image shown - NO LIVE DETECTION"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"session_recording_{timestamp}.mp4"
        recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
        
        print(f"Creating simple recording at: {recording_path}")
        
        # Create simple recording like the image
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(recording_path, fourcc, 30, (640, 480))
        
        if not out.isOpened():
            print("Failed to open video writer")
            return None
        
        # Create frames similar to the image shown (dark blue background with text)
        total_frames = 150  # 5 seconds at 30fps
        for i in range(total_frames):
            # Create dark blue background like in the image
            frame = np.full((480, 640, 3), [42, 52, 74], dtype=np.uint8)  # Dark blue background
            
            # Title
            cv.putText(frame, "Smart Focus Alert - Session Recording", (50, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Frame counter (green)
            cv.putText(frame, f"Frame: {i+1}/{total_frames}", (50, 120), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Session time (yellow)
            elapsed_time = i / 30  # seconds
            cv.putText(frame, f"Session Time: {elapsed_time:.1f}s", (50, 180), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Alerts count (cyan)
            cv.putText(frame, f"Alerts: {len(session_data.get('alerts', []))}", (50, 240), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
            
            # Detections count (magenta)
            cv.putText(frame, f"Detections: {session_data['focus_statistics']['total_detections']}", (50, 300), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
            
            # Add animated circle like in the image (red/orange)
            circle_x = int(320 + 100 * np.sin(i * 0.1))
            circle_y = int(360 + 30 * np.cos(i * 0.1))
            cv.circle(frame, (circle_x, circle_y), 25, (80, 80, 255), -1)  # Red-orange circle
            
            # Generated timestamp at bottom
            current_time = datetime.now().strftime('%H:%M:%S')
            cv.putText(frame, f"Generated: {current_time}", (50, 450), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            out.write(frame)
        
        out.release()
        
        # Verify file was created and has content
        if os.path.exists(recording_path):
            file_size = os.path.getsize(recording_path)
            print(f"Simple recording created successfully: {recording_path} (Size: {file_size} bytes)")
            if file_size > 0:
                return recording_path
            else:
                print("Recording file is empty")
                return None
        else:
            print("Recording file was not created")
            return None
            
    except Exception as e:
        print(f"Error creating simple recording: {str(e)}")
        return None

# Download endpoints
@application.route('/download/reports/<filename>')
def download_report_file(filename):
    """Download report files"""
    try:
        file_path = os.path.join(application.config['REPORTS_FOLDER'], filename)
        print(f"Attempting to download report: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")
            
            if file_size > 0:
                return send_file(file_path, as_attachment=True, download_name=filename, mimetype='application/pdf')
            else:
                return jsonify({"error": "File is empty"}), 404
        else:
            return jsonify({"error": "Report file not found"}), 404
    except Exception as e:
        print(f"Error downloading report file {filename}: {str(e)}")
        return jsonify({"error": "Download failed"}), 500

@application.route('/download/recordings/<filename>')
def download_recording_file(filename):
    """Download recording files"""
    try:
        file_path = os.path.join(application.config['RECORDINGS_FOLDER'], filename)
        print(f"Attempting to download recording: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")
            
            if file_size > 0:
                return send_file(file_path, as_attachment=True, download_name=filename, mimetype='video/mp4')
            else:
                return jsonify({"error": "File is empty"}), 404
        else:
            return jsonify({"error": "Recording file not found"}), 404
    except Exception as e:
        print(f"Error downloading recording file {filename}: {str(e)}")
        return jsonify({"error": "Download failed"}), 500

# Static file serving routes for viewing
@application.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files for viewing"""
    try:
        return send_from_directory(application.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@application.route('/static/detected/<filename>')
def detected_file(filename):
    """Serve detected/processed files for viewing"""
    try:
        return send_from_directory(application.config['DETECTED_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@application.route('/static/reports/<filename>')
def report_file(filename):
    """Serve report files for viewing"""
    try:
        return send_from_directory(application.config['REPORTS_FOLDER'], filename, mimetype='application/pdf')
    except FileNotFoundError:
        return jsonify({"error": "Report file not found"}), 404

@application.route('/static/recordings/<filename>')
def recording_file(filename):
    """Serve recording files for viewing"""
    try:
        return send_from_directory(application.config['RECORDINGS_FOLDER'], filename, mimetype='video/mp4')
    except FileNotFoundError:
        return jsonify({"error": "Recording file not found"}), 404

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
                    
                    pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                    
                    file_info = {'filename': filename, 'type': file_ext.upper()}
                    pdf_result = generate_upload_pdf_report(detections, file_info, pdf_path)
                    
                    if pdf_result and os.path.exists(pdf_path):
                        result["pdf_report"] = f"/download/reports/{pdf_filename}"
                
            elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                output_path, detections = process_video_file(file_path)
                
                if os.path.exists(output_path):
                    result["processed_video"] = f"/static/detected/{os.path.basename(output_path)}"
                    result["detections"] = detections
                    result["type"] = "video"
                    
                    pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                    
                    file_info = {'filename': filename, 'type': file_ext.upper()}
                    pdf_result = generate_upload_pdf_report(detections, file_info, pdf_path)
                    
                    if pdf_result and os.path.exists(pdf_path):
                        result["pdf_report"] = f"/download/reports/{pdf_filename}"
            
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
    
    person_state_timers = {}
    person_current_state = {}
    last_alert_time = {}
    
    live_monitoring_active = True
    recording_active = True
    
    print("Live monitoring started successfully")
    return jsonify({"status": "success", "message": "Monitoring started"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, session_data, recording_active
    
    if not live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring not active"})
    
    live_monitoring_active = False
    recording_active = False
    session_data['end_time'] = datetime.now()
    
    print("Stopping live monitoring and generating files...")
    
    # Generate PDF report
    pdf_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
    
    pdf_result = generate_pdf_report(session_data, pdf_path)
    
    response_data = {
        "status": "success", 
        "message": "Monitoring stopped"
    }
    
    if pdf_result and os.path.exists(pdf_path):
        response_data["pdf_report"] = f"/download/reports/{pdf_filename}"
        print(f"PDF report generated successfully: {pdf_filename}")
    else:
        print("Failed to generate PDF report")
    
    # Create simple recording file (like the image shown)
    simple_recording_path = create_simple_recording_file()
    if simple_recording_path and os.path.exists(simple_recording_path):
        response_data["video_file"] = f"/download/recordings/{os.path.basename(simple_recording_path)}"
        session_data['recording_path'] = simple_recording_path
        print(f"Simple recording generated successfully: {os.path.basename(simple_recording_path)}")
    else:
        print("Failed to generate simple recording")
    
    print("Stop monitoring completed")
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

@application.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@application.route('/debug/files')
def debug_files():
    """Debug route to check file existence"""
    try:
        debug_info = {
            "reports_folder": application.config['REPORTS_FOLDER'],
            "recordings_folder": application.config['RECORDINGS_FOLDER'],
            "reports_exists": os.path.exists(application.config['REPORTS_FOLDER']),
            "recordings_exists": os.path.exists(application.config['RECORDINGS_FOLDER']),
            "reports_files": [],
            "recordings_files": []
        }
        
        if os.path.exists(application.config['REPORTS_FOLDER']):
            debug_info["reports_files"] = os.listdir(application.config['REPORTS_FOLDER'])
        
        if os.path.exists(application.config['RECORDINGS_FOLDER']):
            debug_info["recordings_files"] = os.listdir(application.config['RECORDINGS_FOLDER'])
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Smart Focus Alert application on port {port}")
    print(f"Reports folder: {application.config['REPORTS_FOLDER']}")
    print(f"Recordings folder: {application.config['RECORDINGS_FOLDER']}")
    application.run(host='0.0.0.0', port=port, debug=False)
