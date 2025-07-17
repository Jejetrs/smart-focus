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
import glob

application = Flask(__name__)

# Configuration
application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['MAX_CONTENT_PATH'] = 10000000

# Ensure all directories exist with proper permissions
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        os.chmod(folder, 0o755)
    print(f"Directory created/verified: {folder}")

# Global variables for session management
live_monitoring_active = False
current_session_id = None
current_pdf_path = None
current_video_path = None

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
    'recording_frames': []
}

# Person tracking variables
person_state_timers = {}
person_current_state = {}
last_alert_time = {}

# Detection thresholds
DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,
    'YAWNING': 3.5,
    'NOT FOCUSED': 10
}

def cleanup_old_session_files():
    """Clean up old session files to prevent accumulation"""
    try:
        # Clean old reports (keep last 3)
        report_files = glob.glob(os.path.join(application.config['REPORTS_FOLDER'], 'session_report_*.pdf'))
        if len(report_files) > 3:
            report_files.sort(key=os.path.getmtime)
            for old_file in report_files[:-3]:
                try:
                    os.remove(old_file)
                    print(f"Removed old report: {old_file}")
                except:
                    pass
        
        # Clean old recordings (keep last 3)
        recording_files = glob.glob(os.path.join(application.config['RECORDINGS_FOLDER'], 'session_recording_*.mp4'))
        if len(recording_files) > 3:
            recording_files.sort(key=os.path.getmtime)
            for old_file in recording_files[:-3]:
                try:
                    os.remove(old_file)
                    print(f"Removed old recording: {old_file}")
                except:
                    pass
                    
        # Clean old detected files (keep last 20)
        detected_files = glob.glob(os.path.join(application.config['DETECTED_FOLDER'], 'person_*'))
        if len(detected_files) > 20:
            detected_files.sort(key=os.path.getmtime)
            for old_file in detected_files[:-20]:
                try:
                    os.remove(old_file)
                except:
                    pass
                    
    except Exception as e:
        print(f"Error cleaning old files: {str(e)}")

def reset_session_data():
    """Reset all session data for new monitoring session"""
    global session_data, person_state_timers, person_current_state, last_alert_time
    
    session_data = {
        'session_id': current_session_id,
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
        'recording_frames': []
    }
    
    # Reset person tracking
    person_state_timers = {}
    person_current_state = {}
    last_alert_time = {}
    
    print(f"Session data reset for session: {current_session_id}")

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

    # Draw all landmarks with face mesh
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

    # Calculate ratios for detection
    ratio_left_eye = get_aspect_ratio(frame, landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
    ratio_right_eye = get_aspect_ratio(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
    eye_ratio = (ratio_left_eye + ratio_right_eye) / 2
    
    ratio_lips = get_aspect_ratio(frame, landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
    
    iris_focused = check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points)
    
    # Determine states
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
            
            # Ensure bounding box is within image bounds
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
            
            # Match detection with face mesh for landmark analysis
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
            
            # Process drowsiness detection with landmarks
            if matched_face_idx != -1:
                attention_status, state = detect_drowsiness(
                    image, 
                    mesh_results.multi_face_landmarks[matched_face_idx]
                )
            
            status_text = attention_status.get("state", "FOCUSED")
            person_key = f"person_{i+1}"
            
            # Calculate duration for live monitoring
            duration = 0
            if mode == "video" and live_monitoring_active:
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
            
            # Draw bounding box and labels with complete visualization
            if mode == "video" and live_monitoring_active:
                status_colors = {
                    "FOCUSED": (0, 255, 0),      # Green
                    "NOT FOCUSED": (0, 165, 255), # Orange
                    "YAWNING": (0, 255, 255),     # Yellow
                    "SLEEPING": (0, 0, 255)       # Red
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                
                # Draw bounding box
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Create status text with timer
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
                
                # Create semi-transparent background for text
                overlay = image.copy()
                cv.rectangle(overlay, (x, text_y - text_height - 5), (x + text_width + 10, text_y + 5), (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                # Draw the text
                cv.putText(image, timer_text, (x + 5, text_y), font, font_scale, main_color, thickness)
            
            # Handle alerts for live monitoring
            if (mode == "video" and live_monitoring_active and status_text in DISTRACTION_THRESHOLDS and 
                person_key in person_state_timers and status_text in person_state_timers[person_key]):
                
                if duration >= DISTRACTION_THRESHOLDS[status_text]:
                    alert_cooldown = 5  # 5 seconds between alerts
                    if current_time - last_alert_time[person_key] >= alert_cooldown:
                        last_alert_time[person_key] = current_time
                        
                        # Generate alert message
                        if status_text == 'SLEEPING':
                            alert_message = f'Person {i+1} is sleeping - please wake up!'
                        elif status_text == 'YAWNING':
                            alert_message = f'Person {i+1} is yawning - please take a rest!'
                        elif status_text == 'NOT FOCUSED':
                            alert_message = f'Person {i+1} is not focused - please focus on screen!'
                        else:
                            alert_message = f'Person {i+1} attention alert'
                        
                        # Add alert to session data
                        if live_monitoring_active:
                            session_data['alerts'].append({
                                'timestamp': datetime.now().isoformat(),
                                'person': f"Person {i+1}",
                                'detection': status_text,
                                'message': alert_message,
                                'duration': int(duration)
                            })
            
            # Save detected face image
            face_img = image[y:y+h, x:x+w]
            if face_img.size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_filename = f"person_{i+1}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
                face_path = os.path.join(application.config['DETECTED_FOLDER'], face_filename)
                cv.imwrite(face_path, face_img)
            
            # Create detection data
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "image_path": f"/static/detected/{face_filename}" if face_img.size > 0 else None,
                "status": status_text,
                "timestamp": datetime.now().isoformat(),
                "duration": duration if mode == "video" else 0
            })
    
    # Draw detection count on image
    if detections:
        cv.putText(image, f"Persons detected: {len(detections)}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(image, "No persons detected", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image, detections

def update_session_statistics(detections):
    global session_data
    
    if not detections:
        return
    
    session_data['detections'].extend(detections)
    session_data['focus_statistics']['total_detections'] += len(detections)
    session_data['focus_statistics']['total_persons'] = max(
        session_data['focus_statistics']['total_persons'],
        len(detections)
    )

def generate_pdf_report(session_data, output_path):
    """Generate comprehensive PDF report for the session"""
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
        
        # Calculate session duration
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            total_session_seconds = duration.total_seconds()
            duration_str = str(duration).split('.')[0]
        else:
            total_session_seconds = 0
            duration_str = "N/A"
        
        # Session Information
        story.append(Paragraph("Session Information", heading_style))
        
        session_info = [
            ['Session ID', session_data.get('session_id', 'Unknown')],
            ['Start Time', session_data.get('start_time', datetime.now()).strftime('%m/%d/%Y, %I:%M:%S %p')],
            ['Duration', duration_str],
            ['Total Detections', str(session_data['focus_statistics']['total_detections'])],
            ['Total Alerts', str(len(session_data['alerts']))]
        ]
        
        session_table = Table(session_info, colWidths=[3*inch, 2*inch])
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
        story.append(Spacer(1, 20))
        
        # Alert History
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
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(alert_table)
        
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
        
        doc.build(story)
        print(f"PDF report generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return None

def create_session_recording(session_id, output_path):
    """Create recording from session frames or generate demo video"""
    try:
        if session_data.get('recording_frames') and len(session_data['recording_frames']) > 0:
            # Create video from recorded frames
            height, width = session_data['recording_frames'][0].shape[:2]
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(output_path, fourcc, 10.0, (width, height))
            
            for frame in session_data['recording_frames']:
                if frame is not None:
                    out.write(frame)
            
            out.release()
        else:
            # Create demo video
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(output_path, fourcc, 30, (640, 480))
            
            for i in range(150):  # 5 seconds at 30fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Add session information to demo video
                cv.putText(frame, f"Session Recording - Frame {i+1}", (50, 150), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv.putText(frame, f"Session ID: {session_id}", (50, 200), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.putText(frame, f"Total alerts: {len(session_data['alerts'])}", (50, 250), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if session_data['start_time']:
                    duration = (datetime.now() - session_data['start_time']).total_seconds()
                    cv.putText(frame, f"Duration: {int(duration//60)}m {int(duration%60)}s", (50, 300), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv.putText(frame, "Smart Focus Alert - Live Session Complete", (50, 400), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Session recording created successfully: {output_path}")
            return output_path
        else:
            print("Failed to create session recording")
            return None
            
    except Exception as e:
        print(f"Error creating session recording: {str(e)}")
        return None

# Route handlers
@application.route('/static/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(application.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving uploaded file: {str(e)}")
        return jsonify({"error": "File not found"}), 404

@application.route('/static/detected/<filename>')
def detected_file(filename):
    try:
        return send_from_directory(application.config['DETECTED_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving detected file: {str(e)}")
        return jsonify({"error": "File not found"}), 404

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
            print(f"Report file not found: {file_path}")
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
            print(f"Recording file not found: {file_path}")
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
            
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, current_session_id, current_pdf_path, current_video_path
    
    if live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring already active"})
    
    try:
        # Generate new session ID
        current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Clean up old files
        cleanup_old_session_files()
        
        # Reset session data
        reset_session_data()
        
        # Reset file paths
        current_pdf_path = None
        current_video_path = None
        
        # Start monitoring
        live_monitoring_active = True
        
        print(f"Monitoring started for session: {current_session_id}")
        
        return jsonify({
            "status": "success", 
            "message": "Monitoring started",
            "session_id": current_session_id
        })
        
    except Exception as e:
        print(f"Error starting monitoring: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to start monitoring: {str(e)}"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, current_session_id, current_pdf_path, current_video_path
    
    if not live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring not active"})
    
    try:
        live_monitoring_active = False
        session_data['end_time'] = datetime.now()
        
        if not current_session_id:
            return jsonify({"status": "error", "message": "No active session found"})
        
        # Generate PDF report
        pdf_filename = f"session_report_{current_session_id}.pdf"
        pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
        
        pdf_result = generate_pdf_report(session_data, pdf_path)
        if pdf_result:
            current_pdf_path = pdf_path
            pdf_url = f"/static/reports/{pdf_filename}"
        else:
            pdf_url = None
        
        # Generate recording
        recording_filename = f"session_recording_{current_session_id}.mp4"
        recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
        
        video_result = create_session_recording(current_session_id, recording_path)
        if video_result:
            current_video_path = recording_path
            video_url = f"/static/recordings/{recording_filename}"
        else:
            video_url = None
        
        response_data = {
            "status": "success", 
            "message": "Monitoring stopped and files generated",
            "session_id": current_session_id,
            "pdf_report": pdf_url,
            "video_file": video_url
        }
        
        print(f"Session {current_session_id} stopped. PDF: {pdf_url}, Video: {video_url}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error stopping monitoring: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to stop monitoring: {str(e)}"})

@application.route('/get_monitoring_data')
def get_monitoring_data():
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
    
    # Calculate current status from recent detections
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
        'latest_alerts': formatted_alerts,
        'session_id': current_session_id
    })

@application.route('/monitoring_status')
def monitoring_status():
    return jsonify({
        "is_active": live_monitoring_active,
        "session_id": current_session_id
    })

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
        
        # Store frame for recording if monitoring is active
        if live_monitoring_active:
            session_data['recording_frames'].append(frame.copy())
            # Keep only last 300 frames to manage memory
            if len(session_data['recording_frames']) > 300:
                session_data['recording_frames'] = session_data['recording_frames'][-300:]
        
        # Process frame with full detection and landmarks
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        # Update session statistics if monitoring is active
        if live_monitoring_active and detections:
            update_session_statistics(detections)
        
        # Convert processed frame to base64
        _, buffer = cv.imencode('.jpg', processed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections,
            "session_id": current_session_id
        })
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({"error": str(e)}), 500

@application.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "session_id": current_session_id,
        "monitoring_active": live_monitoring_active,
        "directories": {
            "uploads": os.path.exists(application.config['UPLOAD_FOLDER']),
            "detected": os.path.exists(application.config['DETECTED_FOLDER']),
            "reports": os.path.exists(application.config['REPORTS_FOLDER']),
            "recordings": os.path.exists(application.config['RECORDINGS_FOLDER'])
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Smart Focus Alert server on port {port}")
    application.run(host='0.0.0.0', port=port, debug=False)
