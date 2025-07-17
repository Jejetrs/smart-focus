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

application = Flask(__name__)

application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['MAX_CONTENT_PATH'] = 10000000

for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        os.chmod(folder, 0o755)

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
    'recording_frames': []
}

video_writer = None
recording_active = False
recording_frames = []

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
            
            if mode == "video" and live_monitoring_active:
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
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=50, rightMargin=50, 
                               topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        story = []
        
        # Define custom colors
        header_blue = colors.HexColor('#2563EB')
        light_blue = colors.HexColor('#EBF5FF')
        dark_gray = colors.HexColor('#374151')
        medium_gray = colors.HexColor('#6B7280')
        light_gray = colors.HexColor('#F3F4F6')
        
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            textColor=header_blue
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            textColor=dark_gray
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            fontName='Helvetica',
            textColor=colors.black
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
            ('BACKGROUND', (0, 0), (0, -1), light_blue),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, medium_gray),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [light_blue, colors.white]),
        ]))
        
        story.append(session_table)
        story.append(Spacer(1, 30))
        
        story.append(Paragraph("Focus Accuracy Summary", heading_style))
        
        # Colored accuracy percentage
        if focus_accuracy >= 90:
            accuracy_color = colors.HexColor('#10B981')  # Green
            focus_rating = "Excellent"
        elif focus_accuracy >= 75:
            accuracy_color = header_blue  # Blue
            focus_rating = "Good"
        elif focus_accuracy >= 60:
            accuracy_color = colors.HexColor('#F59E0B')  # Yellow
            focus_rating = "Fair"
        else:
            accuracy_color = colors.HexColor('#EF4444')  # Red
            focus_rating = "Poor"
        
        accuracy_style = ParagraphStyle(
            'AccuracyStyle',
            parent=normal_style,
            fontSize=28,
            fontName='Helvetica-Bold',
            textColor=accuracy_color,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph(f"{focus_accuracy:.1f}%", accuracy_style))
        story.append(Spacer(1, 10))
        
        rating_style = ParagraphStyle(
            'RatingStyle',
            parent=normal_style,
            fontSize=16,
            fontName='Helvetica-Bold',
            textColor=accuracy_color,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph(f"Focus Quality: {focus_rating}", rating_style))
        story.append(Spacer(1, 30))
        
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        
        def format_percentage(value, total):
            if total > 0:
                return f"{(value/total)*100:.1f}%"
            return "0.0%"
        
        metric_data = [
            ['Metric', 'Time', 'Percentage'],
            ['Total Focused Time', format_time(focused_time), format_percentage(focused_time, total_session_seconds)],
            ['Total Distraction Time', format_time(total_distraction_time), format_percentage(total_distraction_time, total_session_seconds)],
            ['- Unfocused Time', format_time(unfocused_time), format_percentage(unfocused_time, total_session_seconds)],
            ['- Yawning Time', format_time(yawning_time), format_percentage(yawning_time, total_session_seconds)],
            ['- Sleeping Time', format_time(sleeping_time), format_percentage(sleeping_time, total_session_seconds)]
        ]
        
        metric_table = Table(metric_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        metric_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), header_blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, medium_gray),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [light_gray, colors.white]),
        ]))
        
        story.append(metric_table)
        story.append(Spacer(1, 30))
        
        story.append(Paragraph("Detailed Focus Statistics", heading_style))
        
        detail_data = [
            ['Total Session Duration', format_time(total_session_seconds)],
            ['Focus Accuracy Score', f"{focus_accuracy:.2f}%"],
            ['Focus Quality Rating', focus_rating],
            ['Distraction Frequency', f"{len(session_data['alerts'])} alerts in {format_time(total_session_seconds)}"]
        ]
        
        if session_data['alerts']:
            most_common = {}
            for alert in session_data['alerts']:
                detection = alert.get('detection', 'Unknown')
                duration = alert.get('duration', 0)
                if detection not in most_common:
                    most_common[detection] = {'count': 0, 'total_duration': 0}
                most_common[detection]['count'] += 1
                most_common[detection]['total_duration'] += duration
            
            if most_common:
                max_detection = max(most_common, key=lambda x: most_common[x]['count'])
                detail_data.append(['Most Common Distraction', f"{max_detection} ({most_common[max_detection]['count']} times, {most_common[max_detection]['total_duration']}s total)"])
        
        detail_table = Table(detail_data, colWidths=[2.5*inch, 2.5*inch])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), light_blue),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, medium_gray),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [light_blue, colors.white]),
        ]))
        
        story.append(detail_table)
        story.append(Spacer(1, 30))
        
        if session_data['alerts']:
            story.append(Paragraph("Alert History", heading_style))
            
            alert_headers = ['Time', 'Person', 'Detection', 'Duration', 'Message']
            alert_data = [alert_headers]
            
            for alert in session_data['alerts']:
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%I:%M:%S %p')
                except:
                    alert_time = alert['timestamp']
                
                alert_data.append([
                    alert_time,
                    alert['person'],
                    alert['detection'],
                    f"{alert.get('duration', 0)}s",
                    alert['message']
                ])
            
            # Adjusted column widths - made message column wider
            alert_table = Table(alert_data, colWidths=[0.8*inch, 0.8*inch, 1*inch, 0.6*inch, 2.8*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), header_blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, medium_gray),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [light_gray, colors.white]),
            ]))
            
            story.append(alert_table)
        
        story.append(Spacer(1, 50))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            fontName='Helvetica',
            textColor=medium_gray
        )
        
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - Focus Monitoring Report"
        story.append(Paragraph(footer_text, footer_style))
        
        try:
            doc.build(story)
            print(f"PDF report created: {output_path}")
            return output_path
        except Exception as pdf_error:
            print(f"PDF error: {str(pdf_error)}")
            return None
            
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

def generate_upload_pdf_report(detections, file_info, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'Title',
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
        story.append(Spacer(1, 20))
        
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
        
        try:
            doc.build(story)
            print(f"Upload PDF report created: {output_path}")
            return output_path
        except Exception as pdf_error:
            print(f"PDF error: {str(pdf_error)}")
            return None
            
    except Exception as e:
        print(f"Error generating upload PDF: {str(e)}")
        return None

def process_video_file(video_path):
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
    try:
        if not recording_frames:
            print("No frames to create video")
            return None
        
        height, width = recording_frames[0].shape[:2]
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, 10.0, (width, height))
        
        for frame in recording_frames:
            if frame is not None:
                out.write(frame)
        
        out.release()
        
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
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"session_recording_{timestamp}.mp4"
        recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(recording_path), exist_ok=True)
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(recording_path, fourcc, 30, (640, 480))
        
        for i in range(150):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
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
                mimetype='application/pdf'
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
                mimetype='video/mp4'
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
                        result["pdf_report"] = f"/download_report/{pdf_filename}"
                
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
                        result["pdf_report"] = f"/download_report/{pdf_filename}"
            
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
        'recording_path': None,
        'recording_frames': []
    }
    
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
        response_data["pdf_report"] = f"/download_report/{pdf_filename}"
        print(f"PDF report created: {pdf_path}")
    
    # Generate video recording
    recording_filename = f"session_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
    
    if session_data.get('recording_frames') and len(session_data['recording_frames']) > 0:
        video_result = create_session_recording_from_frames(session_data['recording_frames'], recording_path)
        print(f"Created recording from {len(session_data['recording_frames'])} frames")
    else:
        video_result = create_demo_recording_file()
        if video_result:
            recording_path = video_result
        print("Created demo recording file")
    
    if video_result and os.path.exists(recording_path):
        response_data["video_file"] = f"/download_recording/{os.path.basename(recording_path)}"
        session_data['recording_path'] = recording_path
        print(f"Video recording created: {recording_path}")
    
    return jsonify(response_data)

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
    global session_data
    
    try:
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400
        
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        if live_monitoring_active and recording_active:
            session_data['recording_frames'].append(processed_frame.copy())
            if len(session_data['recording_frames']) > 300:
                session_data['recording_frames'] = session_data['recording_frames'][-300:]
        
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

@application.route('/download_report/<filename>')
def download_report(filename):
    try:
        file_path = os.path.join(application.config['REPORTS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(
                file_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({"error": "Report file not found"}), 404
    except Exception as e:
        print(f"Error downloading report: {str(e)}")
        return jsonify({"error": "Error downloading report file"}), 500

@application.route('/download_recording/<filename>')
def download_recording(filename):
    try:
        file_path = os.path.join(application.config['RECORDINGS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(
                file_path,
                mimetype='video/mp4',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({"error": "Recording file not found"}), 404
    except Exception as e:
        print(f"Error downloading recording: {str(e)}")
        return jsonify({"error": "Error downloading recording file"}), 500

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
