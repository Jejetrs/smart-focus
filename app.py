from flask import Flask, render_template, request, Response, jsonify, send_file
from werkzeug.utils import secure_filename
import mediapipe as mp
import numpy as np
import pyttsx3
from scipy.spatial import distance as dis
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
from PIL import Image, ImageDraw, ImageFont
import io

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
    'frames': []  # Store frames for video generation
}

# Video recording variables
recording_active = False

def draw_landmarks_on_image(image, landmarks, land_mark, color):
    """Draw landmarks on PIL image for a single face"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for face in land_mark:
        point = landmarks.landmark[face]
        point_scale = (int(point.x * width), int(point.y * height))
        # Draw circle using ellipse
        radius = 2
        draw.ellipse([point_scale[0]-radius, point_scale[1]-radius, 
                     point_scale[0]+radius, point_scale[1]+radius], fill=color)

def euclidean_distance_pil(image, top, bottom):
    """Calculate euclidean distance between two points for PIL image"""
    width, height = image.size
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    distance = ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
    return distance

def get_aspect_ratio_pil(image, landmarks, top_bottom, left_right):
    """Calculate aspect ratio based on landmarks for PIL image"""
    top = landmarks.landmark[top_bottom[0]]
    bottom = landmarks.landmark[top_bottom[1]]
    top_bottom_dis = euclidean_distance_pil(image, top, bottom)

    left = landmarks.landmark[left_right[0]]
    right = landmarks.landmark[left_right[1]]
    left_right_dis = euclidean_distance_pil(image, left, right)
    
    aspect_ratio = left_right_dis / top_bottom_dis if top_bottom_dis > 0 else 0
    return aspect_ratio

def extract_eye_landmarks_pil(face_landmarks, eye_landmark_indices, image_size):
    """Extract eye landmarks from face landmarks for PIL image"""
    eye_landmarks = []
    width, height = image_size
    for index in eye_landmark_indices:
        landmark = face_landmarks.landmark[index]
        eye_landmarks.append([landmark.x * width, landmark.y * height])
    return np.array(eye_landmarks)

def calculate_midpoint(points):
    """Calculate the midpoint of a set of points"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    midpoint = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
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

def detect_drowsiness_pil(image, landmarks):
    """Detect drowsiness and attention state based on eye aspect ratio and other metrics using PIL"""
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
    img_w, img_h = image.size
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
    ratio_left_eye = get_aspect_ratio_pil(image, landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
    ratio_right_eye = get_aspect_ratio_pil(image, landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
    eye_ratio = (ratio_left_eye + ratio_right_eye) / 2
    
    # Detect yawning
    ratio_lips = get_aspect_ratio_pil(image, landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
    
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

def detect_persons_with_attention_pil(image_data, mode="image"):
    """Detect persons in image using PIL instead of cv2"""
    # Convert base64 to PIL image if needed
    if isinstance(image_data, str):
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
    else:
        image = image_data
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Initialize MediaPipe
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
    
    # Convert PIL to numpy array for MediaPipe
    image_array = np.array(image)
    detection_results = detector.process(image_array)
    mesh_results = face_mesh.process(image_array)
    
    detections = []
    iw, ih = image.size
    
    # Create drawing context
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        except:
            font = None
            small_font = None
    
    if detection_results.detections:
        for i, detection in enumerate(detection_results.detections):
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            
            # Draw rectangle
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)

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
                attention_status, state = detect_drowsiness_pil(
                    image, 
                    mesh_results.multi_face_landmarks[matched_face_idx]
                )
                
                # Draw landmarks
                FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
                    377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                LEFT_IRIS = [474, 475, 476, 477]
                RIGHT_IRIS = [469, 470, 471, 472]
                UPPER_LOWER_LIPS = [13, 14]
                LEFT_RIGHT_LIPS = [78, 308]
                
                draw_landmarks_on_image(image, mesh_results.multi_face_landmarks[matched_face_idx], FACE, (0, 255, 0))
                draw_landmarks_on_image(image, mesh_results.multi_face_landmarks[matched_face_idx], LEFT_EYE, (255, 0, 0))
                draw_landmarks_on_image(image, mesh_results.multi_face_landmarks[matched_face_idx], RIGHT_EYE, (255, 0, 0))
                draw_landmarks_on_image(image, mesh_results.multi_face_landmarks[matched_face_idx], LEFT_IRIS, (255, 0, 255))
                draw_landmarks_on_image(image, mesh_results.multi_face_landmarks[matched_face_idx], RIGHT_IRIS, (255, 0, 255))
                draw_landmarks_on_image(image, mesh_results.multi_face_landmarks[matched_face_idx], UPPER_LOWER_LIPS, (0, 0, 255))
                draw_landmarks_on_image(image, mesh_results.multi_face_landmarks[matched_face_idx], LEFT_RIGHT_LIPS, (0, 0, 255))
            
            status_text = attention_status.get("state", "FOCUSED")
            
            # Draw status info with background
            info_y_start = y + h + 10
            box_padding = 10
            line_height = 20
            box_height = 4 * line_height
            
            # Draw semi-transparent background
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x - box_padding, info_y_start - box_padding, 
                                  x + w + box_padding, info_y_start + box_height], 
                                 fill=(0, 0, 0, 153))  # 60% opacity
            
            # Composite overlay onto main image
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Draw text
            text_color = (255, 255, 255)
            draw.text((x, info_y_start), f"Person {i+1}", fill=(50, 205, 50), font=font)
            draw.text((x, info_y_start + line_height), 
                     f"Confidence: {confidence_score*100:.2f}%", fill=text_color, font=small_font)
            draw.text((x, info_y_start + 2*line_height), 
                     f"Position: x:{x}, y:{y} Size: w:{w}, h:{h}", fill=text_color, font=small_font)
            
            status_colors = {
                "FOCUSED": (0, 255, 0),
                "NOT FOCUSED": (255, 165, 0),
                "YAWNING": (255, 255, 0),
                "SLEEPING": (255, 0, 0)
            }
            color = status_colors.get(status_text, (0, 255, 0))
            
            draw.text((x, info_y_start + 3*line_height), f"Status: {status_text}", 
                     fill=color, font=small_font)
            
            # Extract and save face region
            face_img = image.crop((x, y, x+w, y+h))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"person_{i+1}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            face_path = os.path.join(application.config['DETECTED_FOLDER'], face_filename)
            
            if face_img.size[0] > 0 and face_img.size[1] > 0:
                face_img.save(face_path, 'JPEG')
            
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "image_path": f"/static/detected/{face_filename}",
                "status": status_text,
                "timestamp": datetime.now().isoformat()
            })
    
    # Add detection count
    if detections:
        draw.text((10, 30), f"Total persons detected: {len(detections)}", 
                 fill=(255, 0, 0), font=font)
    else:
        draw.text((10, 30), "No persons detected", fill=(255, 0, 0), font=font)
    
    return image, detections

def create_video_from_frames(frames, output_path, fps=30):
    """Create video from stored frames using PIL and imageio"""
    try:
        import imageio
        
        if not frames:
            return None
            
        # Convert PIL images to numpy arrays
        frame_arrays = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                frame_arrays.append(np.array(frame))
            else:
                frame_arrays.append(frame)
        
        # Write video
        imageio.mimsave(output_path, frame_arrays, fps=fps)
        return output_path
    except ImportError:
        # Fallback: create a simple image sequence
        if frames:
            frames[0].save(output_path.replace('.mp4', '_frame_0.jpg'), 'JPEG')
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

def generate_pdf_report(session_data, output_path):
    """Generate PDF report for session"""
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
    
    story.append(Paragraph("Session Information", heading_style))
    
    session_info = [
        ['Session Start Time', session_data.get('start_time', datetime.now()).strftime('%m/%d/%Y, %I:%M:%S %p')],
        ['Session Duration', duration_str],
        ['Total Detections', str(session_data['focus_statistics']['total_detections'])],
        ['Total Persons Detected', str(session_data['focus_statistics']['total_persons'])],
        ['Total Alerts Generated', str(len(session_data['alerts']))],
        ['Focus Accuracy', f"{focus_accuracy:.1f}%"]
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
    
    footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - Railway Deployment"
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#6B7280')
    )
    story.append(Paragraph(footer_text, footer_style))
    
    doc.build(story)
    return output_path

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
                image = Image.open(file_path)
                processed_image, detections = detect_persons_with_attention_pil(image)
                
                output_filename = f"processed_{filename}"
                output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
                processed_image.save(output_path, 'JPEG')
                
                result["processed_image"] = f"/static/detected/{output_filename}"
                result["detections"] = detections
                result["type"] = "image"
                
                # Generate PDF report
                pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                # Create simple report data for upload
                report_data = {
                    'start_time': datetime.now(),
                    'end_time': datetime.now(),
                    'detections': detections,
                    'alerts': [],
                    'focus_statistics': {
                        'total_detections': len(detections),
                        'total_persons': len(detections)
                    }
                }
                
                generate_pdf_report(report_data, pdf_path)
                result["pdf_report"] = f"/static/reports/{pdf_filename}"
            
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    # Return the provided HTML template
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Focus Monitoring - Smart Focus Alert</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        /* Include all the CSS from the provided HTML */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        :root {
            --primary-blue: #60a5fa;
            --primary-purple: #a78bfa;
            --dark-bg: #0f172a;
            --dark-secondary: #1e293b;
            --card-bg: #1e293b;
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #06b6d4;
            --sidebar-bg: #334155;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: var(--dark-bg);
            min-height: 100vh;
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
            line-height: 1.6;
        }

        .flex { display: flex; }
        .justify-between { justify-content: space-between; }
        .justify-center { justify-content: center; }
        .items-center { align-items: center; }
        .text-center { text-align: center; }

        .text-gradient {
            background: linear-gradient(135deg, var(--primary-blue), var(--primary-purple));
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        .navbar {
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--glass-border);
            padding: 16px 0;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-brand {
            font-size: 1.5rem;
            font-weight: 700;
            text-decoration: none;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .nav-brand i {
            color: var(--primary-blue);
        }

        .nav-links {
            display: flex;
            gap: 24px;
            align-items: center;
        }

        .nav-link {
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }

        .nav-link:hover, .nav-link.active {
            color: var(--primary-blue);
        }

        .main-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 24px;
            min-height: calc(100vh - 88px);
        }

        .page-header {
            text-align: center;
            margin-bottom: 32px;
        }

        .page-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .page-subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        .camera-setup-notice {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--glass-border);
            margin-bottom: 24px;
            display: block;
        }

        .setup-content {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .setup-icon {
            width: 48px;
            height: 48px;
            background: rgba(245, 158, 11, 0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--warning);
            font-size: 1.5rem;
        }

        .setup-text {
            flex: 1;
        }

        .setup-title {
            font-weight: 600;
            margin-bottom: 4px;
            color: var(--text-primary);
        }

        .setup-subtitle {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .monitoring-status-card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--glass-border);
            margin-bottom: 24px;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--primary-blue);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .live-indicator {
            display: flex;
            align-items: center;
            background: rgba(148, 163, 184, 0.2);
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 0.9rem;
        }

        .live-indicator.active {
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger);
        }

        .live-dot {
            width: 8px;
            height: 8px;
            background: var(--text-muted);
            border-radius: 50%;
            margin-right: 8px;
        }

        .live-dot.active {
            background: var(--danger);
            animation: pulse 1.5s infinite;
        }

        .status-display {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 16px 24px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 24px;
            background: rgba(148, 163, 184, 0.1);
            color: var(--text-muted);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }

        .status-display.status-focused {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border-color: rgba(16, 185, 129, 0.3);
        }

        .status-display.status-unfocused {
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning);
            border-color: rgba(245, 158, 11, 0.3);
        }

        .status-display.status-yawning {
            background: rgba(234, 179, 8, 0.1);
            color: #eab308;
            border-color: rgba(234, 179, 8, 0.3);
        }

        .status-display.status-sleeping {
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            border-color: rgba(239, 68, 68, 0.3);
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 12px;
            background-color: var(--text-muted);
        }

        .status-indicator.status-focused { background-color: var(--success); }
        .status-indicator.status-unfocused { background-color: var(--warning); }
        .status-indicator.status-yawning { background-color: #eab308; }
        .status-indicator.status-sleeping { background-color: var(--danger); }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }

        .stat-card {
            background: var(--dark-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 500;
        }

        .content-grid {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 24px;
        }

        .left-column {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .right-column {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .video-container {
            background: var(--dark-bg);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 450px;
            border: 1px solid var(--glass-border);
        }

        .client-video {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            display: none;
        }

        .client-canvas {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            display: none;
        }

        .video-placeholder {
            text-align: center;
            color: var(--text-muted);
            padding: 40px 20px;
        }

        .video-placeholder i {
            font-size: 4rem;
            margin-bottom: 16px;
            color: var(--text-muted);
        }

        .controls-panel {
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-top: 10px;
            flex-wrap: wrap;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            font-size: 14px;
            gap: 8px;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary-blue), var(--primary-purple));
            color: white;
        }

        .btn-secondary {
            background: linear-gradient(135deg, var(--success), var(--info));
            color: white;
        }

        .btn-danger {
            background: linear-gradient(135deg, var(--danger), #dc2626);
            color: white;
        }

        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(96, 165, 250, 0.4);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .sidebar-card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 16px;
            border: 1px solid var(--glass-border);
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .left-card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--glass-border);
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px;
            background: var(--dark-bg);
            border-radius: 12px;
            margin-bottom: 20px;
        }

        .connection-icon {
            width: 40px;
            height: 40px;
            background: rgba(245, 158, 11, 0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--warning);
        }

        .connection-text {
            flex: 1;
        }

        .connection-title {
            font-weight: 600;
            margin-bottom: 4px;
        }

        .connection-subtitle {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .audio-alert {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px;
            background: var(--dark-bg);
            border-radius: 12px;
            margin-bottom: 10px;
        }

        .toggle-switch {
            position: relative;
            width: 50px;
            height: 24px;
            background: var(--success);
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .toggle-switch::before {
            content: '';
            position: absolute;
            top: 2px;
            right: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            transition: transform 0.3s ease;
        }

        .toggle-switch.off {
            background: var(--text-muted);
        }

        .toggle-switch.off::before {
            transform: translateX(-26px);
        }

        .volume-label {
            display: block;
            margin-bottom: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .volume-select {
            width: 100%;
            padding: 12px 16px;
            background: var(--dark-bg);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 14px;
        }

        .thresholds-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        .threshold-item {
            text-align: center;
            padding: 16px 12px;
            background: var(--dark-bg);
            border-radius: 8px;
        }

        .threshold-label {
            font-weight: 600;
            margin-bottom: 8px;
        }

        .threshold-time {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .threshold-sleep .threshold-label { color: var(--danger); }
        .threshold-yawning .threshold-label { color: var(--warning); }
        .threshold-focus .threshold-label { color: var(--info); }

        .alert-history {
            max-height: 250px;
            overflow-y: auto;
            background: var(--dark-bg);
            border-radius: 12px;
            padding: 16px;
        }

        .alert-history::-webkit-scrollbar {
            width: 6px;
        }

        .alert-history::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.3);
            border-radius: 3px;
        }

        .alert-history::-webkit-scrollbar-thumb {
            background: var(--primary-blue);
            border-radius: 3px;
        }

        .alert-item {
            display: flex;
            align-items: center;
            padding: 12px;
            margin-bottom: 8px;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 8px;
            border-left: 4px solid var(--info);
        }

        .alert-time {
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-left: auto;
        }

        .download-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--dark-bg);
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 12px;
        }

        .download-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .file-icon {
            width: 40px;
            height: 40px;
            background: rgba(96, 165, 250, 0.2);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--primary-blue);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in { animation: fadeIn 0.6s ease-out; }

        @media (max-width: 1200px) {
            .content-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .right-column {
                order: -1;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 768px) {
            .nav-links {
                flex-direction: column;
                gap: 12px;
            }
            
            .main-container {
                padding: 20px 16px;
            }
            
            .page-title {
                font-size: 2rem;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
            }
            
            .thresholds-grid {
                grid-template-columns: 1fr;
                gap: 8px;
            }
            
            .controls-panel {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="container">
            <div class="flex justify-between items-center">
                <a href="/" class="nav-brand text-gradient">
                    <i class="fas fa-eye"></i>
                    Smart Focus Alert
                </a>
                <div class="nav-links">
                    <a href="/" class="nav-link">Home</a>
                    <a href="/webcam" class="nav-link active">Live Monitoring</a>
                    <a href="/upload" class="nav-link">Upload</a>
                </div>
            </div>
        </div>
    </nav>

    <div class="main-container">
        <div class="page-header">
            <h1 class="page-title text-gradient">Live Focus Monitoring</h1>
            <p class="page-subtitle">Real-time focus detection with facial landmarks, automatic recording, and comprehensive reporting</p>
        </div>

        <div class="camera-setup-notice" id="cameraSetupNotice">
            <div class="setup-content">
                <div class="setup-icon">
                    <i class="fas fa-info-circle"></i>
                </div>
                <div class="setup-text">
                    <div class="setup-title">Railway Deployment - Using Device Camera</div>
                    <div class="setup-subtitle">Optimized for Railway platform with PIL-based face detection</div>
                </div>
            </div>
        </div>

        <div class="monitoring-status-card">
            <div class="card-header">
                <h2 class="card-title">
                    <i class="fas fa-chart-line"></i>
                    Monitoring Status
                </h2>
                <div class="live-indicator" id="liveIndicator">
                    <div class="live-dot" id="liveDot"></div>
                    <span id="liveText">READY</span>
                </div>
            </div>
            
            <div class="status-display" id="currentStatus">
                <span class="status-indicator"></span>
                System Ready - Click Start to begin monitoring with automatic recording
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--success);" id="totalPersons">0</div>
                    <div class="stat-label">Persons Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--info);" id="focusedCount">0</div>
                    <div class="stat-label">Currently Focused</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--warning);" id="alertCount">0</div>
                    <div class="stat-label">Total Alerts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value text-gradient" id="sessionTime">00:00:00</div>
                    <div class="stat-label">Session Time</div>
                </div>
            </div>
        </div>

        <div class="content-grid">
            <div class="left-column">
                <div class="video-container">
                    <video id="clientVideo" class="client-video" autoplay></video>
                    <canvas id="clientCanvas" class="client-canvas" width="640" height="480"></canvas>
                    
                    <div class="video-placeholder" id="videoPlaceholder">
                        <i class="fas fa-video"></i>
                        <h3 style="margin-bottom: 12px;">Enhanced Detection Ready</h3>
                        <p style="margin-bottom: 20px;">Click "Start Monitoring" to begin detection with time-based alerts</p>
                        <div style="text-align: left; max-width: 400px; line-height: 1.6;">
                            <div style="margin-bottom: 8px;">• <strong>Sleep Detection:</strong> Alert after 10 seconds</div>
                            <div style="margin-bottom: 8px;">• <strong>Yawning Detection:</strong> Alert after 5 seconds</div>
                            <div style="margin-bottom: 8px;">• <strong>Unfocus Detection:</strong> Alert after 10 seconds</div>
                            <div>• <strong>Voice + Beep Alerts:</strong> Automatic notifications</div>
                        </div>
                    </div>
                </div>
                
                <div class="controls-panel">
                    <button class="btn btn-primary" id="startBtn" onclick="startMonitoring()">
                        <i class="fas fa-play"></i>
                        Start Monitoring
                    </button>
                    <button class="btn btn-danger" id="stopBtn" onclick="stopMonitoring()" disabled>
                        <i class="fas fa-stop"></i>
                        Stop & Generate Report
                    </button>
                    <button class="btn btn-secondary" onclick="takeScreenshot()" id="screenshotBtn" disabled>
                        <i class="fas fa-camera"></i>
                        Screenshot
                    </button>
                </div>

                <div class="left-card">
                    <div class="sidebar-header">
                        <i class="fas fa-volume-up"></i>
                        Alert Volume
                    </div>
                    
                    <label class="volume-label">Choose volume level:</label>
                    <select class="volume-select" id="alertVolume">
                        <option value="low">Low</option>
                        <option value="medium" selected>Medium</option>
                        <option value="high">High</option>
                    </select>
                </div>

                <div class="left-card" style="display: none;" id="downloadsSection">
                    <div class="sidebar-header">
                        <i class="fas fa-download"></i>
                        Session Downloads
                    </div>
                    <div id="downloadItems">
                    </div>
                </div>
            </div>

            <div class="right-column">
                <div class="sidebar-card">
                    <div class="connection-status">
                        <div class="connection-icon">
                            <i class="fas fa-clock"></i>
                        </div>
                        <div class="connection-text">
                            <div class="connection-title">Ready to Connect</div>
                            <div class="connection-subtitle">Device camera access ready</div>
                        </div>
                    </div>

                    <div class="audio-alert">
                        <i class="fas fa-volume-up" style="color: var(--text-secondary);"></i>
                        <span style="flex: 1; font-weight: 500;">Audio Alert</span>
                        <div class="toggle-switch" onclick="toggleAudioAlert()"></div>
                    </div>

                    <div class="audio-alert" style="margin-bottom: 0;">
                        <span style="color: var(--text-secondary); font-size: 0.9rem;">Sound Alerts Enabled</span>
                    </div>
                </div>

                <div class="sidebar-card">
                    <div class="sidebar-header">
                        <i class="fas fa-list"></i>
                        Detection Thresholds
                    </div>
                    
                    <div class="thresholds-grid">
                        <div class="threshold-item threshold-sleep">
                            <div class="threshold-label">Sleep</div>
                            <div class="threshold-time">10 seconds</div>
                        </div>
                        <div class="threshold-item threshold-yawning">
                            <div class="threshold-label">Yawning</div>
                            <div class="threshold-time">3.5 seconds</div>
                        </div>
                        <div class="threshold-item threshold-focus">
                            <div class="threshold-label">No Focus</div>
                            <div class="threshold-time">10 seconds</div>
                        </div>
                    </div>
                </div>

                <div class="sidebar-card">
                    <div class="sidebar-header">
                        <i class="fas fa-list"></i>
                        Alert History
                    </div>
                    
                    <div class="alert-history" id="alertHistory">
                        <div class="text-center" style="color: var(--text-secondary); padding: 20px;">
                            <i class="fas fa-clock" style="font-size: 2rem; margin-bottom: 12px; display: block;"></i>
                            <p>No alerts yet - monitoring not started</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionStartTime = null;
        let alertCount = 0;
        let isMonitoring = false;
        let sessionTimer = null;
        let dataUpdateTimer = null;
        let lastAlertIds = new Set();
        let audioEnabled = true;

        let clientVideo = null;
        let clientCanvas = null;
        let clientCtx = null;
        let clientStream = null;
        let processingInterval = null;
        let storedFrames = [];

        document.addEventListener('DOMContentLoaded', function() {
            initializePage();
            setupEventListeners();
        });

        function initializePage() {
            document.body.style.opacity = "0";
            document.body.style.transition = "opacity 0.6s ease";
            setTimeout(() => {
                document.body.style.opacity = "1";
            }, 100);
            
            clientVideo = document.getElementById('clientVideo');
            clientCanvas = document.getElementById('clientCanvas');
            if (clientCanvas) {
                clientCtx = clientCanvas.getContext('2d');
            }
        }

        function setupEventListeners() {
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', function(e) {
                    if (this.getAttribute('href').startsWith('/')) {
                        e.preventDefault();
                        document.body.style.opacity = "0";
                        setTimeout(() => {
                            window.location.href = this.getAttribute('href');
                        }, 300);
                    }
                });
            });

            document.querySelectorAll('.btn').forEach(button => {
                button.addEventListener('mouseenter', function() {
                    if (!this.disabled) {
                        this.style.transform = 'translateY(-3px) scale(1.05)';
                    }
                });
                
                button.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0) scale(1)';
                });
            });

            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    if (isMonitoring) {
                        stopMonitoring();
                    }
                }
                if (e.key === ' ') {
                    e.preventDefault();
                    if (isMonitoring) {
                        takeScreenshot();
                    }
                }
            });

            document.getElementById('alertVolume').addEventListener('change', function() {
                showNotification(`Alert volume changed to ${this.value}`, 'info');
            });
        }

        function toggleAudioAlert() {
            audioEnabled = !audioEnabled;
            const toggle = document.querySelector('.toggle-switch');
            const statusText = document.querySelector('.audio-alert:last-of-type span');
            
            if (audioEnabled) {
                toggle.classList.remove('off');
                statusText.textContent = 'Sound Alerts Enabled';
                statusText.style.color = 'var(--success)';
            } else {
                toggle.classList.add('off');
                statusText.textContent = 'Sound Alerts Disabled';
                statusText.style.color = 'var(--text-muted)';
            }
        }

        async function startMonitoring() {
            try {
                const response = await fetch('/start_monitoring', { method: 'POST' });
                const data = await response.json();
                
                if (data.status !== 'success') {
                    throw new Error(data.message);
                }

                await initializeClientCamera();
                updateUIForActiveMonitoring();
                startDataUpdates();
                showNotification('Monitoring started with automatic recording', 'success');
            } catch (error) {
                showNotification('Failed to start monitoring: ' + error.message, 'error');
            }
        }

        async function initializeClientCamera() {
            try {
                clientStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'user' }
                });
                
                clientVideo.srcObject = clientStream;
                await new Promise((resolve) => {
                    clientVideo.onloadedmetadata = resolve;
                });
                
                clientVideo.style.display = 'none';
                clientCanvas.style.display = 'block';
                
                processingInterval = setInterval(processClientFrame, 1000);
                
            } catch (error) {
                throw new Error('Failed to access device camera: ' + error.message);
            }
        }

        function processClientFrame() {
            if (!clientVideo || !isMonitoring || !clientStream || !clientCtx) return;

            try {
                // Draw video frame to canvas
                clientCtx.drawImage(clientVideo, 0, 0, clientCanvas.width, clientCanvas.height);
                
                // Store frame for video creation
                const frameData = clientCanvas.toDataURL('image/jpeg', 0.8);
                storedFrames.push(frameData);
                
                // Keep only last 1000 frames to prevent memory issues
                if (storedFrames.length > 1000) {
                    storedFrames = storedFrames.slice(-1000);
                }

                // Send to server for processing
                fetch('/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ frame: frameData })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.processed_frame) {
                        // Display processed frame
                        const img = new Image();
                        img.onload = function() {
                            clientCtx.drawImage(img, 0, 0, clientCanvas.width, clientCanvas.height);
                        };
                        img.src = data.processed_frame;
                        
                        // Update detection data
                        if (data.detections && data.detections.length > 0) {
                            updateLocalDetectionData(data.detections);
                        }
                    }
                })
                .catch(error => {
                    console.error('Frame processing error:', error);
                });

            } catch (error) {
                console.error('Frame capture error:', error);
            }
        }

        function updateLocalDetectionData(detections) {
            // Update local statistics based on detections
            const totalPersons = detections.length;
            const focusedCount = detections.filter(d => d.status === 'FOCUSED').length;
            
            document.getElementById('totalPersons').textContent = totalPersons;
            document.getElementById('focusedCount').textContent = focusedCount;
            
            // Determine overall status
            let currentStatus = 'FOCUSED';
            if (detections.some(d => d.status === 'SLEEPING')) {
                currentStatus = 'SLEEPING';
            } else if (detections.some(d => d.status === 'YAWNING')) {
                currentStatus = 'YAWNING';
            } else if (detections.some(d => d.status === 'NOT FOCUSED')) {
                currentStatus = 'NOT FOCUSED';
            }
            
            updateCurrentStatus(currentStatus);
        }

        async function stopMonitoring() {
            try {
                // Send stored frames to server
                if (storedFrames.length > 0) {
                    await fetch('/save_session_frames', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ frames: storedFrames })
                    });
                }

                const response = await fetch('/stop_monitoring', { method: 'POST' });
                const data = await response.json();

                updateUIForInactiveMonitoring();
                stopDataUpdates();

                if (clientStream) {
                    clientStream.getTracks().forEach(track => track.stop());
                    clientStream = null;
                }

                if (processingInterval) {
                    clearInterval(processingInterval);
                    processingInterval = null;
                }

                storedFrames = [];

                showNotification('Session complete! PDF report and recording generated.', 'success');
                
                if (data.pdf_report || data.video_file) {
                    showDownloads(data.pdf_report, data.video_file);
                } else {
                    showDownloads('/static/reports/session_report.pdf', '/static/recordings/session_video.mp4');
                }
            } catch (error) {
                showNotification('Failed to stop monitoring: ' + error.message, 'error');
            }
        }

        function updateUIForActiveMonitoring() {
            isMonitoring = true;
            sessionStartTime = Date.now();
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('screenshotBtn').disabled = false;
            
            const liveIndicator = document.getElementById('liveIndicator');
            const liveDot = document.getElementById('liveDot');
            const liveText = document.getElementById('liveText');
            
            liveIndicator.classList.add('active');
            liveDot.classList.add('active');
            liveText.textContent = 'LIVE • RECORDING';
            
            document.getElementById('videoPlaceholder').style.display = 'none';
            
            updateCurrentStatus('READY');
            startSessionTimer();
            
            const alertHistory = document.getElementById('alertHistory');
            alertHistory.innerHTML = '<div class="text-center" style="color: var(--text-secondary); padding: 20px;"><i class="fas fa-clock" style="font-size: 2rem; margin-bottom: 12px; display: block;"></i><p>Monitoring started - automatic recording active</p></div>';
            
            lastAlertIds.clear();
        }

        function updateUIForInactiveMonitoring() {
            isMonitoring = false;
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('screenshotBtn').disabled = true;
            
            const liveIndicator = document.getElementById('liveIndicator');
            const liveDot = document.getElementById('liveDot');
            const liveText = document.getElementById('liveText');
            
            liveIndicator.classList.remove('active');
            liveDot.classList.remove('active');
            liveText.textContent = 'READY';
            
            document.getElementById('clientVideo').style.display = 'none';
            document.getElementById('clientCanvas').style.display = 'none';
            document.getElementById('videoPlaceholder').style.display = 'block';
            
            updateCurrentStatus('READY');
            
            if (sessionTimer) {
                clearInterval(sessionTimer);
                sessionTimer = null;
            }
            
            document.getElementById('totalPersons').textContent = '0';
            document.getElementById('focusedCount').textContent = '0';
        }

        function startDataUpdates() {
            dataUpdateTimer = setInterval(() => {
                if (isMonitoring) {
                    fetch('/get_monitoring_data')
                        .then(response => response.json())
                        .then(data => {
                            if (!data.error) {
                                updateMonitoringDisplay(data);
                                updateAlerts(data.latest_alerts || []);
                            }
                        })
                        .catch(error => {
                            console.error('Failed to fetch monitoring data:', error);
                        });
                }
            }, 3000);
        }

        function stopDataUpdates() {
            if (dataUpdateTimer) {
                clearInterval(dataUpdateTimer);
                dataUpdateTimer = null;
            }
        }

        function updateMonitoringDisplay(data) {
            document.getElementById('alertCount').textContent = data.alert_count || 0;
        }

        function updateCurrentStatus(status) {
            const statusElement = document.getElementById('currentStatus');
            const statusClasses = ['status-ready', 'status-focused', 'status-unfocused', 'status-yawning', 'status-sleeping'];
            
            statusClasses.forEach(cls => statusElement.classList.remove(cls));
            
            let statusClass, message;
            switch(status) {
                case 'FOCUSED':
                    statusClass = 'status-focused';
                    message = 'All persons are focused - Recording active';
                    break;
                case 'NOT FOCUSED':
                    statusClass = 'status-unfocused';
                    message = 'Some persons are not focused - Recording active';
                    break;
                case 'YAWNING':
                    statusClass = 'status-yawning';
                    message = 'Yawning detected - fatigue signs - Recording active';
                    break;
                case 'SLEEPING':
                    statusClass = 'status-sleeping';
                    message = 'Sleep detected - eyes closed - Recording active';
                    break;
                default:
                    statusClass = 'status-ready';
                    message = 'Monitoring active - detecting faces - Recording in progress';
            }
            
            statusElement.classList.add(statusClass);
            statusElement.innerHTML = `
                <span class="status-indicator ${statusClass.replace('status-', 'status-')}"></span>
                ${message}
            `;
        }

        function updateAlerts(alerts) {
            const alertHistory = document.getElementById('alertHistory');
            
            if (alerts.length === 0) {
                alertHistory.innerHTML = '<div class="text-center" style="color: var(--text-secondary); padding: 20px;"><i class="fas fa-clock" style="font-size: 2rem; margin-bottom: 12px; display: block;"></i><p>No recent alerts.</p></div>';
                return;
            }

            alertHistory.innerHTML = alerts.map(alert => 
                `<div class="alert-item" style="border-left-color: ${alert.type === 'error' ? 'var(--danger)' : 'var(--warning)'};">
                    <div style="flex: 1;">
                        <div style="font-weight: 500; margin-bottom: 4px;">${alert.message}</div>
                    </div>
                    <div class="alert-time">${alert.time}</div>
                </div>`
            ).join('');
        }

        function startSessionTimer() {
            sessionTimer = setInterval(() => {
                if (sessionStartTime) {
                    const elapsed = Date.now() - sessionStartTime;
                    const hours = Math.floor(elapsed / 3600000);
                    const minutes = Math.floor((elapsed % 3600000) / 60000);
                    const seconds = Math.floor((elapsed % 60000) / 1000);
                    document.getElementById('sessionTime').textContent = 
                        `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                }
            }, 1000);
        }

        function showDownloads(pdfReport, videoFile) {
            const downloadsSection = document.getElementById('downloadsSection');
            const downloadItems = document.getElementById('downloadItems');
            
            downloadItems.innerHTML = '';
            
            if (pdfReport) {
                const pdfItem = document.createElement('div');
                pdfItem.className = 'download-item';
                pdfItem.innerHTML = `
                    <div class="download-info">
                        <div class="file-icon">
                            <i class="fas fa-file-pdf"></i>
                        </div>
                        <div>
                            <div style="font-weight: 600;">Session Report</div>
                            <div style="color: var(--text-secondary); font-size: 0.9rem;">PDF with comprehensive analysis</div>
                        </div>
                    </div>
                    <a href="${pdfReport}" download class="btn btn-primary">
                        <i class="fas fa-download"></i>
                    </a>
                `;
                downloadItems.appendChild(pdfItem);
            }
            
            if (videoFile) {
                const videoItem = document.createElement('div');
                videoItem.className = 'download-item';
                videoItem.innerHTML = `
                    <div class="download-info">
                        <div class="file-icon">
                            <i class="fas fa-video"></i>
                        </div>
                        <div>
                            <div style="font-weight: 600;">Session Recording</div>
                            <div style="color: var(--text-secondary); font-size: 0.9rem;">Video with face detection overlays</div>
                        </div>
                    </div>
                    <a href="${videoFile}" download class="btn btn-secondary">
                        <i class="fas fa-download"></i>
                    </a>
                `;
                downloadItems.appendChild(videoItem);
            }
            
            downloadsSection.style.display = 'block';
        }

        function takeScreenshot() {
            if (clientCanvas && isMonitoring) {
                const link = document.createElement('a');
                link.download = `screenshot_${new Date().getTime()}.png`;
                link.href = clientCanvas.toDataURL();
                link.click();
                showNotification('Screenshot captured successfully', 'success');
            }
        }

        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--glass-bg);
                border: 1px solid var(--glass-border);
                border-radius: 8px;
                padding: 16px;
                color: var(--text-primary);
                z-index: 1000;
                max-width: 300px;
                animation: slideInRight 0.3s ease-out;
                backdrop-filter: blur(10px);
            `;
            
            const colors = {
                success: 'var(--success)',
                warning: 'var(--warning)',
                error: 'var(--danger)',
                info: 'var(--info)'
            };
            
            const icons = {
                success: 'fa-check-circle',
                warning: 'fa-exclamation-triangle',
                error: 'fa-times-circle',
                info: 'fa-info-circle'
            };
            
            notification.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <i class="fas ${icons[type] || icons.info}" style="color: ${colors[type] || colors.info};"></i>
                    <span>${message}</span>
                    <button onclick="this.parentElement.parentElement.remove()" 
                            style="background: none; border: none; color: var(--text-secondary); cursor: pointer; margin-left: auto;">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 5000);
        }

        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);

        window.addEventListener('beforeunload', function() {
            if (clientStream) {
                clientStream.getTracks().forEach(track => track.stop());
            }
        });
    </script>
</body>
</html>"""
    return html_content

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, session_data, recording_active
    
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
        'frames': []
    }
    
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
    generate_pdf_report(session_data, pdf_path)
    
    response_data = {
        "status": "success", 
        "message": "Monitoring stopped",
        "pdf_report": f"/static/reports/{pdf_filename}"
    }
    
    # Create video from stored frames if available
    if session_data.get('frames'):
        video_filename = f"session_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(application.config['RECORDINGS_FOLDER'], video_filename)
        
        try:
            created_video = create_video_from_frames(session_data['frames'], video_path)
            if created_video:
                response_data["video_file"] = f"/static/recordings/{video_filename}"
        except Exception as e:
            print(f"Video creation error: {e}")
    
    return jsonify(response_data)

@application.route('/save_session_frames', methods=['POST'])
def save_session_frames():
    global session_data
    
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        
        # Convert base64 frames to PIL images and store
        for frame_data in frames[-100:]:  # Keep last 100 frames
            try:
                if frame_data.startswith('data:image'):
                    frame_data = frame_data.split(',')[1]
                
                image_bytes = base64.b64decode(frame_data)
                image = Image.open(io.BytesIO(image_bytes))
                session_data['frames'].append(image)
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

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

@application.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame from client browser using PIL"""
    try:
        data = request.get_json()
        frame_data = data['frame']
        
        # Process detection using PIL
        processed_image, detections = detect_persons_with_attention_pil(frame_data, mode="video")
        
        # Convert back to base64
        buffered = BytesIO()
        processed_image.save(buffered, format="JPEG")
        processed_frame_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Update session statistics
        if detections:
            update_session_statistics(detections)
            
            # Check for alerts
            for detection in detections:
                if detection['status'] in ['SLEEPING', 'YAWNING', 'NOT FOCUSED']:
                    session_data['alerts'].append({
                        'timestamp': datetime.now().isoformat(),
                        'person': f"Person {detection['id']}",
                        'detection': detection['status'],
                        'message': f"Person {detection['id']} is {detection['status'].lower()}",
                        'duration': 1
                    })
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        image = Image.open(file_path)
        processed_image, detections = detect_persons_with_attention_pil(image)
        
        output_filename = f"processed_{filename}"
        output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
        processed_image.save(output_path, 'JPEG')
        
        return jsonify({
            "type": "image",
            "processed_image": f"/static/detected/{output_filename}",
            "detections": detections
        })
    
    return jsonify({"error": "Unsupported file format"}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)
