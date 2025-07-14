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
import tempfile

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

def draw_landmarks_on_pil(image, landmarks, land_mark, color):
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

def calculate_midpoint(points):
    """Calculate the midpoint of a set of points"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    midpoint = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    return midpoint

def check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
    """Check if iris is in the middle of the eye"""
    try:
        left_eye_midpoint = calculate_midpoint(left_eye_points)
        right_eye_midpoint = calculate_midpoint(right_eye_points)
        left_iris_midpoint = calculate_midpoint(left_iris_points)
        right_iris_midpoint = calculate_midpoint(right_iris_points)
        deviation_threshold_horizontal = 2.8
        
        return (abs(left_iris_midpoint[0] - left_eye_midpoint[0]) <= deviation_threshold_horizontal 
                and abs(right_iris_midpoint[0] - right_eye_midpoint[0]) <= deviation_threshold_horizontal)
    except:
        return True  # Default to focused if calculation fails

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
    
    try:
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
        
        # Determine state based on conditions - adjusted thresholds for better accuracy
        eyes_closed = eye_ratio > 4.5  # More sensitive
        yawning = ratio_lips < 2.0     # More sensitive
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
    except Exception as e:
        print(f"Detection error: {e}")
        return {"state": "FOCUSED"}, "FOCUSED"

def detect_persons_with_attention_pil(image_data, mode="image"):
    """Detect persons in image using PIL with improved accuracy"""
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
    
    # Initialize MediaPipe with more conservative settings for Railway
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,  # Use model 0 for better performance
        min_detection_confidence=0.7  # Higher confidence for better accuracy
    )

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=(mode == "image"),
        max_num_faces=5,  # Reduced for better performance
        refine_landmarks=True,
        min_detection_confidence=0.7,
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
            
            # Improved bounding box validation
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            
            # Skip invalid detections
            if w <= 0 or h <= 0:
                continue
            
            # Draw rectangle with improved visibility
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)

            confidence_score = detection.score[0]
            
            attention_status = {
                "eyes_closed": False,
                "yawning": False,
                "not_focused": False,
                "state": "FOCUSED"
            }
            
            # Improved face matching algorithm
            matched_face_idx = -1
            if mesh_results.multi_face_landmarks:
                min_distance = float('inf')
                for face_idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                    # Calculate face center from landmarks
                    face_center_x = sum([lm.x for lm in face_landmarks.landmark]) / len(face_landmarks.landmark) * iw
                    face_center_y = sum([lm.y for lm in face_landmarks.landmark]) / len(face_landmarks.landmark) * ih
                    
                    # Detection center
                    det_center_x = x + w / 2
                    det_center_y = y + h / 2
                    
                    # Calculate distance between centers
                    distance = ((face_center_x - det_center_x)**2 + (face_center_y - det_center_y)**2)**0.5
                    
                    # Check if this is the closest match within reasonable bounds
                    if distance < min_distance and distance < max(w, h) * 0.5:
                        min_distance = distance
                        matched_face_idx = face_idx
            
            if matched_face_idx != -1:
                try:
                    attention_status, state = detect_drowsiness_pil(
                        image, 
                        mesh_results.multi_face_landmarks[matched_face_idx]
                    )
                    
                    # Draw facial landmarks for better visualization
                    landmarks = mesh_results.multi_face_landmarks[matched_face_idx]
                    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
                    
                    # Draw key landmarks
                    draw_landmarks_on_pil(image, landmarks, FACE_OVAL, (0, 255, 0))
                    draw_landmarks_on_pil(image, landmarks, LEFT_EYE, (255, 0, 0))
                    draw_landmarks_on_pil(image, landmarks, RIGHT_EYE, (255, 0, 0))
                    draw_landmarks_on_pil(image, landmarks, LIPS, (0, 0, 255))
                    
                except Exception as e:
                    print(f"Landmark processing error: {e}")
                    attention_status = {"state": "FOCUSED"}
            
            status_text = attention_status.get("state", "FOCUSED")
            
            # Draw status info with improved visibility
            info_y_start = y + h + 15
            
            # Draw semi-transparent background
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Calculate text box size
            text_lines = [
                f"Person {i+1}",
                f"Confidence: {confidence_score*100:.1f}%",
                f"Status: {status_text}",
                f"Position: ({x}, {y})"
            ]
            
            max_width = 0
            total_height = 0
            line_height = 20
            
            for line in text_lines:
                if font:
                    bbox = draw.textbbox((0, 0), line, font=small_font)
                    line_width = bbox[2] - bbox[0]
                else:
                    line_width = len(line) * 8
                max_width = max(max_width, line_width)
                total_height += line_height
            
            # Draw background rectangle
            overlay_draw.rectangle([x - 5, info_y_start - 5, 
                                  x + max_width + 10, info_y_start + total_height + 5], 
                                 fill=(0, 0, 0, 180))
            
            # Composite overlay onto main image
            image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # Draw text with status-based colors
            status_colors = {
                "FOCUSED": (0, 255, 0),
                "NOT FOCUSED": (255, 165, 0),
                "YAWNING": (255, 255, 0),
                "SLEEPING": (255, 0, 0)
            }
            
            for idx, line in enumerate(text_lines):
                y_pos = info_y_start + (idx * line_height)
                if "Status:" in line:
                    color = status_colors.get(status_text, (255, 255, 255))
                else:
                    color = (255, 255, 255)
                
                draw.text((x, y_pos), line, fill=color, font=small_font)
            
            # Extract and save face region
            try:
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
            except Exception as e:
                print(f"Face extraction error: {e}")
                detections.append({
                    "id": i+1,
                    "confidence": float(confidence_score),
                    "bbox": [x, y, w, h],
                    "image_path": "",
                    "status": status_text,
                    "timestamp": datetime.now().isoformat()
                })
    
    # Add detection count
    count_text = f"Total persons detected: {len(detections)}" if detections else "No persons detected"
    draw.text((10, 30), count_text, fill=(255, 0, 0), font=font)
    
    return image, detections

def create_video_from_frames(frames, output_path, fps=20):
    """Create video from stored frames using imageio"""
    try:
        import imageio
        
        if not frames:
            return None
            
        # Convert PIL images to numpy arrays
        frame_arrays = []
        for frame in frames[-300:]:  # Use last 300 frames to avoid memory issues
            if isinstance(frame, Image.Image):
                frame_arrays.append(np.array(frame))
            else:
                frame_arrays.append(frame)
        
        # Write video with better codec
        imageio.mimsave(output_path, frame_arrays, fps=fps, format='mp4', codec='libx264')
        return output_path
    except Exception as e:
        print(f"Video creation error: {e}")
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
    return render_template('webcam.html')

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
    
    try:
        generate_pdf_report(session_data, pdf_path)
        pdf_success = True
    except Exception as e:
        print(f"PDF generation error: {e}")
        pdf_success = False
    
    response_data = {
        "status": "success", 
        "message": "Monitoring stopped"
    }
    
    if pdf_success:
        response_data["pdf_report"] = f"/static/reports/{pdf_filename}"
    
    # Create video from stored frames if available
    if session_data.get('frames'):
        video_filename = f"session_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(application.config['RECORDINGS_FOLDER'], video_filename)
        
        try:
            created_video = create_video_from_frames(session_data['frames'], video_path)
            if created_video and os.path.exists(created_video):
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
        
        # Convert base64 frames to PIL images and store (with limit)
        for frame_data in frames[-50:]:  # Keep last 50 frames to manage memory
            try:
                if frame_data.startswith('data:image'):
                    frame_data = frame_data.split(',')[1]
                
                image_bytes = base64.b64decode(frame_data)
                image = Image.open(io.BytesIO(image_bytes))
                # Resize to reduce memory usage
                image = image.resize((320, 240))
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
    """Process frame from client browser using PIL with improved performance"""
    try:
        data = request.get_json()
        frame_data = data['frame']
        
        # Process detection using PIL with improved algorithms
        processed_image, detections = detect_persons_with_attention_pil(frame_data, mode="video")
        
        # Convert back to base64
        buffered = BytesIO()
        processed_image.save(buffered, format="JPEG", quality=70)  # Reduced quality for better performance
        processed_frame_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Update session statistics
        if detections:
            update_session_statistics(detections)
            
            # Check for alerts with improved thresholds
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
        print(f"Frame processing error: {e}")
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

# Download endpoint for generated files
@application.route('/download/<path:filename>')
def download_file(filename):
    """Secure file download endpoint"""
    try:
        # Determine file type and directory
        if filename.startswith('reports/'):
            directory = application.config['REPORTS_FOLDER']
            file_path = os.path.join(directory, filename.split('/')[-1])
        elif filename.startswith('recordings/'):
            directory = application.config['RECORDINGS_FOLDER']
            file_path = os.path.join(directory, filename.split('/')[-1])
        else:
            return jsonify({"error": "Invalid file type"}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)
