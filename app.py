from flask import Flask, render_template, request, Response, jsonify, send_file
from werkzeug.utils import secure_filename
import mediapipe as mp
import numpy as np
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
from PIL import Image
import scipy.spatial.distance as distance

# Use opencv-python-headless for Railway compatibility
try:
    import cv2 as cv
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

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
    'recording_path': None
}

# Voice alerts data
voice_alerts = []

def euclidean_distance(point1, point2):
    """Calculate euclidean distance between two points"""
    return distance.euclidean(point1, point2)

def get_aspect_ratio(landmarks, top_bottom_indices, left_right_indices, img_w, img_h):
    """Calculate aspect ratio based on landmarks"""
    top = landmarks.landmark[top_bottom_indices[0]]
    bottom = landmarks.landmark[top_bottom_indices[1]]
    left = landmarks.landmark[left_right_indices[0]]
    right = landmarks.landmark[left_right_indices[1]]
    
    top_point = (int(top.x * img_w), int(top.y * img_h))
    bottom_point = (int(bottom.x * img_w), int(bottom.y * img_h))
    left_point = (int(left.x * img_w), int(left.y * img_h))
    right_point = (int(right.x * img_w), int(right.y * img_h))
    
    vertical_dist = euclidean_distance(top_point, bottom_point)
    horizontal_dist = euclidean_distance(left_point, right_point)
    
    if vertical_dist == 0:
        return 100  # Avoid division by zero
    
    return horizontal_dist / vertical_dist

def check_iris_position(left_eye_landmarks, right_eye_landmarks, left_iris_landmarks, right_iris_landmarks, img_w, img_h):
    """Check if iris is centered in eyes for focus detection"""
    deviation_threshold = 0.02  # Adjust this value as needed
    
    # Calculate eye centers
    left_eye_center_x = np.mean([landmark.x for landmark in left_eye_landmarks])
    right_eye_center_x = np.mean([landmark.x for landmark in right_eye_landmarks])
    
    # Calculate iris centers
    left_iris_center_x = np.mean([landmark.x for landmark in left_iris_landmarks])
    right_iris_center_x = np.mean([landmark.x for landmark in right_iris_landmarks])
    
    # Check if iris is centered
    left_deviation = abs(left_iris_center_x - left_eye_center_x)
    right_deviation = abs(right_iris_center_x - right_eye_center_x)
    
    return left_deviation < deviation_threshold and right_deviation < deviation_threshold

def detect_drowsiness_and_attention(landmarks, img_w, img_h):
    """Detect drowsiness and attention state using MediaPipe landmarks"""
    # Eye landmark indices for MediaPipe Face Mesh
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    LEFT_IRIS_INDICES = [474, 475, 476, 477]
    RIGHT_IRIS_INDICES = [469, 470, 471, 472]
    
    # Mouth landmark indices
    UPPER_LIP_INDICES = [13, 14]
    LOWER_LIP_INDICES = [15, 16]
    
    try:
        # Calculate eye aspect ratios
        left_eye_ratio = get_aspect_ratio(
            landmarks, [385, 380], [362, 263], img_w, img_h
        )
        right_eye_ratio = get_aspect_ratio(
            landmarks, [160, 144], [33, 133], img_w, img_h
        )
        avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        # Calculate mouth aspect ratio for yawning detection
        mouth_ratio = get_aspect_ratio(
            landmarks, [13, 15], [61, 291], img_w, img_h
        )
        
        # Get eye and iris landmarks for focus detection
        left_eye_landmarks = [landmarks.landmark[i] for i in LEFT_EYE_INDICES]
        right_eye_landmarks = [landmarks.landmark[i] for i in RIGHT_EYE_INDICES]
        left_iris_landmarks = [landmarks.landmark[i] for i in LEFT_IRIS_INDICES]
        right_iris_landmarks = [landmarks.landmark[i] for i in RIGHT_IRIS_INDICES]
        
        # Check if person is focused (iris centered)
        is_focused = check_iris_position(
            left_eye_landmarks, right_eye_landmarks,
            left_iris_landmarks, right_iris_landmarks,
            img_w, img_h
        )
        
        # Determine state based on thresholds
        eyes_closed = avg_eye_ratio > 5.5  # Eyes closed threshold
        yawning = mouth_ratio < 1.8  # Yawning threshold
        not_focused = not is_focused
        
        # State priority: SLEEPING > YAWNING > NOT FOCUSED > FOCUSED
        if eyes_closed:
            state = "SLEEPING"
        elif yawning:
            state = "YAWNING"
        elif not_focused:
            state = "NOT FOCUSED"
        else:
            state = "FOCUSED"
        
        return {
            "state": state,
            "eyes_closed": eyes_closed,
            "yawning": yawning,
            "not_focused": not_focused,
            "focused": is_focused,
            "eye_ratio": avg_eye_ratio,
            "mouth_ratio": mouth_ratio
        }
        
    except Exception as e:
        print(f"Error in drowsiness detection: {e}")
        return {
            "state": "FOCUSED",
            "eyes_closed": False,
            "yawning": False,
            "not_focused": False,
            "focused": True,
            "eye_ratio": 0,
            "mouth_ratio": 0
        }

def process_image_with_mediapipe(image_data):
    """Process image using MediaPipe for face detection and analysis"""
    try:
        # Convert base64 to PIL Image
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        pil_image = Image.open(BytesIO(img_bytes))
        
        # Convert PIL to RGB numpy array
        rgb_image = np.array(pil_image.convert('RGB'))
        img_h, img_w = rgb_image.shape[:2]
        
        # Initialize MediaPipe
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Process image
        detection_results = face_detection.process(rgb_image)
        mesh_results = face_mesh.process(rgb_image)
        
        detections = []
        processed_image = rgb_image.copy()
        
        if detection_results.detections:
            for i, detection in enumerate(detection_results.detections):
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * img_w)
                y = int(bbox.ymin * img_h)
                w = int(bbox.width * img_w)
                h = int(bbox.height * img_h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                
                confidence = detection.score[0]
                
                # Default analysis
                analysis = {
                    "state": "FOCUSED",
                    "eyes_closed": False,
                    "yawning": False,
                    "not_focused": False,
                    "focused": True
                }
                
                # Try to get corresponding face mesh for detailed analysis
                if mesh_results.multi_face_landmarks and i < len(mesh_results.multi_face_landmarks):
                    face_landmarks = mesh_results.multi_face_landmarks[i]
                    analysis = detect_drowsiness_and_attention(face_landmarks, img_w, img_h)
                
                # Draw bounding box with color based on state
                color_map = {
                    "FOCUSED": (0, 255, 0),      # Green
                    "NOT FOCUSED": (255, 165, 0), # Orange
                    "YAWNING": (255, 255, 0),     # Yellow
                    "SLEEPING": (255, 0, 0)       # Red
                }
                
                color = color_map.get(analysis["state"], (0, 255, 0))
                
                # Draw rectangle on processed image
                if CV_AVAILABLE:
                    cv.rectangle(processed_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Add label
                    label = f"Person {i+1}: {analysis['state']}"
                    cv.putText(processed_image, label, (x, y - 10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Add confidence
                    conf_text = f"Conf: {confidence*100:.1f}%"
                    cv.putText(processed_image, conf_text, (x, y + h + 20), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Store detection data
                detections.append({
                    "id": i + 1,
                    "confidence": float(confidence),
                    "bbox": [x, y, w, h],
                    "status": analysis["state"],
                    "timestamp": datetime.now().isoformat(),
                    "analysis": analysis
                })
        
        # Convert processed image back to base64
        if CV_AVAILABLE:
            _, buffer = cv.imencode('.jpg', processed_image)
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            # Fallback to original image if CV not available
            pil_processed = Image.fromarray(processed_image)
            buffered = BytesIO()
            pil_processed.save(buffered, format="JPEG")
            processed_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "processed_image": f"data:image/jpeg;base64,{processed_base64}",
            "detections": detections,
            "total_persons": len(detections)
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {
            "success": False,
            "error": str(e),
            "processed_image": image_data,  # Return original
            "detections": [],
            "total_persons": 0
        }

def add_voice_alert(message, alert_type="info"):
    """Add voice alert to queue (web-based alerts)"""
    global voice_alerts
    voice_alerts.append({
        "message": message,
        "type": alert_type,
        "timestamp": datetime.now().isoformat()
    })
    # Keep only last 10 alerts
    voice_alerts = voice_alerts[-10:]

def calculate_distraction_time_from_alerts(alerts):
    """Calculate actual distraction time based on alert history"""
    distraction_times = {
        'unfocused_time': 0,
        'yawning_time': 0,
        'sleeping_time': 0
    }
    
    if not alerts:
        return distraction_times
    
    for alert in alerts:
        detection = alert.get('detection', 'Unknown')
        duration = alert.get('duration', 0)
        
        if detection == 'NOT FOCUSED':
            distraction_times['unfocused_time'] += duration
        elif detection == 'YAWNING':
            distraction_times['yawning_time'] += duration
        elif detection == 'SLEEPING':
            distraction_times['sleeping_time'] += duration
    
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
    
    # Update distraction times
    distraction_times = calculate_distraction_time_from_alerts(session_data['alerts'])
    session_data['focus_statistics'].update(distraction_times)

def generate_pdf_report(session_data, output_path):
    """Generate PDF report for session"""
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
    
    # Title
    story.append(Paragraph("Smart Focus Alert - Session Report", title_style))
    story.append(Spacer(1, 20))
    
    # Session Information
    if session_data['start_time'] and session_data['end_time']:
        duration = session_data['end_time'] - session_data['start_time']
        total_session_seconds = duration.total_seconds()
        duration_str = str(duration).split('.')[0]
    else:
        total_session_seconds = 0
        duration_str = "N/A"
    
    session_info = [
        ['Session Start Time', session_data.get('start_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')],
        ['Session Duration', duration_str],
        ['Total Detections', str(session_data['focus_statistics']['total_detections'])],
        ['Total Persons', str(session_data['focus_statistics']['total_persons'])],
        ['Total Alerts', str(len(session_data['alerts']))]
    ]
    
    session_table = Table(session_info, colWidths=[3*inch, 2*inch])
    session_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    
    story.append(Paragraph("Session Information", styles['Heading2']))
    story.append(session_table)
    story.append(Spacer(1, 20))
    
    # Alert History
    if session_data['alerts']:
        story.append(Paragraph("Alert History", styles['Heading2']))
        
        alert_data = [['Time', 'Person', 'Detection', 'Duration', 'Message']]
        for alert in session_data['alerts'][-20:]:  # Last 20 alerts
            try:
                alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
            except:
                alert_time = alert['timestamp']
            
            alert_data.append([
                alert_time,
                alert['person'],
                alert['detection'],
                f"{alert.get('duration', 0)}s",
                alert['message']
            ])
        
        alert_table = Table(alert_data)
        alert_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        
        story.append(alert_table)
    
    # Footer
    story.append(Spacer(1, 30))
    footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    doc.build(story)
    return output_path

# Flask Routes
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
            
            # For now, return basic result for uploaded files
            result = {
                "filename": filename,
                "file_path": f"/static/uploads/{filename}",
                "detections": [],
                "type": "image"
            }
            
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

@application.route('/video_feed')
def video_feed():
    """Video feed endpoint - disabled for Railway"""
    return jsonify({"error": "Server camera not available on Railway"}), 404

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, session_data
    
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
    
    return jsonify({"status": "success", "message": "Monitoring started"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, session_data
    
    if not live_monitoring_active:
        return jsonify({"status": "error", "message": "Monitoring not active"})
    
    live_monitoring_active = False
    session_data['end_time'] = datetime.now()
    
    # Generate PDF report
    pdf_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
    generate_pdf_report(session_data, pdf_path)
    
    return jsonify({
        "status": "success", 
        "message": "Monitoring stopped",
        "pdf_report": f"/static/reports/{pdf_filename}"
    })

@application.route('/get_monitoring_data')
def get_monitoring_data():
    global session_data, voice_alerts
    
    if not live_monitoring_active:
        return jsonify({"error": "Monitoring not active"})
    
    # Get recent alerts
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
    
    # Calculate current status
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
    
    response_data = {
        'total_persons': total_persons,
        'focused_count': focused_count,
        'alert_count': len(session_data['alerts']),
        'current_status': current_status,
        'latest_alerts': formatted_alerts
    }
    
    # Add voice alerts if any
    if voice_alerts:
        response_data['voice_alerts'] = voice_alerts
        voice_alerts = []  # Clear after sending
    
    return jsonify(response_data)

@application.route('/monitoring_status')
def monitoring_status():
    return jsonify({"is_active": live_monitoring_active})

@application.route('/check_camera')
def check_camera():
    """Check camera availability - always return false on Railway"""
    return jsonify({"camera_available": False})

@application.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame from client browser with enhanced detection"""
    global session_data
    
    try:
        data = request.get_json()
        frame_data = data['frame']
        
        # Process the frame
        result = process_image_with_mediapipe(frame_data)
        
        if result["success"] and result["detections"]:
            # Update session data
            if live_monitoring_active:
                update_session_statistics(result["detections"])
                
                # Check for alerts
                current_time = time.time()
                for detection in result["detections"]:
                    person_id = detection["id"]
                    status = detection["status"]
                    
                    # Generate alerts for distraction states
                    if status in ["SLEEPING", "YAWNING", "NOT FOCUSED"]:
                        alert_message = f"Person {person_id} is {status.lower()}"
                        
                        # Add to session alerts
                        session_data['alerts'].append({
                            'timestamp': datetime.now().isoformat(),
                            'person': f"Person {person_id}",
                            'detection': status,
                            'message': alert_message,
                            'duration': 5  # Default duration
                        })
                        
                        # Add voice alert
                        add_voice_alert(alert_message, "warning" if status != "SLEEPING" else "error")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "processed_frame": data.get('frame', ''),
            "detections": []
        }), 500

@application.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for file detection"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Process uploaded file (simplified for Railway)
    filename = secure_filename(file.filename)
    file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # For now, return basic detection result
    return jsonify({
        "type": "image",
        "processed_image": f"/static/uploads/{filename}",
        "detections": []
    })

# Health check endpoint for Railway
@application.route('/health')
def health_check():
    return jsonify({"status": "healthy", "cv_available": CV_AVAILABLE})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)
