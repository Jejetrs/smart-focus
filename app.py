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
import base64
import tempfile
import shutil

application = Flask(__name__)

# Simplified configuration
application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['MAX_CONTENT_PATH'] = 10000000

# Ensure directories exist
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# SIMPLIFIED GLOBAL VARIABLES
monitoring_active = False
current_session = {
    'start_time': None,
    'alerts': [],
    'total_detections': 0,
    'recording_frames': [],
    'last_pdf_path': None,
    'last_video_path': None
}

# Detection thresholds
ALERT_THRESHOLDS = {
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

def detect_drowsiness(frame, landmarks):
    LEFT_EYE_TOP_BOTTOM = [386, 374]
    LEFT_EYE_LEFT_RIGHT = [263, 362]
    RIGHT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_LEFT_RIGHT = [133, 33]
    UPPER_LOWER_LIPS = [13, 14]
    LEFT_RIGHT_LIPS = [78, 308]

    ratio_left_eye = get_aspect_ratio(frame, landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
    ratio_right_eye = get_aspect_ratio(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
    eye_ratio = (ratio_left_eye + ratio_right_eye) / 2
    ratio_lips = get_aspect_ratio(frame, landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
    
    eyes_closed = eye_ratio > 5.0
    yawning = ratio_lips < 1.8
    
    if eyes_closed:
        state = "SLEEPING"
    elif yawning:
        state = "YAWNING"
    else:
        state = "FOCUSED"
    
    return state

def detect_persons_with_attention(image):
    global current_session
    
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
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
    
    if detection_results.detections:
        for i, detection in enumerate(detection_results.detections):
            if i >= 3:  # Limit to 3 faces max for performance
                break
                
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            x, y = max(0, x), max(0, y)
            w, h = min(w, iw - x), min(h, ih - y)
            
            confidence_score = detection.score[0]
            status_text = "FOCUSED"
            
            # Simple face mesh matching
            if mesh_results.multi_face_landmarks and i < len(mesh_results.multi_face_landmarks):
                status_text = detect_drowsiness(image, mesh_results.multi_face_landmarks[i])
            
            # Draw detection box
            status_colors = {
                "FOCUSED": (0, 255, 0),
                "NOT FOCUSED": (0, 165, 255),
                "YAWNING": (0, 255, 255),
                "SLEEPING": (0, 0, 255)
            }
            
            color = status_colors.get(status_text, (0, 255, 0))
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv.putText(image, f"Person {i+1}: {status_text}", (x, y-10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "status": status_text,
                "timestamp": datetime.now().isoformat()
            })
    
    # Update session stats
    if monitoring_active:
        current_session['total_detections'] += len(detections)
    
    return image, detections

def generate_simple_pdf_report(output_path):
    """Generate a simple PDF report"""
    global current_session
    
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Smart Focus Alert - Session Report", title_style))
        story.append(Spacer(1, 20))
        
        # Session info
        if current_session['start_time']:
            duration = datetime.now() - current_session['start_time']
            duration_str = str(duration).split('.')[0]
        else:
            duration_str = "N/A"
        
        session_info = [
            ['Session Start', current_session['start_time'].strftime('%Y-%m-%d %H:%M:%S') if current_session['start_time'] else 'N/A'],
            ['Duration', duration_str],
            ['Total Detections', str(current_session['total_detections'])],
            ['Total Alerts', str(len(current_session['alerts']))]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 2*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(session_table)
        story.append(Spacer(1, 20))
        
        # Alerts
        if current_session['alerts']:
            story.append(Paragraph("Recent Alerts", styles['Heading2']))
            alert_data = [['Time', 'Message']]
            for alert in current_session['alerts'][-10:]:
                alert_data.append([alert['time'], alert['message']])
            
            alert_table = Table(alert_data, colWidths=[1.5*inch, 4*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(alert_table)
        
        doc.build(story)
        return output_path
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

def create_simple_video(output_path):
    """Create a simple demo video"""
    global current_session
    
    try:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, 10.0, (640, 480))
        
        # Create simple frames
        for i in range(60):  # 6 seconds at 10fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            cv.putText(frame, f"Session Recording - Frame {i+1}", (50, 150), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(frame, f"Total Alerts: {len(current_session['alerts'])}", (50, 200), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if current_session['start_time']:
                duration = (datetime.now() - current_session['start_time']).total_seconds()
                cv.putText(frame, f"Duration: {int(duration//60)}m {int(duration%60)}s", (50, 250), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            out.write(frame)
        
        out.release()
        return output_path if os.path.exists(output_path) else None
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        return None

# SIMPLIFIED ROUTES
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

@application.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Handle file upload (simplified)
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No file selected')
        
        # Process file (basic implementation)
        filename = secure_filename(file.filename)
        file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        result = {"filename": filename, "message": "File uploaded successfully"}
        return render_template('result.html', result=result)
    
    return render_template('upload.html')

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global monitoring_active, current_session
    
    # Reset session
    current_session = {
        'start_time': datetime.now(),
        'alerts': [],
        'total_detections': 0,
        'recording_frames': [],
        'last_pdf_path': None,
        'last_video_path': None
    }
    
    monitoring_active = True
    print("Monitoring started successfully")
    
    return jsonify({"status": "success", "message": "Monitoring started"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring_active, current_session
    
    print("Stopping monitoring...")
    monitoring_active = False
    
    # Generate files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate PDF
    pdf_filename = f"session_report_{timestamp}.pdf"
    pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
    pdf_result = generate_simple_pdf_report(pdf_path)
    
    # Generate Video
    video_filename = f"session_video_{timestamp}.mp4"
    video_path = os.path.join(application.config['RECORDINGS_FOLDER'], video_filename)
    video_result = create_simple_video(video_path)
    
    response_data = {
        "status": "success",
        "message": "Session completed successfully"
    }
    
    if pdf_result:
        current_session['last_pdf_path'] = pdf_path
        response_data["pdf_report"] = f"/reports/{pdf_filename}"
    
    if video_result:
        current_session['last_video_path'] = video_path
        response_data["video_file"] = f"/recordings/{video_filename}"
    
    print(f"Session stopped. Files: PDF={pdf_filename}, Video={video_filename}")
    return jsonify(response_data)

@application.route('/process_frame', methods=['POST'])
def process_frame():
    global current_session
    
    try:
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid frame"}), 400
        
        # Store frame (limit to last 50 frames to save memory)
        if monitoring_active:
            current_session['recording_frames'].append(frame.copy())
            if len(current_session['recording_frames']) > 50:
                current_session['recording_frames'] = current_session['recording_frames'][-50:]
        
        # Process frame
        processed_frame, detections = detect_persons_with_attention(frame)
        
        # Encode result
        _, buffer = cv.imencode('.jpg', processed_frame)
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections
        })
        
    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@application.route('/get_monitoring_data')
def get_monitoring_data():
    global current_session
    
    if not monitoring_active:
        return jsonify({"error": "Monitoring not active"})
    
    return jsonify({
        'total_persons': min(3, current_session['total_detections']),  # Limit display
        'focused_count': 0,
        'alert_count': len(current_session['alerts']),
        'current_status': 'ACTIVE',
        'latest_alerts': current_session['alerts'][-5:] if current_session['alerts'] else []
    })

@application.route('/check_camera')
def check_camera():
    return jsonify({"camera_available": False})

# File serving routes
@application.route('/reports/<filename>')
def serve_report(filename):
    return send_from_directory(application.config['REPORTS_FOLDER'], filename)

@application.route('/recordings/<filename>')
def serve_recording(filename):
    return send_from_directory(application.config['RECORDINGS_FOLDER'], filename)

@application.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(application.config['UPLOAD_FOLDER'], filename)

@application.route('/static/detected/<filename>')
def serve_detected(filename):
    return send_from_directory(application.config['DETECTED_FOLDER'], filename)

@application.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "monitoring_active": monitoring_active,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)
