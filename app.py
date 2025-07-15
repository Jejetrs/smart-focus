<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Focus Monitoring - Smart Focus Alert</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global CSS Variables */
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

        /* Reset and Base Styles */
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

        /* Utility Classes */
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

        /* Container Component */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        /* Navbar Component */
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

        /* Card Component */
        .card {
            background: var(--card-bg);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            transition: all 0.3s ease;
        }

        .card:hover {
            border-color: rgba(96, 165, 250, 0.2);
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card-padding {
            padding: 24px;
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

        /* Button Component */
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

        /* Camera Setup Card */
        .camera-setup-notice {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--glass-border);
            margin-bottom: 24px;
            display: none;
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

        /* Sidebar Card Component */
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

        /* Live Indicator Component */
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

        /* Status Components */
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

        /* Statistics Grid Component */
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

        /* Toggle Switch Component */
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

        /* Page Layout */
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

        /* Monitoring Status - Full Width */
        .monitoring-status-card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--glass-border);
            margin-bottom: 24px;
        }

        /* Two Column Layout */
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

        .video-stream {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        /* Client Camera Elements */
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

        /* Animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in { animation: fadeIn 0.6s ease-out; }

        /* Responsive Design */
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
    <!-- Navbar Component -->
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
        <!-- Page Header -->
        <div class="page-header">
            <h1 class="page-title text-gradient">Live Focus Monitoring</h1>
            <p class="page-subtitle">Real-time focus detection with facial landmarks, automatic recording, and comprehensive reporting</p>
        </div>

        <!-- Camera Setup Notice -->
        <div class="camera-setup-notice" id="cameraSetupNotice">
            <div class="setup-content">
                <div class="setup-icon">
                    <i class="fas fa-info-circle"></i>
                </div>
                <div class="setup-text">
                    <div class="setup-title">Using Device Camera</div>
                    <div class="setup-subtitle" id="connectionSubtitle">Server camera not available. Using your device camera for Railway/ngrok compatibility.</div>
                </div>
            </div>
        </div>

        <!-- Monitoring Status Card - Full Width -->
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

        <!-- Two Column Layout -->
        <div class="content-grid">
            <!-- Left Column -->
            <div class="left-column">
                <!-- Video Panel -->
                <div class="video-container">
                    <!-- Server Camera (original) -->
                    <img src="" class="video-stream" id="videoStream" alt="Live Video Feed" style="display: none;">
                    
                    <!-- Client Camera (for Railway/ngrok) -->
                    <video id="clientVideo" class="client-video" autoplay muted></video>
                    <canvas id="clientCanvas" class="client-canvas" width="640" height="480"></canvas>
                    
                    <div class="video-placeholder" id="videoPlaceholder">
                        <i class="fas fa-video"></i>
                        <h3 style="margin-bottom: 12px;">Enhanced Detection Ready</h3>
                        <p style="margin-bottom: 20px;">Click "Start Monitoring" to begin detection with time-based alerts</p>
                        <div style="text-align: left; max-width: 400px; line-height: 1.6;">
                            <div style="margin-bottom: 8px;">• <strong>Sleep Detection:</strong> Alert after 10 seconds</div>
                            <div style="margin-bottom: 8px;">• <strong>Yawning Detection:</strong> Alert after 3.5 seconds</div>
                            <div style="margin-bottom: 8px;">• <strong>Unfocus Detection:</strong> Alert after 10 seconds</div>
                            <div>• <strong>Visual + Audio Alerts:</strong> Automatic notifications</div>
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

                <!-- Alert Volume Card -->
                <div class="left-card">
                    <div class="sidebar-header">
                        <i class="fas fa-volume-up"></i>
                        Alert Volume
                    </div>
                    
                    <label class="volume-label">Choose volume level:</label>
                    <select class="volume-select" id="alertVolume">
                        <option value="0.3">Low</option>
                        <option value="0.6" selected>Medium</option>
                        <option value="1.0">High</option>
                    </select>
                </div>

                <!-- Session Downloads Card -->
                <div class="left-card" style="display: none;" id="downloadsSection">
                    <div class="sidebar-header">
                        <i class="fas fa-download"></i>
                        Session Downloads
                    </div>
                    <div id="downloadItems">
                        <!-- Download items will be populated here -->
                    </div>
                </div>
            </div>

            <!-- Right Column -->
            <div class="right-column">
                <!-- Connection Status -->
                <div class="sidebar-card">
                    <div class="connection-status">
                        <div class="connection-icon">
                            <i class="fas fa-clock"></i>
                        </div>
                        <div class="connection-text">
                            <div class="connection-title">Ready to Connect</div>
                            <div class="connection-subtitle" id="connectionSubtitle">Webcam access ready</div>
                        </div>
                    </div>

                    <div class="audio-alert">
                        <i class="fas fa-volume-up" style="color: var(--text-secondary);"></i>
                        <span style="flex: 1; font-weight: 500;">Audio Alert</span>
                        <div class="toggle-switch" onclick="toggleAudioAlert()"></div>
                    </div>

                    <div class="audio-alert" style="margin-bottom: 0;">
                        <span style="color: var(--text-secondary); font-size: 0.9rem;" id="audioStatus">Sound Alerts Enabled</span>
                    </div>
                </div>

                <!-- Detection Thresholds -->
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

                <!-- Alert History -->
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
        // ===== COMPLETE SESSION VARIABLES =====
        let sessionStartTime = null;
        let alertCount = 0;
        let isMonitoring = false;
        let sessionTimer = null;
        let dataUpdateTimer = null;
        let lastAlertIds = new Set();
        let audioEnabled = true;
        let currentSessionId = null;

        // ===== CAMERA VARIABLES =====
        let usingClientCamera = false;
        let clientVideo = null;
        let clientCanvas = null;
        let clientCtx = null;
        let clientStream = null;
        let processingInterval = null;

        // ===== RECORDING VARIABLES =====
        let mediaRecorder = null;
        let recordedChunks = [];
        let isRecording = false;

        // ===== ALERT TRACKING VARIABLES =====
        let clientAlerts = [];
        let personStates = {};  
        let personTimers = {};  
        let alertThresholds = {
            'SLEEPING': 10000,      // 10 seconds in milliseconds
            'YAWNING': 3500,        // 3.5 seconds in milliseconds  
            'NOT FOCUSED': 10000    // 10 seconds in milliseconds
        };

        // ===== AUDIO SYSTEM =====
        let audioContext = null;
        let alertAudio = null;

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            initializePage();
            setupEventListeners();
            checkCameraSetup();
            initializeAudioSystem();
        });

        function initializePage() {
            document.body.style.opacity = "0";
            document.body.style.transition = "opacity 0.6s ease";
            setTimeout(() => {
                document.body.style.opacity = "1";
            }, 100);
            
            // Initialize client camera elements
            clientVideo = document.getElementById('clientVideo');
            clientCanvas = document.getElementById('clientCanvas');
            clientCtx = clientCanvas.getContext('2d');
        }

        // ===== Initialize Audio System =====
        function initializeAudioSystem() {
            try {
                // Create audio context for beep sounds
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                
                // Create alert audio for speech synthesis
                if ('speechSynthesis' in window) {
                    console.log('Speech synthesis available');
                } else {
                    console.log('Speech synthesis not available, using beep alerts only');
                }
            } catch (error) {
                console.log('Audio initialization failed:', error);
            }
        }

        // ===== Play Alert Sound =====
        function playAlertSound(alertType = 'default') {
            if (!audioEnabled || !audioContext) return;

            try {
                const volume = parseFloat(document.getElementById('alertVolume').value);
                
                // Create beep sound
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                // Different frequencies for different alert types
                const frequencies = {
                    'SLEEPING': 800,    // High frequency for urgent
                    'YAWNING': 600,     // Medium frequency
                    'NOT FOCUSED': 400,  // Lower frequency
                    'default': 500
                };
                
                oscillator.frequency.setValueAtTime(frequencies[alertType] || frequencies.default, audioContext.currentTime);
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0, audioContext.currentTime);
                gainNode.gain.linearRampToValueAtTime(volume, audioContext.currentTime + 0.1);
                gainNode.gain.linearRampToValueAtTime(0, audioContext.currentTime + 0.5);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.5);
                
            } catch (error) {
                console.error('Audio playback error:', error);
            }
        }

        // ===== Speak Alert Message =====
        function speakAlertMessage(message) {
            if (!audioEnabled || !window.speechSynthesis) return;

            try {
                const utterance = new SpeechSynthesisUtterance(message);
                const volume = parseFloat(document.getElementById('alertVolume').value);
                
                utterance.volume = volume;
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                
                window.speechSynthesis.speak(utterance);
            } catch (error) {
                console.error('Speech synthesis error:', error);
            }
        }

        // Check camera setup
        async function checkCameraSetup() {
            try {
                const response = await fetch('/check_camera');
                const data = await response.json();
                
                if (!data.camera_available) {
                    usingClientCamera = true;
                    document.getElementById('cameraSetupNotice').style.display = 'block';
                    document.getElementById('connectionSubtitle').textContent = 'Using device camera (Railway/ngrok mode)';
                    console.log('Using client-side camera for Railway/ngrok compatibility');
                }
            } catch (error) {
                usingClientCamera = true;
                document.getElementById('cameraSetupNotice').style.display = 'block';
                document.getElementById('connectionSubtitle').textContent = 'Using device camera (fallback mode)';
                console.log('Fallback to client-side camera');
            }
        }

        function setupEventListeners() {
            // Navigation links
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

            // Button hover effects
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

            // Keyboard shortcuts
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

            // Volume change notification
            document.getElementById('alertVolume').addEventListener('change', function() {
                showNotification(`Alert volume changed to ${this.selectedOptions[0].text}`, 'info');
            });

            // Enable audio context on user interaction
            document.addEventListener('click', function() {
                if (audioContext && audioContext.state === 'suspended') {
                    audioContext.resume();
                }
            }, { once: true });
        }

        function toggleAudioAlert() {
            audioEnabled = !audioEnabled;
            const toggle = document.querySelector('.toggle-switch');
            const statusText = document.getElementById('audioStatus');
            
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

        // ===== FIXED - startMonitoring function with recording =====
        async function startMonitoring() {
            try {
                // Reset alert tracking
                clientAlerts = [];
                personStates = {};
                personTimers = {};
                alertCount = 0;
                recordedChunks = [];

                // Start server monitoring
                const response = await fetch('/start_monitoring', { method: 'POST' });
                const data = await response.json();
                
                if (data.status !== 'success') {
                    throw new Error(data.message);
                }

                currentSessionId = data.session_id;

                if (usingClientCamera) {
                    await initializeClientCamera();
                    await startClientRecording();
                } else {
                    initializeServerCamera();
                }

                updateUIForActiveMonitoring();
                startDataUpdates();
                showNotification('Monitoring started with automatic recording and alert tracking', 'success');
            } catch (error) {
                showNotification('Failed to start monitoring: ' + error.message, 'error');
            }
        }

        // ===== Initialize client camera =====
        async function initializeClientCamera() {
            try {
                clientStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'user' },
                    audio: true  // Enable audio for recording
                });
                
                clientVideo.srcObject = clientStream;
                await new Promise((resolve) => {
                    clientVideo.onloadedmetadata = resolve;
                });
                
                clientVideo.style.display = 'none';
                clientCanvas.style.display = 'block';
                
                // Start processing frames
                processingInterval = setInterval(processClientFrame, 1000);
                
            } catch (error) {
                throw new Error('Failed to access device camera: ' + error.message);
            }
        }

        // ===== FIXED - Start Client Recording =====
        async function startClientRecording() {
            try {
                if (!clientStream) {
                    throw new Error('No camera stream available for recording');
                }

                // Create media recorder for the stream
                mediaRecorder = new MediaRecorder(clientStream, {
                    mimeType: 'video/webm; codecs=vp9'
                });

                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = function() {
                    console.log('Recording stopped, processing data...');
                    if (recordedChunks.length > 0) {
                        saveRecordingToServer();
                    }
                };

                // Start recording
                mediaRecorder.start(1000); // Collect data every 1 second
                isRecording = true;
                
                console.log('Client-side recording started');
                
            } catch (error) {
                console.error('Failed to start recording:', error);
                showNotification('Recording unavailable, but monitoring continues', 'warning');
            }
        }

        // ===== FIXED - Save Recording to Server =====
        async function saveRecordingToServer() {
            try {
                if (recordedChunks.length === 0) {
                    console.log('No recording data to save');
                    return null;
                }

                // Create blob from recorded chunks
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                console.log(`Recording blob size: ${blob.size} bytes`);

                // Convert to base64
                const reader = new FileReader();
                reader.onload = async function(e) {
                    const base64Data = e.target.result.split(',')[1];
                    
                    // Send to server
                    try {
                        const response = await fetch('/save_recording', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                recording_data: base64Data,
                                session_id: currentSessionId
                            })
                        });

                        const result = await response.json();
                        
                        if (result.status === 'success') {
                            console.log('Recording saved successfully');
                            return result.recording_url;
                        } else {
                            console.error('Failed to save recording:', result.error);
                            return null;
                        }
                    } catch (error) {
                        console.error('Error sending recording to server:', error);
                        return null;
                    }
                };

                reader.readAsDataURL(blob);
                
            } catch (error) {
                console.error('Error processing recording:', error);
                return null;
            }
        }

        // Initialize server camera
        function initializeServerCamera() {
            document.getElementById('videoStream').src = '/video_feed';
            document.getElementById('videoStream').style.display = 'block';
        }

        // ===== Process client frame with Alert Detection =====
        function processClientFrame() {
            if (!clientVideo || !isMonitoring || !clientStream) return;

            try {
                // Draw video frame to canvas
                clientCtx.drawImage(clientVideo, 0, 0, clientCanvas.width, clientCanvas.height);
                
                // Convert to base64
                const frameData = clientCanvas.toDataURL('image/jpeg', 0.8);

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

                        // Process detections for alerts
                        if (data.detections && data.detections.length > 0) {
                            processDetectionAlerts(data.detections);
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

        // ===== Process Detection Alerts =====
        function processDetectionAlerts(detections) {
            const currentTime = Date.now();

            detections.forEach(detection => {
                const personId = `person_${detection.id}`;
                const currentState = detection.status;

                // Initialize person tracking if not exists
                if (!personStates[personId]) {
                    personStates[personId] = null;
                    personTimers[personId] = {};
                }

                // Check if state changed
                if (personStates[personId] !== currentState) {
                    // State changed, reset timers
                    personStates[personId] = currentState;
                    personTimers[personId] = {};
                    
                    // Start timer for new state if it's a distraction state
                    if (alertThresholds[currentState]) {
                        personTimers[personId][currentState] = currentTime;
                    }
                } else {
                    // Same state continues, check if we need to trigger alert
                    if (alertThresholds[currentState] && personTimers[personId][currentState]) {
                        const duration = currentTime - personTimers[personId][currentState];
                        
                        // Check if duration exceeds threshold
                        if (duration >= alertThresholds[currentState]) {
                            triggerClientAlert(detection.id, currentState, Math.floor(duration / 1000));
                            
                            // Reset timer to prevent repeated alerts
                            personTimers[personId][currentState] = currentTime;
                        }
                    }
                }
            });

            // Update UI with current detection data
            updateUIWithDetections(detections);
        }

        // ===== Trigger Client Alert =====
        function triggerClientAlert(personId, alertType, duration) {
            const alertTime = new Date().toLocaleTimeString();
            const alertMessage = getAlertMessage(personId, alertType);
            
            // Create alert object
            const alert = {
                id: `${personId}_${alertType}_${Date.now()}`,
                time: alertTime,
                person: `Person ${personId}`,
                detection: alertType,
                message: alertMessage,
                duration: duration,
                type: alertType === 'SLEEPING' ? 'error' : 'warning'
            };

            // Add to local alerts array
            clientAlerts.unshift(alert);
            
            // Keep only last 20 alerts
            if (clientAlerts.length > 20) {
                clientAlerts = clientAlerts.slice(0, 20);
            }

            // Update alert count
            alertCount++;
            document.getElementById('alertCount').textContent = alertCount;

            // Play audio alert
            playAlertSound(alertType);
            
            // Speak alert message (with delay to avoid overlap)
            setTimeout(() => {
                speakAlertMessage(alertMessage);
            }, 500);

            // Update alert history UI
            updateClientAlertHistory();

            // Show notification
            showNotification(alertMessage, alert.type);

            console.log('Alert triggered:', alert);
        }

        // ===== Get Alert Message =====
        function getAlertMessage(personId, alertType) {
            const messages = {
                'SLEEPING': `Person ${personId} is sleeping - please wake up!`,
                'YAWNING': `Person ${personId} is yawning - please take a rest!`,
                'NOT FOCUSED': `Person ${personId} is not focused - please focus on screen!`
            };
            return messages[alertType] || `Person ${personId} attention alert`;
        }

        // ===== Update UI with Detections =====
        function updateUIWithDetections(detections) {
            // Update statistics
            document.getElementById('totalPersons').textContent = detections.length;
            
            const focusedCount = detections.filter(d => d.status === 'FOCUSED').length;
            document.getElementById('focusedCount').textContent = focusedCount;

            // Update current status based on detections
            let overallStatus = 'FOCUSED';
            if (detections.some(d => d.status === 'SLEEPING')) {
                overallStatus = 'SLEEPING';
            } else if (detections.some(d => d.status === 'YAWNING')) {
                overallStatus = 'YAWNING';
            } else if (detections.some(d => d.status === 'NOT FOCUSED')) {
                overallStatus = 'NOT FOCUSED';
            }

            updateCurrentStatus(overallStatus);
        }

        // ===== Update Client Alert History =====
        function updateClientAlertHistory() {
            const alertHistory = document.getElementById('alertHistory');
            
            if (clientAlerts.length === 0) {
                alertHistory.innerHTML = '<div class="text-center" style="color: var(--text-secondary); padding: 20px;"><i class="fas fa-clock" style="font-size: 2rem; margin-bottom: 12px; display: block;"></i><p>No alerts yet in this session.</p></div>';
                return;
            }

            alertHistory.innerHTML = clientAlerts.slice(0, 10).map(alert => 
                `<div class="alert-item" style="border-left-color: ${alert.type === 'error' ? 'var(--danger)' : 'var(--warning)'};">
                    <div style="flex: 1;">
                        <div style="font-weight: 500; margin-bottom: 4px;">${alert.message}</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">Duration: ${alert.duration}s</div>
                    </div>
                    <div class="alert-time">${alert.time}</div>
                </div>`
            ).join('');
        }

        // ===== FIXED - stopMonitoring function with recording save =====
        async function stopMonitoring() {
            try {
                // Stop recording first
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                }

                const response = await fetch('/stop_monitoring', { method: 'POST' });
                const data = await response.json();

                updateUIForInactiveMonitoring();
                stopDataUpdates();

                // Cleanup client camera
                if (clientStream) {
                    clientStream.getTracks().forEach(track => track.stop());
                    clientStream = null;
                }

                if (processingInterval) {
                    clearInterval(processingInterval);
                    processingInterval = null;
                }

                // Reset alert tracking
                personStates = {};
                personTimers = {};

                showNotification(`Session complete! Generated ${alertCount} alerts. PDF report and recording available.`, 'success');
                
                // Wait a moment for recording to process, then show downloads
                setTimeout(() => {
                    if (data.pdf_report || data.video_file) {
                        showDownloads(data.pdf_report, data.video_file);
                    } else {
                        // Show default download links
                        showDownloads(
                            `/static/reports/session_report_${currentSessionId}.pdf`,
                            `/static/recordings/session_recording_${currentSessionId}.webm`
                        );
                    }
                }, 2000);

            } catch (error) {
                showNotification('Failed to stop monitoring: ' + error.message, 'error');
            }
        }

        // ===== updateUIForInactiveMonitoring function =====
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
            
            // Hide all video elements
            document.getElementById('videoStream').style.display = 'none';
            document.getElementById('videoStream').src = '';
            document.getElementById('clientVideo').style.display = 'none';
            document.getElementById('clientCanvas').style.display = 'none';
            document.getElementById('videoPlaceholder').style.display = 'block';
            
            updateCurrentStatus('READY');
            
            // Clear session timer properly
            if (sessionTimer) {
                clearInterval(sessionTimer);
                sessionTimer = null;
            }
            
            document.getElementById('totalPersons').textContent = '0';
            document.getElementById('focusedCount').textContent = '0';
        }

        // Rest of functions remain the same as previous version
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
            alertHistory.innerHTML = '<div class="text-center" style="color: var(--text-secondary); padding: 20px;"><i class="fas fa-clock" style="font-size: 2rem; margin-bottom: 12px; display: block;"></i><p>Monitoring started - watching for alerts...</p></div>';
            
            lastAlertIds.clear();
        }

        function startDataUpdates() {
            dataUpdateTimer = setInterval(() => {
                if (isMonitoring && !usingClientCamera) {
                    // Only fetch server data if not using client camera
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
            }, 2000);
        }

        function stopDataUpdates() {
            if (dataUpdateTimer) {
                clearInterval(dataUpdateTimer);
                dataUpdateTimer = null;
            }
        }

        function updateMonitoringDisplay(data) {
            document.getElementById('totalPersons').textContent = data.total_persons || 0;
            document.getElementById('focusedCount').textContent = data.focused_count || 0;
            document.getElementById('alertCount').textContent = data.alert_count || 0;
            
            const currentStatus = data.current_status || 'READY';
            updateCurrentStatus(currentStatus);
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
            // This function is for server-side alerts (when not using client camera)
            if (usingClientCamera) return; // Skip if using client camera
            
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

        // ===== Session Timer function =====
        function startSessionTimer() {
            sessionTimer = setInterval(() => {
                if (sessionStartTime && isMonitoring) {
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
                            <div style="color: var(--text-secondary); font-size: 0.9rem;">PDF with ${alertCount} alerts tracked</div>
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
                            <div style="color: var(--text-secondary); font-size: 0.9rem;">WebM recording with face detection</div>
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
            if (clientCanvas) {
                // Create download link for screenshot
                const link = document.createElement('a');
                link.download = `screenshot_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.png`;
                link.href = clientCanvas.toDataURL();
                link.click();
                showNotification('Screenshot captured and downloaded', 'success');
            } else {
                showNotification('Screenshot not available', 'error');
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

        // Add slideInRight animation
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

        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            if (clientStream) {
                clientStream.getTracks().forEach(track => track.stop());
            }
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
            }
        });
    </script>
</body>
</html>
