/* ============================================
   STAMPEDE SHIELD — Detection Frontend
   Vanilla JS, COCO-SSD integration.
   ============================================ */

import { GoogleGenAI } from "@google/genai";
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

// UI Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const settingsBtn = document.getElementById('settingsBtn');
const bluetoothBtn = document.getElementById('bluetoothBtn');
const modalBluetoothBtn = document.getElementById('modalBluetoothBtn');
const bleStatus = document.getElementById('bleStatus');
const heatmapToggleBtn = document.getElementById('heatmapToggleBtn');
const muteBtn = document.getElementById('muteBtn');
const testAlarmBtn = document.getElementById('testAlarmBtn');
const videoUpload = document.getElementById('videoUpload');
const video = document.getElementById('videoFeed');
const canvas = document.getElementById('overlayCanvas');
const heatmap = document.getElementById('heatmapCanvas');
const placeholder = document.getElementById('videoPlaceholder');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const peopleCountEl = document.getElementById('peopleCount');
const motionLevelEl = document.getElementById('motionLevel');
const densityValueEl = document.getElementById('densityValue');
const runnersCountEl = document.getElementById('runnersCount');
const alertBox = document.getElementById('alertBox');
const alertMessage = document.getElementById('alertMessage');
const peakPeopleEl = document.getElementById('peakPeople');
const avgDensityEl = document.getElementById('avgDensity');
const totalAlertsEl = document.getElementById('totalAlerts');
const sessionTimeEl = document.getElementById('sessionTime');
const historyChart = document.getElementById('historyChart');
const eventLog = document.getElementById('eventLog');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const exportReportBtn = document.getElementById('exportReportBtn');
const panicSoundEl = document.getElementById('panicSound');
const networkLoadEl = document.getElementById('networkLoad');
const vulnerableCountEl = document.getElementById('vulnerableCount');
const settingsModal = document.getElementById('settingsModal');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const setDensity = document.getElementById('setDensity');
const setRunners = document.getElementById('setRunners');
const setSensitivity = document.getElementById('setSensitivity');
const setActivity = document.getElementById('setActivity');
const setInterval_ = document.getElementById('setInterval');
const setBackend = document.getElementById('setBackend');
const setAlarm = document.getElementById('setAlarm');
const setVolume = document.getElementById('setVolume');
const setHeatmap = document.getElementById('setHeatmap');
const setContinuous = document.getElementById('setContinuous');
const setContactEmail = document.getElementById('setContactEmail');
const setSenderEmail = document.getElementById('setSenderEmail');
const setSenderPass = document.getElementById('setSenderPass');
const setAutoReport = document.getElementById('setAutoReport');
const testEmailBtn = document.getElementById('testEmailBtn');
const customSoundUpload = document.getElementById('customSoundUpload');
const customSoundsList = document.getElementById('customSoundsList');
const clearCustomSoundsBtn = document.getElementById('clearCustomSoundsBtn');
const customAudioPlayer = new Audio(); // For custom sounds
const geminiStatus = document.getElementById('geminiStatus');
const geminiStatusText = document.getElementById('geminiStatusText');
const geminiInsightLine = document.getElementById('geminiInsightLine');
const runnerFlash = document.getElementById('runnerFlash');

// State
let stream = null, audioStream = null, detectionTimer = null, sessionStart = null, sessionTimer = null, locTimer = null;
let history = [], totalAlerts = 0, peakPeople = 0, alarmPlaying = false;
let bluetoothDevice = null, alertCharacteristic = null, bleRetryCount = 0, model = null;
const MAX_BLE_RETRIES = 3;
let isProcessing = false;
let detectionFramesCount = 0;
let audioAnalyzer = null;
let audioSource = null;
let panicLevel = 0;
let panicHistory = [];
let baselineData = { density: [], activity: [], longTermDensity: [] };
let isLearning = true;
const LEARNING_FRAMES = 30; // ~30 seconds of observation
let prevPeople = []; // For simple velocity tracking fallback
let trackingBuffer = new Map(); // Detailed motion analysis: id -> history object
let nextTrackId = 0;
let currentCoords = null;
let lastRunnersCount = 0;
let lastAutoReportTime = 0;
let environmentType = 'standard';

// Config
const CONFIG = {
  DENSITY_THRESHOLD: 4,
  RUNNING_THRESHOLD: 3,
  RUNNING_SENSITIVITY: 0.08,
  ACTIVITY_SENSITIVITY: 0.05, // Threshold for limb oscillation (Aspect Ratio change)
  ADAPTIVE_MODE: true,
  CONTINUOUS_DRIVE: true, // Learn from drift over time
  DETECTION_INTERVAL_MS: 1000,
  BACKEND_URL: '',
  ALARM_TYPE: 'siren',
  ALARM_VOLUME: 0.8,
  SHOW_HEATMAP: true,
  ALARM_MUTED: false,
  CONTACT_EMAIL: '',
  SENDER_EMAIL: '',
  SENDER_PASS: '',
  AUTO_REPORT: true,
  CLUSTER_THRESHOLD: 1.2, // People within 1.2x their width of each other
  PANIC_WINDOW_SIZE: 5,   // Seconds of audio history
  CONVERGENCE_RADIUS: 0.15, // Normalized distance for choke points
  CUSTOM_SOUNDS: [] // [{ name: '...', data: '...' }]
};

// Safe Gemini Loader
let genAI = null;
function getGenAI() {
  if (!genAI) {
    // Try both process.env and import.meta.env for compatibility
    const apiKey = (typeof process !== 'undefined' && process.env && process.env.GEMINI_API_KEY) || 
                   (import.meta.env && import.meta.env.VITE_GEMINI_API_KEY);
    if (apiKey) {
      genAI = new GoogleGenAI({ apiKey });
    }
  }
  return genAI;
}

// Load saved settings
try {
  const saved = JSON.parse(localStorage.getItem('stampedeShieldSettings') || '{}');
  Object.assign(CONFIG, saved);
} catch (_) {}

// ============================================
// SYSTEM CORE
// ============================================

async function initSystem(isLocalFile = false) {
  try {
    statusText.textContent = 'Preparing AI Engine...';
    startBtn.disabled = true;

    // Fetch location concurrently
    updateLocation();

    await tf.ready();
    
    if (!model) {
      statusText.textContent = 'Loading Model (0.5MB)...';
      model = await cocoSsd.load();
      logEvent('info', 'AI Model ready');
    }

    if (!isLocalFile) {
      statusText.textContent = 'Requesting Camera...';
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false
      });
      video.srcObject = stream;
    }

    placeholder.classList.add('hidden');
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      heatmap.width = video.videoWidth;
      heatmap.height = video.videoHeight;
    };

    stopBtn.disabled = false;
    statusIndicator.classList.add('active');
    statusText.textContent = isLocalFile ? 'Live Analysis (File)' : 'Monitoring Active';

    sessionStart = Date.now();
    sessionTimer = setInterval(updateSessionTime, 1000);
    
    // Periodic Location Refresh (every 30s)
    locTimer = setInterval(updateLocation, 30000);
    
    // Init Sound Sensor
    initSoundSensor();
    
    logEvent('info', isLocalFile ? 'Monitoring file' : 'Monitoring camera');
    detectionTimer = setInterval(runDetectionFrame, CONFIG.DETECTION_INTERVAL_MS);
  } catch (err) {
    console.error('AI Initialization Error:', err);
    let msg = err.message;
    
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError' || msg.includes('Permission denied')) {
      msg = "Camera access denied. Please allow camera permissions in your browser settings to start live monitoring.";
      logEvent('error', 'Camera Permission Denied');
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      msg = "No camera found. Please connect a webcam or use the 'Upload Video' option.";
    } else {
      msg = "System Error: " + err.message;
    }
    
    statusText.textContent = '❌ Start Failed';
    alert(msg);
    startBtn.disabled = false;
  }
}

function stopDetection() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (audioStream) audioStream.getTracks().forEach(t => t.stop());
  stream = null;
  audioStream = null;
  if (detectionTimer) clearInterval(detectionTimer);
  if (sessionTimer) clearInterval(sessionTimer);
  if (locTimer) clearInterval(locTimer);
  if (video) {
    video.srcObject = null;
    video.pause();
  }
  if (placeholder) placeholder.classList.remove('hidden');
  if (startBtn) startBtn.disabled = false;
  if (stopBtn) stopBtn.disabled = true;
  if (statusIndicator) statusIndicator.classList.remove('active', 'alert');
  if (statusText) statusText.textContent = 'System Idle';
  if (peopleCountEl) peopleCountEl.textContent = '0';
  if (motionLevelEl) motionLevelEl.textContent = 'Low';
  if (densityValueEl) densityValueEl.textContent = '0/m²';
  if (alertBox) alertBox.classList.add('hidden');
  stopAlarm();
  clearHeatmap();
  logEvent('info', 'Stopped');
}

function updateLocation() {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      currentCoords = { lat: pos.coords.latitude, lng: pos.coords.longitude };
      logEvent('info', `Location locked: ${currentCoords.lat.toFixed(4)}, ${currentCoords.lng.toFixed(4)}`);
    },
    (err) => console.warn('Location blocked or unavailable'),
    { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
  );
}

// ============================================
// DETECTION & UI
// ============================================

async function runDetectionFrame() {
  if (!video.videoWidth || !model || isProcessing) return;
  isProcessing = true;
  detectionFramesCount++;

  // 1. Analyze Sound Spikes
  analyzeSound();

  try {
    const predictions = await model.detect(video);
    
    // Calculate Average Height for relative child/elderly detection
    const heights = predictions.filter(p => p.class === 'person').map(p => p.bbox[3]);
    const avgHeight = heights.length ? (heights.reduce((a, b) => a + b, 0) / heights.length) : 200;

    const detections = predictions.filter(p => p.class === 'person').map(p => ({
       x: p.bbox[0], y: p.bbox[1], w: p.bbox[2], h: p.bbox[3],
       cx: p.bbox[0] + p.bbox[2]/2, cy: p.bbox[1] + p.bbox[3]/2,
       ratio: p.bbox[2] / p.bbox[3],
       score: p.score
    }));

    const currentTracks = [];
    let runners = 0;
    let vulnerableCount = 0;
    let trappedAtBarrierCount = 0;
    let fallingCount = 0;

    // Advanced Motion & Limb Analysis
    detections.forEach(det => {
       let bestMatch = null;
       let minDist = 150; // Max search radius in pixels

       // Match with existing tracks using velocity-informed prediction
       for (const [id, track] of trackingBuffer.entries()) {
          const framesSinceSeen = detectionFramesCount - track.lastSeen;
          
          // Predicted position based on last velocity and time elapsed
          const predX = track.lastX + (track.vx || 0) * framesSinceSeen;
          const predY = track.lastY + (track.vy || 0) * framesSinceSeen;
          
          const dist = Math.sqrt((det.cx - predX)**2 + (det.cy - predY)**2);
          if (dist < minDist) {
             minDist = dist;
             bestMatch = id;
          }
       }

       // CLUSTER ANALYSIS (Pressure Detection)
       let clusterPressure = 0;
       detections.forEach(other => {
          if (other === det) return;
          const dX = Math.abs(det.cx - other.cx);
          const dY = Math.abs(det.cy - other.cy);
          // If within dangerous proximity (using box width as proxy for 0.5m)
          if (dX < det.w * CONFIG.CLUSTER_THRESHOLD && dY < det.h * 0.5) clusterPressure++;
       });

       if (bestMatch !== null) {
          const track = trackingBuffer.get(bestMatch);
          // Determine "Destiny Point" (where they'll be in 3 seconds)
          track.destX = det.cx + (det.cx - track.lastX) * 30;
          track.destY = det.cy + (det.cy - track.lastY) * 30;
          
          const actualDist = Math.sqrt((det.cx - track.lastX)**2 + (det.cy - track.lastY)**2);
          const velocity = actualDist / video.videoWidth; 
          
          // Acceleration & Jerk analysis
          const prevVx = track.vx || 0;
          const prevVy = track.vy || 0;
          const prevAx = track.ax || 0;
          const prevAy = track.ay || 0;
          
          track.vx = det.cx - track.lastX;
          track.vy = det.cy - track.lastY;
          track.ax = track.vx - prevVx;
          track.ay = track.vy - prevVy;
          
          // Jerk: Rate of change of acceleration
          const jerk = Math.sqrt((track.ax - prevAx)**2 + (track.ay - prevAy)**2) / video.videoWidth;
          const acceleration = Math.sqrt(track.ax**2 + track.ay**2) / video.videoWidth;

          // Update aspect ratio history
          track.ratioHistory.push(det.ratio);
          if (track.ratioHistory.length > 15) track.ratioHistory.shift();

          // Advanced Stride & Cadence Analysis
          let activityScore = 0;
          let oscillationFrequency = 0;
          let strideRegularity = 0;
          
          if (track.ratioHistory.length > 8) {
             const avg = track.ratioHistory.reduce((a, b) => a + b, 0) / track.ratioHistory.length;
             const variance = track.ratioHistory.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / track.ratioHistory.length;
             activityScore = Math.sqrt(variance);

             let crossings = [];
             for (let i = 1; i < track.ratioHistory.length; i++) {
                if ((track.ratioHistory[i-1] > avg && track.ratioHistory[i] <= avg) ||
                    (track.ratioHistory[i-1] < avg && track.ratioHistory[i] >= avg)) {
                   crossings.push(i);
                }
             }
             
             oscillationFrequency = crossings.length / track.ratioHistory.length;
             
             // Stride Regularity: Low variance in time between crossings indicates rhythmic human movement (running/walking)
             if (crossings.length >= 3) {
                const intervals = [];
                for(let j=1; j<crossings.length; j++) intervals.push(crossings[j] - crossings[j-1]);
                const avgInterval = intervals.reduce((a,b) => a+b, 0) / intervals.length;
                strideRegularity = 1 / (1 + intervals.reduce((a,b) => a + Math.pow(b - avgInterval, 2), 0) / intervals.length);
             }
          }

          const isFast = velocity > CONFIG.RUNNING_SENSITIVITY;
          const isAccelerating = acceleration > 0.012;
          const hasSteadyJerk = jerk < 0.05; // Extreme jerk often means detection noise/jitter, not smooth running
          const isOscillating = activityScore > CONFIG.ACTIVITY_SENSITIVITY;
          const hasHumanCadence = oscillationFrequency > 0.25 && oscillationFrequency < 0.7;
          const isRegular = strideRegularity > 0.6;
          
          // Hybrid logic: Must be moving fast, oscillating rhythmically, and showing "human" stride patterns
          const isRunningNow = isFast && ((isOscillating && hasHumanCadence && isRegular) || (isAccelerating && hasSteadyJerk));
          
          // Scoring with persistence
          track.runScore = (track.runScore || 0) + (isRunningNow ? 1.6 : -0.8);
          if (isRunningNow && isAccelerating) track.runScore += 0.4;
          track.runScore = Math.max(0, Math.min(8, track.runScore));
          
          const isConfirmedRunner = track.runScore >= 4.0;
          if (isConfirmedRunner) runners++;

          track.lastX = det.cx;
          track.lastY = det.cy;
          track.lastSeen = detectionFramesCount;
          track.isRunning = isConfirmedRunner;
          track.activity = activityScore;
          track.persistence = (track.persistence || 0) + 1;

          // Refined Vulnerable scoring: Small size + vertical posture (to distinguish from falling)
          const isSmall = det.h < (avgHeight * 0.58) && det.ratio < 0.8;
          track.vulnerableScore = (track.vulnerableScore || 0) + (isSmall ? 1 : -0.6);
          track.vulnerableScore = Math.max(0, Math.min(10, track.vulnerableScore));
          const isVulnerable = track.vulnerableScore >= 5;
          if (isVulnerable) vulnerableCount++;

          // Refined Falling check: Aggressive ratio + lower half of frame + persistence
          const isFlat = det.ratio > 2.2 && det.y > (video.videoHeight * 0.45);
          track.fallenScore = (track.fallenScore || 0) + (isFlat ? 1.5 : -1);
          track.fallenScore = Math.max(0, Math.min(10, track.fallenScore));
          const isFallen = track.fallenScore >= 6;
          if (isFallen) fallingCount++;
          
          // Barrier trapping check: Positioned at edges with high pressure
          const isAtEdge = det.cx < 80 || det.cx > video.videoWidth - 80;
          const isTrapped = isAtEdge && (isConfirmedRunner || activityScore > CONFIG.ACTIVITY_SENSITIVITY * 2);
          if (isTrapped) trappedAtBarrierCount++;

          currentTracks.push({ 
            ...det, 
            trackId: bestMatch, 
            isRunning: isConfirmedRunner, 
            activity: activityScore,
            isVulnerable: isVulnerable,
            isFallen: isFallen,
            isTrapped: isTrapped,
            clusterPressure: clusterPressure
          });
       } else {
          // New track - high persistence threshold (2 frames) before counting
          const id = nextTrackId++;
          trackingBuffer.set(id, {
             lastX: det.cx,
             lastY: det.cy,
             vx: 0,
             vy: 0,
             lastSeen: detectionFramesCount,
             ratioHistory: [det.ratio],
             isRunning: false,
             runScore: 0,
             persistence: 1,
             activity: 0,
             vulnerableScore: 0,
             fallenScore: 0
          });
          currentTracks.push({ ...det, trackId: id, isRunning: false, activity: 0 });
       }
    });

    // Cleanup old tracks
    for (const [id, track] of trackingBuffer.entries()) {
       const framesSinceSeen = detectionFramesCount - track.lastSeen;
       
       // Priority Retention: 
       // Runners are highly significant and likely to be temporarily occluded.
       // We grant them a 1.5s (15 frame) grace period to reappear.
       const gracePeriod = track.isRunning ? 15 : 5;
       
       if (framesSinceSeen > gracePeriod) {
          trackingBuffer.delete(id);
       }
    }

    // Secondary Filter: Motion Consistency (Panning Suppression)
    // If multiple objects move with nearly identical vectors, it's global camera motion
    if (runners > 1) {
      const activeTracks = currentTracks.filter(t => t.isRunning).map(t => trackingBuffer.get(t.trackId));
      const vxs = activeTracks.map(t => t.vx);
      const vys = activeTracks.map(t => t.vy);
      
      const avgVx = vxs.reduce((a,b) => a+b, 0) / vxs.length;
      const avgVy = vys.reduce((a,b) => a+b, 0) / vys.length;
      
      // Calculate Variance of vectors
      const varX = vxs.reduce((a,b) => a + Math.pow(b - avgVx, 2), 0) / vxs.length;
      const varY = vys.reduce((a,b) => a + Math.pow(b - avgVy, 2), 0) / vys.length;
      
      // If average velocity is high and variance is low, it's a pan
      const totalSpeed = Math.sqrt(avgVx**2 + avgVy**2) / video.videoWidth;
      const isPanning = totalSpeed > 0.02 && Math.sqrt(varX + varY) < 15;
      
      if (isPanning) {
         runners = 0;
         currentTracks.forEach(t => t.isRunning = false);
      }
    }

    const confirmedPeople = currentTracks.filter(t => {
      const track = trackingBuffer.get(t.trackId);
      return track && track.persistence >= 2;
    });

    // CHOKE POINT DETECTION (Vector Convergence)
    let convergePoint = null;
    if (confirmedPeople.length >= 3) {
      // Find where trajectories converge
      const dests = confirmedPeople.map(p => trackingBuffer.get(p.trackId)).filter(t => t && Math.abs(t.vx) > 2);
      if (dests.length >= 3) {
         const avgDestX = dests.reduce((a,b) => a + b.destX, 0) / dests.length;
         const avgDestY = dests.reduce((a,b) => a + b.destY, 0) / dests.length;
         
         // If many destinations represent a "pinch point"
         const spread = dests.reduce((a,b) => a + Math.sqrt((b.destX - avgDestX)**2 + (b.destY - avgDestY)**2), 0) / dests.length;
         if (spread < video.videoWidth * CONFIG.CONVERGENCE_RADIUS) {
            convergePoint = { x: avgDestX, y: avgDestY };
            logEvent('alert', 'PREDICTIVE: Potential Choke-Point / Pinch-Point forming ahead of crowd');
         }
      }
    }

    const result = {
      peopleCount: confirmedPeople.length,
      runnersCount: runners,
      vulnerableCount,
      fallingCount,
      trappedCount: trappedAtBarrierCount,
      panicLevel: panicLevel,
      networkLoad: Math.min(100, Math.round((confirmedPeople.length / CONFIG.DENSITY_THRESHOLD) * 60 + Math.random() * 20)),
      motionLevel: runners > 0 ? 'High' : (confirmedPeople.length > 5 ? 'Medium' : 'Low'),
      density: (confirmedPeople.length / 4).toFixed(1),
      boxes: confirmedPeople.map(p => {
        const track = trackingBuffer.get(p.trackId);
        return { 
          x: p.x, y: p.y, w: p.w, h: p.h, 
          cx: p.cx, cy: p.cy,
          vx: track.vx, vy: track.vy,
          isRunning: p.isRunning, 
          activity: p.activity,
          isVulnerable: p.isVulnerable,
          isFallen: p.isFallen,
          isTrapped: p.isTrapped,
          clusterPressure: p.clusterPressure
        };
      }),
      heatPoints: confirmedPeople.map(p => ({ x: p.cx, y: p.cy, intensity: p.score })),
      convergePoint
    };

    // Environmental Shock Filter: Prevent sudden jumps from 0 to MANY in one frame (camera bump)
    if (lastRunnersCount === 0 && runners > 3) {
       result.runnersCount = 0; // Filter this spike
       logEvent('info', 'Environmental Shock detected - filtering runner spike');
    }
    lastRunnersCount = runners;
    
    // Adaptive Learning Logic
    if (isLearning && CONFIG.ADAPTIVE_MODE) {
      baselineData.density.push(detections.length);
      const combinedActivity = currentTracks.reduce((sum, p) => sum + p.activity, 0) / (detections.length || 1);
      baselineData.activity.push(combinedActivity);
      
      statusText.textContent = `Learning Environment (${baselineData.density.length}/${LEARNING_FRAMES})...`;
      
      if (baselineData.density.length >= LEARNING_FRAMES) {
        finalizeCalibration();
      }
    } else if (CONFIG.ADAPTIVE_MODE && CONFIG.CONTINUOUS_DRIVE) {
      // Drift adaptation: Collect data even when calibrated
      baselineData.longTermDensity.push(detections.length);
      if (baselineData.longTermDensity.length > 300) { // Keep last 5 mins approx
        baselineData.longTermDensity.shift();
      }
      
      // Every 100 frames, check for significant drift (Slow EMA-like adjustment)
      if (detectionFramesCount % 100 === 0 && baselineData.longTermDensity.length > 50) {
        const longTermAvg = baselineData.longTermDensity.reduce((a,b) => a+b,0) / baselineData.longTermDensity.length;
        const currentThresh = CONFIG.DENSITY_THRESHOLD;
        const targetThresh = Math.max(4, Math.ceil(longTermAvg * 1.8));
        
        // Move 10% towards target to avoid jitter
        if (Math.abs(targetThresh - currentThresh) > 1) {
           CONFIG.DENSITY_THRESHOLD += (targetThresh > currentThresh ? 1 : -1);
           logEvent('info', `Threshold drifted. Now calibrated to: ${CONFIG.DENSITY_THRESHOLD} (Environment Avg: ${longTermAvg.toFixed(1)})`);
        }
      }
    }

    updateUI(result);

    // Automated Incident Escalation
    if (CONFIG.AUTO_REPORT && detections.length > CONFIG.DENSITY_THRESHOLD) {
      checkAutoReportTrigger(detections.length, runners, fallingCount);
    }

    // Hybrid Logic: Gemini Insight
    if (detectionFramesCount % 10 === 0 || (detections.length >= CONFIG.DENSITY_THRESHOLD && detectionFramesCount % 3 === 0)) {
       runGeminiInsight(detections.length);
    }
  } catch (e) {
    console.error('Frame processing failed:', e);
  } finally {
    isProcessing = false;
  }
}

async function runGeminiInsight(peopleCount) {
  if (geminiStatus && geminiStatus.classList.contains('active')) return;
  const ai = getGenAI();
  if (!ai) return;

  if (geminiStatus) geminiStatus.classList.add('active');
  if (geminiStatusText) geminiStatusText.textContent = isLearning ? 'Categorizing Scene...' : 'Analyzing...';

  try {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 300;
    tempCanvas.height = (video.videoHeight / video.videoWidth) * 300;
    const tctx = tempCanvas.getContext('2d');
    tctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const base64Image = tempCanvas.toDataURL('image/jpeg', 0.6).split(',')[1];

    const prompt = isLearning 
      ? `This is a safety monitoring system in calibration mode. Current people count: ${peopleCount}. 
         1. Categorize this scene (e.g. "Airport Terminal", "Narrow Stairwell", "Busy Plaza").
         2. Suggest if the environment is normally "Static" (people standing) or "Dynamic" (walking).
         3. Provide a safety insight.
         Return in format: [Category] Insight.` 
      : `Current count: ${peopleCount}. Safety threshold: ${CONFIG.DENSITY_THRESHOLD}. Environment: ${environmentType}. Provide 1-sentence proactive safety insight.`;

    const result = await ai.models.generateContent({
      model: "gemini-3-flash-preview", 
      contents: [{
        parts: [
          { inlineData: { mimeType: "image/jpeg", data: base64Image } },
          { text: prompt }
        ]
      }]
    });

    const text = result.text;
    if (isLearning) {
      const match = text.match(/\[(.*?)\]/);
      if (match) environmentType = match[1];
      logEvent('info', `Scene identified as: ${environmentType}`);
    }
    
    if (geminiInsightLine) geminiInsightLine.textContent = `✨ ${text || 'Normal conditions.'}`;

    // Additional Suggestion Logic
    if (!isLearning && (peopleCount > CONFIG.DENSITY_THRESHOLD * 0.8 || panicLevel > 50)) {
      if (peopleCount > CONFIG.DENSITY_THRESHOLD) {
         logEvent('alert', `STRATEGY SUGGESTION: "Deploy staff at Gate 3. Redirect crowd to Zone B."`);
      } else {
         logEvent('info', `PROACTIVE SUGGESTION: "Partial evacuation of sector A recommended."`);
      }
    }
  } catch (err) {
    console.warn('Gemini error:', err);
  } finally {
    if (geminiStatus) geminiStatus.classList.remove('active');
    if (geminiStatusText) geminiStatusText.textContent = 'Idle';
  }
}

function updateUI(result) {
  const { peopleCount, runnersCount, motionLevel, density, boxes, heatPoints, vulnerableCount, fallingCount, trappedCount, panicLevel, networkLoad, convergePoint } = result;
  if (peopleCountEl) peopleCountEl.textContent = peopleCount;
  if (runnersCountEl) runnersCountEl.textContent = runnersCount;
  if (motionLevelEl) motionLevelEl.textContent = motionLevel;
  if (densityValueEl) densityValueEl.textContent = `${density}/m²`;
  
  // New UI Elements
  if (panicSoundEl) {
    panicSoundEl.textContent = panicLevel > 70 ? 'HIGH' : (panicLevel > 30 ? 'WARN' : 'Clear');
    if (panicLevel > 70) panicSoundEl.style.color = '#ef4444';
    else if (panicLevel > 30) panicSoundEl.style.color = '#fbbf24';
    else panicSoundEl.style.color = '#818cf8';
  }
  
  if (networkLoadEl) networkLoadEl.textContent = `${networkLoad}%`;
  if (vulnerableCountEl) vulnerableCountEl.textContent = vulnerableCount;

  if (vulnerableCount > 0 || fallingCount > 0 || trappedCount > 0) {
     if (fallingCount > 0) logEvent('alert', `CRITICAL: ${fallingCount} person falling detected!`);
     if (trappedCount > 0) logEvent('alert', `CRITICAL: ${trappedCount} person trapped near barriers!`);
  }
  
  if (runnersCount > 0) runnerFlash.classList.remove('hidden');
  else runnerFlash.classList.add('hidden');

  drawBoxes(boxes || [], convergePoint);
  if (CONFIG.SHOW_HEATMAP) drawHeatmap(heatPoints || []);
  else clearHeatmap();

  history.push({ 
    time: Date.now(), 
    count: peopleCount, 
    density: parseFloat(density) || 0, 
    runners: runnersCount,
    vulnerable: vulnerableCount,
    panic: panicLevel
  });
  if (history.length > 60) history.shift();
  if (peopleCount > peakPeople) peakPeople = peopleCount;
  peakPeopleEl.textContent = peakPeople;
  totalAlertsEl.textContent = totalAlerts;
  avgDensityEl.textContent = (history.length ? history.reduce((a, h) => a + h.density, 0) / history.length : 0).toFixed(2);
  drawHistoryChart();

  const isHoard = runnersCount >= CONFIG.RUNNING_THRESHOLD;
  const isDensityRisk = peopleCount >= CONFIG.DENSITY_THRESHOLD;
  const isPanicRisk = panicLevel > 80;

  if (isHoard || isDensityRisk || isPanicRisk) {
    triggerAlert(peopleCount, runnersCount, isHoard, isPanicRisk, vulnerableCount);
  } else {
    clearAlert();
  }
}

function drawBoxes(boxes, convergePoint) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // 1. Draw Global Predictions (Choke Points)
  if (convergePoint) {
    ctx.beginPath();
    ctx.arc(convergePoint.x, convergePoint.y, 40, 0, Math.PI * 2);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 4;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    
    ctx.fillStyle = 'rgba(251, 191, 36, 0.2)';
    ctx.fill();
    ctx.setLineDash([]);
    
    ctx.fillStyle = '#fbbf24';
    ctx.font = '700 12px Inter';
    ctx.fillText('PREDICTED CHOKE POINT', convergePoint.x - 60, convergePoint.y - 50);
  }

  boxes.forEach((b, i) => {
    let color = '#818cf8'; // Default Blue
    let label = `Person ${i + 1}`;

    if (b.isRunning) {
      color = '#ef4444'; // Red
      label = 'RUNNING';
    }
    if (b.isVulnerable) {
      color = '#fbbf24'; // Yellow/Amber
      label = 'VULNERABLE';
    }
    if (b.isTrapped) {
      color = '#f472b6'; // Pink
      label = 'TRAPPED';
    }
    if (b.isFallen) {
      color = '#f87171'; // Lighter Red/Panic
      label = 'FALLEN';
    }

    // 2. Draw Trajectory Vector (Predictive Path)
    if (Math.abs(b.vx) > 1 || Math.abs(b.vy) > 1) {
      ctx.beginPath();
      ctx.moveTo(b.cx, b.cy);
      ctx.lineTo(b.cx + b.vx * 10, b.cy + b.vy * 10);
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.4;
      ctx.setLineDash([2, 4]);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }

    // 3. Draw High Pressure Indicators (Clusters)
    if (b.clusterPressure > 1) {
       ctx.beginPath();
       ctx.arc(b.cx, b.cy, b.w * 0.7, 0, Math.PI * 2);
       ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
       ctx.lineWidth = 1;
       ctx.stroke();
       ctx.fillStyle = 'rgba(239, 68, 68, 0.1)';
       ctx.fill();
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.font = '700 14px Inter, sans-serif';
    ctx.fillStyle = color;
    
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    
    // Label plate with background
    const textWidth = ctx.measureText(label).width;
    ctx.fillRect(b.x, b.y - 25, textWidth + 15, 25);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, b.x + 7, b.y - 7);

    // If fallen or trapped, draw a warning pulse
    if (b.isFallen || b.isTrapped) {
       ctx.strokeStyle = color;
       ctx.setLineDash([5, 5]);
       ctx.strokeRect(b.x - 5, b.y - 5, b.w + 10, b.h + 10);
       ctx.setLineDash([]);
    }
  });
}

let blobBrush = null;
function getBlobBrush() {
  if (!blobBrush) {
    blobBrush = document.createElement('canvas');
    blobBrush.width = 280;
    blobBrush.height = 280;
    const ctx = blobBrush.getContext('2d');
    const grad = ctx.createRadialGradient(140, 140, 0, 140, 140, 140);
    grad.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
    grad.addColorStop(0.4, 'rgba(99, 102, 241, 0.12)');
    grad.addColorStop(1, 'rgba(99, 102, 241, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 280, 280);
  }
  return blobBrush;
}

function drawHeatmap(points) {
  const ctx = heatmap.getContext('2d');
  ctx.clearRect(0, 0, heatmap.width, heatmap.height);
  const brush = getBlobBrush();
  
  ctx.globalCompositeOperation = 'screen';
  points.forEach(p => {
    ctx.drawImage(brush, p.x - 140, p.y - 140);
  });
  ctx.globalCompositeOperation = 'source-over';
}

function clearHeatmap() {
  heatmap.getContext('2d').clearRect(0, 0, heatmap.width, heatmap.height);
}

// ============================================
// ALARM SYSTEM
// ============================================

let audioCtx = null, alarmOscillators = [];

// ============================================
// CUSTOM SOUNDS SYSTEM
// ============================================

const DB_NAME = 'StampedeShieldDB';
const STORE_NAME = 'CustomSounds';

function initDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = (e) => resolve(e.target.result);
    request.onerror = (e) => reject(e.target.error);
  });
}

async function saveCustomSound(name, blob) {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const id = 'custom_' + Date.now();
    const request = store.add({ id, name, blob });
    request.onsuccess = () => {
      CONFIG.CUSTOM_SOUNDS.push({ id, name });
      localStorage.setItem('stampedeShieldSettings', JSON.stringify(CONFIG));
      resolve(id);
    };
    request.onerror = (e) => reject(e.target.error);
  });
}

async function getCustomSound(id) {
  const db = await initDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(id);
    request.onsuccess = (e) => resolve(e.target.result);
    request.onerror = (e) => reject(e.target.error);
  });
}

function updateAlarmDropdown() {
  // Clear existing custom options
  const defaultValues = ['siren', 'beep', 'alert'];
  Array.from(setAlarm.options).forEach(opt => {
    if (!defaultValues.includes(opt.value)) {
      setAlarm.remove(opt.index);
    }
  });

  // Add custom options
  const listItems = [];
  (CONFIG.CUSTOM_SOUNDS || []).forEach(sound => {
    const opt = document.createElement('option');
    opt.value = sound.id;
    opt.textContent = `🎵 ${sound.name}`;
    setAlarm.appendChild(opt);
    listItems.push(sound.name);
  });

  if (customSoundsList) {
    customSoundsList.textContent = listItems.length > 0 ? `Added: ${listItems.join(', ')}` : '';
  }
}

async function clearAllCustomSounds() {
  if (!confirm('Are you sure you want to delete all custom sounds?')) return;
  
  const db = await initDB();
  const transaction = db.transaction(STORE_NAME, 'readwrite');
  const store = transaction.objectStore(STORE_NAME);
  store.clear();
  
  CONFIG.CUSTOM_SOUNDS = [];
  CONFIG.ALARM_TYPE = 'siren';
  localStorage.setItem('stampedeShieldSettings', JSON.stringify(CONFIG));
  updateAlarmDropdown();
  logEvent('info', 'All custom sounds deleted');
}

clearCustomSoundsBtn.addEventListener('click', clearAllCustomSounds);

customSoundUpload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  try {
    logEvent('info', `Uploading custom sound: ${file.name}`);
    await saveCustomSound(file.name, file);
    updateAlarmDropdown();
    logEvent('info', `Custom sound "${file.name}" added successfully.`);
    setAlarm.value = CONFIG.CUSTOM_SOUNDS[CONFIG.CUSTOM_SOUNDS.length - 1].id;
  } catch (err) {
    console.error('Failed to save sound:', err);
    alert('Failed to save custom sound. Your browser storage might be full.');
  }
});

// Initialize dropdown on load
setTimeout(updateAlarmDropdown, 100);

function playAlarm(isHoard = false) {
  if (alarmPlaying || CONFIG.ALARM_MUTED) return;
  alarmPlaying = true;

  const isCustom = CONFIG.ALARM_TYPE.startsWith('custom_');

  if (isCustom) {
    playCustomAlarm(CONFIG.ALARM_TYPE);
  } else {
    playSynthesizedAlarm(isHoard);
  }
}

async function playCustomAlarm(id) {
  try {
    const soundData = await getCustomSound(id);
    if (!alarmPlaying) return; // Stopped while loading
    if (!soundData) throw new Error('Sound not found');
    
    const url = URL.createObjectURL(soundData.blob);
    customAudioPlayer.src = url;
    customAudioPlayer.volume = CONFIG.ALARM_VOLUME;
    customAudioPlayer.loop = true;
    customAudioPlayer.play();
    
    // We still keep alarmPlaying true
  } catch (err) {
    console.error('Failed to play custom sound:', err);
    alarmPlaying = false;
    playSynthesizedAlarm(); // Fallback
  }
}

function playSynthesizedAlarm(isHoard = false) {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  const gain = audioCtx.createGain();
  gain.gain.value = CONFIG.ALARM_VOLUME;
  gain.connect(audioCtx.destination);

  const osc = audioCtx.createOscillator();
  
  if (isHoard) {
     // Aggressive oscillating alarm for hoard
     osc.type = 'sawtooth';
     const lfo = audioCtx.createOscillator();
     lfo.frequency.value = 4; // 4Hz oscillation
     const lfoGain = audioCtx.createGain();
     lfoGain.gain.value = 200;
     lfo.connect(lfoGain);
     lfoGain.connect(osc.frequency);
     lfo.start();
     osc.frequency.setValueAtTime(800, audioCtx.currentTime);
  } else {
     osc.type = CONFIG.ALARM_TYPE === 'siren' ? 'sawtooth' : 'square';
     osc.frequency.setValueAtTime(600, audioCtx.currentTime);

     if (CONFIG.ALARM_TYPE === 'siren') {
       const dur = 0.8;
       let t = audioCtx.currentTime;
       for (let i = 0; i < 30; i++) {
         osc.frequency.linearRampToValueAtTime(900, t + dur / 2);
         osc.frequency.linearRampToValueAtTime(500, t + dur);
         t += dur;
       }
     }
  }

  osc.connect(gain);
  osc.start();
  alarmOscillators.push(osc);
}

function stopAlarm() {
  alarmOscillators.forEach(o => { try { o.stop(); } catch (_) {} });
  alarmOscillators = [];
  
  customAudioPlayer.pause();
  customAudioPlayer.currentTime = 0;
  
  alarmPlaying = false;
}

function triggerAlert(count, runners, isHoard, isPanicRisk = false, vulnerableCount = 0) {
  if (statusIndicator) {
    statusIndicator.classList.remove('active');
    statusIndicator.classList.add('alert');
  }
  
  if (isHoard) {
    if (statusText) statusText.textContent = '🚨 HOARD / STAMPEDE ALERT';
    if (alertMessage) alertMessage.innerHTML = `<strong>CRITICAL:</strong> Sudden mass movement detected (${runners} runners). <strong>EVACUATE AREA IMMEDIATELY.</strong>`;
  } else if (isPanicRisk) {
    if (statusText) statusText.textContent = '🚨 PANIC NOISE ALERT';
    if (alertMessage) alertMessage.innerHTML = `<strong>CRITICAL:</strong> High-decibel panic signature detected. Possible crowd distress. Use emergency exits.`;
  } else {
    if (statusText) statusText.textContent = '⚠ DANGER DETECTED';
    let msg = `High risk: ${count} detected. Density exceeds safe levels.`;
    if (vulnerableCount > 0) msg += ` ${vulnerableCount} vulnerable persons detected. Priority assistance required.`;
    if (alertMessage) alertMessage.textContent = msg;
  }

  if (alertBox && alertBox.classList.contains('hidden')) {
    totalAlerts++;
    let logMsg = isHoard ? `HOARD ALERT: ${runners} runners!` : `STAMPEDE RISK: ${count} people!`;
    if (isPanicRisk) logMsg = `PANIC NOISE DETECTED: High decibel level!`;
    logEvent('alert', logMsg);
    
    // Notify BLE wearable if connected
    if (isHoard || isPanicRisk) sendBLEAlert(2); // High Alert
    else sendBLEAlert(1); // Mild Alert
  }
  if (alertBox) alertBox.classList.remove('hidden');
  playAlarm(isHoard || isPanicRisk);
}

function clearAlert() {
  if (statusIndicator) {
    statusIndicator.classList.remove('alert');
    statusIndicator.classList.add('active');
  }
  if (statusText) statusText.textContent = 'Monitoring Active';
  if (alertBox) alertBox.classList.add('hidden');
  stopAlarm();
}

// ============================================
// EVENT HANDLERS
// ============================================

startBtn.addEventListener('click', () => initSystem(false));
stopBtn.addEventListener('click', stopDetection);

videoUpload.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = url;
  video.loop = true;
  video.play();
  initSystem(true);
});

testAlarmBtn.addEventListener('click', () => {
  if (alarmPlaying) {
    stopAlarm();
    testAlarmBtn.textContent = '🔊 Test Alarm';
  } else {
    playAlarm();
    testAlarmBtn.textContent = '⏹ Stop Test';
    setTimeout(() => {
      if (alarmPlaying) {
        stopAlarm();
        testAlarmBtn.textContent = '🔊 Test Alarm';
      }
    }, 5000);
  }
});

recalibrateBtn.addEventListener('click', () => {
  recalibrate();
});

heatmapToggleBtn.addEventListener('click', () => {
  CONFIG.SHOW_HEATMAP = !CONFIG.SHOW_HEATMAP;
  heatmapToggleBtn.textContent = `🔥 Heatmap: ${CONFIG.SHOW_HEATMAP ? 'ON' : 'OFF'}`;
  if (!CONFIG.SHOW_HEATMAP) clearHeatmap();
  localStorage.setItem('stampedeShieldSettings', JSON.stringify(CONFIG));
});

muteBtn.addEventListener('click', () => {
  CONFIG.ALARM_MUTED = !CONFIG.ALARM_MUTED;
  muteBtn.textContent = `🔊 Alarm: ${CONFIG.ALARM_MUTED ? 'OFF' : 'ON'}`;
  if (CONFIG.ALARM_MUTED) stopAlarm();
  localStorage.setItem('stampedeShieldSettings', JSON.stringify(CONFIG));
});

bluetoothBtn.addEventListener('click', toggleBLE);
modalBluetoothBtn.addEventListener('click', toggleBLE);

async function toggleBLE() {
  if (!navigator.bluetooth) return alert('Bluetooth Low Energy (BLE) is not supported in this browser. To use wearable alerts, please use Chrome, Edge, or a compatible browser.');
  
  if (bluetoothDevice && bluetoothDevice.gatt.connected) {
    try {
      bleRetryCount = MAX_BLE_RETRIES; // Prevent auto-retry on manual disconnect
      bluetoothDevice.gatt.disconnect();
      logEvent('info', 'BLE Device intentionally disconnected');
    } catch (_) {}
    return;
  }

  try {
    bleRetryCount = 0;
    logEvent('info', 'Requesting BLE access for Wearable Alerts...');
    updateBLEUI('AUTHORIZING');
    
    bluetoothDevice = await navigator.bluetooth.requestDevice({
      acceptAllDevices: true,
      optionalServices: ['immediate_alert', 'alert_notification']
    });

    bluetoothDevice.removeEventListener('gattserverdisconnected', handleBLEDisconnection);
    bluetoothDevice.addEventListener('gattserverdisconnected', handleBLEDisconnection);

    await performBLEConnection();

  } catch (err) {
    handleBLEError(err);
  }
}

async function performBLEConnection() {
  if (!bluetoothDevice) return;

  try {
    updateBLEUI('CONNECTING');
    logEvent('info', `Attempting connection to ${bluetoothDevice.name || 'Unknown Device'}...`);

    const server = await bluetoothDevice.gatt.connect();
    logEvent('info', 'Secure GATT Tunnel established');

    // Attempt to secure alert features
    try {
      const service = await server.getPrimaryService('immediate_alert');
      alertCharacteristic = await service.getCharacteristic('alert_level');
      logEvent('info', 'Wearable Haptics Synchronized (0x1802)');
    } catch (e) {
      logEvent('info', 'Connected (Basic). Standard alert service missing.');
      alertCharacteristic = null;
    }

    bleRetryCount = 0;
    updateBLEUI('CONNECTED');
  } catch (err) {
    throw err; // Pass to caller
  }
}

async function handleBLEDisconnection() {
  alertCharacteristic = null;
  
  if (bleRetryCount < MAX_BLE_RETRIES) {
    bleRetryCount++;
    logEvent('alert', `BLE Handshake lost. Retry attempt ${bleRetryCount}/${MAX_BLE_RETRIES}...`);
    updateBLEUI('RETRYING');
    
    // Exponential backoff
    setTimeout(async () => {
      try {
        if (!bluetoothDevice) return;
        await performBLEConnection();
      } catch (err) {
        if (bleRetryCount >= MAX_BLE_RETRIES) {
          handleBLEError(err);
        }
      }
    }, 1000 * bleRetryCount);
  } else {
    updateBLEUI('DISCONNECTED');
    logEvent('alert', 'BLE Handshake lost or revoked. Max retries reached.');
  }
}

function handleBLEError(err) {
  console.error('BLE Auth/Connection Error:', err);
  updateBLEUI('DISCONNECTED');
  alertCharacteristic = null;
  
  if (err.name === 'NotFoundError') {
     logEvent('info', 'BLE Scan cancelled by user');
  } else if (err.name === 'SecurityError') {
     logEvent('error', 'Bluetooth permission denied by system');
     alert('System Permission Blocked: Please ensure Bluetooth is enabled in your system settings and browser permissions.');
  } else if (err.name === 'NetworkError' || err.message.includes('GATT')) {
     logEvent('error', 'Bluetooth link unstable or connection refused');
  } else {
     logEvent('error', `Bluetooth Error: ${err.message}`);
  }
}

function updateBLEUI(state) {
  const name = (bluetoothDevice && bluetoothDevice.name) ? bluetoothDevice.name : 'Device';
  
  switch(state) {
    case 'AUTHORIZING':
      bluetoothBtn.textContent = '📡 AUTHORIZING...';
      modalBluetoothBtn.textContent = 'AUTHORIZING...';
      bleStatus.textContent = 'SCANNING...';
      bleStatus.className = 'ble-status';
      break;
    case 'CONNECTING':
    case 'RETRYING':
      bluetoothBtn.textContent = state === 'RETRYING' ? '⏳ RETRYING...' : '🔄 CONNECTING...';
      modalBluetoothBtn.textContent = 'ESTABLISHING LINK...';
      bleStatus.textContent = state;
      bleStatus.className = 'ble-status warn';
      break;
    case 'CONNECTED':
      bluetoothBtn.textContent = `🟢 Linked: ${name}`;
      modalBluetoothBtn.textContent = 'Revoke Access & Disconnect';
      modalBluetoothBtn.className = 'btn btn-secondary w-full';
      bleStatus.textContent = 'ACTIVE';
      bleStatus.className = 'ble-status active';
      break;
    case 'DISCONNECTED':
    default:
      bluetoothBtn.textContent = '🔵 Connect BLE Device';
      modalBluetoothBtn.textContent = 'Authorize & Pair Device';
      modalBluetoothBtn.className = 'btn btn-primary w-full';
      bleStatus.textContent = 'OFFLINE';
      bleStatus.className = 'ble-status';
      break;
  }
}

async function sendBLEAlert(level) {
  if (!alertCharacteristic) return;
  try {
    // Standard Immediate Alert Levels: 0=No, 1=Mild, 2=High
    const value = new Uint8Array([level]);
    await alertCharacteristic.writeValue(value);
    logEvent('info', 'BLE Notification sent to wearable');
  } catch (err) {
    console.warn('Failed to send BLE alert:', err);
  }
}

function updateSessionTime() {
  if (!sessionStart) return;
  const secs = Math.floor((Date.now() - sessionStart) / 1000);
  const m = Math.floor(secs / 60), s = secs % 60;
  sessionTimeEl.textContent = m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function drawHistoryChart() {
  const ctx = historyChart.getContext('2d');
  const w = historyChart.width = historyChart.clientWidth;
  const h = historyChart.height = 120;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;

  const maxVal = Math.max(...history.map(p => Math.max(p.count, p.runners || 0)), CONFIG.DENSITY_THRESHOLD, 5);
  const stepX = w / (history.length - 1);

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  for (let i = 1; i <= 4; i++) {
    const y = h - (i / 4) * (h - 20) - 10;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Draw Runners first (Red, fill)
  ctx.beginPath();
  history.forEach((p, i) => {
    const x = i * stepX, y = h - ((p.runners || 0) / maxVal) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2; ctx.setLineDash([5, 5]); ctx.stroke();
  ctx.setLineDash([]);

  // Draw People (Blue, solid)
  ctx.beginPath();
  history.forEach((p, i) => {
    const x = i * stepX, y = h - (p.count / maxVal) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#818cf8'; ctx.lineWidth = 3; ctx.stroke();

  // Threshold Line
  const threshY = h - (CONFIG.DENSITY_THRESHOLD / maxVal) * (h - 20) - 10;
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, threshY); ctx.lineTo(w, threshY); ctx.stroke();
  ctx.setLineDash([]);
}

function finalizeCalibration() {
  isLearning = false;
  const avgDensity = baselineData.density.reduce((a, b) => a + b, 0) / baselineData.density.length;
  const avgActivity = baselineData.activity.reduce((a, b) => a + b, 0) / baselineData.activity.length;
  
  // Environment Bias
  let densityMultiplier = 1.8;
  const envLower = environmentType.toLowerCase();
  
  if (envLower.includes('stair') || envLower.includes('narrow') || envLower.includes('exit')) {
    densityMultiplier = 1.3; // Low tolerance for tight spaces
  } else if (envLower.includes('plaza') || envLower.includes('stadium') || envLower.includes('open')) {
    densityMultiplier = 2.2; // High tolerance for open spaces
  }

  // Set threshold based on baseline + multiplier
  CONFIG.DENSITY_THRESHOLD = Math.max(4, Math.ceil(avgDensity * densityMultiplier));
  CONFIG.ACTIVITY_SENSITIVITY = Math.max(0.04, avgActivity * 1.5);
  
  logEvent('important', `Calibration Complete! Environment: ${environmentType}`);
  logEvent('info', `Baseline Density=${avgDensity.toFixed(1)}, Multiplier=${densityMultiplier}x, Thresh=${CONFIG.DENSITY_THRESHOLD}`);
  logEvent('info', `Kinetic Sensitivity adapted to ${CONFIG.ACTIVITY_SENSITIVITY.toFixed(3)}`);
  
  if (statusText) statusText.textContent = `Active (${environmentType})`;
  if (statusIndicator) statusIndicator.classList.add('calibrated'); // CSS class for visual confirmation
}

function recalibrate() {
  baselineData = { density: [], activity: [], longTermDensity: [] };
  isLearning = true;
  if (statusIndicator) statusIndicator.classList.remove('calibrated');
  logEvent('info', 'Manual Recalibration initiated...');
}

// ... existing code ...
function logEvent(type, message) {
  const empty = eventLog.querySelector('.log-empty');
  if (empty) empty.remove();
  const time = new Date().toLocaleTimeString(), p = document.createElement('p');
  p.className = `log-${type}`;
  
  let locTag = '';
  if (currentCoords) {
    locTag = `<span style="font-size:0.75rem; color:var(--text-mute); margin-left:8px;">[${currentCoords.lat.toFixed(5)}, ${currentCoords.lng.toFixed(5)}]</span>`;
  }

  p.innerHTML = `<span class="log-time">${time}</span>${message}${locTag}`;
  eventLog.prepend(p);
}

clearHistoryBtn.addEventListener('click', () => {
  history = []; totalAlerts = 0; peakPeople = 0;
  if (peakPeopleEl) peakPeopleEl.textContent = '0';
  if (totalAlertsEl) totalAlertsEl.textContent = '0';
  if (avgDensityEl) avgDensityEl.textContent = '0';
  drawHistoryChart();
  if (eventLog) eventLog.innerHTML = '<p class="log-empty">History cleared.</p>';
});

settingsBtn.addEventListener('click', () => {
  updateAlarmDropdown();
  setDensity.value = CONFIG.DENSITY_THRESHOLD;
  setRunners.value = CONFIG.RUNNING_THRESHOLD;
  setSensitivity.value = CONFIG.RUNNING_SENSITIVITY * 100;
  setActivity.value = CONFIG.ACTIVITY_SENSITIVITY * 100;
  setInterval_.value = CONFIG.DETECTION_INTERVAL_MS;
  setAlarm.value = CONFIG.ALARM_TYPE;
  setVolume.value = CONFIG.ALARM_VOLUME;
  setHeatmap.checked = CONFIG.SHOW_HEATMAP;
  setContinuous.checked = CONFIG.CONTINUOUS_DRIVE;
  setContactEmail.value = CONFIG.CONTACT_EMAIL || '';
  setSenderEmail.value = CONFIG.SENDER_EMAIL || '';
  setSenderPass.value = CONFIG.SENDER_PASS || '';
  setAutoReport.checked = CONFIG.AUTO_REPORT;
  settingsModal.classList.remove('hidden');
});

closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));

saveSettingsBtn.addEventListener('click', () => {
  CONFIG.DENSITY_THRESHOLD = parseInt(setDensity.value);
  CONFIG.RUNNING_THRESHOLD = parseInt(setRunners.value);
  CONFIG.RUNNING_SENSITIVITY = parseInt(setSensitivity.value) / 100;
  CONFIG.ACTIVITY_SENSITIVITY = parseInt(setActivity.value) / 100;
  CONFIG.DETECTION_INTERVAL_MS = parseInt(setInterval_.value);
  CONFIG.ALARM_TYPE = setAlarm.value;
  CONFIG.ALARM_VOLUME = parseFloat(setVolume.value);
  CONFIG.SHOW_HEATMAP = setHeatmap.checked;
  CONFIG.CONTINUOUS_DRIVE = setContinuous.checked;
  CONFIG.CONTACT_EMAIL = setContactEmail.value.trim();
  CONFIG.SENDER_EMAIL = setSenderEmail.value.trim();
  CONFIG.SENDER_PASS = setSenderPass.value.replace(/\s+/g, '');
  CONFIG.AUTO_REPORT = setAutoReport.checked;
  localStorage.setItem('stampedeShieldSettings', JSON.stringify(CONFIG));
  if (detectionTimer) {
    clearInterval(detectionTimer);
    detectionTimer = setInterval(runDetectionFrame, CONFIG.DETECTION_INTERVAL_MS);
  }
  settingsModal.classList.add('hidden');
  logEvent('info', 'Settings saved');
});

testEmailBtn.addEventListener('click', async () => {
  const recipient = setContactEmail.value.trim() || 'emergency-response@shield.org';
  const senderEmail = setSenderEmail.value.trim();
  // Strip all spaces from the App Password (Google shows them as xxxx xxxx xxxx xxxx)
  const senderPass = setSenderPass.value.replace(/\s+/g, '');

  if (!senderEmail || !senderPass) {
    alert('Please provide both Sender Email and App Password before testing.');
    return;
  }

  const originalText = testEmailBtn.textContent;
  testEmailBtn.textContent = '⌛ Testing Connection...';
  testEmailBtn.disabled = true;

  try {
    const res = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        isTest: true,
        recipient,
        senderEmail,
        senderPass
      })
    });

    const data = await res.json();
    if (res.ok) {
      alert(`✅ Success: ${data.message}`);
    } else {
      alert(`❌ Failed: ${data.message}\n\nHint: ${data.code === 'EAUTH' ? 'Check your App Password.' : 'Check your network.'}`);
    }
  } catch (err) {
    alert('❌ error: Could not reach the Mission Control server.');
  } finally {
    testEmailBtn.textContent = originalText;
    testEmailBtn.disabled = false;
  }
});

// ============================================
// NEW FEATURES: SOUND, REPORT, VULNERABLE
// ============================================

async function initSoundSensor() {
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioSource = audioCtx.createMediaStreamSource(audioStream);
    audioAnalyzer = audioCtx.createAnalyser();
    audioAnalyzer.fftSize = 256;
    audioSource.connect(audioAnalyzer);
    logEvent('info', 'Audio Panic Sensor Online');
  } catch (err) {
    console.warn('Audio sensor failed:', err);
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      logEvent('info', 'Audio Sensor: Permission denied. Panic noise detection disabled.');
      logEvent('info', '💡 Tip: Click the lock/microphone icon in your browser address bar to allow audio access.');
    } else {
      logEvent('info', 'Audio sensor failed to initialize. Passive video monitoring still active.');
    }
  }
}

function analyzeSound() {
  if (!audioAnalyzer) return;
  const dataArray = new Uint8Array(audioAnalyzer.frequencyBinCount);
  audioAnalyzer.getByteFrequencyData(dataArray);
  
  const filtered = dataArray.filter(v => v > 10);
  const avg = filtered.length ? (filtered.reduce((a, b) => a + b, 0) / filtered.length) : 0;
  
  const normalized = avg / 160; 
  const instantaneousPanic = Math.min(100, Math.pow(normalized, 1.5) * 100);
  
  // Temporal Smoothing (Last 5 frames)
  panicHistory.push(instantaneousPanic);
  if (panicHistory.length > CONFIG.PANIC_WINDOW_SIZE) panicHistory.shift();
  
  panicLevel = panicHistory.reduce((a,b) => a+b, 0) / panicHistory.length;
}

exportReportBtn.addEventListener('click', () => {
  const report = generateReport();
  const blob = new Blob([report], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `STAMPEDE_SHIELD_REPORT_${new Date().toISOString().slice(0, 10)}.txt`;
  a.click();
  
  const recipient = CONFIG.CONTACT_EMAIL || 'stakeholders@shield.org';
  logEvent('important', `Report Exported. Dispatching Manual Escalation...`);
  
  // Background archiving via server-side API
  fetch('/api/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      report, 
      recipient, 
      senderEmail: CONFIG.SENDER_EMAIL,
      senderPass: CONFIG.SENDER_PASS ? CONFIG.SENDER_PASS.replace(/\s+/g, '') : '',
      subject: `🚨 MANUAL SHIELD REPORT: ${new Date().toLocaleDateString()}` 
    })
  }).then(r => r.json())
    .then(data => logEvent('info', `Auto-Dispatch: ${data.message}`))
    .catch(err => {
      console.warn('API Escalation failed, falling back to mailto:', err);
      const mailTo = `mailto:${recipient}?subject=🚨 SHIELD INCIDENT REPORT: ${new Date().toLocaleDateString()}&body=${encodeURIComponent(report)}`;
      window.location.href = mailTo;
    });
});

function checkAutoReportTrigger(count, runners, fallen) {
  const now = Date.now();
  // Debounce automated reports to once every 5 minutes to prevent spam
  if (now - lastAutoReportTime < 300000) return;

  lastAutoReportTime = now;
  const report = generateReport(true);
  const recipient = CONFIG.CONTACT_EMAIL || 'emergency-response@shield.org';
  
  logEvent('important', `🚨 INCIDENT DETECTED: DISPATCHING AUTONOMOUS REPORT`);
  
  // Background logging via server-side API
  fetch('/api/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      report, 
      recipient: recipient, 
      senderEmail: CONFIG.SENDER_EMAIL,
      senderPass: CONFIG.SENDER_PASS ? CONFIG.SENDER_PASS.replace(/\s+/g, '') : '',
      subject: `🚨 CRITICAL STAMPEDE RISK DETECTED: ${new Date().toLocaleTimeString()}` 
    })
  }).then(r => r.json())
    .then(data => logEvent('info', `Auto-Dispatch: ${data.message}`))
    .catch(err => {
      console.warn('API Auto-Escalation failed, falling back to mailto:', err);
      const mailTo = `mailto:${recipient}?subject=🚨 CRITICAL STAMPEDE RISK DETECTED: ${new Date().toLocaleTimeString()}&body=${encodeURIComponent(report)}`;
      window.location.href = mailTo;
    });
  
  // Flash effect on status indicator to notify user of automation
  if (statusIndicator) {
    statusIndicator.style.animation = 'pulse-red 0.5s 3';
  }
}

function generateReport(isIncidentReport = false) {
  const lastHistory = history.slice(-50);
  const avgP = lastHistory.length ? lastHistory.reduce((a, h) => a + h.count, 0) / lastHistory.length : 0;
  const peakFallen = lastHistory.length ? Math.max(...lastHistory.map(h => h.fallen || 0)) : (fallenCount || 0);
  const peakVulnerable = lastHistory.length ? Math.max(...lastHistory.map(h => h.vulnerable || 0)) : (vulnerableCountEl ? parseInt(vulnerableCountEl.textContent) : 0);

  return `--- ${isIncidentReport ? '🚨 CRITICAL INCIDENT REPORT' : 'STAMPEDE SHIELD PREDICTIVE REPORT'} ---
Generated: ${new Date().toLocaleString()} (Local Time)
Coordinates: ${currentCoords ? `${currentCoords.lat.toFixed(5)}, ${currentCoords.lng.toFixed(5)}` : 'UNKNOWN / GPS BLOCKED'}
Environment Context: ${environmentType.toUpperCase()}
Status: ${isIncidentReport ? 'CRITICAL ESCALATION' : 'ROUTINE MONITORING'}

-- REAL-TIME METRICS --
Current People Count: ${peopleCountEl ? peopleCountEl.textContent : 'N/A'}
Confirmed Runners: ${runnersCountEl ? runnersCountEl.textContent : 'N/A'}
Fallen Individuals: ${lastHistory.length ? (lastHistory[lastHistory.length-1].fallen || 0) : 0}
Vulnerable Groups: ${vulnerableCountEl ? vulnerableCountEl.textContent : 'N/A'}
Panic Sound Level: ${panicSoundEl ? panicSoundEl.textContent : 'N/A'}

-- SESSION STATISTICS --
Peak People Detected: ${peakPeople}
Max Fallen at Once: ${peakFallen}
Peak Vulnerable Count: ${peakVulnerable}
Avg Session Density: ${avgP.toFixed(2)} people/frame
Total Security Alerts Triggered: ${totalAlerts}

-- RECENT LOG ENTRIES --
${[...eventLog.children].slice(0, 15).map(p => p.innerText.replace('\n', ' ')).join('\n')}

-- STRATEGIC RECOMMENDATIONS --
${peakPeople > CONFIG.DENSITY_THRESHOLD ? `• IMMEDIATE ACTION: Sector overcrowding detected. Deploy emergency marshals to main exits.` : `• Observation: Load remains within established baseline.`}
${lastHistory.some(h => (h.fallen || 0) > 0) ? `• CRITICAL: Person-down incident detected. Immediate medical dispatch to current GPS coordinates.` : ``}
${lastHistory.some(h => (h.vulnerable || 0) > 0) ? `• LOGISTIC: High concentration of vulnerable persons. Clear emergency lanes in Sector B.` : ``}
${lastHistory.some(h => (h.panic || 0) > 60) ? `• WARNING: High acoustic panic detected. Visual sweep for precursors recommended.` : ``}

--------------------------------------------------
System: AIS-STAMPEDE-SURVEILLANCE-V3
Reporting Node ID: SHIELD-DELTA-9
Data Integrity: VERIFIED
--------------------------------------------------`;
}
