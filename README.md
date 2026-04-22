# 🛡️ Stampede Shield: Advanced Crowd Safety AI

**Stampede Shield** is a high-performance, AI-driven crowd monitoring and stampede prevention system. Designed for security personnel, event organizers, and terminal managers, it utilizes real-time computer vision to detect early precursors of crowd panic and hazardous density.

---

## 🚀 Key Features

### 1. 🤖 Hybrid AI Intelligence
*   **Local Processing (90%):** Utilizes `COCO-SSD` via TensorFlow.js for immediate, private, and zero-latency person detection directly in your browser.
*   **Gemini Cognitive Layer (10%):** Proxies high-level scene analysis to Google Gemini for complex behavioral insights, predicting potential risks before they escalate.

### 2. 🏃 Kinetic Motion & Running Detection
*   **Limb Oscillation Analysis:** Unlike simple movement trackers, our **Kinetic Aspect Oscillation (KAO)** algorithm analyzes the rhythmic shape changes of human limbs to distinguish between actual running and moving objects (like vehicles or carts).
*   **Pace Sensitivity Controls:** Manually tune the sensitivity for "running" detection to match the environment (e.g., airports vs. stadiums).

### 3. 🚨 Disaster Response System
*   **Hoard Alert (Emergency):** Triggers a high-intensity alarm when a specific number of individuals start running simultaneously (indicative of a stampede onset).
*   **Density Thresholds:** Visual and audio warnings when crowd density exceeds safe parameters.
*   **Audio Panic Sensor (Acoustic Monitoring):** Uses the Web Audio API to analyze ambient noise in real-time. It identifies "Panic Signatures" by looking for high-decibel spikes in frequency ranges typical of human screams or distress shouts.
*   **Visual Peripheral Cues:** A "Red Flash" overlay triggers across the monitoring screen during high-risk events to grab operator attention instantly.

### 4. 🧠 Predictive & Behavioral Analytics
*   **Predictive Choke-Points:** The AI analyzes the trajectory vectors of every person to identify "Convergence Zones" where paths are likely to cross in a narrow space, predicting crush risks 3-5 seconds before they occur.
*   **Vulnerability Detection:** Identify and highlight persons with smaller relative physical profiles (children, elderly) to prioritize evacuation or assistance.
*   **Fallen & Trapped Person Detection:** Detects horizontal postures (falls) or individuals immobilized near barriers under high cluster pressure.

### 5. 📬 Automated Mission Control (Escalation)
*   **Automated Email Reports:** Integrated with `Nodemailer`, the system can dispatch automated incident reports to emergency responders or stakeholders.
*   **Google App Password Integration:** Securely handles escalation through Gmail accounts using application-specific credentials.
*   **Manual Export:** Generate a comprehensive `.txt` audit trail containing geo-spatial data, peak metrics, and event logs for official reporting.

### 6. 🔊 Custom Sound Architecture
*   **User-Defined Alarms:** Upload your own specialized alarm tones or recorded instructions.
*   **Persistence Layer:** Custom audio files are stored locally in the browser's **IndexedDB**, ensuring your specialized instructions are ready even after a reboot without re-uploading.
*   **Dual Engine:** Seamlessly switches between synthesized sirens and custom high-fidelity audio playback.

### 7. 🛰️ Precision Geotagging & Audit Trail
*   **Geospacial Incident Mapping:** Every logged event, alert, and detection is automatically tagged with high-accuracy GPS coordinates (`[Lat, Lng]`).
*   **Spatial Context:** Provides responders with the exact position of a bottle-neck or incident zone.
*   **Post-incident Forensics:** Combined with the event log, this creates a complete timeline and spatial map of crowd movement for post-event analysis.

### 8. 📡 BLE Wearable & Haptic Dispatch
*   **Responder Haptics:** Connect to Bluetooth Low Energy (BLE) wearables (smartwatches, industrial pagers) to send vibration alerts.
*   **Immediate Alert Service (0x1802):** Uses industry-standard GATT services to ensure compatibility with professional security hardware.

### 9. 📊 Historical Analysis & Optimized Heatmaps
*   **Real-time Metrics:** Track People Count, Runner Count, and Motion Levels instantly.
*   **Dual-Line Trend Chart:** Visualize the correlation between total density (Blue) and running intensity (Red).
*   **Offscreen-Accelerated Heatmaps:** Utilizing canvas-based offscreen rendering, our heatmaps provide spatial density visualization without degrading CPU performance, even on high-density 4K feeds.

### 10. 🔄 Adaptive Environmental Calibration
*   **Contextual Intelligence:** The system automatically "learns" the environment for the first 30 seconds of deployment, establishing a baseline for normal density and kinetic activity.
*   **Continuous Recalibration:** Operators can trigger a "Recalibrate" event at any time to adapt to shifting crowd dynamics or lighting changes.

---

## 🛠 How to Use

### Initialization
1.  **Launch:** Click "Start Monitoring" or "Feed Local File".
2.  **Permissions:** Grant **Camera**, **Geolocation**, and **Bluetooth** access when prompted.
3.  **Wait for AI:** The "AI Engine" will initialize. Once the "System Active" green dot appears, the live feed will start.

### Real-time Operation
*   **Visual Boxes:** Blue boxes represent individuals; Red highlighted boxes represent individuals flagged as "Running."
*   **Alarm Management:** Use the "🔊 Alarm: ON/OFF" toggle to silence the siren. Visual alerts will still flash in "Silent Mode."
*   **Incident Log:** Review the bottom panel for a time-stamped, geotagged record of every significant event.

### Configuration (Settings ⚙️)
*   **Density Threshold:** Set how many people trigger a standard "Crowd Alert."
*   **Running Threshold:** Set how many runners trigger the "Hoard Alarm."
*   **Kinetic Activity:** Adjust the **Limb Movement Sensitivity**. Higher values reduce false positives from rigid moving objects.
*   **BLE Management:** Authorize or Revoke access to Bluetooth wearables in the dedicated peripheral section.

---

## 🔒 Privacy & Security

*   **Browser-Side Privacy:** Most AI processing happens on your local hardware. Video frames are never stored or sent to external servers unless Gemini Insights are triggered (which only sends non-identifiable metadata).
*   **Permission Control:** You have granular control to revoke Geolocation or Bluetooth access at any time through the browser or apps settings.

---

## 🖥️ Technical Briefing
*   **Core:** TypeScript/JavaScript + TensorFlow.js
*   **Style:** Tailwind CSS (Modern Hardware Interface)
*   **AI Models:** `coco-ssd` (Local Detection), `gemini-3-flash-preview` (Behavioral Intelligence)
*   **Backend:** Express.js + Nodemailer (Critical Alert Routing)
*   **Storage:** LocalStorage (Config) + IndexedDB (Custom Sound Blobs)
*   **Network:** Geolocation API, Web Bluetooth API (BLE), Web Audio API
