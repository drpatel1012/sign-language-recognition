const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const predictionBox = document.getElementById('prediction');

let isPredicting = false;
let camera = null;

// Initialize MediaPipe Hands
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5
});

// Process results from MediaPipe
hands.onResults(onResults);

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw the raw camera image completely transparently, since MediaPipe draws
    // it into the underlying canvas, or just draw the image feed to canvas
    canvasCtx.drawImage(
        results.image, 0, 0, canvasElement.width, canvasElement.height
    );

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const handLandmarks = results.multiHandLandmarks[0];

        // Draw visual landmarks
        drawConnectors(canvasCtx, handLandmarks, HAND_CONNECTIONS,
                       {color: '#00FF00', lineWidth: 3});
        drawLandmarks(canvasCtx, handLandmarks, {color: '#FF0000', lineWidth: 1});

        if (isPredicting) {
            // Process and normalize landmarks to match Python processing
            let x_min = Infinity, y_min = Infinity;
            let x_max = -Infinity, y_max = -Infinity;

            handLandmarks.forEach(lm => {
                x_min = Math.min(x_min, lm.x);
                y_min = Math.min(y_min, lm.y);
                x_max = Math.max(x_max, lm.x);
                y_max = Math.max(y_max, lm.y);
            });

            const width = x_max - x_min;
            const height = y_max - y_min;

            // 21 landmarks * 3 coords = 63 features
            const normalizedLandmarks = [];
            handLandmarks.forEach(lm => {
                const nx = (lm.x - x_min) / (width + 1e-6);
                const ny = (lm.y - y_min) / (height + 1e-6);
                const nz = lm.z; // Keeping Z same as python
                normalizedLandmarks.push(nx, ny, nz);
            });

            sendForPrediction(normalizedLandmarks);
        }
    } else {
         if(isPredicting) {
              predictionBox.textContent = "--";
         }
    }
    canvasCtx.restore();
}

// Throttle predictions to avoid spamming the backend
let lastPredictionTime = 0;
const PREDICTION_DELAY_MS = 150; 

async function sendForPrediction(landmarks) {
    const now = Date.now();
    if (now - lastPredictionTime < PREDICTION_DELAY_MS) return;
    lastPredictionTime = now;

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ landmarks: landmarks })
        });

        if (response.ok) {
            const result = await response.json();
            if (result.prediction) {
                // Update UI visually
                predictionBox.textContent = result.prediction;
                predictionBox.style.transform = 'scale(1.1)';
                setTimeout(()=> predictionBox.style.transform = 'scale(1)', 100);
            }
        } else {
             console.error("Server error:", response.status);
        }
    } catch (e) {
        console.error("Prediction fetch error:", e);
        statusDiv.textContent = "Error connecting to backend.";
    }
}


// Camera controls
startBtn.addEventListener('click', async () => {
    statusDiv.textContent = "Loading camera and models...";
    startBtn.disabled = true;

    try {
        if (!camera) {
             camera = new Camera(videoElement, {
                onFrame: async () => {
                    await hands.send({image: videoElement});
                },
                width: 640,
                height: 480
            });
        }
        
        await camera.start();
        isPredicting = true;
        
        statusDiv.textContent = "Translating... Show your sign!";
        stopBtn.disabled = false;
    } catch(e) {
        statusDiv.textContent = "Error starting camera: " + e.message;
        startBtn.disabled = false;
    }
});

stopBtn.addEventListener('click', () => {
    isPredicting = false;
    if (camera) {
        camera.stop();
    }
    
    // Clear canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusDiv.textContent = "Stopped.";
    predictionBox.textContent = "--";
});
