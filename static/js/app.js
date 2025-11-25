document.addEventListener('DOMContentLoaded', () => {
    const audioFileInput = document.getElementById('audiofile');
    const uploadButton = document.getElementById('uploadButton');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const statusText = document.getElementById('statusText');
    const emotionResult = document.getElementById('emotionResult');

    let mediaRecorder;
    let audioChunks = [];

    const API_ENDPOINT = '/predict';

    function setStatus(message, isError = false) {
        statusText.textContent = message;
        // Use inline style for quick feedback. Ensure .recording-status-text is defined in CSS.
        statusText.style.color = isError ? '#dc3545' : '#198754';
        if (!isError) {
            // Only clear the result display when starting a new analysis
            if (message.includes('Analyzing')) {
                emotionResult.textContent = 'Analyzing...';
                emotionResult.style.color = '#333';
            }
        }
    }

    function displayResult(emotion) {
        emotionResult.textContent = emotion;
        emotionResult.style.color = '#5b62e4'; // Match primary color
    }
    
    // --- API Sending Function ---
    async function analyzeAudio(audioBlob, filename) {
        setStatus("Sending audio for analysis...");
        const formData = new FormData();
        formData.append('audio_file', audioBlob, filename);

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                setStatus("Analysis complete. Result displayed below.", false);
                displayResult(data.predicted_emotion);
            } else {
                // Display the specific error message from the Flask server
                const errorMsg = data.error || 'Unknown server error.';
                setStatus(`Error: ${errorMsg}`, true);
                displayResult('Prediction failed.');
            }

        } catch (error) {
            console.error('Fetch error:', error);
            setStatus(`Network Error: ${error.message}. Check server connection.`, true);
            displayResult('Analysis failed.');
        }
    }

    // --- 1. File Upload Logic ---
    uploadButton.addEventListener('click', () => {
        const file = audioFileInput.files[0];
        if (!file) {
            setStatus("Please select an audio file first.", true);
            return;
        }
        
        // Disable buttons while analyzing
        uploadButton.disabled = true;
        recordButton.disabled = true;

        analyzeAudio(file, file.name).finally(() => {
            uploadButton.disabled = false;
            recordButton.disabled = false;
        });
    });

    // --- 2. Live Recording Logic ---

    recordButton.addEventListener('click', async () => {
        try {
            // Attempt to get microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: true,
                video: false // Ensure video is explicitly false
            });
            
            // --- CRITICAL FIX REVERSION: Use reliable WebM/Opus ---
            const TARGET_MIME_TYPE = 'audio/webm; codecs=opus';
            let mimeType = TARGET_MIME_TYPE;

            // Fallback check for standard webm
            if (!MediaRecorder.isTypeSupported(TARGET_MIME_TYPE)) {
                if (MediaRecorder.isTypeSupported('audio/webm')) {
                   mimeType = 'audio/webm';
                } else {
                    // This should generally not happen in a modern browser
                    throw new Error("No compatible recording format (WebM/Opus) supported by browser.");
                }
            }
            
            mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                // Once recording stops, close the stream tracks
                stream.getTracks().forEach(track => track.stop()); 

                // Use the successful MIME type from the recorder
                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType }); 
                
                // Use .webm extension for server processing
                const filename = `recording-${Date.now()}.webm`; 
                
                // Start analysis
                analyzeAudio(audioBlob, filename).finally(() => {
                    recordButton.disabled = false;
                    stopButton.disabled = true;
                });
            };

            // Start recording
            mediaRecorder.start();
            setStatus("Recording... Press 'Stop Recording' when finished.", false);
            recordButton.disabled = true;
            stopButton.disabled = false;

        } catch (error) {
            console.error('Microphone access failed:', error);
            
            // --- IMPROVED ERROR REPORTING ---
            let userErrorMsg = "Error: Could not access microphone. ";

            if (error.name === "NotAllowedError") {
                userErrorMsg += "Access denied. Check your browser permissions.";
            } else if (error.name === "NotFoundError" || error.name === "DeviceNotFoundError") {
                userErrorMsg += "No microphone device found.";
            } else if (error.name === "NotReadableError") {
                userErrorMsg += "Microphone is in use by another application.";
            } else if (error.message.includes("No compatible recording format")) {
                 userErrorMsg = "Fatal Error: Browser does not support standard WebM recording.";
            } else {
                userErrorMsg += "Check console for details.";
            }

            setStatus(userErrorMsg, true);
            recordButton.disabled = false;
            stopButton.disabled = true;
        }
    });

    stopButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            setStatus("Recording stopped. Sending audio for analysis...", false);
            // Buttons will be re-enabled after the API call completes in mediaRecorder.onstop
        }
    });
});
