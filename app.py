import cv2
import time
import numpy as np
import HandTrackingModule as htm
import subprocess
import threading
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

################################
wCam, hCam = 640, 480
################################

# Initialize global variables
volBar = 400
volPer = 0
colorVol = (255, 0, 0)
currentVol = 50  # Initial value
last_update_time = time.time()
lock = threading.Lock()  # Thread safety


# Function to update system volume
def update_system_volume():
    global currentVol
    try:
        output = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"],
                                capture_output=True, text=True)
        if output.stdout.strip().isdigit():
            with lock:
                currentVol = int(output.stdout.strip())
    except Exception as e:
        st.error(f"Error updating volume: {str(e)}")


# Function to set system volume
def set_system_volume(volume_percent):
    try:
        subprocess.run(["osascript", "-e", f"set volume output volume {volume_percent}"],
                       check=True)
        return True
    except subprocess.SubprocessError:
        return False


# Fetch initial volume
try:
    update_system_volume()
except Exception:
    st.warning("Could not get initial system volume. Using default value.")


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pTime = 0
        self.detector = htm.handDetector(detectionCon=0.7, maxHands=1)
        self.smoothing = 5  # For volume smoothing
        self.volumeValues = []  # Store recent volume values
        self.last_set_vol = currentVol

    def transform(self, frame):
        global volPer, volBar, colorVol, last_update_time, currentVol

        img = frame.to_ndarray(format="bgr24")

        # Find hands
        img = self.detector.findHands(img)
        lmList, bbox = self.detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            # Calculate hand size
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

            # Only process if hand is at appropriate distance (not too close or far)
            if 200 < area < 1500:
                # Find distance between thumb and index finger
                length, img, lineInfo = self.detector.findDistance(4, 8, img)

                # Convert length to volume range (50-200 pixels maps to 0-100%)
                # Adjusted for better control
                volBar = np.interp(length, [30, 220], [400, 150])
                volPer = np.interp(length, [30, 220], [0, 100])

                # Smooth volume (average of recent values)
                self.volumeValues.append(volPer)
                if len(self.volumeValues) > self.smoothing:
                    self.volumeValues.pop(0)

                smoothedVol = sum(self.volumeValues) / len(self.volumeValues)
                volPer = round(smoothedVol / 5) * 5  # Round to nearest 5%

                # Check finger gesture (pinky down to activate)
                fingers = self.detector.fingersUp()

                # If pinky is down, set volume
                if fingers and not fingers[4]:
                    if abs(self.last_set_vol - volPer) > 2:  # Only update if change is significant
                        success = set_system_volume(volPer)
                        if success:
                            self.last_set_vol = volPer
                            with lock:
                                currentVol = volPer

                        # Visual feedback
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        colorVol = (0, 255, 0)
                else:
                    colorVol = (255, 0, 0)

        # Update system volume periodically
        if time.time() - last_update_time > 1.0:
            thread = threading.Thread(target=update_system_volume, daemon=True)
            thread.start()
            last_update_time = time.time()

        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

        # Display volume percentage
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 2)

        # Display current system volume
        with lock:
            cv2.putText(img, f'System: {int(currentVol)}%', (400, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, colorVol, 2)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if (cTime - self.pTime) > 0 else 0
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    0.8, (255, 0, 0), 2)

        return img


# Streamlit UI
def main():
    st.set_page_config(page_title="Hand Gesture Volume Control",
                       page_icon="🎛️",
                       layout="wide")

    st.title("Hand Gesture Volume Control 🎛️")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Instructions")
        st.markdown("""
        1. Show your hand to the camera
        2. Move your thumb & index finger apart to adjust volume
        3. Lower your pinky finger to set the volume
        """)

        st.subheader("Current Volume")
        vol_text = st.empty()
        vol_bar = st.empty()
        status = st.empty()

        # Update the sidebar with current volume
        def update_sidebar():
            while True:
                with lock:
                    vol = currentVol
                vol_text.metric(label="🔊 System Volume", value=f"{int(vol)}%")
                vol_bar.progress(vol / 100)
                status.markdown(f"Volume Status: {'🟢 Active' if colorVol == (0, 255, 0) else '🔴 Standby'}")
                time.sleep(0.1)

        # Start the sidebar updater in a thread
        threading.Thread(target=update_sidebar, daemon=True).start()

    with col1:
        st.markdown("### Camera Feed")
        webrtc_ctx = webrtc_streamer(
            key="gesture-volume-control",
            video_transformer_factory=VideoTransformer,
            media_stream_constraints={"video": {"width": wCam, "height": hCam}, "audio": False},
            async_processing=True,
        )

        st.markdown("""
        ### Tips for Best Results:
        - Ensure good lighting on your hand
        - Keep your hand at a reasonable distance from the camera
        - Make clear gestures with your fingers
        """)


if __name__ == "__main__":
    main()
