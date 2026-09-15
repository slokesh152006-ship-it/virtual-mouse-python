import cv2
import mediapipe as mp
import pyautogui
import math
import time

# -----------------------------
# Virtual Mouse Using Python
# -----------------------------

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Previous mouse position
prev_x, prev_y = 0, 0

# Smoothing factor
smoothening = 5

# Click delay
last_click_time = 0
click_delay = 0.5


def distance(x1, y1, x2, y2):
    """Calculate distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


while True:

    success, frame = cap.read()

    if not success:
        print("Unable to access webcam.")
        break

    # Flip webcam image
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip (Landmark 8)
            index_finger = hand_landmarks.landmark[8]

            # Thumb tip (Landmark 4)
            thumb = hand_landmarks.landmark[4]

            # Middle finger tip (Landmark 12)
            middle_finger = hand_landmarks.landmark[12]

            # Convert index finger coordinates
            index_x = int(index_finger.x * frame_width)
            index_y = int(index_finger.y * frame_height)

            # Convert thumb coordinates
            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)

            # Convert middle finger coordinates
            middle_x = int(middle_finger.x * frame_width)
            middle_y = int(middle_finger.y * frame_height)

            # -----------------------------
            # Mouse Movement
            # -----------------------------

            # Map webcam coordinates to screen coordinates
            mouse_x = int(index_finger.x * screen_width)
            mouse_y = int(index_finger.y * screen_height)

            # Smooth mouse movement
            current_x = prev_x + (mouse_x - prev_x) / smoothening
            current_y = prev_y + (mouse_y - prev_y) / smoothening

            pyautogui.moveTo(
                int(current_x),
                int(current_y)
            )

            prev_x = current_x
            prev_y = current_y

            # -----------------------------
            # Left Click
            # -----------------------------

            thumb_index_distance = distance(
                thumb_x,
                thumb_y,
                index_x,
                index_y
            )

            if thumb_index_distance < 40:

                current_time = time.time()

                if current_time - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = current_time

                    cv2.putText(
                        frame,
                        "LEFT CLICK",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            # -----------------------------
            # Right Click
            # -----------------------------

            thumb_middle_distance = distance(
                thumb_x,
                thumb_y,
                middle_x,
                middle_y
            )

            if thumb_middle_distance < 40:

                current_time = time.time()

                if current_time - last_click_time > click_delay:
                    pyautogui.rightClick()
                    last_click_time = current_time

                    cv2.putText(
                        frame,
                        "RIGHT CLICK",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

            # -----------------------------
            # Display Instructions
            # -----------------------------

            cv2.putText(
                frame,
                "Index Finger: Move Mouse",
                (20, frame_height - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Thumb + Index: Left Click",
                (20, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Thumb + Middle: Right Click",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # Show webcam
    cv2.imshow(
        "Virtual Mouse Using Python",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
