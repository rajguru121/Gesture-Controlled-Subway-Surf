import pyautogui
import math
import cv2
import numpy as np
import time
import os
import mediapipe as mp
from collections import deque

# ── MediaPipe Tasks API setup ────────────────────────────────────────────────
# The legacy mp.solutions.pose API was removed in mediapipe >= 0.10.x
# We use the new Tasks API and create compatibility wrappers so the rest
# of the code works unchanged.

BaseOptions        = mp.tasks.BaseOptions
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode
_PoseLandmark      = mp.tasks.vision.PoseLandmark
PoseConnections    = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
_draw_landmarks    = mp.tasks.vision.drawing_utils.draw_landmarks
_DrawingSpec       = mp.tasks.vision.drawing_utils.DrawingSpec

# Use FULL model for better accuracy (lite is faster but less precise)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pose_landmarker_full.task")
# Fallback to lite if full doesn't exist
if not os.path.exists(_MODEL_PATH):
    _MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pose_landmarker_lite.task")


# ── Compatibility shim: mimic the old mp.solutions.pose API ──────────────────

class _PoseLandmarkCompat:
    """Mimic mp.solutions.pose.PoseLandmark with integer .value attribute."""
    NOSE            = int(_PoseLandmark.NOSE)
    LEFT_EYE        = int(_PoseLandmark.LEFT_EYE)
    RIGHT_EYE       = int(_PoseLandmark.RIGHT_EYE)
    LEFT_SHOULDER   = int(_PoseLandmark.LEFT_SHOULDER)
    RIGHT_SHOULDER  = int(_PoseLandmark.RIGHT_SHOULDER)
    LEFT_ELBOW      = int(_PoseLandmark.LEFT_ELBOW)
    RIGHT_ELBOW     = int(_PoseLandmark.RIGHT_ELBOW)
    LEFT_WRIST      = int(_PoseLandmark.LEFT_WRIST)
    RIGHT_WRIST     = int(_PoseLandmark.RIGHT_WRIST)
    LEFT_HIP        = int(_PoseLandmark.LEFT_HIP)
    RIGHT_HIP       = int(_PoseLandmark.RIGHT_HIP)
    LEFT_KNEE       = int(_PoseLandmark.LEFT_KNEE)
    RIGHT_KNEE      = int(_PoseLandmark.RIGHT_KNEE)
    LEFT_ANKLE      = int(_PoseLandmark.LEFT_ANKLE)
    RIGHT_ANKLE     = int(_PoseLandmark.RIGHT_ANKLE)


class _LandmarkAttr:
    """Single landmark with .x .y .z .visibility .presence attributes."""
    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z
        self.visibility = lm.visibility
        self.presence = getattr(lm, 'presence', lm.visibility)


class _LandmarkListWrapper:
    """Wraps new-API landmark list to look like old-API pose_landmarks."""
    def __init__(self, landmarks_list):
        self.landmark = [_LandmarkAttr(lm) for lm in landmarks_list]


class _ResultWrapper:
    """Wraps new PoseLandmarkerResult to mimic old API result object."""
    def __init__(self, tasks_result):
        if tasks_result.pose_landmarks and len(tasks_result.pose_landmarks) > 0:
            self.pose_landmarks = _LandmarkListWrapper(tasks_result.pose_landmarks[0])
        else:
            self.pose_landmarks = None


class _PoseCompat:
    """Drop-in replacement for mp.solutions.pose.Pose using Tasks API."""
    def __init__(self, static_image_mode=True, min_detection_confidence=0.5,
                 model_complexity=1, min_tracking_confidence=0.5):
        options = PoseLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def process(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        tasks_result = self._landmarker.detect(mp_image)
        return _ResultWrapper(tasks_result)


class _DrawingCompat:
    """Mimic mp.solutions.drawing_utils."""
    @staticmethod
    def draw_landmarks(image, landmark_list, connections,
                       landmark_drawing_spec=None, connection_drawing_spec=None):
        if landmark_list is None:
            return
        # Extract raw landmark objects for the Tasks drawing API
        raw_lms = landmark_list.landmark if hasattr(landmark_list, 'landmark') else landmark_list
        _draw_landmarks(image, raw_lms, PoseConnections,
                        landmark_drawing_spec or _DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        connection_drawing_spec or _DrawingSpec(color=(0, 0, 255), thickness=2))


# ── Create compatibility objects ─────────────────────────────────────────────
class _mp_pose_ns:
    """Namespace that acts like mp.solutions.pose"""
    PoseLandmark = _PoseLandmarkCompat
    POSE_CONNECTIONS = PoseConnections

    @staticmethod
    def Pose(**kwargs):
        return _PoseCompat(**kwargs)


mp_pose = _mp_pose_ns
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = _DrawingCompat()


# ── EMA (Exponential Moving Average) Smoothing Filter ────────────────────────
class LandmarkSmoother:
    """
    Applies exponential moving average to landmark coordinates to reduce
    jitter/noise frame-to-frame. Higher alpha = more responsive but noisier.
    Lower alpha = smoother but more latency.
    """
    def __init__(self, alpha=0.45, num_landmarks=33):
        self.alpha = alpha
        self._prev = None

    def smooth(self, landmarks_wrapper):
        """Smooth landmark positions using EMA. Operates in-place on the wrapper."""
        if landmarks_wrapper is None or not hasattr(landmarks_wrapper, 'landmark'):
            self._prev = None
            return landmarks_wrapper

        curr = landmarks_wrapper.landmark

        if self._prev is None or len(self._prev) != len(curr):
            # First frame or landmark count changed — store and return as-is
            self._prev = [(lm.x, lm.y, lm.z) for lm in curr]
            return landmarks_wrapper

        for i, lm in enumerate(curr):
            px, py, pz = self._prev[i]
            lm.x = self.alpha * lm.x + (1 - self.alpha) * px
            lm.y = self.alpha * lm.y + (1 - self.alpha) * py
            lm.z = self.alpha * lm.z + (1 - self.alpha) * pz
            self._prev[i] = (lm.x, lm.y, lm.z)

        return landmarks_wrapper


# ── Gesture Majority-Vote Smoother ───────────────────────────────────────────
class GestureSmoother:
    """
    Buffers recent gesture decisions and returns the majority vote.
    Eliminates single-frame flickers.
    """
    def __init__(self, window=5):
        self._buffer = deque(maxlen=window)

    def vote(self, gesture):
        self._buffer.append(gesture)
        counts = {}
        for g in self._buffer:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get)


# ── Cooldown Timer ───────────────────────────────────────────────────────────
class CooldownTimer:
    """
    Prevents rapid-fire keypress spam. After a keypress, blocks the same
    action for `cooldown_sec` seconds.
    """
    def __init__(self, cooldown_sec=0.4):
        self.cooldown = cooldown_sec
        self._last = {}

    def ready(self, action):
        now = time.time()
        if action not in self._last:
            return True
        return (now - self._last[action]) >= self.cooldown

    def trigger(self, action):
        self._last[action] = time.time()


# ── Pose detection functions ─────────────────────────────────────────────────

def detectPose(image, pose, smoother=None, blankImage=False):

    output_image = image.copy()

    if blankImage:
        blank_image = np.zeros((720,1920,3), np.uint8)
        output_image = blank_image

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(imageRGB)

    height, width, _ = image.shape

    landmarks = []

    if results.pose_landmarks:
        # Apply EMA smoothing to reduce jitter
        if smoother is not None:
            results.pose_landmarks = smoother.smooth(results.pose_landmarks)

        # Filter out low-visibility detections (noise)
        avg_visibility = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
        if avg_visibility < 0.4:
            return output_image, [], results

        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:

            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    return output_image, landmarks, results


def calculateAngle(landmark1, landmark2, landmark3):

    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


def classifyPose(landmarks, output_image):

    label = 'Unknown Pose'

    color = (0, 0, 255)

    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])

    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:


            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    label = 'Warrior II Pose'

            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

                label = 'T Pose'

    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:

            label = 'Tree Pose'

    if label != 'Unknown Pose':

        color = (0, 255, 0)

    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    return output_image, label



def checkHandsJoined(img, results, draw=False):
    height, width, _ = img.shape

    output_img = img.copy()

    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)

    distance = int(math.hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                              left_wrist_landmark[1] - right_wrist_landmark[1]))

    if distance < 130:
        hand_status = 'Hands Joined'
        color = (0, 255, 0)

    else:
        hand_status = 'Hands Not Joined'
        color = (0, 0, 255)

    if draw:
        cv2.putText(output_img, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_img, f'Distance: {distance}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

    return output_img, hand_status


def checkLeftRight(img, results, draw=False):
    """
    Uses shoulder midpoint + hip midpoint for more stable left/right detection.
    Falls back to shoulder-only if hips not visible.
    """

    horizontal_position = None

    height, width, c = img.shape

    output_image = img.copy()

    lm = results.pose_landmarks.landmark

    # Use average of shoulders AND hips for more stable center detection
    l_shoulder_x = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x
    r_shoulder_x = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
    l_hip_x = lm[mp_pose.PoseLandmark.LEFT_HIP].x
    r_hip_x = lm[mp_pose.PoseLandmark.RIGHT_HIP].x

    # Body center X (average of all 4 torso points for stability)
    body_center_x = (l_shoulder_x + r_shoulder_x + l_hip_x + r_hip_x) / 4.0
    body_center_px = int(body_center_x * width)

    # Use a dead-zone around center to prevent flickering (10% of frame width)
    dead_zone = int(width * 0.08)
    center = width // 2

    if body_center_px < center - dead_zone:
        horizontal_position = 'Left'
    elif body_center_px > center + dead_zone:
        horizontal_position = 'Right'
    else:
        horizontal_position = 'Center'

    if draw:
        cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (center, 0), (center, height), (255, 255, 255), 2)
        # Draw dead-zone boundaries
        cv2.line(output_image, (center - dead_zone, 0), (center - dead_zone, height), (100, 100, 100), 1)
        cv2.line(output_image, (center + dead_zone, 0), (center + dead_zone, height), (100, 100, 100), 1)
        # Draw body center marker
        cv2.circle(output_image, (body_center_px, height // 2), 8, (0, 255, 255), -1)

    return output_image, horizontal_position


def checkJumpCrouch(img, results, MID_Y=250, draw=False):
    """
    Uses average of BOTH shoulders AND hips for more stable vertical detection.
    Wider dead-zone to prevent false jumps/crouches.
    """

    height, width, _ = img.shape

    output_image = img.copy()

    lm = results.pose_landmarks.landmark

    # Use average of shoulders AND hips for more stable Y position
    l_shoulder_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height
    r_shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height
    l_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * height
    r_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * height

    # Torso center Y (average of 4 points — much more stable)
    actual_mid_y = int((l_shoulder_y + r_shoulder_y + l_hip_y + r_hip_y) / 4.0)

    # Wider dead-zone to reduce false positives
    lower_bound = MID_Y - 25
    upper_bound = MID_Y + 50

    if (actual_mid_y < lower_bound):
        posture = 'Jumping'

    elif (actual_mid_y > upper_bound):
        posture = 'Crouching'

    else:
        posture = 'Standing'

    if draw:
        cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (0, MID_Y),(width, MID_Y),(255, 255, 255), 2)
        # Draw dead-zone
        cv2.line(output_image, (0, lower_bound), (width, lower_bound), (0, 200, 200), 1)
        cv2.line(output_image, (0, upper_bound), (width, upper_bound), (0, 200, 200), 1)
        # Draw current Y marker
        cv2.circle(output_image, (width // 2, actual_mid_y), 8, (0, 0, 255), -1)

    return output_image, posture


if __name__ == '__main__':

    # Higher detection confidence for more reliable poses
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, model_complexity=1)

    cap = cv2.VideoCapture(0)
    # Higher resolution for better landmark precision
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    pTime = 0

    # ── Smoothing & cooldown instances ───────────────────────────────────────
    landmark_smoother = LandmarkSmoother(alpha=0.5)      # EMA on landmark coords
    lr_smoother       = GestureSmoother(window=5)         # majority vote: left/right
    jc_smoother       = GestureSmoother(window=5)         # majority vote: jump/crouch
    cooldown          = CooldownTimer(cooldown_sec=0.35)  # 350ms between keypresses

    game_started = False
    x_pos_index = 1
    y_pos_index = 1
    MID_Y = None
    counter = 0
    num_of_frames = 10


    while True:
        success, img = cap.read()
        if not success:
            print("[Error] Could not read from camera. Check your webcam.")
            break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Pass the landmark smoother for EMA filtering
        img, landmarks, results = detectPose(img, pose_video, smoother=landmark_smoother)

        if landmarks:
            if game_started:
                img, raw_horizontal = checkLeftRight(img, results, draw=True)

                # Smooth left/right with majority vote
                horizontal_position = lr_smoother.vote(raw_horizontal)

                if (horizontal_position=='Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):
                    if cooldown.ready('left'):
                        pyautogui.press('left')
                        cooldown.trigger('left')
                        x_pos_index -= 1

                elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):
                    if cooldown.ready('right'):
                        pyautogui.press('right')
                        cooldown.trigger('right')
                        x_pos_index += 1

            else:

                cv2.putText(img, 'JOIN BOTH HANDS TO START THE GAME.', (5, h - 10), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 3)

            if checkHandsJoined(img, results)[1] == 'Hands Joined':

                counter += 1

                if counter == num_of_frames:

                    if not(game_started):

                        game_started = True

                        # Calibrate MID_Y using torso center (shoulders + hips)
                        lm = results.pose_landmarks.landmark
                        l_sh_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h
                        r_sh_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h
                        l_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
                        r_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
                        MID_Y = int((l_sh_y + r_sh_y + l_hip_y + r_hip_y) / 4.0)

                        pyautogui.click(x=1300, y=800, button='left')
                    else:
                        pyautogui.press('space')


                    counter = 0

            else:

                counter = 0

            if MID_Y:

                img, raw_posture = checkJumpCrouch(img, results, MID_Y, draw=True)

                # Smooth jump/crouch with majority vote
                posture = jc_smoother.vote(raw_posture)

                if posture == 'Jumping' and y_pos_index == 1:
                    if cooldown.ready('up'):
                        pyautogui.press('up')
                        cooldown.trigger('up')
                        y_pos_index += 1

                elif posture == 'Crouching' and y_pos_index == 1:
                    if cooldown.ready('down'):
                        pyautogui.press('down')
                        cooldown.trigger('down')
                        y_pos_index -= 1

                elif posture == 'Standing' and y_pos_index != 1:

                    y_pos_index = 1

        else:

            counter = 0

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('Game', img)
        k = cv2.waitKey(1) & 0xFF
        if(k == 27) or (k == 113):
            break
    cap.release()
    cv2.destroyAllWindows()