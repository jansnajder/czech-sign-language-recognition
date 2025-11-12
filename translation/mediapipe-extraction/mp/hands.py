import cv2
import mediapipe as mp


class MpHands:

    def __init__(self):
        self._hands = mp.solutions.hands.Hands(max_num_hands=2,
                                               model_complexity=1,
                                               static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    def get_results(self, frame, draw: bool = True):
        frame.flags.writeable = False
        results = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True

        if results.multi_hand_landmarks and draw:
            self._draw_results(frame, results)

        return results