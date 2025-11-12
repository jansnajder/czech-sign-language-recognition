import cv2
import mediapipe as mp


class MpPose:

    def __init__(self):
        self._pose = mp.solutions.pose.Pose(static_image_mode=False,
                                            model_complexity=0,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)

    def get_results(self, frame, draw: bool = True):
        frame.flags.writeable = False
        results = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True

        if results.pose_landmarks and draw:
            self._draw_results(frame, results)

        return results
