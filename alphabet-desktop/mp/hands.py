import cv2
import mediapipe as mp


class MpHands:
    '''Encapsulation of MediaPipe Hands solution.'''

    def __init__(self):
        self._hands = mp.solutions.hands.Hands(max_num_hands=1,
                                               static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self._drawing_style = mp.solutions.drawing_styles

    def get_results(self, frame, draw: bool = True):
        '''Get MPH results from the frame.

        :param frame: frame to obtain the results from
        :param draw: boolean indication whehter to draw the landmarks
        '''
        frame.flags.writeable = False
        results = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True

        if results.multi_hand_landmarks and draw:
            self._draw_results(frame, results)

        return results

    def _draw_results(self, frame, results):
        mp.solutions.drawing_utils.draw_landmarks(frame,
                                                  results.multi_hand_landmarks[0],
                                                  mp.solutions.hands.HAND_CONNECTIONS,
                                                  self._drawing_style.get_default_hand_landmarks_style(),
                                                  self._drawing_style.get_default_hand_connections_style())
