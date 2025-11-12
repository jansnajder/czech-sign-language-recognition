from typing import Optional

from mp.hands import MpHands
from camera.capture import CaptureOffline, CaptureOnline
from prediction.prediction_model import PredictionModel
from postprocessing.letter_postprocessing import LetterPostprocessing


class FingerSpellingRecognition:

    def __init__(self, input_path: Optional[str] = None, output_path: Optional[str] = None):
        self._capture = self._get_capture(input_path, output_path)
        self._mp_hands = MpHands()
        self._prediction = PredictionModel()
        self._postprocessing = LetterPostprocessing()

    def _get_capture(self, input_path: Optional[str], output_path: Optional[str]):
        if input_path:
            capture = CaptureOffline(input_path, output_path)
        else:
            capture = CaptureOnline(output_path)

        return capture

    def run(self):
        while self._capture.video.isOpened():
            success, frame = self._capture.read_frame()

            if not success:
                if isinstance(self._capture, CaptureOnline):
                    print('Missing camera frame!')
                    continue
                else:
                    break

            mp_results = self._mp_hands.get_results(frame, draw=True)
            letter, diacritic = self._prediction.predict(mp_results)
            self._postprocessing.process(letter, diacritic)
            frame = self._capture.print(frame, self._postprocessing.predicted_string)
            end_video = self._capture.show(frame)

            if end_video:
                break

        self._capture.release()
        print(self._postprocessing.predicted_string)


if __name__ == '__main__':
    app = FingerSpellingRecognition()
    app.run()
