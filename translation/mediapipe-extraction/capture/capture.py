import cv2

from abc import ABC
from typing import Optional


class Capture(ABC):

    def __init__(self, video, output_path: Optional[str] = None):
        self.out = None
        self.video = video

        if self.video.isOpened():
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

        self._setup_output(output_path)

    def read_frame(self):
        success, frame = self.video.read()
        return success, frame

    def release(self) -> None:
        self.video.release()

        if self.out:
            self.out.release()
        else:
            pass

    def _setup_output(self, output_path: Optional[str]) -> None:
        if output_path:
            self.fourcc = cv2.VideoWriter_fourcc(*'mjpg')
            self.out = cv2.VideoWriter(output_path, self.fourcc, 30, (640, 480))

    def _write_frame(self, frame) -> None:
        if self.out:
            self.out.write(frame)


class CaptureOnline(Capture):

    def __init__(self, output_path: Optional[str] = None):
        video = cv2.VideoCapture(0)
        super().__init__(video, output_path)


class CaptureOffline(Capture):

    def __init__(self, input_path: str, output_path: Optional[str] = None):
        video = cv2.VideoCapture(input_path)
        super().__init__(video, output_path)
