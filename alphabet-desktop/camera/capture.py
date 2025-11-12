import cv2
import math
import numpy as np

from abc import ABC
from typing import Optional
from PIL import ImageFont, ImageDraw, Image


class Capture(ABC):
    '''Abstract class encapsulating capture, i.e. getting the input image from a source.'''

    FONT = ImageFont.truetype(r'C:\Windows\Fonts\calibri.ttf', 40)

    def __init__(self, output_path: Optional[str], video):
        self.out = None
        self.video = video

        if self.video.isOpened():
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

        self._setup_output(output_path)

    def read_frame(self):
        '''Read a frame from the source.'''
        success, frame = self.video.read()
        return success, frame

    def release(self) -> None:
        '''Release the video source.'''
        self.video.release()

        if self.out:
            self.out.release()
        else:
            pass

    def show(self, frame) -> bool:
        '''Show the frame.

        :param frame: frame to display
        '''
        end_video = False
        cv2.imshow('Fingerspelling recognition', frame)
        self._write_frame(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            end_video = True

        return end_video

    def print(self, frame, string: str):
        '''Print string in the top left corner of the frame.

        :param frame: frame to print to
        :param string: string to print
        '''
        pil_frame = Image.fromarray(frame)
        pil_draw = ImageDraw.Draw(pil_frame)
        position = (math.floor(0.05*self.width), math.floor(0.05*self.height))
        pil_draw.text((position), string, font=self.FONT)
        frame = np.asarray(pil_frame)
        return frame

    def _setup_output(self, output_path: Optional[str]) -> None:
        if output_path:
            self.fourcc = cv2.VideoWriter_fourcc(*'mjpg')
            self.out = cv2.VideoWriter(output_path, self.fourcc, 30, (640, 480))
        else:
            print('No output path, capture will not be saved.')

    def _write_frame(self, frame) -> None:
        if self.out:
            self.out.write(frame)
        else:
            pass


class CaptureOnline(Capture):
    '''Online capture from web-cam.

    :param output_path: path to output the video to, None if no output necessary
    '''

    def __init__(self, output_path: Optional[str]):
        video = cv2.VideoCapture(0)
        super().__init__(output_path, video)


class CaptureOffline(Capture):
    '''Offline capture from source file.

    :param input_path: path to video source
    :param output_path: path to output the video to, None if no output necessary
    '''

    def __init__(self, input_path: str, output_path: Optional[str]):
        video = cv2.VideoCapture(input_path)
        super().__init__(output_path, video)
