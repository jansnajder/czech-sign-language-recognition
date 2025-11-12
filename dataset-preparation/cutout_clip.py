import os
import cv2

from skimage.metrics import structural_similarity
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def get_offset(minutes, fps, total_count):
    return (total_count - (minutes * fps *  60))


if __name__ == '__main__':
    '''Script cutting out all segments based on border frames'''
    input_path = r''
    start_frame = 'start.png'
    end_frame = 'end.png'
    start_img = cv2.imread(start_frame)
    end_img = cv2.imread(end_frame)
    start_gray = cv2.cvtColor(start_img, cv2.COLOR_BGR2GRAY)
    end_gray = cv2.cvtColor(end_img, cv2.COLOR_BGR2GRAY)

    start_height, start_width, _ = start_img.shape
    end_height, end_width, _ = end_img.shape

    video_names = os.listdir(input_path)
    video_count = len(video_names)
    fail = []

    for video_name in video_names:
        print(f'Loading file {video_name}')

        target_filename = video_name.rstrip('.mp4') + '_pocasi.mp4'
        target_path = os.path.join('out', target_filename)

        if os.path.isfile(target_path):
            print(f'Skipping file {video_name}(already done)')
            continue

        path = os.path.join(input_path, video_name)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if not ((start_width == width == end_width) and  (start_height == height == end_height)):
            print(f'Skipping file {video_name}(invalid dimensions)')
            fail.append(f'{video_name} - invalid_dimension')
            continue

        # set starting point offset from the end of the video
        minutes = 2
        offset = get_offset(minutes, fps, total_count)
        found = []

        while offset > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, offset)

            found.clear()

            for to_find in (start_gray, end_gray):
                while cap.isOpened():
                    ret, frame = cap.read()

                    if ret:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        score, _ = structural_similarity(to_find, frame_gray, full=True)
                        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)

                        if score < 0.7:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + 12)
                        elif 0.85> score > 0.7:
                            print(f'{frame_num}: {score}')
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + 6)
                        elif score >= 0.85:
                            found.append(frame_num)

                            if len(found) == 1:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + 350)

                            print(f'Found {len(found)}/{len(video_count)}')
                            break

                    else:
                        break

            if len(found) == 2:
                break
            else:
                minutes += 1
                offset = get_offset(minutes, fps, total_count)
                print(f'Could not find both border images, setting offset to {minutes} minutes.')

        cap.release()

        if len(found) != 2:
            print(f'Border images not found in {video_name}. Proceeding to next file.')
            fail.append(video_name)
            continue

        found = [k // fps for k in found]

        ffmpeg_extract_subclip(path, found[0] + 3, found[1], targetname=target_path)
        print(f'Successfully finished with {video_name}.')

    print(f'Finished - failed videos: {", ".join(fail)}.')

    if fail:
        with open('output.txt', 'w') as fh:
            fh.write(', '.join(fail))
