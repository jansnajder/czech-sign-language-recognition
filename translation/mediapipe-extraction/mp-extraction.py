import os
import pandas as pd
from capture.capture import CaptureOffline
from mp.hands import MpHands
from mp.pose import MpPose
from data_handlers.process_results import process_results


if __name__ == '__main__':
    '''Extraction of key-points by MPH/P'''
    folder = ''
    output = 'output'
    files = os.listdir(folder)
    files_count = len(files)
    mp_pose = MpPose()
    mp_hands = MpHands()

    for i, file in enumerate(files):
        print(f'Starting {file}, {i}/{files_count}')
        output_data = []
        video_path = os.path.join(folder, file)
        cap = CaptureOffline(video_path)

        while True:
            success, frame = cap.read_frame()

            if not success:
                cap.release()
                break

            hands_res = mp_hands.get_results(frame, draw=True)
            pose_res = mp_pose.get_results(frame, draw=True)
            output_data.append(process_results(hands_res, pose_res))

            if cap.show(frame, show=False):
                break

        if output_data:
            output_frame = pd.DataFrame(output_data)
            base_name = os.path.splitext(file)[0]
            output_frame.to_csv(os.path.join(output, base_name + '.csv'), index=False, header=False)
