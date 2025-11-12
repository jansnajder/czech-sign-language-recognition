LEFT_HAND = [11, 13, 15, 17, 19, 21]
RIGHT_HAND = [12, 14, 16, 18, 20, 22]
FACE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def process_results(h_results, p_results):
    out_line = []
    left_idx = None
    right_idx = None

    if p_results.pose_landmarks is not None:
        for part in (LEFT_HAND, RIGHT_HAND):

            for i, idx in enumerate(part):
                if i == 0:
                    x_0 = p_results.pose_landmarks.landmark[idx].x
                    y_0 = p_results.pose_landmarks.landmark[idx].y
                else:
                    out_line.append(p_results.pose_landmarks.landmark[idx].x - x_0)
                    out_line.append(p_results.pose_landmarks.landmark[idx].y - y_0)

        if h_results.multi_handedness is not None:
            for handedness in h_results.multi_handedness:
                if handedness.classification[0].label == 'Left':
                    left_idx = handedness.classification[0].index - 1
                elif handedness.classification[0].label == 'Right':
                    right_idx = handedness.classification[0].index - 1

            for idx in (left_idx, right_idx):
                if idx is not None:
                    landmarks = h_results.multi_hand_landmarks[idx]

                    for i, landmark in enumerate(landmarks.landmark):
                        if i == 0:
                            x_0 = landmark.x
                            y_0 = landmark.y
                        else:
                            out_line.append(landmark.x - x_0)
                            out_line.append(landmark.y - y_0)
                else:
                    out_line += 40 * [0]
        else:
            out_line += 80 * [0]
    else:
        out_line += 100 * [0]



    return out_line
