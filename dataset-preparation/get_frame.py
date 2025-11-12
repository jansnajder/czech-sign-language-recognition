import cv2


if __name__ == '__main__':
    '''
    Simple script which goes through video and allow to setup border frames, by clicking "s" the current frame is
    saved, by clicking "q" the capture is ended. Any other key leads to next frame.
    '''

    path = ''

    cap = cv2.VideoCapture(path)
    i = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    minutes = 1
    offset = total_count - (minutes * fps *  60)
    cap.set(cv2.CAP_PROP_POS_FRAMES, offset)

    name = 'start.png'

    while cap.isOpened():
        ret, frame = cap.read()
        print(i:=i+1)

        if ret:
            cv2.imshow('frame', frame)
            k = cv2.waitKey(0) & 0xFF

            if k == ord('s'):
                cv2.imwrite(name, frame)
                name = 'end.png'
            elif k == ord('q'):
                break
        else:
            break

    cap.release()