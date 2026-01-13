import cv2
import numpy as np

from game import Game

def main():

    cap = cv2.VideoCapture("videos/good_1.mp4")

    h, w = 800, 1100
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))

    ret, frame = cap.read()     #INITIAL FRAME TO INITIALIZE GAME OBJECT
    game = Game(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, -1)

    i = 1
    while cap.isOpened():
        if i % 10 == 0:
            print(f"{i}/{cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        ret, frame = cap.read()
        if not ret:
            break   

        vid_frame = game.update(frame)

        out.write(vid_frame)

        i += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()