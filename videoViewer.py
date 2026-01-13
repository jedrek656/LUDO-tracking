import numpy as np
import cv2

class textWindow:
    def __init__(self):
        self.sentences = []
    
    def add_text_to_sentences(self, text):
        if len(self.sentences) == 4:
            self.sentences.pop()
        self.sentences.insert(0, text)

    def get_text_window(self):
        frame = np.zeros((200, 600), "uint8")
        for idx, sen in enumerate(self.sentences):
            cv2.putText(frame, sen, (10, 190 - idx * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame



class videoViewer:
    def __init__(self):
        self.frame = np.zeros((800, 1100, 3), "uint8")
        self.tW = textWindow()

    def paint_board(self, board):
        board = cv2.resize(board, (500, 800))
        self.frame[0:800, 0:500, :] = board

    def paint_board_viz(self, board):
        board = cv2.resize(board, (600, 600))
        self.frame[0:600, 500:1100, :] = board

    def paint_dice(self, dice):
        dice = cv2.resize(dice, (200, 200))
        self.frame[0:200, 500:700, :] = dice

    def print(self, text):
        self.tW.add_text_to_sentences(text)

    def show(self):
        self.frame[600:800, 500:1100, :] = self.tW.get_text_window()

        return self.frame

def test():
    vV = videoViewer()
    vV.print("aaa")
    vV.print("bbb")
    vV.print("ccc")

    vV.show()

if __name__ == "__main__":
    test()