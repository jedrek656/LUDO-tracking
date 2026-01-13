import numpy as np
import cv2
from enum import Enum

class Field(Enum):
    NORMAL = 0
    START = 1
    BONE = 2
    CARD = 3

class Events(Enum):
    NONE = 0
    CAPTURE = 1
    CARD = 2
    BONE = 3
    CAPTURE_AND_CARD = 4

    PAWN_SPAWN = 5
    PAWN_TO_HOME = 6

class boardViz:
    # RED HOME HAS IDX = 0
    def __init__(self, size = 600):
        self.size = size
        self.segment = size // 14
        board = np.zeros((size, size, 3), "uint8")
        self.cx, self.cy = size // 2, size // 2

        self.board = board

        self.fields = [[] for _ in range(48)]

        for i in range(48):
            x, y = self.get_field_idx_to_coords(i)

            if i%12 == 0:
                continue

            self.draw_sqr_on_board(x, y)

        for i, col in enumerate(((0, 0, 255), (0, 255, 0), (255, 150, 50), (0, 255, 255))):
            x, y = self.get_field_idx_to_coords(i*12)
            self.draw_sqr_on_board(x, y, col)

    def get_field_idx_to_coords(self, idx):
        segm = idx // 12
        idx = idx % 12

        y_coeff = 6 - min(5, idx)
        x_coeff = max(5, idx) - 4

        if idx == 11:
            x_coeff -= 1
            y_coeff -= 1

        match segm:
            case 0:
                return self.cx - x_coeff * self.segment, self.cy + y_coeff * self.segment
            case 1:
                return self.cy - y_coeff * self.segment, self.cx - x_coeff * self.segment
            case 2:
                return self.cx + x_coeff * self.segment, self.cy - y_coeff * self.segment
            case 3:
                return self.cy + y_coeff * self.segment, self.cx + x_coeff * self.segment
            

    def draw_sqr_on_board(self, x, y, color = (255, 255, 255)):
        cv2.rectangle(self.board, (x - self.segment // 2, y - self.segment // 2), (x + self.segment // 2, y + self.segment // 2), color, 3)

    def draw_players(self, players, board):
        colors = (((0, 0, 255), (0, 255, 0), (255, 150, 50), (0, 255, 255)))
        r = self.segment // 3
        if len(players) == 1:
            field_idx, player_idx = players[0]
            x, y = self.get_field_idx_to_coords(field_idx)
            cv2.circle(board, (x, y), r, colors[player_idx], -1)
        else:
            for idx, (field_idx, player_idx) in enumerate(players):
                x, y = self.get_field_idx_to_coords(field_idx)
                x -= r // 2
                y -= r // 2
                cv2.circle(board, (x + (idx % 2) * 2 * (r // 2), y + ((idx // 2)) * 2 * (r // 2)), r // 2, colors[player_idx], -1)

    def get_field_type(self, idx):
        if idx % 12 == 0:
            return Field.START
        if idx % 12 == 3 or idx % 12 == 7:
            return Field.BONE
        if idx % 12 == 5:
            return Field.CARD
        return Field.NORMAL
        

    def move_pawn(self, old_idx, new_idx, player_idx):
        #if self.board[old_idx]:
        #    for 
        if player_idx in self.fields[old_idx]:
            self.fields[old_idx].remove(player_idx)
        
        field_type = self.get_field_type(new_idx)
        if field_type == Field.BONE:
            self.fields[new_idx].append(player_idx)
            return Events.BONE
        
        if len(self.fields[new_idx]) != 0:
            self.fields[new_idx] = [player_idx]
            if field_type == Field.CARD:
                return Events.CAPTURE_AND_CARD
            else:
                return Events.CAPTURE
        self.fields[new_idx] = [player_idx]
        return Events.NONE
    
    def single_pawn_action(self, field_idx, player_idx):
        if (field_idx == player_idx * 12):
            self.fields[field_idx].append(player_idx)
            return Events.PAWN_SPAWN
        else:
            self.fields[field_idx].remove(player_idx)
            return Events.PAWN_TO_HOME
        

    def show_board(self):
        cv2.imshow("board", self.board)

    def draw_board_with_players(self):
        board = self.board.copy()
        for idx, field in enumerate(self.fields):
            if field:
                pawns = [(idx, f) for f in field]
                self.draw_players(pawns, board)
        return board



def test():
    brd = boardViz()
    brd.show_board()

    brd.move_pawn(1, 5, 2)
    brd.move_pawn(0, 5, 0)
    #brd.move_pawn(0, 5, 0)
    #brd.move_pawn(0, 5, 0)
    #brd.move_pawn(0, 5, 0)

    brd.draw_board_with_players()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()