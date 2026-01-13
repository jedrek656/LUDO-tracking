import cv2

from detector import Detector
from diceDetector import diceDetector

import numpy as np

from boardViz import boardViz, Events
from videoViewer import videoViewer

class Game:
    def __init__(self, init_frame):

        self.frame_h, self.frame_w = init_frame.shape[:2]

        self.players = [2, 0]   #0 is red, 2 is blue (1, 3 doesnt play)

        self.CURR_PLAYER = -1     #0 is left, 1 is right, -1 is not known yet

        self.LAST_DETECTED_DIGIT = -1
        self.SAME_NUM_DETECTOR_COUNTER = 0
        self.NUM_DETECTED = False
        self.NONE_DIG_COUNTER = 0

        self.dice_scores = {i:0 for i in range(1, 7)}

        self.detector = Detector(init_frame)
        self.diceDetector = diceDetector()
        self.diceDetector.set_board_area(self.detector.board_area)
        self.diceDetector.set_board_cords(self.detector.board_x, self.detector.board_w, self.detector.board_y, self.detector.board_h)

        self.board_outline = self.detector.find_board_outline(init_frame)

        self.prev_frame = init_frame

        self.prev_position = self.detector.return_frame_wit_plus_only(self.board_outline, init_frame)
        self.new_pose_counter = 25

        self.board_viz = boardViz()

        self.video_viewer = videoViewer()


    def track_and_paint_orange(self, frame):

        orange_cnt, orange_area = self.detector.detect_orange(frame)

        # TRACK ORANGE AND PLAYER CHANGE
        if orange_cnt is not None:
            if orange_area < 0.04 and orange_area > 0.02:
                M = cv2.moments(orange_cnt)
                cx = int(M['m10']/M['m00'])

                if cx < (0.4 * self.frame_w) and self.CURR_PLAYER != 0:
                    self.CURR_PLAYER = 0
                    self.video_viewer.print("Player change to player 1")
                if cx > (0.6 * self.frame_w) and self.CURR_PLAYER != 1:
                    self.CURR_PLAYER = 1
                    self.video_viewer.print("Player change to player 2")
        
        cv2.drawContours(frame, [orange_cnt], -1, (255, 0, 0), 8)

    def track_and_show_dice(self, frame):

        dice_rois = self.diceDetector.get_dice_rois(frame, 0.2)

        dice_roi = None if not dice_rois else dice_rois[0]
        dig = self.diceDetector.digit_from_roi(dice_roi)

        # DIGIT DETECTION AND LOGGING
        if dig is None:
            self.NONE_DIG_COUNTER += 1
            self.dice_scores = {i:0 for i in range(1, 7)}
            if self.NONE_DIG_COUNTER == 6:
                self.NUM_DETECTED = False
        else:
            self.NONE_DIG_COUNTER = 0
            numb, score = self.diceDetector.comp_roi_to_gt(dig)
            self.dice_scores[numb] += 1
            if self.NUM_DETECTED == False and self.dice_scores[numb] == 7:
                self.NUM_DETECTED = True
                self.video_viewer.print(f"Dice thrown: detected number: {numb}")

        # DIGIT ROI VIEWING
        if dig is not None:
            dice_roi = cv2.resize(dice_roi, (256, 256))
            self.video_viewer.paint_dice(dice_roi)
        else:
            no_roi = np.zeros((256, 256), "uint8")
            no_roi = cv2.cvtColor(no_roi, cv2.COLOR_GRAY2BGR)
            self.video_viewer.paint_dice(no_roi)

    def find_and_paint_board_outline(self, frame):
        self.board_outline.drawPlus(frame, (255, 0, 0), 8)

    def paint_fields_acc_to_plus(self, frame):
        fields = self.board_outline.get_fields()
        for x, y in fields:
            cv2.circle(frame, (x, y), 10, (255, 0, 0), 10)

    def comp_difference(self, frame):
        fields = []
        plus_frame = self.detector.return_frame_wit_plus_only(self.board_outline, frame)

        plus_frame_bf = self.detector.return_frame_wit_plus_only(self.board_outline, self.prev_frame)


        diff = self.detector.diff_map_ssim(plus_frame, plus_frame_bf)

        diff = (diff > 0.7).astype("uint8")

        if(np.sum(diff) / plus_frame.shape[0]) <= 0.05:
            if self.new_pose_counter == 24:
                pos_diff = self.detector.diff_map_ssim(self.prev_position, plus_frame)
                pos_diff = (pos_diff > 0.7).astype("uint8") * 255
                kern = np.ones((5, 5))
                pos_diff = cv2.morphologyEx(pos_diff, cv2.MORPH_CLOSE, kern, iterations=3)
                pos_diff = cv2.morphologyEx(pos_diff, cv2.MORPH_OPEN, kern)

                h, w = plus_frame.shape[:2]
                cx = w // 2
                cy = h // 2

                for idx, (x, y) in enumerate(self.board_outline.get_fields(cx, cy)):
                    if pos_diff[y, x] == 255:
                        fields.append(idx)
                        if len(fields) == 2:
                            break
                        

                #pos_diff = cv2.resize(pos_diff, (500, 500))
                #cv2.imshow("diff", pos_diff)
                self.prev_position = plus_frame.copy()

            self.new_pose_counter +=1

        else:
            self.new_pose_counter = 0

        return fields

    #def show_bases(self, frame):
    #    bases = self.detector.return_frame_with_bases_only(frame)
    #    bases = cv2.resize(bases, (500, 800))
    #    cv2.imshow("bases", bases)

    def calculate_events_and_show_board(self, fields):

        if len(fields) == 1:
            event = self.board_viz.single_pawn_action(fields[0], self.players[self.CURR_PLAYER])

            match event:
                case Events.PAWN_SPAWN:
                    self.video_viewer.print(f"Player {self.CURR_PLAYER} placed a new.")
                case Events.PAWN_TO_HOME:
                    self.video_viewer.print(f"Player {self.CURR_PLAYER} placed pawn in home.")
        
        if len(fields) == 2:
            players_home_loc = 12 * self.players[self.CURR_PLAYER]
            # WE DON'T KNOW THE ORDER OF FIELDS, SO WE CHECK TO ALWAYS HAVE CLOCKWISE MOVEMENT, AND WE ASSUME PLAYER CAN'T RECIRCLE THE BOARD(THEY GO HOME OR DO NOTHING)
            rel_fields = [(x + players_home_loc) % 48 for x in fields]
            if rel_fields[1] < rel_fields[0]:
                fields[0], fields[1] = fields[1], fields[0]
            event = self.board_viz.move_pawn(fields[0], fields[1], self.players[self.CURR_PLAYER])

            self.video_viewer.print(f"Player {self.CURR_PLAYER} moved from {fields[0]} to {fields[1]}")

            match event:
                case Events.BONE:
                    self.video_viewer.print(f"Player {self.CURR_PLAYER} stands on bone.")
                case Events.CARD:
                    self.video_viewer.print(f"Player {self.CURR_PLAYER} takes card.")
                case Events.CAPTURE:
                    self.video_viewer.print(f"Player {self.CURR_PLAYER} captures pawn.")
                case Events.CAPTURE_AND_CARD:
                    self.video_viewer.print(f"Player {self.CURR_PLAYER} captures pawn and takes card.")
        
        board = self.board_viz.draw_board_with_players()
        self.video_viewer.paint_board_viz(board)
    
    def update(self, frame):
        fields = self.comp_difference(frame)
        

        self.calculate_events_and_show_board(fields)
        self.track_and_show_dice(frame)

        #self.show_bases(frame)

        self.prev_frame = frame.copy()
        
        #PAINTING
        self.track_and_paint_orange(frame)

        self.video_viewer.paint_board(frame)

        

        return self.video_viewer.show()
