import cv2
import numpy as np
from diceAugmentor import augment_dice
from collections import defaultdict

class diceDetector:
    def __init__(self):
        #Get contours for comparions
        self.board_area = None
        self.board_x, self.board_w, self.board_y, self.board_h = 0, 0, 0, 0
        self.test_cnts = defaultdict(list)
        for i in range(1, 7):
            img = cv2.imread(f"diceRefence/{i}.png", cv2.IMREAD_COLOR)
            dice_roi = self.get_dice_rois(img, 0.2)[0]
            digit_roi = self.digit_from_roi(dice_roi)
            cnt, _ = cv2.findContours(digit_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.test_cnts[i].append(cnt)
            # j = 0
            # while j < 10:
            #     aug = augment_dice(img)
            #     try:
            #         dice_roi = self.get_dice_rois(aug, 0.2)[0]
            #         digit_roi = self.digit_from_roi(dice_roi)
            #         cnt, _ = cv2.findContours(digit_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #         self.test_cnts[i].append(cnt)
            #         j += 1
            #     except:
            #         continue

    def set_board_area(self, area):
        self.board_area = area

    def set_board_cords(self, bx, bw, by, bh):
        self.board_x, self.board_w, self.board_y, self.board_h = bx, bw, by, bh

    def get_dice_rois(self, frame, crop = 0.0):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #two masks because red is ~170 - 10 (openCV uses hue 0-179)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([5, 255, 255])

        lower_red2 = np.array([170, 80, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if self.board_area is not None:
            board_mask = np.ones(red_mask.shape, dtype=np.uint8) * 255
            bx, bw, by, bh = self.board_x, self.board_w, self.board_y, self.board_h

            board_mask[by:by+bh, bx:bx+bw] = 0
                  
            red_mask = cv2.bitwise_and(red_mask, board_mask)
        
        contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        dice_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            #if area < 500:
            #    continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)

            if 0.8 < aspect < 1.2:
                dice_candidates.append((x, y, w, h))

        dice_rois = []
        for (x, y, w, h) in dice_candidates:
            y_new = int(y + crop*h)
            y_new_end = int(y + (1-crop)*h)

            x_new = int(x + crop * w)
            x_new_end = int(x + (1-crop)*w)
            dice_roi = frame[y_new:y_new_end, x_new:x_new_end]
            dice_rois.append(dice_roi)

        return dice_rois
    
    def digit_from_roi(self, dice_roi):
        if dice_roi is None:
            return None
        dice_roi = cv2.resize(dice_roi, (90, 90))
        hsv = cv2.cvtColor(dice_roi, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 0])
        upper_white = np.array([180, 90, 255])

        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        contours, _ = cv2.findContours(
            white_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # pick the largest white contour
        if not contours:
            return None
        digit_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(digit_cnt)

        digit_roi = white_mask[y:y+h, x:x+w]

        digit_roi = cv2.copyMakeBorder(
            digit_roi, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, value=0
        )

        return digit_roi
    
    def comp_roi_to_gt(self, digit_roi):

        def smooth_contour(cnt, eps_ratio=0.01):
            eps = eps_ratio * cv2.arcLength(cnt, True)
            return cv2.approxPolyDP(cnt, eps, True)



        cnt, _ = cv2.findContours(digit_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bestScore = float('inf')
        num = 0
        score_dict = {}
        for idx, qCnts in self.test_cnts.items():
            c1 = max(cnt, key=cv2.contourArea)
            c1 = smooth_contour(c1)
            scores = []
            for qCnt in qCnts:
                c2 = max(qCnt, key=cv2.contourArea)
                c2 = smooth_contour(c2)
                score = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
                scores.append(score)

            score = min(scores)
            score_dict[idx] = score
            if score < bestScore:
                num = idx
                bestScore = score
                
        return num, bestScore