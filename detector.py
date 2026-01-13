import cv2
import numpy as np
from collections import defaultdict
from plusShapeAdjuster import PlusShape

from skimage.metrics import structural_similarity

class Detector:
    def __init__(self, init_frame):
        self.plus_mask = None
        self.find_board(init_frame)
    
    def find_board(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        board_lower = np.array([0, 0, 0])
        board_upper = np.array([180, 255, 100])

        mask = cv2.inRange(hsv, board_lower, board_upper)
        kernel = kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_cnt = max(contours, key = cv2.contourArea)

        self.board_area = cv2.contourArea(best_cnt)

        self.board_x, self.board_y, self.board_w, self.board_h = cv2.boundingRect(best_cnt)

        ogBSize = 8912960.0
        self.board_scale_factor = np.sqrt(self.board_area / ogBSize)

        return best_cnt
    
    def detect_orange(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([5, 180, 120])
        upper_bound = np.array([20, 255, 255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        #filter out orange color inside the board
        if self.board_area is not None:
            board_mask = np.ones(mask.shape, dtype=np.uint8) * 255
            bx, bw, by, bh = self.board_x, self.board_w, self.board_y, self.board_h

            board_mask[by:by+bh, bx:bx+bw] = 0
                  
            mask = cv2.bitwise_and(mask, board_mask)


        kernel = kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, None

        best_cnt = max(contours, key = cv2.contourArea)

        area_coeff = cv2.contourArea(best_cnt) / float(self.board_area)

        return best_cnt, area_coeff
    
    def find_player_bases(self, image):

        bx, bw, by, bh = self.board_x, self.board_w, self.board_y, self.board_h
        image = image[by: by + bh, bx: bx + bw, :]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist = self.rescale_number_to_board_size(1000),
            param1 = self.rescale_number_to_board_size(100),
            param2 = self.rescale_number_to_board_size(70),
            minRadius = self.rescale_number_to_board_size(130),
            maxRadius = self.rescale_number_to_board_size(250)
        )

        circles = circles[0]

        correct_circles = []
        midX = bw / 2
        midY = bh / 2
        for circle in circles:
            if abs(circle[0] - midX) < 0.1 * bw or abs(circle[1] - midY) < 0.1 * bh:
                circle[0] += bx
                circle[1] += by
                correct_circles.append(circle)
        return np.array(correct_circles)
    
    def find_game_fields(self, image):
        bx, bw, by, bh = self.board_x, self.board_w, self.board_y, self.board_h
        image = image[by: by + bh, bx: bx + bw, :]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist = self.rescale_number_to_board_size(100),
            param1 = self.rescale_number_to_board_size(100),
            param2 = self.rescale_number_to_board_size(60),
            minRadius = self.rescale_number_to_board_size(40),
            maxRadius = self.rescale_number_to_board_size(100)
        )

        circles = circles[0]

        for circle in circles:
            circle[0] += bx
            circle[1] += by
        return np.array(circles)
    
    def group_fields_by_type(self, image, circles):
        def mean_method(circles, hsv):
            circle_colors = []
            if circles is not None:
                circles = np.round(circles).astype("int")
                for (x, y, r) in circles:
                    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (x, y), int(round(r*0.9)), 255, -1)  # avoid border
                    
                    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
                    circle_colors.append((x, y, r, mean_hsv))
            return circle_colors
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        circle_colors = mean_method(circles, hsv)
        groups = defaultdict(list)

        for x, y, r, (h, s, v) in circle_colors:
            if s < 30 and v > 150:
                groups["white"].append((x, y, r))
            elif 10 < h < 25:
                groups["orange"].append((x, y, r))
            elif 125 < h < 160:
                groups["purple"].append((x, y, r))
            else:
                groups["None"].append((x, y, r))

        return groups
        
    def rescale_number_to_board_size(self, number):
        #ORIGINAL BOARD SIZE = 8912960
        
        return np.round(number * self.board_scale_factor).astype("int")
    
    def find_cards(self, image):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bx, bw, by, bh = self.board_x, self.board_w, self.board_y, self.board_h

        hsv = hsv[:by, :, :]

        lower_cyan = np.array([70, 0, 0])
        upper_cyan = np.array([110, 255, 255])

        mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        correct_cnts = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0.08 * self.board_area or area < 0.02 * self.board_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(h) / float(w)

            if 1.0 < aspect < 1.8:
                correct_cnts.append(cnt)

        return correct_cnts
        
    def find_homes_bases_startields(self, frame, cut = 0.07):

        #SHRINK THE BOARD AREA TO AVOID DETECTING BOARD CONTOUR
        bx, bw, by, bh = self.board_x, self.board_w, self.board_y, self.board_h

        bx = int(round(bx + cut * bw))
        bw = int(round((1-2*cut) * bw))
        by = int(round(by + cut * bh))
        bh = int(round((1-2*cut) * bh))

        elements = {}
        #BLUE
        lower_mask = np.array([90, 120, 100])
        upper_mask = np.array([110, 255, 255])
        elements["blue"] = self.find_home_base_startfield(frame, lower_mask, upper_mask, bx, bw, by, bh)

        #RED
        lower_mask = np.array([0, 120, 100])
        upper_mask = np.array([5, 255, 255])
        elements["red"] = self.find_home_base_startfield(frame, lower_mask, upper_mask, bx, bw, by, bh)

        #YELLOW
        lower_mask = np.array([20, 120, 100])
        upper_mask = np.array([40, 255, 255])
        elements["yellow"] = self.find_home_base_startfield(frame, lower_mask, upper_mask, bx, bw, by, bh)

        #GREEN
        lower_mask = np.array([50, 30, 30])
        upper_mask = np.array([80, 255, 220])
        elements["green"] = self.find_home_base_startfield(frame, lower_mask, upper_mask, bx, bw, by, bh)

        for key, cnts in elements.items():
            for i in range(len(cnts)):
                cnts[i] = cnts[i] + np.array([bx, by])

        return elements
    
    def find_cnt_middle(self, cnt):
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        return(cx, cy)
    
    def find_home_base_startfield(self, frame, lowwer, upper, bx, bw, by, bh):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hsv = hsv[by: by + bh, bx: bx + bw, :]

        mask = cv2.inRange(hsv, lowwer, upper)

        kernel = np.ones((5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        correct_cnts = []

        hsv_h, hsv_w = hsv.shape[:2]

        cx_board = hsv_w / 2
        cy_board = hsv_h / 2
        
        for cnt in cnts:
            cx, cy = self.find_cnt_middle(cnt)
            if abs(cx - cx_board) < 0.05 * bw and abs(cy - cy_board) < 0.05 * bh:
                continue
            correct_cnts.append(cnt)

        correct_cnts = sorted(correct_cnts, key=cv2.contourArea, reverse=True)[:3]
        
        return correct_cnts
    
    def find_board_outline(self, frame, fields = None, bases = None):
        
        #Avoid looking for fields for second time if they are already founn
        if fields is None:
            fields = self.find_game_fields(frame)
        
        if bases is None:
            bases = self.find_player_bases(frame)

        circles_outside = []
        for (x, y, r) in fields:
            if not any(
                np.hypot(x2 - x, y2 - y) + r <= 1.2 * r2
                for (x2, y2, r2) in bases
            ):
                circles_outside.append((x, y, r))
        fields = circles_outside

        points = list((x, y) for x, y, r in fields)

        cx = self.board_x + (self.board_w) / 2
        cy = self.board_y + (self.board_h) / 2

        plus = PlusShape(cx, cy, 100, 300, 45)
        plus.optimize_plus(points)

        return plus

    def diff_map_ssim(self, ref_frame: np.ndarray, dist_frame: np.ndarray) -> np.ndarray:
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
        _, diff = structural_similarity(ref_gray, dist_gray, full=True)
        diff_map = (1.0 - diff).astype(np.float32)
        return diff_map
    
    def diff_map_L2(self, ref_frame: np.ndarray, dist_frame: np.ndarray) -> np.ndarray:
        """Return squared difference map between grayscale versions of two images."""
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
        diff = (ref_gray.astype(np.float32) - dist_gray.astype(np.float32)) ** 2
        return diff.astype(np.uint8)

    def return_frame_wit_plus_only(self, plus, frame):

        if self.plus_mask is None:
            plus_mask = np.zeros_like(frame)
            plus.drawPlus(plus_mask, thickness = 100)
            
            plus_mask = cv2.cvtColor(plus_mask, cv2.COLOR_BGR2GRAY)
            self.plus_mask = plus_mask

        x, y, w, h = cv2.boundingRect(self.plus_mask)

        plus_frame = cv2.bitwise_and(frame, frame, mask = self.plus_mask)

        plus_frame = plus_frame[y: y+h, x: x+w]

        return plus_frame
    
    def return_frame_with_bases_only(self, frame):
        bases = self.find_player_bases(frame)
        bases = np.round(bases).astype("int")
        frames = []
        for (x, y, r) in bases:
            frames.append(frame[y-r:y+r, x-r:x+r, :])
        
        return frames 
