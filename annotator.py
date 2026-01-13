import cv2

from detector import Detector
from diceDetector import diceDetector
from annotator import Annotator

class Annotator:
    def __init__(self):
        pass

    def paint_circles(self, frame, circles):
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 0, 255), 8)
    
    def paint_fields_types(self, frame, groups):
        for key, value in groups.items():
            match key:
                case "white":
                    color = (255, 0, 0)     #white fields should be blue
                case "orange":
                    color = (0, 255, 0)     #orange fields (brown tokens and bones) should be green
                case "purple":
                    color = (0, 0, 255)     #purple fields (eyes) should be red
                case "None":
                    color = (0, 0, 0)       #other should be black
            for x, y, r in value:
                cv2.circle(frame, (x, y), r, color, 8)

    def paint_game_elements(self, frame, groups):
        for key, cnts in groups.items():
            match key:
                case "red":
                    color = (0, 0, 255)    
                case "green":
                    color = (0, 255, 0)  
                case "blue":
                    color = (255, 0, 0)  
                case "yellow":
                    color = (85, 253, 255) 
            cv2.drawContours(frame, cnts, -1, color, 8)