import cv2
import numpy as np
from scipy.optimize import least_squares

class PlusShape:
    def __init__(self, cx, cy, innerSize, outerSizeSize, rotation = 0):
        self.cx = cx
        self.cy = cy
        self.rotation = rotation
        self.outerSize = outerSizeSize
        self.innerSize = innerSize
    
    def _getSegments(self, cx, cy, inner, outer):
        segs = []

        segs += [
            (cx + inner, cy + inner, cx + inner, cy + outer),
            (cx + inner, cy - inner, cx + inner, cy - outer),
            (cx - inner, cy + inner, cx - inner, cy + outer),
            (cx - inner, cy - inner, cx - inner, cy - outer),
        ]

        segs += [
            (cx + inner, cy + inner, cx + outer, cy + inner),
            (cx - inner, cy + inner, cx - outer, cy + inner),
            (cx + inner, cy - inner, cx + outer, cy - inner),
            (cx - inner, cy - inner, cx - outer, cy - inner),
        ]

        segs += [
            (cx + inner, cy + outer, cx - inner, cy + outer),
            (cx - inner, cy - outer, cx + inner, cy - outer),
            (cx + outer, cy - inner, cx + outer, cy + inner),
            (cx - outer, cy + inner, cx - outer, cy - inner),
        ]

        return segs
    
    def _rotatePoints(self, x, y, cx, cy, angle_deg):
        a = np.deg2rad(angle_deg)
        c, s = np.cos(a), np.sin(a)

        xr = c * (x - cx) - s * (y - cy) + cx
        yr = s * (x - cx) + c * (y - cy) + cy
        return xr, yr
    
    def getPlusSegments(self, cx, cy, inner, outer, angle):
        segs = self._getSegments(cx, cy, inner, outer)
        out = []

        for x1, y1, x2, y2 in segs:
            x1r, y1r = self._rotatePoints(x1, y1, cx, cy, angle)
            x2r, y2r = self._rotatePoints(x2, y2, cx, cy, angle)
            out.append((x1r, y1r, x2r, y2r))

        return out

    def pointSegmentDistance(self, px, py, x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1

        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return np.hypot(px - x1, py - y1)

        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return np.hypot(px - x2, py - y2)

        b = c1 / c2
        bx, by = x1 + b * vx, y1 + b * vy
        return np.hypot(px - bx, py - by)
    
    def plusResiduals(self, params, points):
        cx, cy, inner, outer, rot = params
        segs = self.getPlusSegments(cx, cy, inner, outer, rot)

        residuals = []
        for px, py in points:
            d = min(
                self.pointSegmentDistance(px, py, *seg)
                for seg in segs
            )
            residuals.append(d)

        return np.array(residuals)

    def drawPlus(self, frame, color = (255, 255, 255), thickness = 1):
        segs = self.getPlusSegments(self.cx, self.cy, self.innerSize, self.outerSize, self.rotation)
        segs = np.round(segs).astype("int")
        for x1, y1, x2, y2 in segs:
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    
    def get_fields(self, cx = None, cy = None):
        if cx is None:
            cx = self.cx
        if cy is None:
            cy = self.cy
        segs = self.getPlusSegments(cx, cy, self.innerSize, self.outerSize, self.rotation)
        segs = np.round(segs).astype("int")
        corr_idx = np.array([1, 6, 10, 4, 0, 8, 2, 5, 11, 7, 3, 9])
        segs = segs[corr_idx]
        circles = []
        for idx, (x1, y1, x2, y2) in enumerate(segs):
            if (idx % 3) == 0:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            if (idx % 3) == 2:
                for i in range(2):
                    circles.append((round(x1 + i * ((x2 - x1) / 2)), round(y1 + i * ((y2 - y1) / 2))))
            else:
                for i in range(5):
                    circles.append((round(x1 + i * ((x2 - x1) / 5)), round(y1 + i * ((y2 - y1) / 5))))
                    #pass
            

        return circles

    def optimize_plus(self, points):

        x0 = [
            self.cx,
            self.cy,
            self.innerSize,
            self.outerSize,
            self.rotation
        ]

        result = least_squares(
            self.plusResiduals,
            x0,
            args=(points,),
            loss="huber",
            f_scale=5.0
        )

        self.cx, self.cy, self.innerSize, self.outerSize, self.rotation = result.x

class PlusShapeTestGenerator(PlusShape):
    def __init__(self, cx, cy, innerSize, outerSizeSize, rotation=0):
        super().__init__(cx, cy, innerSize, outerSizeSize, rotation)

    def sampleSegment(self, x1, y1, x2, y2, n):
        t = np.random.rand(n)
        xs = x1 + t * (x2 - x1)
        ys = y1 + t * (y2 - y1)
        return np.column_stack([xs, ys])
    
    def addNoise(self, points, sigma):
        angles = np.random.uniform(0, 2 * np.pi, len(points))
        radii = np.random.normal(0, sigma, len(points))

        dx = radii * np.cos(angles)
        dy = radii * np.sin(angles)

        return points + np.column_stack([dx, dy])
    
    def generateNoisyPoints(
        self,
        points_per_segment=8,
        noise_sigma=4.0,
        outliers=0,
        img_size=(600, 600)
    ):
        segments = self.getPlusSegments(self.cx, self.cy, self.innerSize, self.outerSize, self.rotation)

        all_points = []

        for seg in segments:
            pts = self.sampleSegment(*seg, points_per_segment)

            # rotate
            pts[:, 0], pts[:, 1] = self._rotatePoints(pts[:, 0], pts[:, 1], self.cx, self.cy, self.rotation)

            all_points.append(pts)

        points = np.vstack(all_points)

        # add noise
        points = self.addNoise(points, noise_sigma)

        # add random outliers
        if outliers > 0:
            ox = np.random.uniform(0, img_size[1], outliers)
            oy = np.random.uniform(0, img_size[0], outliers)
            points = np.vstack([points, np.column_stack([ox, oy])])

        return points

#Example of problem to show working model
def main():
    IMG_SIZE = (600, 600)

    bckgrnd = np.zeros(IMG_SIZE, "uint8")

    plus = PlusShape(400, 400, 50, 100, rotation = 12)

    testPlus = PlusShapeTestGenerator(380, 380, 30, 120, rotation=20)
    points = testPlus.generateNoisyPoints(img_size = IMG_SIZE)

    points = np.round(points).astype("int")

    plus.optimize_plus(points)

    for point in points:
        cv2.circle(bckgrnd, point, 5, (255, 255, 255), 1)

    plus.drawPlus(bckgrnd, thickness=1)

    cv2.imshow("Plus" ,bckgrnd)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()