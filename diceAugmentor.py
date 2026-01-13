import random
import numpy as np
import cv2

def random_perspective(img, max_warp=0.2):
    h, w = img.shape[:2]

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    def jitter(pt):
        return [
            pt[0] + random.uniform(-max_warp, max_warp) * w,
            pt[1] + random.uniform(-max_warp, max_warp) * h
        ]

    dst = np.float32([jitter(p) for p in src])

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        img, H, (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped

def random_rotation(img, max_angle=15):
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated

def random_anisotropic_scale(img, min_scale=0.85):
    h, w = img.shape[:2]

    sx = random.uniform(min_scale, 1.0)
    sy = random.uniform(min_scale, 1.0)

    resized = cv2.resize(img, None, fx=sx, fy=sy)

    canvas = np.zeros_like(img)
    rh, rw = resized.shape[:2]

    x = (w - rw) // 2
    y = (h - rh) // 2

    canvas[y:y+rh, x:x+rw] = resized
    return canvas

def contour_noise(img, blur_ksize=3):
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # optional threshold jitter
    if len(blurred.shape) == 2:
        t = random.randint(5, 15)
        _, blurred = cv2.threshold(
            blurred, t, 255, cv2.THRESH_BINARY
        )

    return blurred

def augment_dice(img):
    aug = img.copy()

    if random.random() < 0.8:
        aug = random_perspective(aug)

    if random.random() < 0.7:
        aug = random_rotation(aug)

    if random.random() < 0.5:
        aug = random_anisotropic_scale(aug)

    if random.random() < 0.7:
        aug = contour_noise(aug)

    return aug