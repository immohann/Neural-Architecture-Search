import cv2 as cv
import numpy as np
import skimage.exposure as exposure
import copy


def XSobelFiltered(X):
    XSobel = cv.Sobel(X, cv.CV_64F, 1, 0, ksize=5)
    # optionally normalize to range 0 to 255 for proper display
    XSobelNorm = exposure.rescale_intensity(XSobel, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
    return XSobelNorm


def YSobelFiltered(X):
    YSobel = cv.Sobel(X, cv.CV_64F, 0, 1, ksize=5)
    # optionally normalize to range 0 to 255 for proper display
    YSobelNorm = exposure.rescale_intensity(YSobel, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
    return YSobelNorm


def XYSobelFiltered(X):
    return XSobelFiltered(X) + YSobelFiltered(X)


def HarrisRMap(X):
    X = np.float32(X)
    dst = cv.cornerHarris(X, 2, 3, 0.04)
    X_copy = copy.deepcopy(X)
    X_copy[dst > 0.01*dst.max()] = [255]
    return X_copy


def HalfScaledUp(X):
    ScaledUp = cv.resize(X, (0, 0), fx=1.5, fy=1.5)
    return ScaledUp


def HalfScaledDown(X):
    ScaledDown = cv.resize(X, (0, 0), fx=0.5, fy=0.5)
    return ScaledDown


def GaussianBlurred(X):
    Blurred = cv.GaussianBlur(X, (3, 3), 0)
    return Blurred


# def BlobDetector(X):
#     detector = cv.SimpleBlobDetector_create()
#     keypoints = detector.detect(np.uint8(X))
#     ImageWithKeypoints = cv.drawKeypoints(np.uint8(X), keypoints, np.array([]), (255, 255, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     ImageWithKeypoints = ImageWithKeypoints.reshape(3, 32, 32)[0]
#     return ImageWithKeypoints


def RotateImage(X):
    Rotated = cv.rotate(X, cv.ROTATE_90_COUNTERCLOCKWISE)
    return Rotated


# def CannyEdge(X):
#     Edges = cv.Canny(X, 200, 300, True)
#     return Edges


def MedianBlur(X):
    MedianBlurred = cv.medianBlur(X, 5)
    return MedianBlurred


def BilateralFilter(X):
    BilateralFiltered = cv.bilateralFilter(X, 9, 75, 75)
    return BilateralFiltered


operation_set = [XSobelFiltered, YSobelFiltered, XYSobelFiltered, HarrisRMap, HalfScaledUp, HalfScaledDown,
                 GaussianBlurred, RotateImage, MedianBlur, BilateralFilter]


def translate(arr):
    for i in range(len(arr)):
        arr[i] = operation_set[int(arr[i])]
    return arr
