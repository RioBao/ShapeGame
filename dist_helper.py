import math
import cv2
import numpy as np

class Scene:
    def __init__(self, sceneWidth, sceneHeight, videoWidth, sceneDepth=3500):
        self.sceneWidth = sceneWidth
        self.sceneHeight = sceneHeight
        self.sceneDepth = sceneDepth
        self.videoWidth = videoWidth
        self.Image = np.zeros((sceneHeight, sceneWidth, 3), np.uint8)

    def get_scene_position(self, detection):
        # If the frame is available, draw bounding boxes on it and show the frame
        sceneHeight = self.sceneHeight #pixel
        sceneWidth  = self.sceneWidth #pixel
        sceneDepth = self.sceneDepth  #cm. Max depth in the room. Has to be adjusted for every room
        videoWidth = self.videoWidth # camerapixels

        x = int((detection.xmin+detection.xmax)/2 * sceneWidth) + videoWidth
        z = int((sceneDepth-detection.spatialCoordinates.z)/sceneDepth * sceneHeight)

        return (x,z)


def calculate_distance(point1, point2):
    x1, z1 = point1
    x2, z2 = point2
    dx, dz = x1 - x2, z1 - z2
    distance = math.sqrt(dx ** 2 + dz ** 2)
    return distance

# point = (x,z)
def correct_corner_position(point, cornerPoint, distTol = 1):
    distance = calculate_distance(point, cornerPoint)
    if distance < distTol:
        return True
    else:
        return False

def nearest_corner_point(point, cornerPoints):
    dist = np.sqrt(np.sum((cornerPoints - point)**2, axis=1))
    minIdx = np.argmin(dist)
    minVal = dist[minIdx]
    return minVal, minIdx





