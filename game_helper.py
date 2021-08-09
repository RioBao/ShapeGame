import math
import cv2
import numpy as np

# Get next/random shape

# Play countdown

# shape found

# highschore?

def GetResultAndPrintShape(frame,sceneShape):
    for corner in sceneShape.cornerPoints:
        #print(corner.correct_position())
        cv2.circle(frame, corner.coord, 2, (255, 255,255), 1)
    #print(sceneShape.all_positions_correct())
    if sceneShape.all_positions_correct():
        shapeColor = (0,255,0)
        cv2.polylines(frame, sceneShape.get_polyLines(), True, shapeColor, 1)
        return True
    else:
        shapeColor = (0,0,255)
        cv2.polylines(frame, sceneShape.get_polyLines(), True, shapeColor, 1)
        return False

def printVictoryShape(frame,sceneShape):
    shapeColor = (0,255,0)
    cv2.polylines(frame, sceneShape.get_polyLines(), True, shapeColor, 3)


def printText(frame, label, detection, x1, x2, y1, y2,color):
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)