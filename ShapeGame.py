#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import dist_helper
import shape_helper
import game_helper
'''
Spatial detection network demo.
    Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
'''

# MobilenetSSD label texts
#labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#            "diningtable", "dog", "horse", "motorbike   ", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

labelMap = ["background", "player"]


# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('./mobilenet-ssd_openvino_2021.2_6shave.blob')).resolve().absolute())
#nnBlobPath = str((Path(__file__).parent / Path('person/person-detection-retail-0013.blob.sh6cmx6NCE1')).resolve().absolute())
#nnBlobPath = str((Path(__file__).parent / Path('person/tiny-yolo-v3.blob.sh13cmx13NCE1')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

videoWidth = 640*3 # 1920
videoHeight = 400*3 # 1080

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
colorCam = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutRgb = pipeline.createXLinkOut()
xoutNN = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")


colorCam.setPreviewKeepAspectRatio(True) #When true, crops video to keep aspect ratio
colorCam.setPreviewSize(300,300) # 300x300 is used by NN
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
#colorCam.setFps(15)

#colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam_out = pipeline.createXLinkOut()
colorCam_out.setStreamName("rgb_out")
# Link video output to host for higher resolution
colorCam.video.link(colorCam_out.input)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Setting node configs
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(200)
stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_5x5)


spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.2)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.25)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(4000)

# Create outputs
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Get shape and create sceneImage
sceneShape = shape_helper.Triangle2(videoWidth)
#sceneShape = shape_helper.get_random_shape('', videoWidth) 
sceneImage = dist_helper.Scene(350, videoHeight, videoWidth)
#sceneImage.Image = cv2.imread('triangle1.jpg',1)
sceneImage.Image = cv2.imread(sceneShape.imagePath,1)
success = False



# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb_out", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    while True:
        inPreview = previewQueue.get()
        inNN = detectionNNQueue.get()
        depth = depthQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        
        # get frames
        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        detections = inNN.detections
        if len(detections) != 0:
             boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
             roiDatas = boundingBoxMapping.getConfigData()

             for roiData in roiDatas:
                 roi = roiData.roi
                 roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                 topLeft = roi.topLeft()
                 bottomRight = roi.bottomRight()
                 xmin = int(topLeft.x)
                 ymin = int(topLeft.y)
                 xmax = int(bottomRight.x)
                 ymax = int(bottomRight.y)
                 

                 cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        # Add sceneImage to frame
        frame = np.concatenate((frame, sceneImage.Image), axis=1)

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]-sceneImage.sceneWidth
        print(width)
        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            # Get scene position. Then distance to closest corner point 
            scenePoint = sceneImage.get_scene_position(detection)
            #print(scenePoint)
            distToCorner, nearestCorner = dist_helper.nearest_corner_point(scenePoint, sceneShape.cornerPoints_to_array())
            if distToCorner < 50:
                scenePointColor = (0,255,0)
                sceneShape.cornerPoints[nearestCorner].hit()
            else:
                scenePointColor = (0,0,255)
                sceneShape.cornerPoints[nearestCorner].miss()

            try:
                label = labelMap[detection.label]
            except:
                label = "Player"# detection.label
            #cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            #cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            #cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            #cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            #cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            
            #cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.putText(frame, "*", scenePoint, cv2.FONT_HERSHEY_TRIPLEX,1.5,scenePointColor)

        # shape
        if not success: 
            success = game_helper.GetResultAndPrintShape(frame, sceneShape)
            successFrameCounter = 0
        if success:
            cv2.putText(frame, "Success!", (int(videoWidth/3),int(videoHeight/2)), cv2.FONT_HERSHEY_TRIPLEX,8,(0,255,0)) #PutText does not work in other functions?? #PutText does not work in other functions??
            game_helper.printVictoryShape(frame,sceneShape)
            successFrameCounter = successFrameCounter+1
            # Start next shape after 100 frames
            if successFrameCounter > 100:
                sceneShape = shape_helper.get_random_shape(sceneShape.name, width) 
                sceneImage.Image = cv2.imread(sceneShape.imagePath,1)
                success = False

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.putText(frame, str(sceneShape.name), (frame.shape[1]-sceneImage.sceneWidth+20,frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, .65, (255,255,255))
        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", cv2.resize(frame,(1400,720))) #Resize instead of higher resolution?
        #cv2.imshow("rgb", frame) 
         
        if cv2.waitKey(1) == ord('q'):
            break