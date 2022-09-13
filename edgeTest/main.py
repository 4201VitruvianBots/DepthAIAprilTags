#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
# import apriltag

from common.utils import FPSHandler

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
# monoLeft = pipeline.create(dai.node.MonoCamera)
# monoRight = pipeline.create(dai.node.MonoCamera)

# edgeDetectorLeft = pipeline.create(dai.node.EdgeDetector)
# edgeDetectorRight = pipeline.create(dai.node.EdgeDetector)
edgeDetectorRgb = pipeline.create(dai.node.EdgeDetector)

# xoutEdgeLeft = pipeline.create(dai.node.XLinkOut)
# xoutEdgeRight = pipeline.create(dai.node.XLinkOut)
xoutEdgeRgb = pipeline.create(dai.node.XLinkOut)
xinEdgeCfg = pipeline.create(dai.node.XLinkIn)

edgeLeftStr = "edge left"
edgeRightStr = "edge right"
edgeRgbStr = "edge rgb"
edgeCfgStr = "edge cfg"

# xoutEdgeLeft.setStreamName(edgeLeftStr)
# xoutEdgeRight.setStreamName(edgeRightStr)
xoutEdgeRgb.setStreamName(edgeRgbStr)
xinEdgeCfg.setStreamName(edgeCfgStr)

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setFps(60) # Breaks edge detection

# monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
# monoLeft.setFps(200)
# monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
# monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

edgeDetectorRgb.setMaxOutputFrameSize(camRgb.getVideoWidth() * camRgb.getVideoHeight())

# Linking
# monoLeft.out.link(edgeDetectorLeft.inputImage)
# monoRight.out.link(edgeDetectorRight.inputImage)
camRgb.video.link(edgeDetectorRgb.inputImage)

# edgeDetectorLeft.outputImage.link(xoutEdgeLeft.input)
# edgeDetectorRight.outputImage.link(xoutEdgeRight.input)
edgeDetectorRgb.outputImage.link(xoutEdgeRgb.input)

# xinEdgeCfg.out.link(edgeDetectorLeft.inputConfig)
# xinEdgeCfg.out.link(edgeDetectorRight.inputConfig)
xinEdgeCfg.out.link(edgeDetectorRgb.inputConfig)

fps = FPSHandler()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output/input queues
    # edgeLeftQueue = device.getOutputQueue(edgeLeftStr, 8, False)
    # edgeRightQueue = device.getOutputQueue(edgeRightStr, 8, False)
    edgeRgbQueue = device.getOutputQueue(edgeRgbStr, 8, False)
    edgeCfgQueue = device.getInputQueue(edgeCfgStr)

    print("Switch between sobel filter kernels using keys '1' and '2'")

    while(True):
        # edgeLeft = edgeLeftQueue.get()
        # edgeRight = edgeRightQueue.get()
        edgeRgb = edgeRgbQueue.get()

        edgeFrame = edgeRgb.getFrame()
        # edgeRightFrame = edgeRight.getFrame()
        # edgeRgbFrame = edgeRgb.getFrame()

        thresh = cv2.threshold(edgeFrame, 25, 255, cv2.THRESH_BINARY)[1]
        contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        squares = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

            if len(cnt) == 4 and area > 1000:
                squares.append(cnt)

        if len(squares) > 0:
            cv2.drawContours(edgeFrame, squares, -1, color=(255, 255, 255), thickness=cv2.FILLED)

        fps.nextIter()
        cv2.putText(edgeFrame, "{:.2f}".format(fps.fps()), (0, 24), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

        # Show the frame
        cv2.imshow(edgeLeftStr, edgeFrame)
        # cv2.imshow(edgeRightStr, edgeRightFrame)
        # cv2.imshow(edgeRgbStr, edgeRgbFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if key == ord('1'):
            print("Switching sobel filter kernel.")
            cfg = dai.EdgeDetectorConfig()
            sobelHorizontalKernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
            sobelVerticalKernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
            cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
            edgeCfgQueue.send(cfg)

        if key == ord('2'):
            print("Switching sobel filter kernel.")
            cfg = dai.EdgeDetectorConfig()
            sobelHorizontalKernel = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
            sobelVerticalKernel = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
            cfg.setSobelFilterKernels(sobelHorizontalKernel, sobelVerticalKernel)
            edgeCfgQueue.send(cfg)
