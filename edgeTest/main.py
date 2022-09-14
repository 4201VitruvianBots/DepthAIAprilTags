#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

from pupil_apriltags import Detector

import constants

from common.utils import FPSHandler

# Create pipeline
pipeline = dai.Pipeline()

CAMERA_TYPE = 'MONO'


if CAMERA_TYPE == 'RGB':
    # Define sources and outputs
    cam = pipeline.create(dai.node.ColorCamera)
    # monoLeft = pipeline.create(dai.node.MonoCamera)
    # monoRight = pipeline.create(dai.node.MonoCamera)

    edgeDetector = pipeline.create(dai.node.EdgeDetector)
    # edgeDetectorLeft = pipeline.create(dai.node.EdgeDetector)
    # edgeDetectorRight = pipeline.create(dai.node.EdgeDetector)

    # xoutEdgeLeft = pipeline.create(dai.node.XLinkOut)
    # xoutEdgeRight = pipeline.create(dai.node.XLinkOut)
    xoutEdge = pipeline.create(dai.node.XLinkOut)
    xinEdgeCfg = pipeline.create(dai.node.XLinkIn)

    edgeStr = "rgb"
    edgeCfgStr = "edge cfg"

    # xoutEdgeLeft.setStreamName(edgeLeftStr)
    # xoutEdgeRight.setStreamName(edgeRightStr)
    xoutEdge.setStreamName(edgeStr)
    xinEdgeCfg.setStreamName(edgeCfgStr)

    # Properties
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # camRgb.setFps(60) # Breaks edge detection

    # monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    # monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    # monoLeft.setFps(200)
    # monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    # monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    edgeDetector.setMaxOutputFrameSize(cam.getVideoWidth() * cam.getVideoHeight())

    # Linking
    cam.video.link(edgeDetector.inputImage)

    edgeDetector.outputImage.link(xoutEdge.input)

    xinEdgeCfg.out.link(edgeDetector.inputConfig)
else:
    # Define sources and outputs
    cam = pipeline.create(dai.node.MonoCamera)

    edgeDetector = pipeline.create(dai.node.EdgeDetector)

    xoutMono = pipeline.create(dai.node.XLinkOut)
    xoutEdge = pipeline.create(dai.node.XLinkOut)
    xinEdgeCfg = pipeline.create(dai.node.XLinkIn)

    edgeStr = "mono left"
    edgeCfgStr = "edge cfg"

    xoutMono.setStreamName("mono")
    xoutEdge.setStreamName(edgeStr)
    xinEdgeCfg.setStreamName(edgeCfgStr)

    # Properties
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)

    # Linking
    cam.out.link(xoutMono.input)
    cam.out.link(edgeDetector.inputImage)

    edgeDetector.outputImage.link(xoutEdge.input)

    xinEdgeCfg.out.link(edgeDetector.inputConfig)

# # OAK-D Lite @ 1080P
# camera_params = {
#     "fx": 0.00337,
#     "fY": 0.00337,
#     "cx": 960,
#     "cy": 540
# }

# OAK-D Lite @ 480P
camera_params = {
    "fx": 0.00337,
    "fY": 0.00337,
    "cx": 960,
    "cy": 540
}

detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

fps = FPSHandler()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output/input queues
    # edgeLeftQueue = device.getOutputQueue(edgeLeftStr, 8, False)
    # edgeRightQueue = device.getOutputQueue(edgeRightStr, 8, False)
    monoQueue = device.getOutputQueue("mono", 4, False)
    edgeQueue = device.getOutputQueue(edgeStr, 4, False)
    edgeCfgQueue = device.getInputQueue(edgeCfgStr)

    print("Switch between sobel filter kernels using keys '1' and '2'")

    while(True):
        # edgeLeft = edgeLeftQueue.get()
        # edgeRight = edgeRightQueue.get()
        monoOutput = monoQueue.get()
        edgeOutput = edgeQueue.get()

        monoFrame = monoOutput.getFrame()
        edgeFrame = edgeOutput.getFrame()
        # edgeRightFrame = edgeRight.getFrame()
        # edgeRgbFrame = edgeRgb.getFrame()

        thresh = cv2.threshold(edgeFrame, 25, 255, cv2.THRESH_BINARY)[1]
        contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        squares = []
        contourInfo = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

            if len(cnt) == 4 and area > 1000:
                squares.append(cnt)

                # Pad the square and save the coordinates for later
                rect = cv2.minAreaRect(cnt)
                points = cv2.boxPoints(rect)
                xmin = min(x for x in points[:, 0])
                xmax = max(x for x in points[:, 0])
                ymin = min(y for y in points[:, 1])
                ymax = max(y for y in points[:, 1])
                xmin = max(int(xmin - xmin * constants.PADDING_PERCENTAGE), 0)
                xmax = min(int(xmax + xmin * constants.PADDING_PERCENTAGE), 1920)
                ymin = max(int(ymin - ymin * constants.PADDING_PERCENTAGE), 0)
                ymax = min(int(ymax + ymax * constants.PADDING_PERCENTAGE), 1080)

                contourData = {'Contour': cnt,
                               'x_min': xmin,
                               'x_max': xmax,
                               'y_min': ymin,
                               'y_max': ymax,
                               'area': area}

                contourInfo.append(contourData)

        contourInfo = sorted(contourInfo, key=lambda d: d['area'], reverse=True)
        count = 0
        positives = 0
        for contour in contourInfo:
            frameSegment = monoFrame[contour['y_min']:contour['y_max'], contour['x_min']:contour['x_max']]
            tags = detector.detect(frameSegment, estimate_tag_pose=True, camera_params=camera_params.values(), tag_size=0.2)

            if len(tags) > 0:
                for tag in tags:
                    points = tag.corners.astype(np.int32)
                    # Shift points since this is a snapshot
                    points[:, 0] += contour['x_min']
                    points[:, 1] += contour['y_min']
                    cv2.polylines(monoFrame, [points], True, (120, 120, 120), 3)
                    textX = min(points[:, 0])
                    textY = min(points[:, 1]) + 30
                    cv2.putText(monoFrame, "tag_id: {}".format(tag.tag_id), (textX, textY), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

                    # TODO: Shift pose estimation by snapshot shift

                    print(tag)
                    positives += 1

            count += 1
            if count > 5 or positives > 2:
                break

        # if len(squares) > 0:
        #     cv2.drawContours(edgeFrame, squares, -1, color=(255, 255, 255), thickness=cv2.FILLED)

        fps.nextIter()
        cv2.putText(monoFrame, "{:.2f}".format(fps.fps()), (0, 24), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

        # Show the frame
        cv2.imshow("mono", monoFrame)
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
