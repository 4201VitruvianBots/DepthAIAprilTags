
import cv2
import depthai as dai
import logging
import numpy as np

from pupil_apriltags import Detector

import spatialCalculator_pipelines
from common import utils

log = logging.getLogger(__name__)


def main():
    log.info("Starting AprilTag Spatial Detector")

    pipeline, pipeline_info, camera_params = spatialCalculator_pipelines.create_spaitalCalculator_pipeline()

    detector = Detector(families='tag36h11',
                        nthreads=3,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    fps = utils.FPSHandler()

    firstPass = True

    with dai.Device(pipeline) as device:
        depthQueue = device.getOutputQueue(name=pipeline_info["depthQueue"], maxSize=4, blocking=False)
        spatialCalcQueue = device.getOutputQueue(name=pipeline_info["spatialDataQueue"], maxSize=4, blocking=False)
        spatialCalcConfigInQueue = device.getInputQueue(pipeline_info["spatialConfigQueue"])
        qRight = device.getOutputQueue(name=pipeline_info["monoRightQueue"], maxSize=4, blocking=False)

        while True:
            inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
            inDepthAvg = spatialCalcQueue.get()  # blocking call, will wait until a new data has arrived
            inRight = qRight.tryGet()

            spatialData = inDepthAvg.getSpatialLocations()

            if inRight is not None:
                frameRight = inRight.getCvFrame()  # get mono right frame

                tags = detector.detect(frameRight, estimate_tag_pose=False, tag_size=0.2)

                if len(tags) > 0:
                    detectedTags = []
                    for tag in tags:
                        topLeft = dai.Point2f(0.4, 0.4)
                        bottomRight = dai.Point2f(0.6, 0.6)
                        # Frame.size(Y, X)
                        # tag.corners (X, Y)
                        topLeftXY = (min(tag.corners[:, 0]), min(tag.corners[:, 1]))
                        bottomRightXY = (max(tag.corners[:, 0]), max(tag.corners[:, 1]))
                        topLeft.x = topLeftXY[0] / frameRight.shape[1]
                        topLeft.y = topLeftXY[1] / frameRight.shape[0]
                        bottomRight.x = bottomRightXY[0] / frameRight.shape[1]
                        bottomRight.y = bottomRightXY[1] / frameRight.shape[0]

                        tagInfo = {
                            "tagId": tag.tag_id,
                            "tagCorners": tag.corners,
                            "tagCenter": tag.center,
                            "tagTopLeft": topLeft,
                            "tagTopLeftXY": [int(i) for i in topLeftXY],
                            "tagBottomRight": bottomRight,
                            "tagBottomRightXY": [int(i) for i in bottomRightXY]
                        }
                        detectedTags.append(tagInfo)

                    # Sort by center Y, then X coordinates to match with spatial data
                    detectedTags = sorted(detectedTags, key=lambda k: [k["tagCenter"][1], k["tagCenter"][0]])

                    # Only assign spatial data to tags after getting valid tags
                    if len(spatialData) > 0 and len(spatialData) == len(detectedTags) and not firstPass:
                        # Sort by center Y, then X coordinates to match with spatial Data
                        sortedSpatialData = sorted(spatialData, key=lambda d: [d.spatialCoordinates.y, d.spatialCoordinates.x])
                        dataCounter = 0
                        for detectedTag in detectedTags:
                            detectedTag["spatialData"] = sortedSpatialData[dataCounter].spatialCoordinates
                            dataCounter += 1

                    for detectedTag in detectedTags:
                        points = detectedTag["tagCorners"].astype(np.int32)
                        # Shift points since this is a snapshot
                        cv2.polylines(frameRight, [points], True, (120, 120, 120), 3)
                        textX = min(points[:, 0])
                        textY = min(points[:, 1]) + 20
                        cv2.putText(frameRight, "tag_id: {}".format(tag.tag_id), (textX, textY), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.6, (255, 255, 255))

                        # If we have spatial data, print it
                        if "spatialData" in detectedTag.keys():
                            cv2.putText(frameRight, "x: {:.2f}".format(detectedTag["spatialData"].x / 1000), (textX, textY + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                        (120, 120, 120))
                            cv2.putText(frameRight, "y: {:.2f}".format(detectedTag["spatialData"].y / 1000), (textX, textY + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                        (120, 120, 120))
                            cv2.putText(frameRight, "z: {:.2f}".format(detectedTag["spatialData"].z / 1000), (textX, textY + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                        (120, 120, 120))
                            cv2.rectangle(frameRight, detectedTag["tagTopLeftXY"], detectedTag["tagBottomRightXY"], (0, 0, 0), 3)

                    # For each detected tag, generate a ROI to estimate depth to tag
                    if len(detectedTags) > 0:
                        cfg = dai.SpatialLocationCalculatorConfig()
                        for detectedTag in detectedTags:
                            config = dai.SpatialLocationCalculatorConfigData()
                            config.depthThresholds.lowerThreshold = 100
                            config.depthThresholds.upperThreshold = 10000
                            config.roi = dai.Rect(detectedTag["tagTopLeft"], detectedTag["tagBottomRight"])
                            cfg.addROI(config)

                        spatialCalcConfigInQueue.send(cfg)
                        firstPass = False

            depthFrame = inDepth.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            fps.nextIter()
            cv2.putText(frameRight, "{:.2f}".format(fps.fps()), (0, 24), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

            cv2.imshow(pipeline_info["depthQueue"], depthFrameColor)
            cv2.imshow(pipeline_info["monoRightQueue"], frameRight)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    main()
