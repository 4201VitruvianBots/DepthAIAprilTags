#!/usr/bin/env python3
import argparse
import logging

import cv2
import depthai as dai
import numpy as np

from pupil_apriltags import Detector

from common import constants

from common.utils import FPSHandler
from edgeTest import apriltag_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
parser.add_argument('-c', dest='camera_type', action="store", type=str, default='RGB', help='Set camera type '
                                                                                            '(RGB, MONO. Default: RGB)')

args = parser.parse_args()

log = logging.getLogger(__name__)


def main():
    log.info("Starting AprilTag Detector")
    CAMERA_TYPE = args.camera_type

    if (CAMERA_TYPE.upper() != "RGB") and (CAMERA_TYPE.upper() != "MONO"):
        log.error("CAMERA_TYPE not recognized \"{}\". Defaulting to RGB".format(CAMERA_TYPE))

    if CAMERA_TYPE == 'MONO':
        pipeline, pipeline_info, camera_params = apriltag_pipeline.create_pipeline_mono()
    else:
        pipeline, pipeline_info, camera_params = apriltag_pipeline.create_pipeline_rgb()

    detector = Detector(families='tag36h11',
                        nthreads=3,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    fps = FPSHandler()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Output/input queues
        videoQueue = device.getOutputQueue(pipeline_info['videoQueue'], 1, False)
        edgeQueue = device.getOutputQueue(pipeline_info['edgeQueue'], 1, False)
        edgeCfgQueue = device.getInputQueue(pipeline_info['edgeCfgQueue'])

        print("Switch between sobel filter kernels using keys '1' and '2'")

        while True:
            videoOutput = videoQueue.get()
            edgeOutput = edgeQueue.get()

            videoFrame = videoOutput.getFrame()
            edgeFrame = edgeOutput.getFrame()

            tags = detector.detect(videoFrame, estimate_tag_pose=False, tag_size=0.2)

            if len(tags) > 0:
                for tag in tags:
                    print("Tag: {}\nCenter: {}".format(tag.tag_id, tag.center))

            # thresh = cv2.threshold(edgeFrame, 25, 255, cv2.THRESH_BINARY)[1]
            # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            # squares = []
            # contourInfo = []
            # for cnt in contours:
            #     area = cv2.contourArea(cnt)
            #     cnt_len = cv2.arcLength(cnt, True)
            #     cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
            #
            #     # TODO: Figure out better area value
            #     if len(cnt) == 4 and area > 1000:
            #         squares.append(cnt)
            #
            #         # Create a bounding box around the square with padding and save it for later
            #         rect = cv2.minAreaRect(cnt)
            #         points = cv2.boxPoints(rect)
            #         xmin = min(x for x in points[:, 0])
            #         xmax = max(x for x in points[:, 0])
            #         ymin = min(y for y in points[:, 1])
            #         ymax = max(y for y in points[:, 1])
            #         xmin = max(int(xmin - xmin * constants.PADDING_PERCENTAGE), 0)
            #         xmax = min(int(xmax + xmin * constants.PADDING_PERCENTAGE), videoFrame.shape[1])
            #         ymin = max(int(ymin - ymin * constants.PADDING_PERCENTAGE), 0)
            #         ymax = min(int(ymax + ymax * constants.PADDING_PERCENTAGE), videoFrame.shape[0])
            #
            #         contourData = {'Contour': cnt,
            #                        'x_min': xmin,
            #                        'x_max': xmax,
            #                        'y_min': ymin,
            #                        'y_max': ymax,
            #                        'area': area}
            #
            #         contourInfo.append(contourData)
            #
            # contourInfo = sorted(contourInfo, key=lambda d: d['area'], reverse=True)
            # count = 0
            # positives = 0
            # for contour in contourInfo:
            #     frameSegment = videoFrame[contour['y_min']:contour['y_max'], contour['x_min']:contour['x_max']]
            #     # tags = detector.detect(frameSegment, estimate_tag_pose=True, camera_params=camera_params.values(),
            #     #                        tag_size=0.2)
            #     tags = detector.detect(frameSegment, estimate_tag_pose=False, tag_size=0.2)
            #
            #     if len(tags) > 0:
            #         for tag in tags:
            #             points = tag.corners.astype(np.int32)
            #             # Shift points since this is a snapshot
            #             points[:, 0] += contour['x_min']
            #             points[:, 1] += contour['y_min']
            #             cv2.polylines(videoFrame, [points], True, (120, 120, 120), 3)
            #             textX = min(points[:, 0])
            #             textY = min(points[:, 1]) + 20
            #             cv2.putText(videoFrame, "tag_id: {}".format(tag.tag_id), (textX, textY), cv2.FONT_HERSHEY_TRIPLEX,
            #                         0.6, (255, 255, 255))
            #
            #             # pose_x = tag.pose_t[0] * 39.3701
            #             # pose_y = tag.pose_t[1] * 39.3701
            #             # pose_z = tag.pose_t[2] * 39.3701
            #             # cv2.putText(videoFrame, "x: {}".format(pose_x), (textX, textY + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
            #             #             (255, 255, 255))
            #             # cv2.putText(videoFrame, "y: {}".format(pose_y), (textX, textY + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
            #             #             (255, 255, 255))
            #             # cv2.putText(videoFrame, "z: {}".format(pose_z), (textX, textY + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
            #             #             (255, 255, 255))
            #
            #             # TODO: Shift pose estimation by snapshot shift?
            #
            #             # print(tag)
            #             positives += 1
            #
            #     count += 1
            #     if count > 5 or positives > 2:
            #         break

            # if len(squares) > 0:
            #     cv2.drawContours(edgeFrame, squares, -1, color=(255, 255, 255), thickness=cv2.FILLED)

            fps.nextIter()
            cv2.putText(videoFrame, "{:.2f}".format(fps.fps()), (0, 24), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

            # Show the frame
            cv2.imshow(pipeline_info['videoQueue'], videoFrame)

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


if __name__ == '__main__':
    main()
