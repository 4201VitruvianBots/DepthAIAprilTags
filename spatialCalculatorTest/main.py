import argparse

import cv2
import depthai as dai
import logging
from networktables.util import NetworkTables
import math
import numpy as np

from pupil_apriltags import Detector

import spatialCalculator
import spatialCalculator_pipelines
from spatialCalculator import HostSpatialsCalc

from common import constants
from common import utils

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
parser.add_argument('-t', dest='test', action="store_true", default=False, help='Start in Test Mode')

parser.add_argument('-f', dest='family', action="store", type=str, default='tag36h11', help='Tag family (default: tag36h11)')
parser.add_argument('-nt', dest='nthreads', action="store", type=int, default=3, help='nthreads (default: 3)')
parser.add_argument('-qd', dest='quad_decimate', action="store", type=float, default=4.0, help='quad_decimate (default: 4)')
parser.add_argument('-qs', dest='quad_sigma', action="store", type=float, default=0.0, help='quad_sigma (default: 0.0)')
parser.add_argument('-re', dest='refine_edges', action="store", type=float, default=1.0, help='refine_edges (default: 1.0)')
parser.add_argument('-ds', dest='decode_sharpening', action="store", type=float, default=0.25, help='decode_sharpening (default: 0.25)')
parser.add_argument('-dd', dest='detector_debug', action="store", type=int, default=0, help='AprilTag Detector debug mode (default: 0)')

args = parser.parse_args()

log = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
log.addHandler(c_handler)
log.setLevel(logging.INFO)


def init_networktables():
    NetworkTables.startClientTeam(4201)

    if not NetworkTables.isConnected():
        log.debug("Could not connect to team client. Trying other addresses...")
        NetworkTables.startClient([
            '10.42.1.2',
            '127.0.0.1',
            '10.0.0.2',
            '192.168.100.108'
        ])

    if NetworkTables.isConnected():
        log.debug("NT Connected to {}".format(NetworkTables.getRemoteAddress()))
        return True
    else:
        log.error("Could not connect to NetworkTables. Restarting server...")
        return False


def main():
    log.info("Starting AprilTag Spatial Detector")

    DISABLE_VIDEO_OUTPUT = args.test

    pipeline, pipeline_info = spatialCalculator_pipelines.create_spaitalCalculator_pipeline()

    detector = Detector(families=args.family,
                        nthreads=args.nthreads,
                        quad_decimate=args.quad_decimate,
                        quad_sigma=args.quad_sigma,
                        refine_edges=args.refine_edges,
                        decode_sharpening=args.decode_sharpening,
                        debug=args.detector_debug)

    nt_tab = NetworkTables.getTable("DepthAI")

    fps = utils.FPSHandler()

    with dai.Device(pipeline) as device:
        log.info("USB SPEED: {}".format(device.getUsbSpeed()))
        if device.getUsbSpeed() not in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
            log.warning("USB Speed is set to USB 2.0")

        calibData = device.readCalibration()
        productName = calibData.getEepromData().productName
        if len(productName) == 0:
            boardName = calibData.getEepromData().boardName
            if boardName == 'BW1098OBC':
                productName = 'OAK-D'
            else:
                log.warning("Product name could not be determined. Defaulting to OAK-D")
                productName = 'OAK-D'

        camera_params = {
            "hfov": constants.CAMERA_PARAMS[productName]["mono"]["hfov"],
            "vfov": constants.CAMERA_PARAMS[productName]["mono"]["vfov"],
            "mount_angle_radians": math.radians(constants.CAMERA_MOUNT_ANGLE)
        }

        device.setIrLaserDotProjectorBrightness(1200)
        # device.setIrFloodLightBrightness(1500)

        depthQueue = device.getOutputQueue(name=pipeline_info["depthQueue"], maxSize=4, blocking=False)
        qRight = device.getOutputQueue(name=pipeline_info["monoRightQueue"], maxSize=4, blocking=False)

        hostSpatials = HostSpatialsCalc(device)

        # Precalculate this value to save some performance
        camera_params["hfl"] = pipeline_info["resolution_x"] / (2 * math.tan(math.radians(camera_params['hfov']) / 2))
        camera_params["vfl"] = pipeline_info["resolution_y"] / (2 * math.tan(math.radians(camera_params['vfov']) / 2))

        while True:
            inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
            inRight = qRight.tryGet()

            depthFrame = inDepth.getFrame()

            if inRight is not None:
                frameRight = inRight.getCvFrame()  # get mono right frame

                tags = detector.detect(frameRight, estimate_tag_pose=False, tag_size=0.2)

                if len(tags) > 0:
                    detectedTags = []
                    x_pos = []
                    y_pos = []
                    z_pos = []
                    for tag in tags:
                        topLeftXY = (int(min(tag.corners[:, 0])), int(min(tag.corners[:, 1])))
                        bottomRightXY = (int(max(tag.corners[:, 0])), int(max(tag.corners[:, 1])))

                        spatialData, _ = hostSpatials.calc_spatials(depthFrame, (topLeftXY[0], topLeftXY[1], bottomRightXY[0], bottomRightXY[1]))

                        robotPose, tagTranslation = spatialCalculator.estimate_robot_pose_from_apriltag(tag, spatialData, camera_params, frameRight.shape)

                        tagInfo = {
                            "id": tag.tag_id,
                            "corners": tag.corners,
                            "center": tag.center,
                            "XAngle": tagTranslation['x_angle'],
                            "YAngle": tagTranslation['y_angle'],
                            "topLeftXY": topLeftXY,
                            "bottomRightXY": bottomRightXY,
                            "spatialData": spatialData,
                            "translation": tagTranslation,
                            "estimatedRobotPose": robotPose
                        }
                        detectedTags.append(tagInfo)
                        x_pos.append(robotPose['x'])
                        y_pos.append(robotPose['y'])
                        z_pos.append(robotPose['z'])
                        log.info("Tag ID: {}\tCenter: {}\tz: {}".format(tag.tag_id, tag.center, spatialData['z']))

                    # Merge AprilTag measurements to estimate
                    avg_x = sum(x_pos) / len(x_pos)
                    avg_y = sum(y_pos) / len(y_pos)
                    avg_z = sum(z_pos) / len(z_pos)
                    if len(detectedTags) > 1:
                        log.info("Estimated Pose X: {:.2f}\tY: {:.2f}\tZ: {:.2f}".format(avg_x, avg_y, avg_z))
                        log.info("Std dev X: {:.2f}\tY: {:.2f}\tZ: {:.2f}".format(np.std(x_pos), np.std(y_pos), np.std(z_pos)))

                    for detectedTag in detectedTags:
                        points = detectedTag["corners"].astype(np.int32)
                        # Shift points since this is a snapshot
                        cv2.polylines(frameRight, [points], True, (120, 120, 120), 3)
                        textX = min(points[:, 0])
                        textY = min(points[:, 1]) + 20
                        cv2.putText(frameRight, "tag_id: {}".format(detectedTag['id']),
                                    (textX, textY), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                    (255, 255, 255))

                        # If we have spatial data, print it
                        if "spatialData" in detectedTag.keys() and not DISABLE_VIDEO_OUTPUT:
                            cv2.putText(frameRight, "x: {:.2f}".format(detectedTag["translation"]['x']),
                                        (textX, textY + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                        (255, 255, 255))
                            cv2.putText(frameRight, "y: {:.2f}".format(detectedTag["translation"]['y']),
                                        (textX, textY + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                        (255, 255, 255))
                            cv2.putText(frameRight, "z: {:.2f}".format(detectedTag["translation"]['z']),
                                        (textX, textY + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                                        (255, 255, 255))
                            cv2.rectangle(frameRight, detectedTag["topLeftXY"], detectedTag["bottomRightXY"],
                                          (0, 0, 0), 3)

            fps.nextIter()

            if not DISABLE_VIDEO_OUTPUT:
                cv2.putText(frameRight, "{:.2f}".format(fps.fps()), (0, 24), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

                cv2.imshow(pipeline_info["monoRightQueue"], frameRight)

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                cv2.imshow(pipeline_info["depthQueue"], depthFrameColor)
            else:
                print("FPS TEST: {:.2f}".format(fps.fps()))

            key = cv2.waitKey(1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    log.info("Starting AprilTag SpatialCalculator")
    main()
