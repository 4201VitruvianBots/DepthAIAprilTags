import argparse
import copy

import cv2
import depthai as dai
import logging
import math
import numpy as np
from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtGui import QIntValidator
import socket
import sys

from common import constants, mathUtils
from common import utils
import spatialCalculator_pipelines
from common.constants import TAG_DICTIONARY

from common.imu import navX
from networktables.util import NetworkTables
from pupil_apriltags import Detector
from spatialCalculator import HostSpatialsCalc, estimate_robot_pose_with_solvePnP

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='debug', action="store_true", default=False, help='Start in Debug Mode')
parser.add_argument('-t', dest='test', action="store_true", default=False, help='Start in Test Mode')
parser.add_argument('-pt', dest='performance_test', action="store_true", default=False, help='Set Performance Test Mode')

parser.add_argument('-f', dest='family', action="store", type=str, default='tag36h11',
                    help='Tag family (default: tag36h11)')
parser.add_argument('-nt', dest='nthreads', action="store", type=int, default=3,
                    help='nthreads (default: 3)')
parser.add_argument('-qd', dest='quad_decimate', action="store", type=float, default=4.0,
                    help='quad_decimate (default: 4)')
parser.add_argument('-qs', dest='quad_sigma', action="store", type=float, default=0.0,
                    help='quad_sigma (default: 0.0)')
parser.add_argument('-re', dest='refine_edges', action="store", type=float, default=1.0,
                    help='refine_edges (default: 1.0)')
parser.add_argument('-ds', dest='decode_sharpening', action="store", type=float, default=0.25,
                    help='decode_sharpening (default: 0.25)')
parser.add_argument('-dd', dest='detector_debug', action="store", type=int, default=0,
                    help='AprilTag Detector debug mode (default: 0)')

parser.add_argument('-pnp', dest='apriltag_pose', action="store_true", default=False,
                    help='Enable pupil_apriltags Detector Pose Estimation')
parser.add_argument('-imu', dest='imu', action="store_true", default=False, help='Use external IMU')
parser.add_argument('-r', dest='record_video', action="store_true", default=False, help='Record video data')

parser.add_argument('-dic', dest='tag_dictionary', action="store", type=str, default='test', help='Set Tag Dictionary')

args = parser.parse_args()

log = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
log.addHandler(c_handler)
log.setLevel(logging.INFO)


def init_networktables():
    NetworkTables.startClientTeam(4201)
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    if not NetworkTables.isConnected():
        log.debug("Could not connect to team client. Trying other addresses...")
        NetworkTables.startClient([
            ip
            # '10.42.1.2',
            # ('127.0.0.1', 57599),
            # ('localhost', 57823)
            # '10.0.0.2',
            # '192.168.100.25'
            # '192.168.100.25'
            # '172.22.64.1'
            # '169.254.254.200'
        ])

    if NetworkTables.isConnected():
        log.info("NT Connected to {}".format(NetworkTables.getRemoteAddress()))
        return True
    else:
        log.error("Could not connect to NetworkTables. Restarting server...")
        return False


def main():
    if args.performance_test:
        disabledStdOut = logging.StreamHandler(stream=None)
        log.addHandler(disabledStdOut)

    log.info("Starting AprilTag Spatial Detector")

    DISABLE_VIDEO_OUTPUT = args.test
    ENABLE_SOLVEPNP = args.apriltag_pose
    USE_EXTERNAL_IMU = args.imu

    pipeline, pipeline_info = spatialCalculator_pipelines.create_stereoDepth_pipeline()

    detector = Detector(families=args.family,
                        nthreads=args.nthreads,
                        quad_decimate=args.quad_decimate,
                        quad_sigma=args.quad_sigma,
                        refine_edges=args.refine_edges,
                        decode_sharpening=args.decode_sharpening,
                        debug=args.detector_debug)

    init_networktables()
    nt_depthai_tab = NetworkTables.getTable("DepthAI")
    nt_drivetrain_tab = NetworkTables.getTable("Drivetrain")

    fps = utils.FPSHandler()
    latency = np.array([])

    tag_dictionary = constants.TAG_DICTIONARIES[args.tag_dictionary]

    with dai.Device(pipeline) as device:
        log.info("USB SPEED: {}".format(device.getUsbSpeed()))
        if device.getUsbSpeed() not in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
            log.warning("USB Speed is set to USB 2.0")

        deviceID = device.getMxId()
        eepromData = device.readCalibration().getEepromData()
        iMatrix = eepromData.cameraData.get(dai.CameraBoardSocket.RIGHT).intrinsicMatrix

        if deviceID in constants.CAMERA_IDS:
            productName = constants.CAMERA_IDS[deviceID]
        else:
            productName = eepromData.productName

            if len(productName) == 0:
                boardName = eepromData.boardName
                if boardName == 'BW1098OBC':
                    productName = 'OAK-D'
                else:
                    log.warning("Product name could not be determined. Defaulting to OAK-D")
                    productName = 'OAK-D'

        camera_params = {
            "hfov": constants.CAMERA_PARAMS[productName]["mono"]["hfov"],
            "vfov": constants.CAMERA_PARAMS[productName]["mono"]["vfov"],
            "mount_angle_radians": math.radians(constants.CAMERA_MOUNT_ANGLE),
            "iMatrix": np.array(iMatrix).reshape(3, 3),
            "intrinsicValues": (iMatrix[0][0], iMatrix[1][1], iMatrix[0][2], iMatrix[1][2])
        }

        # device.setIrLaserDotProjectorBrightness(200)
        # device.setIrFloodLightBrightness(1500)

        depthQueue = device.getOutputQueue(name=pipeline_info["depthQueue"], maxSize=1, blocking=False)
        qRight = device.getOutputQueue(name=pipeline_info["monoRightQueue"], maxSize=1, blocking=False)
        qInputRight = device.getInputQueue(pipeline_info["monoRightCtrlQueue"])

        gyro = None
        if USE_EXTERNAL_IMU:
            try:
                gyro = navX('COM4')
            except Exception as e:
                log.error("Could not initialize gyro")

        # Precalculate this value to save some performance
        camera_params["hfl"] = pipeline_info["resolution_x"] / (2 * math.tan(math.radians(camera_params['hfov']) / 2))
        camera_params["vfl"] = pipeline_info["resolution_y"] / (2 * math.tan(math.radians(camera_params['vfov']) / 2))

        hostSpatials = HostSpatialsCalc(camera_params, tag_dictionary, log)

        robotAngles = {
            'pitch': None,
            'yaw': None
        }

        stats = {
            'numTags': 0,
            'depthAI': {
                'x_pos': [0],
                'y_pos': [0],
                'z_pos': [0],
            },
            'solvePnP': {
                'x_pos': [0],
                'y_pos': [0],
                'z_pos': [0],
            }
        }

        camera_settings = {
            'manual_exposure_usec': 10000,
            'manual_exposure_iso': 200,
            'brightness': 5,
            'white_balance': 6000
        }
        prev_camera_settings = copy.copy(camera_settings)

        if not DISABLE_VIDEO_OUTPUT:
            app = QtWidgets.QApplication(sys.argv)
            testGui = DebugWindow(gyro, camera_settings, ENABLE_SOLVEPNP)
            # app.exec_()

        lastMonoFrame = None
        lastDepthFrame = None

        if DISABLE_VIDEO_OUTPUT:
            # print("\033[2J", end='')
            pass

        while True:
            try:
                inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
                inRight = qRight.get()
                fps.nextIter()
            except Exception as e:
                log.error("Frame not received")
                continue

            if USE_EXTERNAL_IMU:
                try:
                    robotAngles = {
                        'pitch': math.radians(gyro.get('pitch')),
                        'yaw': math.radians(gyro.get('yaw'))
                    }
                    if not DISABLE_VIDEO_OUTPUT:
                        testGui.updateYawValue(math.degrees(-robotAngles['yaw']))
                        testGui.updatePitchValue(math.degrees(robotAngles['pitch']))
                except Exception as e:
                    # log.error("Could not grab gyro values")
                    pass
            else:
                robotAngles = {
                    'pitch': math.radians(constants.CAMERA_MOUNT_ANGLE),
                    'yaw': math.radians(nt_drivetrain_tab.getNumber("Heading_Degrees", 0.0))
                }

            depthFrame = inDepth.getFrame()
            frameRight = inRight.getCvFrame()  # get mono right frame

            if not DISABLE_VIDEO_OUTPUT:
                ENABLE_SOLVEPNP = testGui.getSolvePnpEnabled()

                if testGui.getPauseResumeState():
                    depthFrame = lastDepthFrame
                    frameRight = lastMonoFrame

            tags = detector.detect(frameRight,
                                   estimate_tag_pose=ENABLE_SOLVEPNP,
                                   camera_params=camera_params['intrinsicValues'],
                                   tag_size=constants.TAG_SIZE_M)

            detectedTags = []
            x_pos = []
            y_pos = []
            z_pos = []
            pnp_tag_id = []
            pnp_x_pos = []
            pnp_y_pos = []
            pnp_z_pos = []
            pose_id = []
            if len(tags) > 0:
                for tag in tags:
                    if not DISABLE_VIDEO_OUTPUT:
                        if tag.tag_id not in testGui.getTagFilter():
                            continue
                        elif tag.tag_id not in TAG_DICTIONARY.keys():
                            log.warning("Tag ID {} found, but not defined".format(tag.tag_id))
                    if tag.decision_margin < 30:
                        log.warning("Tag {} found, but not valid".format(tag.tag_id))
                        continue

                    topLeftXY = (int(min(tag.corners[:, 0])), int(min(tag.corners[:, 1])))
                    bottomRightXY = (int(max(tag.corners[:, 0])), int(max(tag.corners[:, 1])))

                    roi = (topLeftXY[0], topLeftXY[1], bottomRightXY[0], bottomRightXY[1])

                    robotPose, tagTranslation, spatialData, = hostSpatials.calc_spatials(depthFrame, tag, roi, robotAngles)

                    if robotPose is None:
                        log.warning("Could not determine robot pose")
                        continue

                    tagInfo = {
                        "tag": tag,
                        "XAngle": tagTranslation['x_angle'],
                        "YAngle": tagTranslation['y_angle'],
                        "topLeftXY": topLeftXY,
                        "bottomRightXY": bottomRightXY,
                        "spatialData": spatialData,
                        "translation": tagTranslation,
                        "estimatedRobotPose": robotPose,
                    }
                    detectedTags.append(tagInfo)
                    x_pos.append(robotPose['x'])
                    y_pos.append(robotPose['y'])
                    z_pos.append(robotPose['z'])
                    pose_id.append(tag.tag_id)
                    log.info("Tag ID: {}\tCenter: {}\tz: {}".format(tag.tag_id, tag.center, spatialData['z']))

                    # Use this to compare stereoDepth results vs solvePnP
                    if ENABLE_SOLVEPNP:
                        pnpRobotPose = estimate_robot_pose_with_solvePnP(tag, tagInfo, tag_dictionary, camera_params, robotAngles)

                        tagInfo['deltaTranslation'] = {
                            'x': tag.pose_t[0][0] - spatialData['x'],
                            'y': tag.pose_t[1][0] - spatialData['y'],
                            'z': tag.pose_t[2][0] - spatialData['z']
                        }

                        pnp_tag_id.append(tag.tag_id)
                        pnp_x_pos.append(pnpRobotPose['x'])
                        pnp_y_pos.append(pnpRobotPose['y'])

                        log.info("Tag ID: {}\tDelta X: {:.2f}\t"
                                 "Delta Y: {:.2f}\tDelta Z: {:.2f}".format(tag.tag_id,
                                                                           tagInfo['deltaTranslation']['x'],
                                                                           tagInfo['deltaTranslation']['y'],
                                                                           tagInfo['deltaTranslation']['z']))

            fpsValue = fps.fps()
            latencyMs = (dai.Clock.now() - inDepth.getTimestamp()).total_seconds() * 1000.0
            latency = np.append(latency, latencyMs)
            avgLatency = np.average(latency) if len(latency) < 100 else np.average(latency[-100:])
            latencyStd = np.std(latency) if len(latency) < 100 else np.std(latency[-100:])

            nt_depthai_tab.putNumberArray("Pose ID", pose_id)
            nt_depthai_tab.putNumberArray("X Poses", x_pos)
            nt_depthai_tab.putNumberArray("Y Poses", y_pos)
            nt_depthai_tab.putNumberArray("Z Poses", z_pos)

            # Merge AprilTag measurements to estimate
            log.info("Estimated Pose X: {:.2f}\tY: {:.2f}\tZ: {:.2f}".format(np.average(x_pos), np.average(y_pos), np.average(z_pos)))
            log.info("Std dev X: {:.2f}\tY: {:.2f}\tZ: {:.2f}".format(np.std(x_pos),
                                                                      np.std(y_pos),
                                                                      np.std(z_pos)))
            nt_depthai_tab.putNumber("Avg X Pose", np.average(x_pos))
            nt_depthai_tab.putNumber("Avg Y Pose", np.average(x_pos))
            nt_depthai_tab.putNumber("Heading Pose", 0 if robotAngles['yaw'] is None else robotAngles['yaw'])
            nt_depthai_tab.putNumber("latency", latency[-1])

            nt_depthai_tab.putNumberArray("PnP Pose ID", pnp_tag_id)
            nt_depthai_tab.putNumberArray("PnP X Poses", pnp_x_pos)
            nt_depthai_tab.putNumberArray("PnP Y Poses", pnp_y_pos)

            stats['numTags'] = len(detectedTags)
            stats['depthAI'] = {
                'x_pos': x_pos,
                'y_pos': y_pos,
                'z_pos': z_pos,
            }
            stats['solvePnP'] = {
                'x_pos': pnp_x_pos,
                'y_pos': pnp_y_pos,
                'z_pos': [0]
            }

            if not DISABLE_VIDEO_OUTPUT:
                for detectedTag in detectedTags:
                    points = detectedTag["tag"].corners.astype(np.int32)
                    # Shift points since this is a snapshot
                    cv2.polylines(frameRight, [points], True, (120, 120, 120), 3)
                    textX = min(points[:, 0])
                    textY = min(points[:, 1]) + 20
                    cv2.putText(frameRight, "tag_id: {}".format(detectedTag['tag'].tag_id),
                                (textX, textY), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255))

                    if ENABLE_SOLVEPNP:
                        ipoints, _ = cv2.projectPoints(constants.OPOINTS,
                                                       detectedTag["tag"].pose_R,
                                                       detectedTag["tag"].pose_t,
                                                       camera_params['iMatrix'],
                                                       np.zeros(5))

                        ipoints = np.round(ipoints).astype(int)

                        ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

                        for i, j in constants.EDGES:
                            cv2.line(frameRight, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)

                    cv2.putText(frameRight, "x: {:.2f}".format(detectedTag["spatialData"]['x']),
                                (textX, textY + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255))
                    cv2.putText(frameRight, "y: {:.2f}".format(detectedTag["spatialData"]['y']),
                                (textX, textY + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255))
                    cv2.putText(frameRight, "x angle: {:.2f}".format(detectedTag["translation"]['x_angle']),
                                (textX, textY + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255))
                    cv2.putText(frameRight, "y angle: {:.2f}".format(detectedTag["translation"]['y_angle']),
                                (textX, textY + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255))
                    cv2.putText(frameRight, "z: {:.2f}".format(detectedTag["spatialData"]['z']),
                                (textX, textY + 100), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255))
                    cv2.rectangle(frameRight, detectedTag["topLeftXY"], detectedTag["bottomRightXY"],
                                  (0, 0, 0), 3)

                if not testGui.getPauseResumeState():
                    cv2.circle(frameRight, (int(frameRight.shape[1]/2), int(frameRight.shape[0]/2)), 1, (255, 255, 255), 1)
                    cv2.putText(frameRight, "FPS: {:.2f}".format(fpsValue), (0, 24), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
                    cv2.putText(frameRight, "Latency: {:.2f}ms".format(avgLatency), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

                    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depthFrameColor = cv2.equalizeHist(depthFrameColor)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                    # cv2.imshow(pipeline_info["monoRightQueue"], frameRight)
                    # cv2.imshow(pipeline_info["depthQueue"], depthFrameColor)
                    testGui.updateTagIds(pose_id)
                    testGui.updateStatsValue(stats)
                    testGui.updateFrames(frameRight, depthFrameColor)
            else:
                print('FPS: {:.2f}\tLatency: {:.2f} ms\tStd: {:.2f}'.format(fpsValue, avgLatency, latencyStd))
                print('DepthAI')
                print('Tags: {}'.format(pose_id))
                print("         Avg.: {:.6f}\t{:.6f}\t{:.6f}".format(np.average(x_pos), np.average(y_pos), np.average(z_pos)))
                print("         Std.: {:.6f}\t{:.6f}\t{:.6f}".format(np.average(x_pos), np.average(y_pos), np.average(z_pos)))
                # print("\033[2J", end='')

            lastMonoFrame = copy.copy(frameRight)
            lastDepthFrame = copy.copy(depthFrame)

            # Camera control
            if not DISABLE_VIDEO_OUTPUT:
                camera_settings = testGui.getCameraSettings()

            if prev_camera_settings != camera_settings:
                ctrl = dai.CameraControl()
                ctrl.setManualExposure(camera_settings['manual_exposure_usec'], camera_settings['manual_exposure_iso'])
                ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.OFF)
                ctrl.setManualWhiteBalance(camera_settings['white_balance'])
                ctrl.setBrightness(camera_settings['brightness'])
                qInputRight.send(ctrl)
                prev_camera_settings = copy.copy(camera_settings)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord(' '):
                if 'gyro' in locals():
                    gyro.resetAll()

            if not DISABLE_VIDEO_OUTPUT:
                if not testGui.isVisible():
                    break


class DebugWindow(QtWidgets.QWidget):
    def __init__(self, gyro, camera_settings, solvePnp=False):
        super(DebugWindow, self).__init__()
        uic.loadUi('../designer/debugWindow.ui', self)
        self.tagFilter = list(range(0, 8))
        self.tagCheckbox0.stateChanged.connect(lambda: self.updateTagFilter())
        self.tagCheckbox1.stateChanged.connect(lambda: self.updateTagFilter())
        self.tagCheckbox2.stateChanged.connect(lambda: self.updateTagFilter())
        self.tagCheckbox3.stateChanged.connect(lambda: self.updateTagFilter())
        self.tagCheckbox4.stateChanged.connect(lambda: self.updateTagFilter())
        self.tagCheckbox5.stateChanged.connect(lambda: self.updateTagFilter())
        self.tagCheckbox6.stateChanged.connect(lambda: self.updateTagFilter())
        self.tagCheckbox7.stateChanged.connect(lambda: self.updateTagFilter())

        self.gyro = gyro
        if self.gyro is not None:
            self.resetGyroBtn.clicked.connect(lambda: self.resetGyroButtonPressed(self.gyro))
        else:
            self.resetGyroBtn.setEnabled(False)
            self.yawValue.setEnabled(False)
            self.pitchValue.setEnabled(False)

        self.values = None
        self.unitsButtonGroup.buttonClicked.connect(lambda: self.updateUnits())
        self.unitScale = 1.0

        self.solvePnp = solvePnp
        self.solvePnpEnableBtn.setChecked(self.solvePnp)
        self.solvePnpEnableBtn.clicked.connect(lambda: self.toggleSolvePnp())

        self.camera_settings = camera_settings
        self.initializeCameraSettings()
        self.lockCameraUpdates = False

        self.exposureTimeSlider.valueChanged.connect(lambda: self.updateCameraSettings(True))
        # self.exposureTimeValue.textChanged.connect(lambda: self.updateCameraSettings(False))
        self.exposureIsoSlider.valueChanged.connect(lambda: self.updateCameraSettings(True))
        # self.exposureIsoValue.textChanged.connect(lambda: self.updateCameraSettings(False))
        self.brightnessSlider.valueChanged.connect(lambda: self.updateCameraSettings(True))
        # self.brightnessValue.textChanged.connect(lambda: self.updateCameraSettings(False))
        self.whiteBalanceSlider.valueChanged.connect(lambda: self.updateCameraSettings(True))
        # self.whiteBalanceValue.textChanged.connect(lambda: self.updateCameraSettings(False))

        self.pauseResumeBtn.clicked.connect(lambda: self.pauseResumeButtonPressed())
        self.pause = False
        self.show()

    def updateYawValue(self, value):
        self.yawValue.setText("{:.06f}".format(value))

    def updatePitchValue(self, value):
        self.pitchValue.setText("{:.06f}".format(value))

    def updateTagIds(self, tagIds):
        if len(tagIds) > 0:
            self.detectedTagsValue.setText("{}".format(tagIds))
        else:
            self.detectedTagsValue.setText('')

    def updateStatsValue(self, values=None):
        if values is None:
            values = self.values

        self.avgDepthXValue.setText("{:.04f}".format(np.average(values['depthAI']['x_pos']) * self.unitScale))
        self.avgDepthYValue.setText("{:.04f}".format(np.average(values['depthAI']['y_pos']) * self.unitScale))
        self.avgDepthZValue.setText("{:.04f}".format(np.average(values['depthAI']['z_pos']) * self.unitScale))
        self.stdDepthXValue.setText("{:.04f}".format(np.std(values['depthAI']['x_pos']) * self.unitScale))
        self.stdDepthYValue.setText("{:.04f}".format(np.std(values['depthAI']['y_pos']) * self.unitScale))
        self.stdDepthZValue.setText("{:.04f}".format(np.std(values['depthAI']['z_pos']) * self.unitScale))

        self.avgPnpXValue.setText("{:.04f}".format(np.average(values['solvePnP']['x_pos']) * self.unitScale))
        self.avgPnpYValue.setText("{:.04f}".format(np.average(values['solvePnP']['y_pos']) * self.unitScale))
        self.avgPnpZValue.setText("{:.04f}".format(np.average(values['solvePnP']['z_pos']) * self.unitScale))
        self.stdPnpXValue.setText("{:.04f}".format(np.std(values['solvePnP']['x_pos']) * self.unitScale))
        self.stdPnpYValue.setText("{:.04f}".format(np.std(values['solvePnP']['y_pos']) * self.unitScale))
        self.stdPnpZValue.setText("{:.04f}".format(np.std(values['solvePnP']['z_pos']) * self.unitScale))
        self.values = values

    def updateFrames(self, monoFrame, depthFrame):
        activeTab = self.frameWidget.currentIndex()

        if activeTab == 0:
            monoFrame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2RGB)
            img = QtGui.QImage(monoFrame, monoFrame.shape[1], monoFrame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.monoFrame.setMinimumWidth(monoFrame.shape[1])
            self.monoFrame.setMinimumHeight(monoFrame.shape[0])
            self.monoFrame.setMaximumWidth(monoFrame.shape[1])
            self.monoFrame.setMaximumHeight(monoFrame.shape[0])
            self.monoFrame.setPixmap(pix)
        elif activeTab == 1:
            depthFrame = cv2.cvtColor(depthFrame, cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(depthFrame, depthFrame.shape[1], depthFrame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.depthFrame.setMinimumWidth(depthFrame.shape[1])
            self.depthFrame.setMinimumHeight(depthFrame.shape[0])
            self.depthFrame.setMaximumWidth(depthFrame.shape[1])
            self.depthFrame.setMaximumHeight(depthFrame.shape[0])
            self.depthFrame.setPixmap(pix)
        elif activeTab == 2:
            monoFrame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2RGB)
            img = QtGui.QImage(monoFrame, monoFrame.shape[1], monoFrame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.monoFrame2.setMinimumWidth(monoFrame.shape[1])
            self.monoFrame2.setMinimumHeight(monoFrame.shape[0])
            self.monoFrame2.setMaximumWidth(monoFrame.shape[1])
            self.monoFrame2.setMaximumHeight(monoFrame.shape[0])
            self.monoFrame2.setPixmap(pix)

            depthFrame = cv2.cvtColor(depthFrame, cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(depthFrame, depthFrame.shape[1], depthFrame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.depthFrame2.setMinimumWidth(depthFrame.shape[1])
            self.depthFrame2.setMinimumHeight(depthFrame.shape[0])
            self.depthFrame2.setMaximumWidth(depthFrame.shape[1])
            self.depthFrame2.setMaximumHeight(depthFrame.shape[0])
            self.depthFrame2.setPixmap(pix)

    def updateUnits(self):
        if self.unitsButtonGroup.checkedId() == -2:
            self.unitScale = 1.0
        elif self.unitsButtonGroup.checkedId() == -3:
            self.unitScale = 3.28084
        elif self.unitsButtonGroup.checkedId() == -4:
            self.unitScale = 39.3701

        self.updateStatsValue()

    def updateTagFilter(self):
        self.tagFilter = np.array(np.where([self.tagCheckbox0.isChecked(), self.tagCheckbox1.isChecked(),
                                            self.tagCheckbox2.isChecked(), self.tagCheckbox3.isChecked(),
                                            self.tagCheckbox4.isChecked(), self.tagCheckbox5.isChecked(),
                                            self.tagCheckbox6.isChecked(), self.tagCheckbox7.isChecked()])).tolist()[0]

    def getTagFilter(self):
        return self.tagFilter

    def resetGyroButtonPressed(self, gyro):
        if gyro is not None:
            gyro.resetAll()

    def toggleSolvePnp(self):
        self.solvePnp = self.solvePnpEnableBtn.isChecked()

    def getSolvePnpEnabled(self):
        return self.solvePnp

    def pauseResumeButtonPressed(self):
        if self.pause:
            self.pause = False
            self.pauseResumeBtn.setText("Pause")
        else:
            self.pause = True
            self.pauseResumeBtn.setText("Resume")

    def getPauseResumeState(self):
        return self.pause

    def initializeCameraSettings(self):
        # Exposure time (microseconds)
        self.exposureTimeLabel.setToolTip("Min: 0, Max 10000")
        self.exposureTimeSlider.setMinimum(0)
        self.exposureTimeSlider.setMaximum(10000)
        self.exposureTimeSlider.setValue(self.camera_settings['manual_exposure_usec'])
        self.exposureTimeValue.setText(self.camera_settings['manual_exposure_usec'].__str__())
        exposureTimeRange = QIntValidator()
        exposureTimeRange.setRange(0, 10000)
        self.exposureTimeValue.setValidator(exposureTimeRange)

        # Exposure ISO Sensitivity (100, 1600)
        self.exposureIsoLabel.setToolTip("Min: 100, Max 1600")
        self.exposureIsoSlider.setMinimum(100)
        self.exposureIsoSlider.setMaximum(1600)
        self.exposureIsoSlider.setValue(self.camera_settings['manual_exposure_iso'])
        self.exposureIsoValue.setText(self.camera_settings['manual_exposure_iso'].__str__())
        exposureIsoRange = QIntValidator()
        exposureIsoRange.setRange(100, 1600)
        self.exposureIsoValue.setValidator(exposureIsoRange)

        # Temperature in Kelvins (1000, 12000)
        self.whiteBalanceLabel.setToolTip("Min: 1000, Max 12000")
        self.whiteBalanceSlider.setMinimum(1000)
        self.whiteBalanceSlider.setMaximum(12000)
        self.whiteBalanceSlider.setValue(self.camera_settings['white_balance'])
        self.whiteBalanceValue.setText(self.camera_settings['white_balance'].__str__())
        whiteBalanceRange = QIntValidator()
        whiteBalanceRange.setRange(1000, 12000)
        self.whiteBalanceValue.setValidator(whiteBalanceRange)

        # Image Brightness (-10, 10)
        self.brightnessLabel.setToolTip("Min: -10, Max 10")
        self.brightnessSlider.setMinimum(-10)
        self.brightnessSlider.setMaximum(10)
        self.brightnessSlider.setValue(self.camera_settings['brightness'])
        self.brightnessValue.setText(self.camera_settings['brightness'].__str__())
        brightnessRange = QIntValidator()
        brightnessRange.setRange(-10, 10)
        self.brightnessValue.setValidator(brightnessRange)

    def updateCameraSettings(self, slider):
        # if not self.lockCameraUpdates:
        #     self.lockCameraUpdates = True
            if slider:
                self.exposureTimeValue.setText(self.exposureTimeSlider.value().__str__())
                self.exposureIsoValue.setText(self.exposureIsoSlider.value().__str__())
                self.whiteBalanceValue.setText(self.whiteBalanceSlider.value().__str__())
                self.brightnessValue.setText(self.brightnessSlider.value().__str__())
            else:
                try:
                    self.exposureTimeSlider.setValue(int(self.exposureTimeValue.text()))
                    self.exposureIsoSlider.setValue(int(self.exposureIsoValue.text()))
                    self.whiteBalanceSlider.setValue(int(self.whiteBalanceValue.text()))
                    self.brightnessSlider.setValue(int(self.brightnessValue.text()))
                except Exception as e:
                    self.lockCameraUpdates = False
                    return

            self.camera_settings['manual_exposure_usec'] = int(self.exposureTimeValue.text())
            self.camera_settings['manual_exposure_iso'] = int(self.exposureIsoValue.text())
            self.camera_settings['white_balance'] = int(self.whiteBalanceValue.text())
            self.camera_settings['brightness'] = int(self.brightnessValue.text())
        # else:
        #     self.lockCameraUpdates = False

    def getCameraSettings(self):
        return self.camera_settings


if __name__ == '__main__':
    main()
