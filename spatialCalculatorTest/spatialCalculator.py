import math
import numpy as np
import depthai as dai

from common import mathUtils
from common.constants import TAG_DICTIONARY


class HostSpatialsCalc:
    # We need device object to get calibration data
    def __init__(self, camera_params):
        # calibData = device.readCalibration()
        # Required information for calculating spatial coordinates on the host
        self.monoHFOV = np.deg2rad(camera_params['hfov'])
        self.monoVFOV = np.deg2rad(camera_params['vfov'])
        self.mmountAngle = camera_params['mount_angle_radians']

        # Values
        self.DELTA = 5
        self.THRESH_LOW = 200  # 20cm
        self.THRESH_HIGH = 30000  # 30m

    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low

    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low

    def setDeltaRoi(self, delta):
        self.DELTA = delta

    def _check_input(self, roi, frame):  # Check if input is ROI or point. If point, convert to ROI
        if len(roi) == 4: return roi
        if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
        # Limit the point so ROI won't be outside the frame
        self.DELTA = 5  # Take 10x10 depth pixels around point for depth averaging
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x - self.DELTA, y - self.DELTA, x + self.DELTA, y + self.DELTA)

    def _calc_h_angle(self, frame, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    def _calc_v_angle(self, frame, offset):
        return math.atan(math.tan(self.monoVFOV / 2.0) * offset / (frame.shape[0] / 2.0))

    # roi has to be list of ints
    def calc_spatials(self, depthFrame, tag, roi, robot_angle=0, averaging_method=np.mean):
        if tag.tag_id not in TAG_DICTIONARY.keys():
            return None, None, None

        tagPose = TAG_DICTIONARY[tag.tag_id]["pose"]

        # roi = self._check_input(roi, depthFrame)  # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])

        centroid = {  # Get centroid of the ROI
            'x': tag.center[0],
            'y': tag.center[1]
        }

        midW = int(depthFrame.shape[1] / 2)  # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2)  # middle of the depth img height
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        angle_x = self._calc_h_angle(depthFrame, bb_x_pos)
        angle_y = -self._calc_v_angle(depthFrame, bb_y_pos)

        spatials = {
            'z': averageDepth / 1000,
            'x': averageDepth * math.tan(angle_x) / 1000,
            'y': -averageDepth * math.tan(angle_y) / 1000,
        }

        xy_target_distance = math.cos(self.mmountAngle + angle_y) * spatials['z']

        tag_translation = {
            'x': math.cos(angle_x) * xy_target_distance,
            'y': -math.sin(angle_x) * xy_target_distance,
            'z': math.sin(self.mmountAngle + angle_y) * spatials['z'],
            'x_angle': math.degrees(angle_x),
            'y_angle': math.degrees(angle_y)
        }

        rotatedTranslation = mathUtils.rotateTranslation((tag_translation['x'], tag_translation['y']), robot_angle)

        robotPose = {
            'x': tagPose['x'] - rotatedTranslation[0],
            'y': tagPose['y'] - rotatedTranslation[1],
            'z': tagPose['z'] - tag_translation['z']
        }

        return robotPose, tag_translation, spatials


def estimate_robot_pose_from_apriltag(tag, spatialData, camera_params, frame_shape):
    if tag.tag_id not in TAG_DICTIONARY.keys():
        return None, None

    tagPose = TAG_DICTIONARY[tag.tag_id]["pose"]

    horizontal_angle_radians = math.atan((tag.center[0] - (frame_shape[1] / 2.0)) / camera_params["hfl"])
    vertical_angle_radians = -math.atan((tag.center[1] - (frame_shape[0] / 2.0)) / camera_params["vfl"])
    horizontal_angle_degrees = math.degrees(horizontal_angle_radians)
    vertical_angle_degrees = math.degrees(vertical_angle_radians)

    xy_target_distance = math.cos(camera_params['mount_angle_radians'] + vertical_angle_radians) * spatialData['z']

    # Calculate the translation from the camera to the tag, in field coordinates
    tag_translation = {
        'x': math.cos(horizontal_angle_radians) * xy_target_distance,
        'y': -math.sin(horizontal_angle_radians) * xy_target_distance,
        'z': math.sin(camera_params['mount_angle_radians'] + vertical_angle_radians) * spatialData['z'],
        'x_angle': horizontal_angle_degrees,
        'y_angle': vertical_angle_degrees
    }

    robotPose = {
        'x': tagPose['x'] - tag_translation['x'],
        'y': tagPose['y'] - tag_translation['y'],
        'z': tagPose['z'] - tag_translation['z']
    }

    return robotPose, tag_translation