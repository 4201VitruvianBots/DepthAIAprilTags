
import math

from common.constants import TAG_DICTIONARY


def estimate_robot_pose_from_apriltag(detectedTag, camera_params, frame_shape):
    tagPose = TAG_DICTIONARY[detectedTag["tagId"]]["pose"]

    cameraToTagDistance = math.sqrt(detectedTag["spatialData"].x ** 2 +
                                    detectedTag["spatialData"].y ** 2 +
                                    detectedTag["spatialData"].z ** 2)

    horizontal_angle_radians = math.atan(
        (detectedTag['tagCenter'].x - (frame_shape[0] / 2.0)) / (frame_shape[1] / (2 * math.tan(math.radians(camera_params['hfov']) / 2))))
    # horizontal_angle_offset = math.degrees(horizontal_angle_radians)

    vertical_angle_radians = math.atan(
        (detectedTag['tagCenter'].x - (frame_shape[0] / 2.0)) / (frame_shape[1] / (2 * math.tan(math.radians(camera_params['vfov']) / 2))))
    # vertical_angle_offset = -math.degrees(vertical_angle_radians)
    horizontal_angle_radians -=math.pi
    vertical_angle_radians -=math.pi

    robotPose = (tagPose[0] - (cameraToTagDistance * math.sin(vertical_angle_radians * math.cos(vertical_angle_radians))),
                 tagPose[1] - (cameraToTagDistance * math.sin(horizontal_angle_radians)),
                 tagPose[2] - (cameraToTagDistance * math.cos(vertical_angle_radians) * math.cos(horizontal_angle_radians)))

    return robotPose

