
import math

depth = 0.2032
horizontal_angle_degrees = 27.2154731
vertical_angle_degrees = 27.2154731
mount_angle_radians = 0

horizontal_angle_radians = math.radians(horizontal_angle_degrees)
vertical_angle_radians = math.radians(vertical_angle_degrees)

xy_target_distance = math.cos(mount_angle_radians + vertical_angle_radians) * depth

tag_translation = {
    'x': math.cos(horizontal_angle_radians) * xy_target_distance,
    'y': -math.sin(horizontal_angle_radians) * xy_target_distance,
    'z': math.sin(mount_angle_radians + vertical_angle_radians) * depth,
    'x_angle': horizontal_angle_degrees,
    'y_angle': vertical_angle_degrees
}

print("x: {}".format(tag_translation['x']))
print("y: {}".format(tag_translation['y']))
print("z: {}".format(tag_translation['z']))
print("x angle: {}".format(tag_translation['x_angle']))
print("y angle: {}".format(tag_translation['y_angle']))
