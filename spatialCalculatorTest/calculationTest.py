import numpy as np
import spatialCalculator


class Tag():
    def __init__(self, id, center, z):
        self.tag_id = id
        self.center = center
        self.spatialData = {
            'z': z
        }


shape = (720, 1280)
tag_centers = {
    4: [515.68797569, 477.75609537],
    3: [156.92244473, 455.32185017],
    2: [789.71808266, 352.95153391],
    1: [279.92486877, 327.18331638]

}
tag_z = {
    4: 1.3850757810083514,
    3: 1.2724903496503495,
    2: 1.1597064840182647,
    1: 1.7456540808543097

}
camera_params = {'hfov': 128.0, 'vfov': 80.0, 'hfl': 312.14885668215135, 'vfl': 429.03129333391564}

x_pos = []
y_pos = []
z_pos = []

for i in range(4):
    id = i + 1
    tag = Tag(i, tag_centers[id], tag_z[id])

    robotPose, tagTranslation = spatialCalculator.estimate_robot_pose_from_apriltag(tag, tag.spatialData, camera_params,
                                                  shape)
    x_pos.append(robotPose['x'])
    y_pos.append(robotPose['y'])
    z_pos.append(robotPose['z'])

avg_x = sum(x_pos) / len(x_pos)
avg_y = sum(y_pos) / len(y_pos)
avg_z = sum(z_pos) / len(z_pos)

print("Estimated Pose X: {:.2f}\tY: {:.2f}\tZ: {:.2f}".format(avg_x, avg_y, avg_z))
print("Std dev X: {:.2f}\tY: {:.2f}\tZ: {:.2f}".format(np.std(x_pos), np.std(y_pos), np.std(z_pos)))
