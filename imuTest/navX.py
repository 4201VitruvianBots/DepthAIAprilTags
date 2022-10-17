from common import imu

navX = imu.navX()

while True:
    yaw = navX.get("yaw")

    print("Yaw: {}".format(yaw))
