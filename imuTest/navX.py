from common import imu

navX = imu.navX()

while True:
    yaw = navX.get("yaw")
    timestamp = navX.get("timestamp")

    print("Timestamp: {} Yaw: {}".format(timestamp, yaw))
