from common import imu

navX = imu.navX()

while True:
    yaw = navX.get("yaw")
    timestamp = navX.get("timestamp")
    fusedHeading = navX.get("fused_heading")

    # print("\033[1ATimestamp: {} Yaw: {}".format(timestamp, yaw), end='\r')
    print("Timestamp: {} Yaw: {}".format(timestamp, yaw))
    # print("Timestamp: {} Fused Heading: {}".format(timestamp, fusedHeading))
