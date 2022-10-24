import keyboard

from common import imu

navX = imu.navX('COM4')
keyboard.on_press_key(" ", lambda _: navX.resetYaw())
keyboard.on_press_key("a", lambda _: navX.resetAll())

navX.resetAll()
while True:
    yaw = navX.get("yaw")
    timestamp = navX.get("timestamp")
    fusedHeading = navX.get("fused_heading")

    # print("\033[1ATimestamp: {} Yaw: {}".format(timestamp, yaw), end='\r')
    print("Timestamp: {} Yaw: {}".format(timestamp, yaw))
    # print("Timestamp: {} Fused Heading: {}".format(timestamp, fusedHeading))
