from common import utils

imu = utils.AndroidWirelessIMU()

fFormat = "{:.02f}"

while True:
    print("Test")
    imu.update()

    values = imu.readValues()

    print(f"X: {fFormat.format(values['roll'])}\tY: {fFormat.format(values['pitch'])}\tZ: {fFormat.format(values['yaw'])}")
