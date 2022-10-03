
import depthai as dai

from common import utils

pipeline = dai.Pipeline()

imu = pipeline.createIMU()
# imu.enableFirmwareUpdate(True)
# sensorConfig = dai.IMUSensorConfig()
# sensorConfig.sensorId = dai.IMUSensor.GYROSCOPE_RAW
# sensorConfig.reportRate = 100
imu.enableIMUSensor([dai.IMUSensor.GYROSCOPE_RAW], 100)

imu.setBatchReportThreshold(5)
imu.setMaxBatchReports(10)

imuQueueStr = "imu"
xoutIMU = pipeline.createXLinkOut()
xoutIMU.setStreamName(imuQueueStr)
imu.out.link(xoutIMU.input)

with dai.Device(pipeline) as device:
    imuQueue = device.getOutputQueue(name=imuQueueStr, maxSize=50, blocking=False)

    gyro = utils.OakIMU(imuQueue)
    imuData = None
    while True:
        try:
            imuData = imuQueue.get()
        except Exception as e:
            print("Error Reading from IMU queue")

        if imuData is not None:
            # imuPackets = imuData.packets
            #
            imuF = "{:.06f}"
            # tsF  = "{:.03f}"
            #
            # for imuPacket in imuPackets:
            #     gyroValues = imuPacket.gyroscope
            #
            #     gyroTs = gyroValues.timestamp.get().total_seconds()
            #
            #     print(f"Gyroscope timestamp: {tsF.format(gyroTs)} ms")
            #     print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")
            gyro.update()

            angles = gyro.getImuAngles()
            print(f"Gyroscope [rad/s]: x: {imuF.format(angles['roll'])} y: {imuF.format(angles['pitch'])} z: {imuF.format(angles['yaw'])} ")


    # gyro.update()
        # angles = gyro.getImuAngles()
        # log.info("IMU - X: {}\tY: {}\tZ: {}".format(angles['roll'], angles['pitch'], angles['yaw']))