import cv2
import depthai as dai
import math
import queue
import socket
import threading
import time



class FPSHandler:

    def __init__(self, cap=None, maxTicks = 100):
        """
        Args:
            cap (cv2.VideoCapture, Optional): handler to the video file object
            maxTicks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if maxTicks < 2:
            raise ValueError(f"Proviced maxTicks value must be 2 or higher (supplied: {maxTicks})")

        self._maxTicks = maxTicks

    def nextIter(self):
        """
        Marks the next iteration of the processing loop. Will use :obj:`time.sleep` method if initialized with video file
        object
        """
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frameDelay = 1.0 / self._framerate
            delay = (self._timestamp + frameDelay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._timestamp = time.monotonic()
        self._iterCnt += 1

    def tick(self, name):
        """
        Marks a point in time for specified name

        Args:
            name (str): Specifies timestamp name
        """
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tickFps(self, name):
        """
        Calculates the FPS based on specified name

        Args:
            name (str): Specifies timestamps' name

        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            timeDiff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / timeDiff if timeDiff != 0 else 0.0
        else:
            return 0.0

    def fps(self):
        """
        Calculates FPS value based on :func:`nextIter` calls, being the FPS of processing loop

        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        timeDiff = self._timestamp - self._start
        return self._iterCnt / timeDiff if timeDiff != 0 else 0.0

    def printStatus(self):
        """
        Prints total FPS for all names stored in :func:`tick` calls
        """
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tickFps(name):.1f}")


class OakIMU:
    def __init__(self, imuQueue, roll=0, pitch=0, yaw=0):
        self._imuQueue = imuQueue
        self.resetIMU(roll, pitch, yaw)

    def reset(self, roll=0, pitch=0, yaw=0):
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw

        self._lastTimestamp = 0
        self._lastRoll = 0
        self._lastPitch = 0
        self._lastYaw = 0

    def update(self):
        imuData = None
        try:
            imuData = self._imuQueue.get()
        except Exception as e:
            pass

        if imuData is not None:
            imuPackets = imuData.packets

            for imuPacket in imuPackets:

                gyroValues = imuPacket.gyroscope

                if gyroValues.accuracy != dai.IMUReport.Accuracy.HIGH:
                    continue

                gyroTs = gyroValues.timestamp.get().total_seconds()
                if self._lastTimestamp != gyroTs:
                    if self._lastTimestamp != 0:
                        self._roll += (self._lastRoll - gyroValues.x) / (gyroTs - self._lastTimestamp)
                        self._pitch += (self._lastPitch - gyroValues.y) / (gyroTs - self._lastTimestamp)
                        self._yaw += (self._lastYaw - gyroValues.z) / (gyroTs - self._lastTimestamp)

                    self._lastTimestamp = gyroTs
                    self._lastRoll = gyroValues.x
                    self._lastPitch = gyroValues.y
                    self._lastYaw = gyroValues.z
                    break

    def getImuAngles(self):
        return {
            'roll': math.degrees(self._roll),
            'pitch': math.degrees(self._pitch),
            'yaw': math.degrees(self._yaw)
        }


class AndroidWirelessIMU:
    def __init__(self, host=None, port=4201, position=True, roll=0, pitch=0, yaw=0):
        if host is None:
            hostname = socket.gethostname()
            host = socket.gethostbyname(hostname)

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # self._socket.settimeout(0.01)
        self._socket.bind((host, port))
        print("Listening to data coming into {}:{}".format(host, port))

        self._position = position
        self._lastTimestamp = 0
        self._roll = 0
        self._pitch = 0
        self._yaw = 0
        self._lastRoll = 0
        self._lastPitch = 0
        self._lastYaw = 0
        self._rollOffset = 0
        self._pitchOffset = 0
        self._yawOffset = 0

        self.reset(roll, pitch, yaw)

        t = threading.Thread(target=self.update)
        t.daemon = True
        t.start()

    def reset(self, roll=0, pitch=0, yaw=0):
        if self._position:
            self._rollOffset = self._lastRoll + roll
            self._pitchOffset = self._lastPitch + pitch
            self._yawOffset = self._lastYaw + yaw
        else:
            self._lastTimestamp = 0
            self._lastRoll = 0
            self._lastPitch = 0
            self._lastYaw = 0

    def update(self):
        while True:
            try:
                message, address = self._socket.recvfrom(8192)
            except Exception as e:
                # print("Socket Error: {}".format(e))
                return

            if message is not None:
                message = message.decode('utf-8')
                print(message)
                data = message.split(',')
                data = [float(i) for i in data]
                gyroData = data[6:]
                if len(gyroData) == 0:
                    return

                timestamp = time.monotonic()
                if self._lastTimestamp != 0:
                    if self._position:
                        self._roll = gyroData[2]
                        self._pitch = gyroData[1]
                        self._yaw = gyroData[0]
                    else:
                        self._roll += (self._lastRoll - gyroData[0]) / (timestamp - self._lastTimestamp)
                        self._pitch += (self._lastPitch - gyroData[1]) / (timestamp - self._lastTimestamp)
                        self._yaw += (self._lastYaw - gyroData[2]) / (timestamp - self._lastTimestamp)

                self._lastTimestamp = timestamp
                self._lastRoll = gyroData[2]
                self._lastPitch  = gyroData[1]
                self._lastYaw = gyroData[0]

    def readValues(self):
        if self._position:
            return {
                'roll': self._roll - self._rollOffset,
                'pitch': self._pitch - self._pitchOffset,
                'yaw': -(self._yaw - self._yawOffset)
            }
        else:
            return {
                'roll': math.degrees(self._roll - self._rollOffset),
                'pitch': math.degrees(self._pitch - self._pitchOffset),
                'yaw': -math.degrees(self._yaw - self._yawOffset)
            }
