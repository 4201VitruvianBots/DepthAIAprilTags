import cv2
import math
import socket
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
        try:
            imuData = self._imuQueue.get()
        except Exception as e:
            pass

        if imuData is not None:
            imuPackets = imuData.packets

            for imuPacket in imuPackets:
                gyroValues = imuPacket.gyroscope

                gyroTs = gyroValues.timestamp.get().total_seconds()

                if self._lastTimestamp != 0:
                    self._roll += gyroValues.x / (gyroTs - self._lastTimestamp)
                    self._pitch += gyroValues.y / (gyroTs - self._lastTimestamp)
                    self._yaw += gyroValues.z / (gyroTs - self._lastTimestamp)

                self._lastTimestamp = gyroTs
                self._lastRoll = gyroValues.x
                self._lastPitch = gyroValues.y
                self._lastYaw = gyroValues.z

    def getImuAngles(self):
        return {
            'roll': math.degrees(self._roll),
            'pitch': math.degrees(self._pitch),
            'yaw': math.degrees(self._yaw)
        }

class AndroidWirelessIMU:
    def __init__(self, host='localhost', port=4201, roll=0, pitch=0, yaw=0):
        
        self._socket = socket.socket(socket.AF_NET, socket.SOCK_DGRAM)
        
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self._socket.bind((host, port))
        
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
        message = None
        try:
            message, address = s.recvfrom(8192)
        except Exception as e:
            pass

        if message is not None:
            print(message)
            data = message.split(',')
            gyroTs = data[0]

            if self._lastTimestamp != 0:
                self._roll += data[1] / (gyroTs - self._lastTimestamp)
                self._pitch += data[2] / (gyroTs - self._lastTimestamp)
                self._yaw += data[3] / (gyroTs - self._lastTimestamp)

            self._lastTimestamp = gyroTs
            self._lastRoll = data[1]
            self._lastPitch = data[2]
            self._lastYaw = data[3]

    def getImuAngles(self):
        return {
            'roll': math.degrees(self._roll),
            'pitch': math.degrees(self._pitch),
            'yaw': math.degrees(self._yaw)
        }