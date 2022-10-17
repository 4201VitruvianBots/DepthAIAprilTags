import depthai as dai
import math
import serial
import socket
import threading

from .AHRSProtocol import *


class navX:
    def __init__(self):
        self.s = serial.Serial('COM4')
        self.data = dict()

        t = threading.Thread(target=self.update)
        t.daemon = True
        t.start()

    def update(self):
        WRITE_TO_BUFFER = False
        READ_MSG = False

        readBuffer = bytearray()
        msgBuffer = bytearray()

        while True:
            serialData = self.s.read()

            if serialData == PACKET_START_CHAR:
                # print('Got New Packet')
                WRITE_TO_BUFFER = True

            if WRITE_TO_BUFFER:
                readBuffer += serialData

                if readBuffer[-1] == ord('\n') and readBuffer[-2] == ord('\r'):
                    WRITE_TO_BUFFER = False
                    msgBuffer = readBuffer
                    readBuffer = bytearray()
                    READ_MSG = True

            if READ_MSG:
                msgLen = msgBuffer[2]
                msgID = chr(msgBuffer[3])
                bufferLen = len(msgBuffer)

                if msgID == 't' and bufferLen == msgLen + 2:
                    self.data['yaw'] = self.decodeProtocol1616Float(
                        msgBuffer[AHRSPOS_TS_UPDATE_YAW_VALUE_INDEX:AHRSPOS_TS_UPDATE_YAW_VALUE_INDEX+4])
                    self.data['pitch'] = self.decodeProtocol1616Float(
                        msgBuffer[AHRSPOS_TS_UPDATE_PITCH_VALUE_INDEX:AHRSPOS_TS_UPDATE_PITCH_VALUE_INDEX + 4])
                    self.data['roll'] = self.decodeProtocol1616Float(
                        msgBuffer[AHRSPOS_TS_UPDATE_ROLL_VALUE_INDEX:AHRSPOS_TS_UPDATE_ROLL_VALUE_INDEX + 4])
                    self.data['compass_heading'] = self.decodeProtocol1616Float(
                        msgBuffer[AHRSPOS_TS_UPDATE_HEADING_VALUE_INDEX:AHRSPOS_TS_UPDATE_HEADING_VALUE_INDEX + 4])
                    self.data['altitude'] = self.decodeProtocol1616Float(
                        msgBuffer[AHRSPOS_TS_UPDATE_ALTITUDE_VALUE_INDEX:AHRSPOS_TS_UPDATE_ALTITUDE_VALUE_INDEX + 4])
                    self.data['fused_heading'] = self.decodeProtocol1616Float(
                        msgBuffer[AHRSPOS_TS_UPDATE_FUSED_HEADING_VALUE_INDEX:AHRSPOS_TS_UPDATE_FUSED_HEADING_VALUE_INDEX + 4])
                    self.data['timestamp'] = self.decodeProtocol1616Float(
                        msgBuffer[AHRSPOS_TS_UPDATE_TIMESTAMP_INDEX:AHRSPOS_TS_UPDATE_TIMESTAMP_INDEX + 4])

    def decodeBinaryUint32(self, buffer):
        return buffer[3] << 24 | buffer[2] << 16 | buffer[1] << 8 | buffer[0]

    def decodeProtocol1616Float(self, buffer):
        # b = buffer.reverse()
        # b=bytearray(len(buffer))
        # for i, v in enumerate(buffer):
        #     b[i] = 0xFF & ~v
        result = self.decodeBinaryUint32(buffer)
        bits = '{:032b}'.format(result)
        # bits = bytearrayToBits(buffer)
        return self.bits_to_num(bits, 16, 16)

    def bits_to_num(self, s, exp, sig, signed=True):
        mantissa = 0
        if signed:
            neg = int(s[0], 2)
            if neg:
                exponent = 2 ** exp - int(s[0:exp], 2)
            else:
                exponent = int(s[1:exp], 2)
            exponent = (((-1) ** neg) * exponent)
        else:
            exponent = int(s[:exp], 2)
        if sig != 0:
            mantissa = int(s[exp:exp + sig], 2) * 2 ** (-sig)

        return exponent + mantissa

    def get(self, keyValue):
        if keyValue not in self.data.keys():
            pass
        else:
            return self.data[keyValue]


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
