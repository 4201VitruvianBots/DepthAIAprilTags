import depthai as dai
import math
import platform
import serial
import socket
import threading

from .AHRSProtocol import *


class navX:
    def __init__(self, port=None):
        if port is None:
            if platform.system() == 'Windows':
                port = 'COM4'
            elif platform.system() == 'Linux':
                port = '/dev/ttyACM0'
        self.s = serial.Serial(port)
        self.data = dict()
        self.offsets = dict()

        t = threading.Thread(target=self.update)
        t.daemon = True
        t.start()
        # self.resetYaw()
        # self.resetAll()
        # self.update()

    def get(self, keyValue):
        if keyValue not in self.data.keys():
            return None
        else:
            return self.data[keyValue]

    def resetYaw(self):
        # TODO: Fix this to be not hard-coded. Main issue is generating proper checksum from message
        resetMsg = bytearray(b''.join(b'\x00' for i in range(9)))
        resetMsg[0] = ord('!')
        resetMsg[1] = ord('#')
        resetMsg[2] = INTEGRATION_CONTROL_CMD_MESSAGE_LENGTH - 2
        resetMsg[3] = ord('I')
        resetMsg[4] = NAVX_INTEGRATION_CTL_RESET_YAW

        # resetMsg[-4] = checksumHex >> 4
        # resetMsg[-3] = checksumHex & 0x0F
        # resetMsg = '!#I{:01b}{:04b}{:02b}\r\n'.format(NAVX_INTEGRATION_CTL_RESET_YAW, 0, 0)
        # resetMsg = b'!#%bI%b' % ((INTEGRATION_CONTROL_CMD_MESSAGE_LENGTH - 2).to_bytes(1, byteorder='big'), NAVX_INTEGRATION_CTL_RESET_YAW.to_bytes(2, byteorder='big'))

        resetMsg = self.encoderTermination(resetMsg)
        self.s.write(resetMsg)

    def resetAll(self):
        resetMsg = bytearray(b''.join(b'\x00' for i in range(9)))
        resetMsg[0] = ord('!')
        resetMsg[1] = ord('#')
        resetMsg[2] = INTEGRATION_CONTROL_CMD_MESSAGE_LENGTH - 2
        resetMsg[3] = ord('I')
        resetMsg[4] = NAVX_INTEGRATION_CTL_RESET_YAW | NAVX_INTEGRATION_CTL_RESET_VEL_X | \
                      NAVX_INTEGRATION_CTL_RESET_VEL_Y | NAVX_INTEGRATION_CTL_RESET_VEL_Z | \
                      NAVX_INTEGRATION_CTL_RESET_DISP_X | NAVX_INTEGRATION_CTL_RESET_DISP_Y | \
                      NAVX_INTEGRATION_CTL_RESET_DISP_Z

        # resetMsg = '!#I{:01b}{:04b}{:02b}\r\n'.format(NAVX_INTEGRATION_CTL_RESET_YAW, 0, 0)
        # resetMsg = b'!#%bI%b' % ((INTEGRATION_CONTROL_CMD_MESSAGE_LENGTH - 2).to_bytes(1, byteorder='big'), NAVX_INTEGRATION_CTL_RESET_YAW.to_bytes(2, byteorder='big'))

        resetMsg = self.encoderTermination(resetMsg)
        self.s.write(resetMsg)

    def update(self):
        WRITE_TO_BUFFER = False
        READ_MSG = False

        msgBuffer = bytearray()
        readBuffer = bytearray()

        while True:
            serialData = self.s.read()

            if serialData == PACKET_START_CHAR:
                # print('Got New Packet')
                WRITE_TO_BUFFER = True

            if WRITE_TO_BUFFER:
                readBuffer += serialData

                if len(readBuffer) > 1:
                    if readBuffer[0] == ord(PACKET_START_CHAR) and readBuffer[1] != ord(BINARY_PACKET_INDICATOR_CHAR):
                        readBuffer = bytearray()
                        WRITE_TO_BUFFER = False
                    elif readBuffer[-1] == ord('\n') and readBuffer[-2] == ord('\r'):
                        WRITE_TO_BUFFER = False
                        msgBuffer = readBuffer
                        readBuffer = bytearray()
                        READ_MSG = True
                    elif len(readBuffer) > 100:
                        readBuffer = bytearray()
                        WRITE_TO_BUFFER = False

            if READ_MSG:
                msgLen = msgBuffer[2]
                msgID = chr(msgBuffer[3])
                bufferLen = len(msgBuffer)

                if msgID == 'p' and bufferLen == (msgLen + 2) == AHRSPOS_UPDATE_MESSAGE_LENGTH:
                    self.data['yaw'] = self.decodeProtocolSignedHundredthsFloat(
                        msgBuffer[AHRSPOS_UPDATE_YAW_VALUE_INDEX:AHRSPOS_UPDATE_YAW_VALUE_INDEX+2])
                    self.data['pitch'] = self.decodeProtocolSignedHundredthsFloat(
                        msgBuffer[AHRSPOS_UPDATE_PITCH_VALUE_INDEX:AHRSPOS_UPDATE_PITCH_VALUE_INDEX + 2])
                    self.data['roll'] = self.decodeProtocolSignedHundredthsFloat(
                        msgBuffer[AHRSPOS_UPDATE_ROLL_VALUE_INDEX:AHRSPOS_UPDATE_ROLL_VALUE_INDEX + 2])
                    self.data['compass_heading'] = self.decodeProtocolUnsignedHundredthsFloat(
                        msgBuffer[AHRSPOS_UPDATE_HEADING_VALUE_INDEX:AHRSPOS_UPDATE_HEADING_VALUE_INDEX + 2])
                    self.data['altitude'] = self.decodeProtocol1616Float(
                        msgBuffer[AHRSPOS_UPDATE_ALTITUDE_VALUE_INDEX:AHRSPOS_UPDATE_ALTITUDE_VALUE_INDEX + 4])
                    self.data['fused_heading'] = self.decodeProtocolUnsignedHundredthsFloat(
                        msgBuffer[AHRSPOS_UPDATE_FUSED_HEADING_VALUE_INDEX:AHRSPOS_UPDATE_FUSED_HEADING_VALUE_INDEX + 2])
                elif msgID == 't' and bufferLen == (msgLen + 2) == AHRSPOS_TS_UPDATE_MESSAGE_LENGTH:
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
                    self.data['timestamp'] = self.decodeBinaryUint32(
                        msgBuffer[AHRSPOS_TS_UPDATE_TIMESTAMP_INDEX:AHRSPOS_TS_UPDATE_TIMESTAMP_INDEX + 4])
                elif msgID == 'j' and bufferLen == (msgLen + 2) == INTEGRATION_CONTROL_RESP_MESSAGE_LENGTH:
                    print("Received reset message response")
                else:
                    # print("Error - Could not process message. msgId: '{}' msgLen: {} bufferLen: {}".format(msgID, msgLen, bufferLen))
                    pass

                READ_MSG = False

    def decodeBinaryUint16(self, buffer):
        return buffer[1] << 8 | buffer[0]

    def decodeProtocolSignedHundredthsFloat(self, buffer):
        result = self.decodeBinaryUint16(buffer)
        bits = '{:016b}'.format(result)

        if bits[0] == '1':
            return -1 * (65536.0 - result) / 100.0
        else:
            return result / 100.0

    def decodeProtocolUnsignedHundredthsFloat(self, buffer):
        result = self.decodeBinaryUint16(buffer)

        if result < 0:
            result += 65536

        return result / 100.0

    def decodeBinaryUint32(self, buffer):
        return buffer[3] << 24 | buffer[2] << 16 | buffer[1] << 8 | buffer[0]

    def decodeProtocol1616Float(self, buffer):
        result = self.decodeBinaryUint32(buffer)
        bits = '{:032b}'.format(result)

        # return self.bits_to_num(bits, 16, 16)
        if bits[0] == '1':
            return -1 * (4294967296.0 - result) / 65536.0
        else:
            return result / 65536.0

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

    def byteToHex(self, byte):
        return byte & 0xFF

    def encoderTermination(self, buffer):
        checksum = 0
        for i in buffer:
            if isinstance(i, int):
                checksum += i
            else:
                checksum += ord(i)

        hex = self.byteToHex(checksum)
        buffer += '{:02X}\r\n'.format(hex).encode()
        return buffer


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
