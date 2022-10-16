
import navx
import numpy as np
import struct
import serial
import time

# gyro = navx.AHRS.create_i2c()
#
# while True:
#     try:
#         print("navX Yaw: {}".format(gyro.getYaw()))
#     except Exception as e:
#         print("Error: Could not read from navX")
#
#     time.sleep(0.001)

s = serial.Serial('COM4')
# s = serial.Serial('COM4', baudrate=9600, bytesize=8, timeout=2)
WRITE_TO_BUFFER = False
CHECK_MSG_TYPE = False
CHECK_END_OF_MSG = False
MSG_TYPE = ''
SEND_MSG = False
buffer = bytearray()

MSG_TYPES = [
    'y'.encode('utf-8'),
    'q'.encode('utf-8'),
    'g'.encode('utf-8'),
    'S'.encode('utf-8'),
    's'.encode('utf-8'),
]


def decodeBinaryUint32(buffer):
    return buffer[3] << 24 | buffer[2] << 16 | buffer[1] << 8 | buffer[0]


def decodeProtocol1616Float(buffer):
    result = decodeBinaryUint32(buffer)
    uint32Bytes = int.to_bytes(result, 4, 'little')
    # sign = bytes(uint32Bytes & b'80000000')
    return result

testBuffer = bytearray()
testBuffer.append(127)
testBuffer.append(255)
testBuffer.append(255)
testBuffer.append(255)

testValue =decodeProtocol1616Float(testBuffer)
print(testValue)
counter = 0

UP = '\033[1A'
CLEAR = '\x1b[2K'

while True:
    res = s.read()

    if res == '!'.encode('utf-8'):
        # print('Got New Packet')
        WRITE_TO_BUFFER = True

    if WRITE_TO_BUFFER:
        buffer += res
        if len(buffer) == 4:
            msgLen = buffer[2]
            msgID = chr(buffer[3])
            # print("\033[3FMsg Len: {}\tMsg ID: {}".format(msgLen, msgID))
        if len(buffer) == 8:
            yawValue = decodeProtocol1616Float(buffer[4:8])
            # print(len(buffer[4:8]))
            # print(buffer[4:8])
            print("\033[FYaw: {}\r".format(yawValue), end='')
        counter += 1

        # if buffer[-1] == '\n'.encode('utf-8') and buffer[-2] == '\r'.encode('utf-8'):

        # if buffer[-1] == '\n'.encode('utf-8'):
        if buffer[-1] == ord('\n') and buffer[-2] == ord('\r'):
            WRITE_TO_BUFFER = False
            output = buffer
            buffer = bytearray()
            SEND_MSG = True
            counter = 0

    if SEND_MSG:
        # print(len(output))
        # print(output)
        SEND_MSG = False
