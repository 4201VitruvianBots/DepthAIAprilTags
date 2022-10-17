import serial

from common.AHRSProtocol import *

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

def clear_line(n=1):
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for i in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def decodeBinaryUint32(buffer):
    return buffer[3] << 24 | buffer[2] << 16 | buffer[1] << 8 | buffer[0]


def decodeProtocol1616Float(buffer):
    # b = buffer.reverse()
    # b=bytearray(len(buffer))
    # for i, v in enumerate(buffer):
    #     b[i] = 0xFF & ~v
    result = decodeBinaryUint32(buffer)
    bits = '{:032b}'.format(result)
    # bits = bytearrayToBits(buffer)
    return bits_to_num(bits, 16, 16)


def bits_to_num(s, exp, sig, signed=True):
    mantissa = 0
    if signed:
        neg = int(s[0], 2)
        if neg:
            exponent = 2**exp - int(s[0:exp], 2)
        else:
            exponent = int(s[1:exp], 2)
        exponent = (((-1)**neg) * exponent)
    else:
        exponent = int(s[:exp], 2)
    if sig != 0:
        mantissa = int(s[exp:exp+sig], 2)*2**(-sig)

    return exponent + mantissa

# def bits_to_num(s, exp, sig):
#     neg = int(s[0],2)
#     if(int(s[1:1+exp],2)!=0):
#         exponent = int(s[1:1+exp],2)-int('1'*(exp-1), 2)
#         mantissa = int(s[1+exp:],2)*2**(-sig)+1
#     else: #subnormal
#         exponent = 1-int('1'*(exp-1), 2)
#         mantissa = int(s[1+exp:],2)*2**(-sig)
#     return ((-1)**neg)*(2**exponent)*mantissa

yawValue = 0
headingValue = 0
fusedHeadingValue = 0
while True:
    res = s.read()

    if res == PACKET_START_CHAR:
        # print('Got New Packet')
        WRITE_TO_BUFFER = True

    if WRITE_TO_BUFFER:
        buffer += res

        if buffer[-1] == ord('\n') and buffer[-2] == ord('\r'):
            WRITE_TO_BUFFER = False
            output = buffer
            buffer = bytearray()
            SEND_MSG = True
            counter = 0

    if SEND_MSG:
        msgLen = output[2]
        msgID = chr(output[3])
        bufferLen = len(output)

        if msgID == 't' and bufferLen == msgLen + 2:
            yawValue = decodeProtocol1616Float(output[AHRSPOS_TS_UPDATE_YAW_VALUE_INDEX:AHRSPOS_TS_UPDATE_YAW_VALUE_INDEX+4])

            # headingValue = decodeProtocol1616Float(output[16:20])
            #
            # fusedHeadingValue = decodeProtocol1616Float(output[24:28])

            # print("Yaw: {:.8f}\tHeading: {:.8f}\tFused Heading: {:.8f}".format(yawValue, headingValue, fusedHeadingValue), end='\r')
        else:
            # print("Could not process message - MsgID: '{}' MsgLen: {} BufferLen: {}".format(msgID, msgLen, bufferLen))
            pass

        SEND_MSG = False
