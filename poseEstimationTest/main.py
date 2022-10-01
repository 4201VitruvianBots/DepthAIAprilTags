import cv2
import depthai as dai
import numpy as np

from pupil_apriltags import Detector

from common import utils

pipeline = dai.Pipeline()
monoRight = pipeline.createMonoCamera()

xoutRight = pipeline.createXLinkOut()

monoRightStr = "right"
xoutRight.setStreamName(monoRightStr)

monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setFps(120)
monoRight.out.link(xoutRight.input)


with dai.Device(pipeline) as device:
    qRight = device.getOutputQueue(name=monoRightStr, maxSize=4, blocking=False)

    fps = utils.FPSHandler()

    detector = Detector(families='tag36h11',
                        nthreads=3,
                        quad_decimate=4.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    camera_params = (
        570.4937744140625,
        569.6653442382813,
        643.565185546875,
        403.140625
    )

    opoints = np.array([
        -1, -1, 0,
         1, -1, 0,
         1,  1, 0,
        -1,  1, 0,
        -1, -1, -2*1,
         1, -1, -2*1,
         1,  1, -2*1,
        -1,  1, -2*1,
    ]).reshape(-1, 1, 3) * 0.5*0.2

    edges = np.array([
        0, 1,
        1, 2,
        2, 3,
        3, 0,
        0, 4,
        1, 5,
        2, 6,
        3, 7,
        4, 5,
        5, 6,
        6, 7,
        7, 4
    ]).reshape(-1, 2)

    while True:
        inRight = qRight.tryGet()

        if inRight is not None:
            frameRight = inRight.getCvFrame()  # get mono right frame

            tags = detector.detect(frameRight, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.2)

            for tag in tags:
                points = tag.corners.astype(np.int32)
                # Shift points since this is a snapshot
                cv2.polylines(frameRight, [points], True, (120, 120, 120), 3)
                textX = min(points[:, 0])
                textY = min(points[:, 1]) + 20

                K = np.array([camera_params[0], 0, camera_params[2], 0, camera_params[1], camera_params[3], 0, 0, 1]).reshape(3, 3)

                dcoeffs = np.zeros(5)

                ipoints, _ = cv2.projectPoints(opoints, tag.pose_R, tag.pose_t, K, dcoeffs)

                ipoints = np.round(ipoints).astype(int)

                ipoints = [tuple(pt) for pt in ipoints.reshape(-1, 2)]

                for i, j in edges:
                    cv2.line(frameRight, ipoints[i], ipoints[j], (0, 255, 0), 1, 16)

                cv2.putText(frameRight, "x: {:.2f}".format(tag.pose_t[0][0]),
                            (textX, textY + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                            (255, 255, 255))
                cv2.putText(frameRight, "y: {:.2f}".format(tag.pose_t[1][0]),
                            (textX, textY + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                            (255, 255, 255))
                cv2.putText(frameRight, "z: {:.2f}".format(tag.pose_t[2][0]),
                            (textX, textY + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                            (255, 255, 255))

            fps.nextIter()
            cv2.circle(frameRight, (int(frameRight.shape[1] / 2), int(frameRight.shape[0] / 2)), 1, (255, 255, 255), 1)
            cv2.putText(frameRight, "{:.2f}".format(fps.fps()), (0, 24), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
            cv2.imshow(monoRightStr, frameRight)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break
