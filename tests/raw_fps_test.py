import cv2
import depthai as dai
import logging
import numpy as np
import queue
import threading

from common import utils
from spatialCalculatorTest import spatialCalculator_pipelines


class FPS_Test:
    def __init__(self):
        self.log = logging.getLogger(__name__)

        self.pipeline, self.pipeline_info = spatialCalculator_pipelines.create_stereoDepth_pipeline()

        self.pipeline.setXLinkChunkSize(0)
        self.frame_queue = queue.Queue()
        self.fps = utils.FPSHandler()

        THREADDING = 1

        if THREADDING:
            self.readThread = threading.Thread(target=self.read)
            self.writeThread = threading.Thread(target=self.write)

            self.readThread.start()
            self.writeThread.start()
        else:
            self.process()
        # self.write()

    def read(self):
        with dai.Device(self.pipeline) as device:
            self.log.info("USB SPEED: {}".format(device.getUsbSpeed()))
            if device.getUsbSpeed() not in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
                self.log.warning("WARNING: USB Speed is set to USB 2.0")

            depthQueue = device.getOutputQueue(name=self.pipeline_info["depthQueue"], maxSize=1, blocking=False)
            qRight = device.getOutputQueue(name=self.pipeline_info["monoRightQueue"], maxSize=1, blocking=False)

            while True:
                try:
                    inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
                    inRight = qRight.tryGet()

                    depthFrame = inDepth.getFrame()
                    self.frame_queue.put(inRight)
                except Exception as e:
                    self.log.warning("WARMING: FRAME SKIP - READ")

    def write(self):
        diffs = np.array([])
        while True:
            try:
                while not self.frame_queue.empty():
                    inRight = self.frame_queue.get()
                    self.fps.nextIter()
                    latencyMs = (dai.Clock.now() - inRight.getTimestamp()).total_seconds() * 1000
                    diffs = np.append(diffs, latencyMs)

                    print("FPS: {:.2f}\tLatency: {:.2f} ms\t Avg. Latency: {:.2f} ms\t Latency Std: {:.2f}".format(self.fps.fps(),
                                                                                                                   latencyMs,
                                                                                                                   np.average(
                                                                                                                       diffs),
                                                                                                                   np.std(
                                                                                                                       diffs)))

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
            except Exception as e:
                print("WARMING: FRAME SKIP - WRITE")

        print("Exiting write")

    def process(self):
        diffs = np.array([])

        with dai.Device(self.pipeline) as device:
            self.log.info("USB SPEED: {}".format(device.getUsbSpeed()))
            if device.getUsbSpeed() not in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
                self.log.warning("WARNING: USB Speed is set to USB 2.0")

            depthQueue = device.getOutputQueue(name=self.pipeline_info["depthQueue"], maxSize=1, blocking=False)
            qRight = device.getOutputQueue(name=self.pipeline_info["monoRightQueue"], maxSize=1, blocking=False)

            while True:
                try:
                    inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
                    inRight = qRight.tryGet()

                    depthFrame = inDepth.getFrame()

                    self.fps.nextIter()
                    latencyMs = (dai.Clock.now() - inRight.getTimestamp()).total_seconds() * 1000
                    diffs = np.append(diffs, latencyMs)

                    print("FPS: {:.2f}\tLatency: {:.2f} ms\t Avg. Latency: {:.2f} ms\t Latency Std: {:.2f}".format(
                        self.fps.fps(),
                        latencyMs,
                        np.average(
                            diffs),
                        np.std(
                            diffs)))

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                except Exception as e:
                    self.log.warning("WARMING: FRAME SKIP")


if __name__ == '__main__':
    FPS_Test()
