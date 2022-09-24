import cv2
import depthai as dai
import numpy as np

from common import utils
from spatialCalculatorTest import spatialCalculator_pipelines


def main():

    pipeline, pipeline_info = spatialCalculator_pipelines.create_spaitalCalculator_pipeline()

    pipeline.setXLinkChunkSize(0)

    fps = utils.FPSHandler()

    with dai.Device(pipeline) as device:
        print("USB SPEED: {}".format(device.getUsbSpeed()))
        if device.getUsbSpeed() not in [dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS]:
            print("WARNING: USB Speed is set to USB 2.0")

        depthQueue = device.getOutputQueue(name=pipeline_info["depthQueue"], maxSize=4, blocking=False)
        qRight = device.getOutputQueue(name=pipeline_info["monoRightQueue"], maxSize=4, blocking=False)

        diffs = np.array([])
        while True:
            inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
            inRight = qRight.tryGet()

            depthFrame = inDepth.getFrame()

            fps.nextIter()
            latencyMs = (dai.Clock.now() - inRight.getTimestamp()).total_seconds() * 1000
            diffs = np.append(diffs, latencyMs)

            print("FPS: {:.2f}\tLatency: {:.2f} ms\t Avg. Latency: {:.2f} ms\t Latency Std: {:.2f}".format(fps.fps(), latencyMs, np.average(diffs), np.std(diffs)))

            key = cv2.waitKey(1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    main()
