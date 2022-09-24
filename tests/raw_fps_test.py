import cv2
import depthai as dai

from common import utils
from spatialCalculatorTest import spatialCalculator_pipelines


def main():

    pipeline, pipeline_info = spatialCalculator_pipelines.create_spaitalCalculator_pipeline()

    fps = utils.FPSHandler()

    with dai.Device(pipeline) as device:
        depthQueue = device.getOutputQueue(name=pipeline_info["depthQueue"], maxSize=4, blocking=False)
        qRight = device.getOutputQueue(name=pipeline_info["monoRightQueue"], maxSize=4, blocking=False)

        while True:
            inDepth = depthQueue.get()  # blocking call, will wait until a new data has arrived
            inRight = qRight.tryGet()

            depthFrame = inDepth.getFrame()

            fps.nextIter()

            print("FPS TEST: {:.2f}".format(fps.fps()))

            key = cv2.waitKey(1)
            if key == ord('q'):
                break


if __name__ == '__main__':
    main()
