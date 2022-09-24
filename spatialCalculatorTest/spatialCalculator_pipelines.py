
import depthai as dai


def create_spaitalCalculator_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - two mono (grayscale) cameras
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()

    xoutDepth = pipeline.createXLinkOut()
    xoutRight = pipeline.createXLinkOut()

    depthStr = "depth"
    monoRightStr = "right"

    xoutDepth.setStreamName(depthStr)
    xoutRight.setStreamName(monoRightStr)


    # MonoCamera
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.setFps(120)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setFps(120)
    monoRight.out.link(xoutRight.input)

    outputDepth = True
    outputRectified = False
    lrcheck = False
    subpixel = False

    # StereoDepth
    stereo.setOutputDepth(outputDepth)
    stereo.setOutputRectified(outputRectified)
    stereo.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_RIGHT)
    stereo.setConfidenceThreshold(255)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    stereo.setLeftRightCheck(lrcheck)
    stereo.setSubpixel(subpixel)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(xoutDepth.input)

    pipeline_info = {
        'resolution_x': monoRight.getResolutionWidth(),
        'resolution_y': monoRight.getResolutionHeight(),
        'depthQueue': depthStr,
        'monoRightQueue': monoRightStr
    }

    return pipeline, pipeline_info