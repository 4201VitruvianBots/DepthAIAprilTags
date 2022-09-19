
import depthai as dai

def create_spaitalCalculator_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a source - two mono (grayscale) cameras
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

    xoutDepth = pipeline.createXLinkOut()
    xoutSpatialData = pipeline.createXLinkOut()
    xinSpatialCalcConfig = pipeline.createXLinkIn()
    xoutRight = pipeline.createXLinkOut()

    depthStr = "depth"
    spatialStr = "spatialData"
    spatialCfgStr = "spatialCalcConfig"
    monoRightStr = "right"

    xoutDepth.setStreamName(depthStr)
    xoutSpatialData.setStreamName(spatialStr)
    xinSpatialCalcConfig.setStreamName(spatialCfgStr)
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
    stereo.setConfidenceThreshold(255)

    stereo.setLeftRightCheck(lrcheck)
    stereo.setSubpixel(subpixel)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
    stereo.depth.link(spatialLocationCalculator.inputDepth)

    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.6, 0.6)

    spatialLocationCalculator.setWaitForConfigInput(False)
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(topLeft, bottomRight)
    spatialLocationCalculator.initialConfig.addROI(config)
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    pipeline_info = {
        'resolution_x': monoRight.getResolutionWidth(),
        'resolution_y': monoRight.getResolutionHeight(),
        'depthQueue': depthStr,
        'spatialDataQueue': spatialStr,
        'spatialConfigQueue': spatialCfgStr,
        'monoRightQueue': monoRightStr
    }

    # OAK-D Lite Mono @480P
    camera_params = {
        # "fx": 0.00337,
        # "fY": 0.00337,
        # "cx": int(cam.getVideoWidth() / 2),
        # "cy": int(cam.getVideoHeight() / 2)
    }

    return pipeline, pipeline_info, camera_params