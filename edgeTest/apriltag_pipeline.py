
import depthai as dai


def create_pipeline_rgb():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    cam = pipeline.create(dai.node.ColorCamera)
    manip = pipeline.create(dai.node.ImageManip)
    edgeDetector = pipeline.create(dai.node.EdgeDetector)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutEdge = pipeline.create(dai.node.XLinkOut)
    xinEdgeCfg = pipeline.create(dai.node.XLinkIn)

    videoStr = "rgb"
    edgeStr = "rgb edge"
    edgeCfgStr = "edge cfg"

    xoutRgb.setStreamName(videoStr)
    xoutEdge.setStreamName(edgeStr)
    xinEdgeCfg.setStreamName(edgeCfgStr)

    # Properties
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # cam.setPreviewSize(1920, 1080)
    # camRgb.setFps(60) # Breaks edge detection

    # manip.initialConfig.setResize(cam.getVideoWidth(), cam.getVideoHeight())
    manip.initialConfig.setResize(960, 540)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
    manip.setMaxOutputFrameSize(cam.getVideoWidth() * cam.getVideoHeight())

    edgeDetector.setMaxOutputFrameSize(cam.getVideoWidth() * cam.getVideoHeight())

    # Linking
    cam.video.link(manip.inputImage)
    manip.out.link(edgeDetector.inputImage)
    manip.out.link(xoutRgb.input)

    edgeDetector.outputImage.link(xoutEdge.input)

    xinEdgeCfg.out.link(edgeDetector.inputConfig)

    pipeline_info = {
        'resolution_x': cam.getVideoWidth(),
        'resolution_y': cam.getVideoHeight(),
        'videoQueue': videoStr,
        'edgeQueue': edgeStr,
        'edgeCfgQueue': edgeCfgStr
    }

    # OAK-D Lite @ 1080P
    camera_params = {
        "fx": 0.00337,
        "fY": 0.00337,
        "cx": int(cam.getVideoWidth() / 2),
        "cy": int(cam.getVideoHeight() / 2)
    }

    return pipeline, pipeline_info, camera_params


def create_pipeline_mono():
    pipeline = dai.Pipeline()

    # Define sources and outputs
    cam = pipeline.create(dai.node.MonoCamera)

    edgeDetector = pipeline.create(dai.node.EdgeDetector)

    xoutMono = pipeline.create(dai.node.XLinkOut)
    xoutEdge = pipeline.create(dai.node.XLinkOut)
    xinEdgeCfg = pipeline.create(dai.node.XLinkIn)

    videoStr = "mono"
    edgeStr = "mono left"
    edgeCfgStr = "edge cfg"

    xoutMono.setStreamName(videoStr)
    xoutEdge.setStreamName(edgeStr)
    xinEdgeCfg.setStreamName(edgeCfgStr)

    # Properties
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    cam.setFps(120)

    # Linking
    cam.out.link(xoutMono.input)
    cam.out.link(edgeDetector.inputImage)

    edgeDetector.outputImage.link(xoutEdge.input)

    xinEdgeCfg.out.link(edgeDetector.inputConfig)

    pipeline_info = {
        'resolution_x': cam.getResolutionWidth(),
        'resolution_y': cam.getResolutionHeight(),
        'videoQueue': videoStr,
        'edgeQueue': edgeStr,
        'edgeCfgQueue': edgeCfgStr
    }

    # # OAK-D Lite @ 480P
    # camera_params = {
    #     "fx": 0.00337,
    #     "fY": 0.00337,
    #     "cx": int(cam.getVideoWidth() / 2),
    #     "cy": int(cam.getVideoHeight() / 2)
    # }
    # OAK-D Lite @ 800P
    camera_params = {
        "fx": 0.00337 * 1000,
        "fY": 0.00337 * 1000,
        "cx": int(cam.getResolutionWidth() / 2),
        "cy": int(cam.getResolutionHeight() / 2)
    }

    return pipeline, pipeline_info, camera_params
