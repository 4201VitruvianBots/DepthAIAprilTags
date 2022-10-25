
import numpy as np

TAG_SIZE_M = 0.15
PADDING_PERCENTAGE = 0.31

CAMERA_MOUNT_ANGLE = 0.0
CAMERA_MOUNT_HEIGHT = 0.0

CAMERA_IDS = {
    '14442C10218CCCD200': 'OAK-D',
    '18443010B1FA0C1300': 'OAK-D Lite',
    '18443010110CC71200': 'OAK-D Pro W'
}

CAMERA_PARAMS = {
    "OAK-D": {
        "rgb": {
            "hfov": 69.0,
            "vfov": 55.0
        },
        "mono": {
            "hfov": 72.0,
            "vfov": 50.0
        },
    },
    "OAK-D PoE": {
        "rgb": {
            "hfov": 69.0,
            "vfov": 55.0
        },
        "mono": {
            "hfov": 72.0,
            "vfov": 50.0
        },
    },
    "OAK-D Lite": {
        "rgb": {
            "hfov": 69.0,
            "vfov": 54.0
        },
        "mono": {
            "hfov": 73.0,
            "vfov": 58.0
        },
    },
    "OAK-D Pro W": {
    # "boardName": "OAK-D Pro W 120",
        "rgb": {
            "hfov": 95.0,
            "vfov": 70.0
        },
        "mono": {
            "hfov": 128.0,
            "vfov": 80.0,
            "rhfov": 97.0,
            "rvfov": 70.0
        },
    },
    "OAK-D Pro PoE FF": {
        "rgb": {
            "hfov": 69.0,
            "vfov": 55.0
        },
        "mono": {
            "hfov": 80.0,
            "vfov": 55.0,
        },
    },
    "OAK-D Pro W PoE 120": {
        "rgb": {
            "hfov": 95.0,
            "vfov": 70.0
        },
        "mono": {
            "hfov": 128.0,
            "vfov": 80.0,
            "rhfov": 97.0,
            "rvfov": 70.0
        },
    }
}

TAG_DICTIONARY = {
    0: {
        "name": "test",
        "pose": {
            'x': 0,
            'y': 0,
            'z': 0,
        }
    },
    1: {
        "name": "test",
        "pose": {
            'x': 2.84480569,
            'y': 2.77019304,
            'z': 1.86995174,
        }
    },
    2: {
        "name": "test",
        "pose": {
            'x': 3.165481331,
            'y': 1.33,
            'z': 1.714503429,
        }
    },
    3: {
        "name": "test",
        "pose": {
            'x': 2.536830074,
            'y': 2.77019304,
            'z': 1.363068326,
        }
    },
    4: {
        "name": "test",
        "pose": {
            'x': 3.165481331,
            'y': 1.9002413,
            'z': 1.460502921,
        }
    },
}

OPOINTS = np.array([
    -1, -1, 0,
     1, -1, 0,
     1,  1, 0,
    -1,  1, 0,
    -1, -1, -2 * 1,
     1, -1, -2 * 1,
     1,  1, -2 * 1,
    -1,  1, -2 * 1,
]).reshape(-1, 1, 3) * 0.5 * TAG_SIZE_M

EDGES = np.array([
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
