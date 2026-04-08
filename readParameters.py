import json
import numpy as np
import cv2  # OpenCV for Python

def read_calib_parameters():
    file_path = "parameters.json"
    K = []
    k = []
    cam_rvecs = []
    cam_tvecs = []

    try:
        with open(file_path, 'r') as file:
            json_struct = json.load(file)
    except Exception:
        return False, K, k, cam_rvecs, cam_tvecs

    try:
        cameras = json_struct["Calibration"]["cameras"]
        assert cameras[0]["model"]["polymorphic_name"] == "libCalib::CameraModelOpenCV"
    except (KeyError, AssertionError):
        return False, K, k, cam_rvecs, cam_tvecs

    n_cameras = len(cameras)
    if n_cameras < 1:
        return False, K, k, cam_rvecs, cam_tvecs

    for cam in cameras:
        intrinsics = cam["model"]["ptr_wrapper"]["data"]["parameters"]

        f = intrinsics["f"]["val"]
        ar = intrinsics["ar"]["val"]
        cx = intrinsics["cx"]["val"]
        cy = intrinsics["cy"]["val"]
        k1 = intrinsics["k1"]["val"]
        k2 = intrinsics["k2"]["val"]
        k3 = intrinsics["k3"]["val"]
        k4 = intrinsics["k4"]["val"]
        k5 = intrinsics["k5"]["val"]
        k6 = intrinsics["k6"]["val"]
        p1 = intrinsics["p1"]["val"]
        p2 = intrinsics["p2"]["val"]
        s1 = intrinsics["s1"]["val"]
        s2 = intrinsics["s2"]["val"]
        s3 = intrinsics["s3"]["val"]
        s4 = intrinsics["s4"]["val"]
        tauX = intrinsics["tauX"]["val"]
        tauY = intrinsics["tauY"]["val"]

        # Camera intrinsic matrix
        K.append(np.array([[f, 0.0, cx],
                           [0.0, f * ar, cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64))

        # Distortion coefficients (14 params)
        k.append(np.array([k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tauX, tauY], dtype=np.float64))

        transform = cam["transform"]
        rot = transform["rotation"]
        cam_rvecs.append(np.array([rot["rx"], rot["ry"], rot["rz"]], dtype=np.float64))

        t = transform["translation"]
        cam_tvecs.append(np.array([t["x"], t["y"], t["z"]], dtype=np.float64))

    return True, K, k, cam_rvecs, cam_tvecs

