import os
import cv2
import numpy as np
import json
from readParameters import read_calib_parameters

# === Load calibration data ===
success, Ks, Ds, rvecs, tvecs = read_calib_parameters()
if not success:
    raise RuntimeError("Could not read calibration data.")

# Left and right camera matrices and distortion
K1, K2 = Ks[0], Ks[1]
D1, D2 = Ds[0], Ds[1]

# Convert rotation vectors to matrices
R1_mat, _ = cv2.Rodrigues(rvecs[0])
R2_mat, _ = cv2.Rodrigues(rvecs[1])

# Relative rotation and translation
R = R2_mat @ R1_mat.T
T = tvecs[1] - tvecs[0]

# Image size from JSON
image_size = (1920, 1080)

# Stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)

np.save("Q_matrix.npy", Q)
print("Saved Q matrix to Q_matrix.npy.")

# Compute rectification maps
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

# Load calibration_project.json
with open("calibration_project.json", "r") as f:
    pairs = json.load(f)["fileInfo"]

# Output folder
os.makedirs("rectified/left", exist_ok=True)
os.makedirs("rectified/right", exist_ok=True)

# Process each stereo pair
for i, pair in enumerate(pairs):
    left_path = pair[0]["filePath"].replace("/home/cirp-laptop-1/code/", "")
    right_path = pair[1]["filePath"].replace("/home/cirp-laptop-1/code/", "")

    left_path = os.path.join("left_parts", os.path.basename(left_path))
    right_path = os.path.join("right_parts", os.path.basename(right_path))


    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"Skipping missing pair {i}: {left_path}, {right_path}")
        continue

    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    left_rect = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)

    # Save rectified images
    left_name = os.path.basename(left_path).replace('.jpg', '')
    right_name = os.path.basename(right_path).replace('.jpg', '')

    cv2.imwrite(f"rectified/left/left_{left_name}_rect.jpg", left_rect)
    cv2.imwrite(f"rectified/right/right_{right_name}_rect.jpg", right_rect)

    print(f"Rectified pair {i} saved.")

print("\n All done. Check the 'rectified' folder.")
