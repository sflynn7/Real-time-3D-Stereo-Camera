# Real-Time 3D Stereo Camera System

ICS 691D Computational Imaging Final Project

A real-time stereo vision pipeline for 3D depth estimation and point cloud generation, built with the goal of deployment on the NVIDIA Jetson AGX Xavier.

---

## Overview

This project implements a real-time stereo camera system capable of capturing synchronized stereo images, computing depth maps, and visualizing 3D structure. Stereo vision systems allow machines to perceive depth and reconstruct their environment in 3D, with applications in autonomous robotics, AR/VR, smart infrastructure, digital twin pipelines, environmental mapping, smart agriculture, and underwater robotics.

---

## Hardware
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/4ec23fb6-5ecb-49e7-8bf1-59cffc21c17e" />

- **Stereo Camera** – USB stereo camera
- **Target Platform** – NVIDIA Jetson AGX Xavier (for real-time on-device processing)
- **Development Platform** – Gaming laptop (Ubuntu / Windows)

---

## Pipeline

### 1. Camera Calibration

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/91b101cf-e622-4828-b249-f14495c01ce5" />

- Captured multiple images of a calibration checkerboard from different angles and views.
- Loaded images into [calib.io](https://calib.io) to extract intrinsic and extrinsic calibration parameters.
- Used the calibration parameters in Python to **rectify** the stereo image pair, aligning both images onto the same plane.
- Constructed a **Q matrix** from the calibration parameters to generate a 3D point cloud of the checkerboard scene.
  - Result was described as "roughly correct" — minor inaccuracies were likely caused by motion blur during calibration or slight camera movement.
- Re-calibrated the camera on a separate day to evaluate consistency and improve accuracy.

### 2. Stereo Depth Estimation (CREStereo)

- Integrated the **CREStereo** model from the paper:
  > *"Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation"* — CVPR 2022
- Used CREStereo to compute **disparity maps** from rectified stereo image pairs.
- Modified the CREStereo codebase to:
  - Accept **real-time input** from the USB stereo camera
  - Automatically rectify incoming frames
  - Output a **live disparity video stream**

---

## Challenges
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/169a16b2-c774-4d78-99f0-ed7e4a7c6073" />

### NVIDIA Jetson AGX Xavier

The Jetson came with an existing account whose password was unknown, requiring a full OS reinstall via the JetPack SDK. This proved difficult due to OS compatibility issues:

- **JetPack SDK** requires Ubuntu 18.04 or 20.04; the development laptop ran Ubuntu 24.04.
- Attempted Docker on Windows — failed to cooperate.
- Attempted VirtualBox running Ubuntu 20.04 — USB passthrough to the Jetson was unreliable or non-functional.
- Jetson deployment was ultimately not completed due to these constraints.

### FoundationStereo

Attempted to integrate **FoundationStereo** (*"FoundationStereo: Zero-Shot Stereo Matching"*) as an upgraded depth estimation model, but encountered installation failures and compatibility issues on available hardware.

---

## Results

Despite being unable to run the full pipeline on the Jetson, the project successfully implemented:

- A **roughly accurate depth estimation pipeline** using stereo feature matching and disparity approximation.
- Real-time disparity video output on a gaming laptop (not fully real-time due to hardware limitations).

---

## Future Work

- Deploy the optimized pipeline on the **NVIDIA Jetson AGX Xavier** using hardware acceleration (CUDA, TensorRT).
- Explore **lightweight neural network architectures** for faster on-device disparity estimation.
- Improve calibration accuracy to reduce point cloud error.
- Re-attempt **FoundationStereo** integration for zero-shot stereo matching.

---

## References

- Jiankun Li et al. *"Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation."* CVPR 2022.
- *FoundationStereo: Zero-Shot Stereo Matching.*
- [calib.io](https://calib.io) — Camera calibration tooling
