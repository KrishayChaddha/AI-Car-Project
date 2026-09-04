# 🚗 Autonomous AI Car Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ROS](https://img.shields.io/badge/ROS-Noetic%2FGalactic-orange.svg)](https://www.ros.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source autonomous vehicle navigation and perception system built using computer vision, machine learning, and embedded hardware.

## 🚀 Features

- **Real-Time Perception**: Object and lane detection powered by custom-trained PyTorch models and OpenCV.
- **Path Planning**: Dynamic obstacle avoidance and waypoint tracking using A* and behavior trees.
- **Hardware Integration**: Precise steering and throttle actuation via PID controllers on microcontroller hardware.
- **Telemetry & Logging**: Live system health monitoring and sensor data recording.

---

## 🛠️ System Architecture

```text
[ Camera / LiDAR ] ---> [ Perception Node (PyTorch/OpenCV) ]
                                    |
                                    v
[ Microcontroller ] <--- [ Control / PID ] <--- [ Path Planning (A*) ]
