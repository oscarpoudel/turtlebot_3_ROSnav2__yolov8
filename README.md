# ROS 2 Object Detection with YOLOv8 and Qdrant 

This repository contains a **ROS 2** package for **object detection** using **YOLOv8** and **Qdrant** to store and retrieve detected object embeddings. The system processes camera feed data, detects objects, and associates detections with the robot's position.

## Features
- **Real-time object detection** using **YOLOv8**.
- **Integration with ROS 2 topics** (`/camera/image_raw`, `/detection_image`, `/amcl_pose`).
- **Stores object embeddings** in a **Qdrant** vector database for retrieval.
- **Uses ResNet50 embeddings** for storing image features.
- **Optimized with MultiThreadedExecutor** for better performance.

---

## Installation & Setup

### 1. Clone the Repository Inside ROS 2 Workspace

Ensure you have a **ROS 2 workspace** set up. Then, navigate to your workspace and clone the repository inside `src`:

```bash
cd ~/ros_ws/src
git clone https://github.com/oscarpoudel/turtlebot_3_ROSnav2__yolov8.git
```


### 2. Install Dependencies

Before building the package, install the required dependencies:

```bash
sudo apt update && sudo apt install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-nav-msgs
```

Then, install the necessary Python libraries:

```bash
pip install \
    torch torchvision torchaudio \
    ultralytics \
    qdrant-client \
    opencv-python \
    numpy
```

### 3. Build the Package

After installing dependencies, navigate to the ROS 2 workspace and build the package:

```bash
cd ~/ros_ws
colcon build 
```


### 4. Sourcing the Workspace

Once the package is built, source the ROS 2 environment:

```bash
source ~/ros_ws/install/setup.bash
```

To automatically source this on every terminal session, add it to your `~/.bashrc`:

```bash
echo "source ~/ros_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Running the Object Detection Node

Start the ROS 2 node:

```bash
ros2 run your_package_name object_detection_node
```

---

## ROS 2 Topics Used

- **Subscribed Topics:**
  - `/camera/image_raw` → Receives live camera feed.
  - `/amcl_pose` → Retrieves robot's current position.

- **Published Topics:**
  - `/detection_image` → Publishes images with detected object bounding boxes.

---

## Qdrant Integration

This package stores object detections in **Qdrant**, a vector database used for efficient retrieval of object embeddings. It uses **ResNet50** for feature extraction before storing object detection embeddings.

To run Qdrant locally:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

---

