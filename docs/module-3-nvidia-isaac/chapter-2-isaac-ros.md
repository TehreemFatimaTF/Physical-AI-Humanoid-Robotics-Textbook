---
sidebar_label: 'Chapter 2: Isaac ROS and VSLAM'
title: 'Chapter 2: Isaac ROS and VSLAM'
description: 'Understanding Isaac ROS and Visual SLAM for humanoid robotics applications'
slug: '/module-3-nvidia-isaac/chapter-2-isaac-ros'
difficulty: 'advanced'
requiredHardware: ['computer', 'nvidia_gpu', 'camera']
recommendedHardware: ['jetson_orin', 'realsense_camera']
---

# Chapter 2: Isaac ROS and VSLAM

Isaac ROS is NVIDIA's collection of hardware-accelerated ROS 2 packages that leverage the power of NVIDIA GPUs for robotics perception and navigation. It includes optimized implementations of common robotics algorithms, particularly focused on visual perception and simultaneous localization and mapping (VSLAM). This chapter explores Isaac ROS packages and their application to humanoid robotics, with a focus on Visual SLAM capabilities.

## Introduction to Isaac ROS

Isaac ROS bridges the gap between NVIDIA's GPU-accelerated computing capabilities and the ROS 2 robotics framework. Key features include:

- **Hardware Acceleration**: GPU-accelerated processing for real-time robotics applications
- **Optimized Perception**: High-performance computer vision and sensor processing
- **Real-time Performance**: Low-latency processing for time-critical robotics tasks
- **Standard Interfaces**: ROS 2-compliant interfaces for easy integration
- **Modular Design**: Standalone packages that can be combined as needed

### Key Isaac ROS Packages

1. **Isaac ROS Image Pipeline**: Optimized image processing and rectification
2. **Isaac ROS Apriltag**: GPU-accelerated fiducial marker detection
3. **Isaac ROS Stereo Dense Depth**: Real-time depth estimation from stereo cameras
4. **Isaac ROS Visual Slam**: GPU-accelerated visual SLAM
5. **Isaac ROS NITROS**: Network Interface for Transforming and Routing of Sensors
6. **Isaac ROS DNN Inference**: Optimized deep learning inference pipelines

## Installing Isaac ROS

### Prerequisites

Before installing Isaac ROS, ensure your system meets the requirements:

- **Hardware**: NVIDIA GPU (Jetson Orin, RTX series, or similar)
- **CUDA**: CUDA 11.8 or later
- **OS**: Ubuntu 20.04 with ROS 2 Humble Hawksbill
- **Drivers**: NVIDIA GPU drivers installed (version 525 or later)

### Installation Methods

#### Method 1: Debian Package Installation

```bash
# Add NVIDIA package repository
wget https://nvidia.github.io/nvidia-isaac-ros/setup_isaac_ros.sh
bash setup_isaac_ros.sh

# Install Isaac ROS common packages
sudo apt update
sudo apt install nvidia-isaac-ros-common

# Install specific packages needed for humanoid robotics
sudo apt install nvidia-isaac-ros-visual-slam
sudo apt install nvidia-isaac-ros-stereo-image-pipeline
sudo apt install nvidia-isaac-ros-apriltag
```

#### Method 2: Docker Installation

```bash
# Pull Isaac ROS Docker image
docker pull nvcr.io/nvidia/isaac-ros:latest

# Run Isaac ROS container
docker run -it --gpus all --net=host --rm \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  nvcr.io/nvidia/isaac-ros:latest
```

#### Method 3: Build from Source

```bash
# Create ROS 2 workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Clone Isaac ROS repositories
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git src/isaac_ros_visual_slam
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_stereo_image_pipeline.git src/isaac_ros_stereo_image_pipeline
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git src/isaac_ros_apriltag

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build packages
colcon build --symlink-install --packages-select \
  isaac_ros_visual_slam \
  isaac_ros_stereo_image_pipeline \
  isaac_ros_apriltag
```

## Isaac ROS Image Pipeline

The Isaac ROS Image Pipeline provides optimized image processing capabilities essential for humanoid robot perception:

### Image Rectification

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacImageRectifier(Node):
    def __init__(self):
        super().__init__('isaac_image_rectifier')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        # Publishers
        self.rectified_pub = self.create_publisher(
            Image, '/camera/image_rectified', 10)

        # Internal state
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.map1 = None
        self.map2 = None

        self.get_logger().info('Isaac Image Rectifier initialized')

    def info_callback(self, msg):
        """Process camera calibration info"""
        if self.camera_matrix is not None:
            return  # Already initialized

        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

        # Precompute rectification maps for efficiency
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            (msg.width, msg.height),
            cv2.CV_32FC1
        )

    def image_callback(self, msg):
        """Process and rectify incoming images"""
        if self.map1 is None or self.map2 is None:
            return  # Wait for calibration info

        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Apply rectification using precomputed maps
        rectified_image = cv2.remap(
            cv_image,
            self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR
        )

        # Convert back to ROS image
        rectified_msg = self.bridge.cv2_to_imgmsg(rectified_image, encoding='bgr8')
        rectified_msg.header = msg.header

        # Publish rectified image
        self.rectified_pub.publish(rectified_msg)

def main(args=None):
    rclpy.init(args=args)
    rectifier = IsaacImageRectifier()

    try:
        rclpy.spin(rectifier)
    except KeyboardInterrupt:
        pass
    finally:
        rectifier.destroy_node()
        rclpy.shutdown()
```

### Isaac ROS Stereo Pipeline

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import numpy as np

class IsaacStereoProcessor(Node):
    def __init__(self):
        super().__init__('isaac_stereo_processor')

        # Subscribers for stereo pair
        self.left_sub = self.create_subscription(
            Image, '/camera/left/image_rect', self.left_callback, 10)
        self.right_sub = self.create_subscription(
            Image, '/camera/right/image_rect', self.right_callback, 10)

        # Publisher for disparity map
        self.disparity_pub = self.create_publisher(
            DisparityImage, '/disparity_map', 10)

        # Internal state
        self.bridge = CvBridge()
        self.left_image = None
        self.right_image = None

        # Stereo processing parameters
        self.block_size = 5
        self.min_disparity = 0
        self.num_disparities = 64
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.get_logger().info('Isaac Stereo Processor initialized')

    def left_callback(self, msg):
        """Process left camera image"""
        self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    def right_callback(self, msg):
        """Process right camera image"""
        self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    def process_stereo(self):
        """Process stereo pair to generate disparity map"""
        if self.left_image is None or self.right_image is None:
            return

        # Compute disparity using SGBM
        disparity = self.sgbm.compute(self.left_image, self.right_image).astype(np.float32) / 16.0

        # Create disparity message
        disp_msg = DisparityImage()
        disp_msg.header.stamp = self.get_clock().now().to_msg()
        disp_msg.header.frame_id = 'camera_link'
        disp_msg.image = self.bridge.cv2_to_imgmsg(disparity, encoding='32FC1')
        disp_msg.f = 320.0  # Focal length (example value)
        disp_msg.T = 0.075   # Baseline (example value)
        disp_msg.min_disparity = float(self.min_disparity)
        disp_msg.max_disparity = float(self.min_disparity + self.num_disparities)

        self.disparity_pub.publish(disp_msg)

def main(args=None):
    rclpy.init(args=args)
    stereo_processor = IsaacStereoProcessor()

    try:
        rclpy.spin(stereo_processor)
    except KeyboardInterrupt:
        pass
    finally:
        stereo_processor.destroy_node()
        rclpy.shutdown()
```

## Visual SLAM with Isaac ROS

Visual SLAM (Simultaneous Localization and Mapping) is crucial for humanoid robots to navigate unknown environments. Isaac ROS provides hardware-accelerated VSLAM capabilities:

### Isaac ROS Visual SLAM Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R

class IsaacVisualSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_pose', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Internal state
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.latest_image = None
        self.previous_features = None
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = R.from_quat([0, 0, 0, 1])

        # Feature detection parameters
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.get_logger().info('Isaac Visual SLAM initialized')

    def info_callback(self, msg):
        """Process camera calibration info"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def image_callback(self, msg):
        """Process incoming images for VSLAM"""
        # Convert to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Detect features in current image
        current_keypoints, current_descriptors = self.feature_detector.detectAndCompute(cv_image, None)

        if current_keypoints is not None and len(current_keypoints) > 10:
            if self.previous_features is not None:
                # Match features with previous frame
                matches = self.feature_matcher.knnMatch(
                    self.previous_features[1], current_descriptors, k=2)

                # Apply Lowe's ratio test for good matches
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= 10:
                    # Extract matched points
                    prev_pts = np.float32([self.previous_features[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    curr_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Estimate motion using essential matrix
                    E, mask = cv2.findEssentialMat(
                        curr_pts, prev_pts, self.camera_matrix,
                        method=cv2.RANSAC, prob=0.999, threshold=1.0)

                    if E is not None:
                        # Recover pose from essential matrix
                        _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, self.camera_matrix)

                        # Convert rotation matrix to quaternion
                        r = R.from_matrix(R)
                        quat = r.as_quat()

                        # Update position based on translation
                        self.current_position += t.flatten() * 0.1  # Scale factor

                        # Update orientation
                        self.current_orientation = self.current_orientation * r

                        # Publish odometry
                        self.publish_odometry(msg.header.stamp)

            # Store current features for next iteration
            self.previous_features = (current_keypoints, current_descriptors)

    def publish_odometry(self, stamp):
        """Publish odometry and pose messages"""
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = float(self.current_position[0])
        odom_msg.pose.pose.position.y = float(self.current_position[1])
        odom_msg.pose.pose.position.z = float(self.current_position[2])

        # Set orientation
        quat = self.current_orientation.as_quat()
        odom_msg.pose.pose.orientation.x = float(quat[0])
        odom_msg.pose.pose.orientation.y = float(quat[1])
        odom_msg.pose.pose.orientation.z = float(quat[2])
        odom_msg.pose.pose.orientation.w = float(quat[3])

        # Set velocities (estimated)
        odom_msg.twist.twist.linear.x = 0.1  # Placeholder
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom_msg)

        # Create and publish pose message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = float(self.current_position[0])
        t.transform.translation.y = float(self.current_position[1])
        t.transform.translation.z = float(self.current_position[2])

        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    slam_node = IsaacVisualSLAMNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()
```

## Isaac ROS AprilTag Detection

AprilTag detection is useful for humanoid robots for localization and object recognition:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import numpy as np
import cv2
from pupil_apriltags import Detector

class IsaacAprilTagDetector(Node):
    def __init__(self):
        super().__init__('isaac_april_tag_detector')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Publishers
        self.tag_poses_pub = self.create_publisher(
            PoseArray, '/april_tags/poses', 10)

        # Internal state
        self.bridge = CvBridge()

        # AprilTag detector
        self.detector = Detector(
            families='tag36h11',
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

        # Camera parameters (should match your camera calibration)
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self.get_logger().info('Isaac AprilTag Detector initialized')

    def image_callback(self, msg):
        """Process image for AprilTag detection"""
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to grayscale for AprilTag detection
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        tags = self.detector.detect(
            gray_image,
            estimate_tag_pose=True,
            camera_params=[self.camera_matrix[0,0], self.camera_matrix[1,1],
                          self.camera_matrix[0,2], self.camera_matrix[1,2]],
            tag_size=0.16  # Tag size in meters (adjust as needed)
        )

        # Create PoseArray message
        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.header.frame_id = 'camera_link'

        for tag in tags:
            # Create pose for the detected tag
            pose = Pose()

            # Position from tag pose estimation
            pose.position.x = float(tag.pose_t[0])
            pose.position.y = float(tag.pose_t[1])
            pose.position.z = float(tag.pose_t[2])

            # Orientation from tag pose estimation
            rotation_matrix = tag.pose_R
            # Convert rotation matrix to quaternion
            qw = np.sqrt(1 + rotation_matrix[0,0] + rotation_matrix[1,1] + rotation_matrix[2,2]) / 2
            qx = (rotation_matrix[2,1] - rotation_matrix[1,2]) / (4 * qw)
            qy = (rotation_matrix[0,2] - rotation_matrix[2,0]) / (4 * qw)
            qz = (rotation_matrix[1,0] - rotation_matrix[0,1]) / (4 * qw)

            pose.orientation.x = float(qx)
            pose.orientation.y = float(qy)
            pose.orientation.z = float(qz)
            pose.orientation.w = float(qw)

            pose_array.poses.append(pose)

        # Publish detected tag poses
        self.tag_poses_pub.publish(pose_array)

        # Optional: Draw detected tags on image and publish visualization
        self.draw_tags_on_image(cv_image, tags)

    def draw_tags_on_image(self, image, tags):
        """Draw detected tags on image for visualization"""
        for tag in tags:
            # Draw tag outline
            for idx in range(len(tag.corners)):
                pt1 = tuple(tag.corners[idx][0].astype(int))
                pt2 = tuple(tag.corners[(idx + 1) % len(tag.corners)][0].astype(int))
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)

            # Draw tag ID
            tag_center = tuple(tag.center.astype(int))
            cv2.putText(image, str(tag.tag_id), tag_center,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main(args=None):
    rclpy.init(args=args)
    detector = IsaacAprilTagDetector()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()
```

## Isaac ROS NITROS (Network Interface for Transforming and Routing of Sensors)

NITROS optimizes data transport and transformation in the perception pipeline:

```python
# Example NITROS configuration for optimized perception pipeline
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class IsaacNITROSExample(Node):
    def __init__(self):
        super().__init__('isaac_nitros_example')

        # Configure QoS for optimized transport
        qos_profile = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Subscribers with optimized QoS
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.optimized_image_callback, qos_profile)

        # Publishers
        self.processed_image_pub = self.create_publisher(
            Image, '/processed_image', qos_profile)

        self.bridge = CvBridge()

        self.get_logger().info('Isaac NITROS Example initialized')

    def optimized_image_callback(self, msg):
        """Process image with optimized pipeline"""
        # Convert image efficiently
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Apply optimized processing (example: edge detection)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Convert back to ROS image
        result_msg = self.bridge.cv2_to_imgmsg(edges, encoding='mono8')
        result_msg.header = msg.header

        # Publish processed image
        self.processed_image_pub.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    nitros_example = IsaacNITROSExample()

    try:
        rclpy.spin(nitros_example)
    except KeyboardInterrupt:
        pass
    finally:
        nitros_example.destroy_node()
        rclpy.shutdown()
```

## GPU-Accelerated Deep Learning Inference

Isaac ROS includes optimized DNN inference capabilities:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

class IsaacDNNInference(Node):
    def __init__(self):
        super().__init__('isaac_dnn_inference')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Publishers
        self.result_pub = self.create_publisher(
            String, '/dnn_result', 10)

        # Internal state
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model (example: MobileNetV2)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # ImageNet class labels (simplified)
        self.imagenet_classes = {
            0: 'tench', 1: 'goldfish', 2: 'great_white_shark',
            # ... (simplified for example)
        }

        self.get_logger().info('Isaac DNN Inference initialized')

    def image_callback(self, msg):
        """Process image with DNN inference"""
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess image
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                # Get top prediction
                top_prob, top_catid = torch.topk(probabilities, 1)
                predicted_class = self.imagenet_classes.get(top_catid.item(), f"class_{top_catid.item()}")
                confidence = float(top_prob.item())

            # Create result string
            result_str = f"Predicted: {predicted_class}, Confidence: {confidence:.2f}"

            # Publish result
            result_msg = String()
            result_msg.data = result_str
            self.result_pub.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'DNN inference error: {e}')

def main(args=None):
    rclpy.init(args=args)
    dnn_inference = IsaacDNNInference()

    try:
        rclpy.spin(dnn_inference)
    except KeyboardInterrupt:
        pass
    finally:
        dnn_inference.destroy_node()
        rclpy.shutdown()
```

## Integration with Humanoid Robot Control

### Perception-Action Integration

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import MarkerArray
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidPerceptionAction(Node):
    def __init__(self):
        super().__init__('humanoid_perception_action')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/visual_pose', self.pose_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)

        # Internal state
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.joint_positions = np.zeros(28)  # Example: 28 DOF humanoid
        self.target_position = np.array([1.0, 1.0])  # Example target
        self.avoid_obstacles = True

        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        self.get_logger().info('Humanoid Perception-Action Integration initialized')

    def image_callback(self, msg):
        """Process image for obstacle detection and navigation"""
        # This would integrate with Isaac ROS perception pipeline
        # For example, using depth data to detect obstacles
        pass

    def pose_callback(self, msg):
        """Update robot pose from VSLAM"""
        self.current_pose[0] = msg.pose.position.x
        self.current_pose[1] = msg.pose.position.y

        # Extract orientation from quaternion
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz')
        self.current_pose[2] = euler[2]  # Yaw angle

    def joint_state_callback(self, msg):
        """Update joint positions"""
        self.joint_positions = np.array(msg.position)

    def control_loop(self):
        """Main perception-action control loop"""
        # Calculate distance to target
        current_pos = self.current_pose[:2]
        distance_to_target = np.linalg.norm(self.target_position - current_pos)

        # Navigation control
        cmd_vel = Twist()

        if distance_to_target > 0.1:  # 10cm tolerance
            # Calculate desired direction
            direction = self.target_position - current_pos
            distance = np.linalg.norm(direction)
            direction_normalized = direction / distance if distance > 0 else direction

            # Calculate angle to target
            target_angle = np.arctan2(direction_normalized[1], direction_normalized[0])
            angle_diff = target_angle - self.current_pose[2]

            # Normalize angle difference
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # Set velocity commands
            if abs(angle_diff) > 0.1:  # Turn to face target
                cmd_vel.angular.z = np.clip(angle_diff * 0.5, -0.5, 0.5)
            else:  # Move forward toward target
                cmd_vel.linear.x = min(0.5, distance * 0.5)

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

        # Joint control for balance and locomotion
        joint_commands = self.generate_locomotion_commands()

        # Publish joint commands
        joint_cmd_msg = Float64MultiArray()
        joint_cmd_msg.data = joint_commands.tolist()
        self.joint_cmd_pub.publish(joint_cmd_msg)

    def generate_locomotion_commands(self):
        """Generate joint commands for humanoid locomotion"""
        # This would integrate with Isaac ROS perception to generate
        # appropriate walking patterns based on terrain and obstacles
        commands = self.joint_positions.copy()  # Start with current positions

        # Example: Simple walking gait (simplified)
        time_step = self.get_clock().now().nanoseconds / 1e9
        gait_freq = 1.0  # Hz

        # Apply periodic adjustments for walking
        for i in range(len(commands)):
            if i % 2 == 0:  # Alternate joints for gait
                commands[i] += 0.1 * np.sin(2 * np.pi * gait_freq * time_step)

        return commands

def main(args=None):
    rclpy.init(args=args)
    perception_action = HumanoidPerceptionAction()

    try:
        rclpy.spin(perception_action)
    except KeyboardInterrupt:
        pass
    finally:
        perception_action.destroy_node()
        rclpy.shutdown()
```

## Performance Optimization

### GPU Memory Management

```python
# GPU memory optimization for Isaac ROS
import torch
import gc

def optimize_gpu_memory():
    """Optimize GPU memory usage for Isaac ROS applications"""

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()

def setup_tensorrt_optimization():
    """Setup TensorRT optimization for deep learning inference"""

    # This would involve converting models to TensorRT format
    # for optimal inference performance on NVIDIA hardware
    pass
```

## Troubleshooting and Best Practices

### Common Issues and Solutions

1. **GPU Memory Exhaustion**:
   - Reduce batch sizes in DNN inference
   - Use lower resolution images when possible
   - Implement proper memory management

2. **Latency Issues**:
   - Optimize data transport with NITROS
   - Use appropriate QoS settings
   - Consider processing at lower frequency for non-critical tasks

3. **Calibration Problems**:
   - Ensure proper camera calibration
   - Verify coordinate frame transformations
   - Check timing synchronization

### Best Practices

1. **Modular Design**: Keep perception and action components modular
2. **Real-time Constraints**: Consider timing requirements for humanoid control
3. **Safety First**: Implement safety checks in perception-action loops
4. **Validation**: Test perception algorithms in simulation before real deployment
5. **Fallback Strategies**: Have backup plans when perception fails

## Summary

In this chapter, you learned:
- How to install and configure Isaac ROS for humanoid robotics
- Key Isaac ROS packages and their applications
- Visual SLAM implementation with GPU acceleration
- AprilTag detection for localization and object recognition
- GPU-accelerated deep learning inference
- Integration of perception with humanoid robot control
- Performance optimization techniques
- Troubleshooting strategies for Isaac ROS systems

Isaac ROS provides powerful GPU-accelerated capabilities that are essential for real-time perception in humanoid robotics applications, enabling complex visual processing and navigation tasks that would be impossible on CPU alone.

---
**Continue to [Chapter 3: Nav2 Path Planning](/docs/module-3-nvidia-isaac/chapter-3-nav2-path-planning)**