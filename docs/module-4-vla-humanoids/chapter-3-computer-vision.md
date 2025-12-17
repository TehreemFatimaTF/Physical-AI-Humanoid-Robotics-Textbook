---
sidebar_label: 'Chapter 3: Computer Vision for Object Manipulation'
title: 'Chapter 3: Computer Vision for Object Manipulation'
description: 'Understanding computer vision techniques for object manipulation in humanoid robotics'
slug: '/module-4-vla-humanoids/chapter-3-computer-vision'
difficulty: 'advanced'
requiredHardware: ['computer', 'camera', 'nvidia_gpu']
recommendedHardware: ['realsense_camera', 'jetson_orin', 'cuda_enabled_device']
---

# Chapter 3: Computer Vision for Object Manipulation

Computer vision is fundamental to humanoid robotics, enabling robots to perceive, understand, and interact with objects in their environment. For object manipulation tasks, computer vision systems must accurately detect, recognize, segment, and track objects to enable precise manipulation. This chapter explores advanced computer vision techniques specifically tailored for object manipulation in humanoid robotics.

## Introduction to Computer Vision for Manipulation

Object manipulation requires computer vision systems to provide:

- **Object Detection**: Locating objects in the robot's field of view
- **Pose Estimation**: Determining the 6D pose (position and orientation) of objects
- **Grasp Point Detection**: Identifying optimal points for robotic grasping
- **Semantic Segmentation**: Understanding object parts and relationships
- **Depth Perception**: Measuring distances for precise manipulation
- **Real-time Processing**: Fast inference for dynamic manipulation tasks

### Why Computer Vision for Manipulation?

Computer vision enables humanoid robots to:
- Identify and locate objects for manipulation
- Understand object properties (size, shape, weight)
- Plan safe and effective grasps
- Adapt to novel objects and environments
- Perform complex manipulation tasks autonomously

## 3D Vision and Depth Perception

### RGB-D Cameras and Depth Sensing

RGB-D cameras provide both color and depth information, crucial for manipulation tasks:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class RGBDProcessor(Node):
    def __init__(self):
        super().__init__('rgbd_processor')

        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.info_callback, 10)

        # Internal state
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.camera_matrix = None
        self.depth_scale = 0.001  # Default for most depth cameras

        # Processing parameters
        self.min_depth = 0.1  # meters
        self.max_depth = 3.0  # meters

        self.get_logger().info('RGB-D Processor initialized')

    def info_callback(self, msg):
        """Process camera calibration info"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.width = msg.width
            self.height = msg.height

    def color_callback(self, msg):
        """Process color image"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing color image: {e}')

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            # Convert to float32 and scale to meters
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.depth_image = self.depth_image * self.depth_scale
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def get_3d_point(self, u, v):
        """Convert 2D pixel coordinates to 3D world coordinates"""
        if self.depth_image is None or self.camera_matrix is None:
            return None

        # Get depth at pixel (u, v)
        depth = self.depth_image[v, u]

        if depth == 0 or np.isnan(depth) or np.isinf(depth):
            return None

        # Convert to 3D coordinates using camera matrix
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return np.array([x, y, z])

    def get_object_3d_position(self, mask):
        """Get 3D position of object using segmentation mask"""
        if self.depth_image is None:
            return None

        # Find center of mass in the mask
        y_coords, x_coords = np.where(mask)

        if len(x_coords) == 0:
            return None

        # Calculate centroid
        u = int(np.mean(x_coords))
        v = int(np.mean(y_coords))

        # Get 3D position at centroid
        return self.get_3d_point(u, v)

    def get_object_bounding_box_3d(self, mask):
        """Get 3D bounding box of object"""
        if self.depth_image is None:
            return None

        y_coords, x_coords = np.where(mask)

        if len(x_coords) == 0:
            return None

        # Get 2D bounding box
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Get 3D coordinates at corners
        corners_3d = []
        for u, v in [(x_min, y_min), (x_max, y_min),
                      (x_max, y_max), (x_min, y_max)]:
            point_3d = self.get_3d_point(u, v)
            if point_3d is not None:
                corners_3d.append(point_3d)

        if len(corners_3d) == 0:
            return None

        corners_3d = np.array(corners_3d)
        min_coords = np.min(corners_3d, axis=0)
        max_coords = np.max(corners_3d, axis=0)

        return {
            'center': np.mean(corners_3d, axis=0),
            'min': min_coords,
            'max': max_coords,
            'size': max_coords - min_coords
        }

def main(args=None):
    rclpy.init(args=args)
    processor = RGBDProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()
```

### Point Cloud Processing

Point clouds provide rich 3D information for manipulation tasks:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('point_cloud_processor')

        # Subscriber
        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera/depth/color/points', self.pointcloud_callback, 10)

        # Publisher for processed point clouds
        self.processed_pc_pub = self.create_publisher(
            PointCloud2, '/processed_pointcloud', 10)

        # Internal state
        self.voxel_size = 0.01  # 1cm voxel size
        self.table_height = 0.8  # Assumed table height in meters

        self.get_logger().info('Point Cloud Processor initialized')

    def pointcloud_callback(self, msg):
        """Process incoming point cloud"""
        try:
            # Convert ROS PointCloud2 to numpy array
            points = []
            for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([point[0], point[1], point[2]])

            if len(points) == 0:
                return

            points = np.array(points)

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Preprocess point cloud
            processed_pcd = self.preprocess_pointcloud(pcd)

            # Extract objects from table
            objects_pcd = self.extract_objects_from_table(processed_pcd)

            # Segment individual objects
            object_clusters = self.segment_objects(objects_pcd)

            # Process each object
            for i, obj_pcd in enumerate(object_clusters):
                self.process_object(obj_pcd, i)

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')

    def preprocess_pointcloud(self, pcd):
        """Preprocess point cloud for manipulation"""
        # Remove statistical outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_filtered = pcd.select_by_index(ind)

        # Downsample using voxel grid
        pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size=self.voxel_size)

        # Estimate normals
        pcd_downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        return pcd_downsampled

    def extract_objects_from_table(self, pcd):
        """Extract objects above the table surface"""
        points = np.asarray(pcd.points)

        # Segment table plane using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        # Extract objects (points not on the table)
        object_indices = np.setdiff1d(np.arange(len(points)), inliers)
        objects_pcd = pcd.select_by_index(object_indices)

        return objects_pcd

    def segment_objects(self, pcd):
        """Segment individual objects using DBSCAN clustering"""
        # Perform DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))

        # Extract individual clusters
        unique_labels = np.unique(labels)
        clusters = []

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            # Get points belonging to this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_pcd = pcd.select_by_index(cluster_indices)

            # Filter small clusters (likely noise)
            if len(cluster_pcd.points) > 50:  # Minimum 50 points
                clusters.append(cluster_pcd)

        return clusters

    def process_object(self, pcd, object_id):
        """Process individual object for manipulation"""
        # Calculate bounding box
        aabb = pcd.get_axis_aligned_bounding_box()
        obb = pcd.get_oriented_bounding_box()

        # Get object properties
        center = np.array(obb.center)
        extent = np.array(obb.extent)  # [width, length, height]

        # Estimate object orientation
        rotation_matrix = np.array(obb.R)
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz')

        # Calculate grasp points
        grasp_points = self.calculate_grasp_points(pcd, center, extent)

        self.get_logger().info(
            f'Object {object_id}: '
            f'Center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), '
            f'Size=({extent[0]:.2f}x{extent[1]:.2f}x{extent[2]:.2f})m'
        )

    def calculate_grasp_points(self, pcd, center, extent):
        """Calculate potential grasp points for the object"""
        points = np.asarray(pcd.points)

        # Find points at the top of the object (highest Z values)
        top_points = points[points[:, 2] > center[2] + extent[2]/4]

        if len(top_points) == 0:
            return []

        # Find points near the center horizontally
        center_points = top_points[
            (np.abs(top_points[:, 0] - center[0]) < extent[0]/2) &
            (np.abs(top_points[:, 1] - center[1]) < extent[1]/2)
        ]

        # If no center points, use all top points
        if len(center_points) == 0:
            center_points = top_points

        # Calculate potential grasp points
        grasp_points = []
        if len(center_points) > 0:
            grasp_center = np.mean(center_points, axis=0)
            grasp_points.append(grasp_center)

        return grasp_points

def main(args=None):
    rclpy.init(args=args)
    processor = PointCloudProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()
```

## Object Detection and Recognition

### Deep Learning-based Object Detection

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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import json

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)

        # Publishers
        self.detection_pub = self.create_publisher(
            String, '/detected_objects', 10)

        # Internal state
        self.bridge = CvBridge()

        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Load pre-trained object detection model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # COCO dataset class names
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.3

        self.get_logger().info('Object Detection Node initialized')

    def image_callback(self, msg):
        """Process incoming images for object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run object detection
            detections = self.detect_objects(cv_image)

            # Filter detections by confidence
            filtered_detections = [
                det for det in detections
                if det['confidence'] > self.confidence_threshold
            ]

            # Publish detections
            self.publish_detections(filtered_detections)

            # Optionally visualize detections
            if self.get_logger().level <= 10:  # Debug level
                self.visualize_detections(cv_image, filtered_detections)

        except Exception as e:
            self.get_logger().error(f'Error in image processing: {e}')

    def detect_objects(self, image):
        """Run object detection on image"""
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(image_rgb).to(self.device)

        # Add batch dimension
        image_batch = image_tensor.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = self.model(image_batch)

        # Process outputs
        detections = []
        for i in range(len(outputs[0]['boxes'])):
            box = outputs[0]['boxes'][i].cpu().numpy()
            score = outputs[0]['scores'][i].cpu().item()
            label = outputs[0]['labels'][i].cpu().item()

            if score > self.confidence_threshold:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                detection = {
                    'class_id': label,
                    'class_name': self.coco_names[label] if label < len(self.coco_names) else f'unknown_{label}',
                    'confidence': score,
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                }

                detections.append(detection)

        return detections

    def publish_detections(self, detections):
        """Publish object detections"""
        detection_msg = String()
        detection_msg.data = json.dumps({
            'timestamp': self.get_clock().now().nanoseconds,
            'detections': detections
        })
        self.detection_pub.publish(detection_msg)

    def visualize_detections(self, image, detections):
        """Visualize detections on image (for debugging)"""
        vis_image = image.copy()

        for det in detections:
            x, y, w, h = det['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(vis_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish or save visualization as needed
        # For now, just log that visualization was done
        self.get_logger().debug(f'Visualized {len(detections)} detections')

def main(args=None):
    rclpy.init(args=args)
    detector = ObjectDetectionNode()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()
```

### Custom Object Detection for Manipulation

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
from torch import nn
import json

class ManipulationObjectDetector(nn.Module):
    def __init__(self, num_classes=10):
        super(ManipulationObjectDetector, self).__init__()

        # Feature extractor (simplified ResNet-like)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4)  # [dx, dy, dw, dh]
        )

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)

        class_logits = self.classifier(features)
        bbox_deltas = self.bbox_regressor(features)

        return class_logits, bbox_deltas

class CustomObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('custom_object_detection_node')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)

        # Publishers
        self.detection_pub = self.create_publisher(
            String, '/custom_detected_objects', 10)

        # Internal state
        self.bridge = CvBridge()

        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize custom model
        self.model = ManipulationObjectDetector(num_classes=10)
        self.model.to(self.device)
        self.model.eval()

        # Object class names for manipulation
        self.class_names = [
            'mug', 'bottle', 'cup', 'book', 'box',
            'phone', 'remote', 'toy', 'tool', 'unknown'
        ]

        # Detection parameters
        self.confidence_threshold = 0.6

        self.get_logger().info('Custom Object Detection Node initialized')

    def image_callback(self, msg):
        """Process incoming images for custom object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image
            image_tensor = self.preprocess_image(cv_image)

            # Run detection
            class_logits, bbox_deltas = self.model(image_tensor)

            # Process results
            detections = self.process_detections(
                class_logits, bbox_deltas, cv_image.shape[1], cv_image.shape[0])

            # Filter by confidence
            filtered_detections = [
                det for det in detections
                if det['confidence'] > self.confidence_threshold
            ]

            # Publish results
            self.publish_detections(filtered_detections)

        except Exception as e:
            self.get_logger().error(f'Error in custom detection: {e}')

    def preprocess_image(self, image):
        """Preprocess image for the model"""
        # Resize image to model input size (224x224)
        image_resized = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        return image_tensor

    def process_detections(self, class_logits, bbox_deltas, img_width, img_height):
        """Process model outputs to get detections"""
        # Apply softmax to get probabilities
        class_probs = torch.softmax(class_logits, dim=1)
        max_probs, class_ids = torch.max(class_probs, dim=1)

        # Convert to numpy for further processing
        class_ids = class_ids.cpu().numpy()
        confidences = max_probs.cpu().numpy()
        bbox_deltas = bbox_deltas.cpu().numpy()

        detections = []
        for i in range(len(class_ids)):
            if confidences[i] > self.confidence_threshold:
                # Convert bounding box deltas to absolute coordinates
                # This is a simplified version - in practice, you'd use anchor boxes
                center_x = img_width / 2
                center_y = img_height / 2
                width = img_width / 4
                height = img_height / 4

                # Apply deltas (simplified)
                dx, dy, dw, dh = bbox_deltas[i]
                center_x += dx * img_width
                center_y += dy * img_height
                width *= (1 + dw)
                height *= (1 + dh)

                # Convert to x1, y1, w, h format
                x1 = max(0, center_x - width/2)
                y1 = max(0, center_y - height/2)
                w = min(img_width - x1, width)
                h = min(img_height - y1, height)

                detection = {
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else 'unknown',
                    'confidence': float(confidences[i]),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'center': [float(x1 + w/2), float(y1 + h/2)]
                }

                detections.append(detection)

        return detections

    def publish_detections(self, detections):
        """Publish object detections"""
        detection_msg = String()
        detection_msg.data = json.dumps({
            'timestamp': self.get_clock().now().nanoseconds,
            'detections': detections
        })
        self.detection_pub.publish(detection_msg)

def main(args=None):
    rclpy.init(args=args)
    detector = CustomObjectDetectionNode()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()
```

## Grasp Detection and Planning

### Grasp Point Detection

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import json

class GraspDetectionNetwork(nn.Module):
    def __init__(self):
        super(GraspDetectionNetwork, self).__init__()

        # Encoder for RGB image
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Encoder for depth image
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 128*2 from RGB and depth
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Grasp detection head
        self.grasp_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 5, kernel_size=1),  # 5 channels: confidence + 4 grasp params
        )

    def forward(self, rgb, depth):
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)

        # Concatenate features
        fused_features = torch.cat([rgb_features, depth_features], dim=1)
        fused_features = self.fusion(fused_features)

        grasp_output = self.grasp_head(fused_features)
        return grasp_output

class GraspDetectionNode(Node):
    def __init__(self):
        super().__init__('grasp_detection_node')

        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)

        # Publishers
        self.grasp_pub = self.create_publisher(
            String, '/grasp_candidates', 10)

        # Internal state
        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.camera_matrix = None

        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize grasp detection model
        self.model = GraspDetectionNetwork()
        self.model.to(self.device)
        self.model.eval()

        # Grasp detection parameters
        self.grasp_threshold = 0.7
        self.min_grasp_distance = 10  # pixels between grasp candidates

        self.get_logger().info('Grasp Detection Node initialized')

    def color_callback(self, msg):
        """Process color image"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing color image: {e}')

    def depth_callback(self, msg):
        """Process depth image and detect grasps"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

            # Only process if we have both color and depth
            if self.color_image is not None and self.depth_image is not None:
                grasp_candidates = self.detect_grasps(self.color_image, self.depth_image)
                self.publish_grasps(grasp_candidates)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def detect_grasps(self, color_image, depth_image):
        """Detect grasp candidates in the image"""
        # Preprocess images
        rgb_tensor = self.preprocess_rgb(color_image).to(self.device)
        depth_tensor = self.preprocess_depth(depth_image).to(self.device)

        # Run grasp detection
        with torch.no_grad():
            grasp_output = self.model(rgb_tensor, depth_tensor)

        # Process grasp output
        grasp_candidates = self.process_grasp_output(grasp_output, color_image.shape)

        return grasp_candidates

    def preprocess_rgb(self, image):
        """Preprocess RGB image for the model"""
        # Resize to model input size (e.g., 224x224)
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return image_tensor

    def preprocess_depth(self, depth_image):
        """Preprocess depth image for the model"""
        # Resize to model input size
        depth_resized = cv2.resize(depth_image, (224, 224))

        # Normalize depth values
        depth_normalized = (depth_resized - np.min(depth_resized)) / (np.max(depth_resized) - np.min(depth_resized) + 1e-6)

        # Convert to tensor
        depth_tensor = torch.from_numpy(depth_normalized).float().unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions
        return depth_tensor

    def process_grasp_output(self, grasp_output, image_shape):
        """Process the model output to get grasp candidates"""
        # Output shape: [batch, 5, H, W] where 5 = [confidence, angle, width, height, depth]
        output = grasp_output[0].cpu().numpy()  # Remove batch dimension

        confidence_map = output[0]  # Grasp confidence
        angle_map = output[1]       # Grasp angle
        width_map = output[2]       # Grasp width
        height_map = output[3]      # Grasp height
        depth_map = output[4]       # Grasp depth

        # Find high-confidence grasp candidates
        high_conf_indices = np.where(confidence_map > self.grasp_threshold)

        grasp_candidates = []
        for y, x in zip(high_conf_indices[0], high_conf_indices[1]):
            if confidence_map[y, x] > self.grasp_threshold:
                # Convert to original image coordinates
                orig_h, orig_w = image_shape[0], image_shape[1]
                orig_x = int(x * orig_w / 224)
                orig_y = int(y * orig_h / 224)

                # Get grasp parameters
                angle = angle_map[y, x]
                width = width_map[y, x]
                height = height_map[y, x]
                depth = depth_map[y, x]

                grasp_candidate = {
                    'x': orig_x,
                    'y': orig_y,
                    'angle': float(angle),
                    'width': float(width),
                    'height': float(height),
                    'depth': float(depth),
                    'confidence': float(confidence_map[y, x])
                }

                grasp_candidates.append(grasp_candidate)

        # Filter grasp candidates to avoid clustering
        filtered_candidates = self.filter_grasp_candidates(grasp_candidates)

        return filtered_candidates

    def filter_grasp_candidates(self, candidates):
        """Filter grasp candidates to avoid clustering"""
        if len(candidates) <= 1:
            return candidates

        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)

        filtered = []
        for candidate in candidates:
            # Check if this candidate is too close to existing ones
            too_close = False
            for existing in filtered:
                dist = np.sqrt((candidate['x'] - existing['x'])**2 + (candidate['y'] - existing['y'])**2)
                if dist < self.min_grasp_distance:
                    too_close = True
                    break

            if not too_close:
                filtered.append(candidate)

        return filtered

    def publish_grasps(self, grasp_candidates):
        """Publish grasp candidates"""
        grasp_msg = String()
        grasp_msg.data = json.dumps({
            'timestamp': self.get_clock().now().nanoseconds,
            'grasps': grasp_candidates
        })
        self.grasp_pub.publish(grasp_msg)

        self.get_logger().info(f'Published {len(grasp_candidates)} grasp candidates')

def main(args=None):
    rclpy.init(args=args)
    grasp_detector = GraspDetectionNode()

    try:
        rclpy.spin(grasp_detector)
    except KeyboardInterrupt:
        pass
    finally:
        grasp_detector.destroy_node()
        rclpy.shutdown()
```

## Semantic Segmentation for Manipulation

### Instance Segmentation for Object Parts

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
from torchvision.models.segmentation import fcn_resnet50
import json

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)

        # Publishers
        self.segmentation_pub = self.create_publisher(
            String, '/segmentation_result', 10)

        # Internal state
        self.bridge = CvBridge()

        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load segmentation model
        self.model = fcn_resnet50(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # COCO class names for segmentation
        self.coco_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.get_logger().info('Segmentation Node initialized')

    def image_callback(self, msg):
        """Process incoming images for segmentation"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run segmentation
            segmentation_result = self.segment_image(cv_image)

            # Process segmentation result for manipulation
            manipulation_objects = self.process_segmentation_for_manipulation(segmentation_result)

            # Publish results
            self.publish_segmentation(manipulation_objects)

        except Exception as e:
            self.get_logger().error(f'Error in segmentation: {e}')

    def segment_image(self, image):
        """Run semantic segmentation on image"""
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(image_rgb).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)['out'][0]
            predicted = torch.argmax(output, dim=0).cpu().numpy()

        return predicted

    def process_segmentation_for_manipulation(self, segmentation_mask):
        """Process segmentation mask for object manipulation"""
        objects = []

        # Get unique classes in the segmentation
        unique_classes = np.unique(segmentation_mask)

        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue

            # Create mask for this class
            class_mask = (segmentation_mask == class_id).astype(np.uint8)

            # Find contours to get individual objects
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Filter by size (avoid very small detections)
                area = cv2.contourArea(contour)
                if area < 100:  # Minimum area threshold
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate center of mass
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    center_x = int(moments['m10'] / moments['m00'])
                    center_y = int(moments['m01'] / moments['m00'])
                else:
                    center_x, center_y = x + w//2, y + h//2

                # Calculate bounding box center in 3D if depth is available
                # This would require depth information which we don't have in this simplified example

                object_info = {
                    'class_id': int(class_id),
                    'class_name': self.coco_names[class_id] if class_id < len(self.coco_names) else f'unknown_{class_id}',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'center': [int(center_x), int(center_y)],
                    'area': int(area),
                    'mask': contour.tolist()  # Contour points
                }

                objects.append(object_info)

        return objects

    def publish_segmentation(self, objects):
        """Publish segmentation results"""
        segmentation_msg = String()
        segmentation_msg.data = json.dumps({
            'timestamp': self.get_clock().now().nanoseconds,
            'objects': objects
        })
        self.segmentation_pub.publish(segmentation_msg)

def main(args=None):
    rclpy.init(args=args)
    segmenter = SegmentationNode()

    try:
        rclpy.spin(segmenter)
    except KeyboardInterrupt:
        pass
    finally:
        segmenter.destroy_node()
        rclpy.shutdown()
```

## Manipulation Planning Integration

### Vision-Based Manipulation Planning

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, PoseStamped, Point
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
import numpy as np
import cv2
import json
from typing import Dict, List, Optional

class VisionManipulationPlanner(Node):
    def __init__(self):
        super().__init__('vision_manipulation_planner')

        # Subscribers
        self.detection_sub = self.create_subscription(
            String, '/detected_objects', self.detection_callback, 10)
        self.segmentation_sub = self.create_subscription(
            String, '/segmentation_result', self.segmentation_callback, 10)
        self.grasp_sub = self.create_subscription(
            String, '/grasp_candidates', self.grasp_callback, 10)

        # Publishers
        self.manipulation_plan_pub = self.create_publisher(
            String, '/manipulation_plan', 10)
        self.target_object_pub = self.create_publisher(
            PoseStamped, '/target_object_pose', 10)

        # Internal state
        self.bridge = CvBridge()
        self.detected_objects = []
        self.segmented_objects = []
        self.grasp_candidates = []

        # Manipulation planning parameters
        self.reachability_threshold = 1.0  # meters
        self.preferred_grasp_height = 0.8  # meters (table height)

        self.get_logger().info('Vision Manipulation Planner initialized')

    def detection_callback(self, msg):
        """Process object detections"""
        try:
            data = json.loads(msg.data)
            self.detected_objects = data.get('detections', [])
            self.get_logger().debug(f'Updated detections: {len(self.detected_objects)} objects')
        except Exception as e:
            self.get_logger().error(f'Error processing detections: {e}')

    def segmentation_callback(self, msg):
        """Process segmentation results"""
        try:
            data = json.loads(msg.data)
            self.segmented_objects = data.get('objects', [])
            self.get_logger().debug(f'Updated segmentation: {len(self.segmented_objects)} objects')
        except Exception as e:
            self.get_logger().error(f'Error processing segmentation: {e}')

    def grasp_callback(self, msg):
        """Process grasp candidates"""
        try:
            data = json.loads(msg.data)
            self.grasp_candidates = data.get('grasps', [])
            self.get_logger().debug(f'Updated grasps: {len(self.grasp_candidates)} candidates')
        except Exception as e:
            self.get_logger().error(f'Error processing grasps: {e}')

    def plan_manipulation(self, target_object_class: str):
        """Plan manipulation for a target object class"""
        # Find target object in detections
        target_obj = self.find_target_object(target_object_class)
        if not target_obj:
            self.get_logger().warn(f'No {target_object_class} found for manipulation')
            return None

        # Find grasp candidates for this object
        object_grasps = self.find_grasps_for_object(target_obj)

        if not object_grasps:
            self.get_logger().warn(f'No suitable grasps found for {target_object_class}')
            return None

        # Select best grasp based on criteria
        best_grasp = self.select_best_grasp(object_grasps, target_obj)

        if not best_grasp:
            self.get_logger().warn(f'No optimal grasp found for {target_object_class}')
            return None

        # Create manipulation plan
        plan = self.create_manipulation_plan(target_obj, best_grasp)

        # Publish the plan
        self.publish_manipulation_plan(plan)

        return plan

    def find_target_object(self, target_class: str):
        """Find an object of the target class"""
        # Look in both detections and segmentation
        for obj in self.detected_objects:
            if obj['class_name'].lower() == target_class.lower():
                return obj

        for obj in self.segmented_objects:
            if obj['class_name'].lower() == target_class.lower():
                return obj

        return None

    def find_grasps_for_object(self, target_obj):
        """Find grasp candidates for a specific object"""
        if 'bbox' in target_obj:
            obj_x, obj_y, obj_w, obj_h = target_obj['bbox']
            obj_center_x, obj_center_y = obj_x + obj_w/2, obj_y + obj_h/2

            # Find grasps near this object
            object_grasps = []
            for grasp in self.grasp_candidates:
                grasp_x, grasp_y = grasp['x'], grasp['y']

                # Calculate distance to object center
                dist = np.sqrt((grasp_x - obj_center_x)**2 + (grasp_y - obj_center_y)**2)

                # Consider grasps within the object bounding box or nearby
                if dist < max(obj_w, obj_h) * 1.5:  # 1.5x object size tolerance
                    grasp['distance_to_object'] = dist
                    object_grasps.append(grasp)

            return object_grasps

        return []

    def select_best_grasp(self, grasps, target_obj):
        """Select the best grasp based on multiple criteria"""
        if not grasps:
            return None

        # Sort grasps by multiple criteria
        def grasp_score(grasp):
            score = 0

            # Higher confidence is better
            score += grasp['confidence'] * 10

            # Prefer grasps closer to object center
            score -= grasp['distance_to_object'] / 100  # Normalize by image size

            # Prefer grasps at reasonable angles for manipulation
            angle_score = 1 - abs(grasp['angle']) / np.pi  # Prefer angles near 0
            score += angle_score * 2

            # Prefer grasps with good width/height ratios
            aspect_ratio = min(grasp['width'], grasp['height']) / max(grasp['width'], grasp['height'])
            score += aspect_ratio * 3

            return score

        # Sort grasps by score
        sorted_grasps = sorted(grasps, key=grasp_score, reverse=True)

        return sorted_grasps[0]  # Return the best grasp

    def create_manipulation_plan(self, target_obj, best_grasp):
        """Create a manipulation plan based on vision data"""
        plan = {
            'target_object': {
                'class': target_obj.get('class_name', 'unknown'),
                'bbox': target_obj.get('bbox', [0, 0, 0, 0]),
                'center': target_obj.get('center', [0, 0])
            },
            'grasp_point': {
                'x': best_grasp['x'],
                'y': best_grasp['y'],
                'angle': best_grasp['angle'],
                'confidence': best_grasp['confidence']
            },
            'steps': [
                {
                    'action': 'approach_object',
                    'description': 'Move arm to position above object',
                    'parameters': {
                        'x': best_grasp['x'],
                        'y': best_grasp['y'],
                        'height_offset': 0.1  # 10cm above object
                    }
                },
                {
                    'action': 'descend_to_grasp',
                    'description': 'Lower arm to grasp position',
                    'parameters': {
                        'x': best_grasp['x'],
                        'y': best_grasp['y'],
                        'z': best_grasp.get('depth', 0.5)  # Use depth if available
                    }
                },
                {
                    'action': 'grasp_object',
                    'description': 'Close gripper to grasp object',
                    'parameters': {
                        'angle': best_grasp['angle'],
                        'force': 50  # 50% gripper force
                    }
                },
                {
                    'action': 'lift_object',
                    'description': 'Lift object to safe height',
                    'parameters': {
                        'height_offset': 0.15  # 15cm above object
                    }
                }
            ]
        }

        return plan

    def publish_manipulation_plan(self, plan):
        """Publish the manipulation plan"""
        plan_msg = String()
        plan_msg.data = json.dumps(plan)
        self.manipulation_plan_pub.publish(plan_msg)

        self.get_logger().info(f'Published manipulation plan for {plan["target_object"]["class"]}')

def main(args=None):
    rclpy.init(args=args)
    planner = VisionManipulationPlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()
```

## Performance Optimization

### Optimized Computer Vision Pipeline

```python
import cv2
import numpy as np
import torch
from typing import Tuple, Optional
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end-start)*1000:.2f}ms")
        return result
    return wrapper

class OptimizedVisionPipeline:
    def __init__(self, input_size=(640, 480)):
        self.input_size = input_size
        self.model_input_size = (224, 224)  # Standard for most models

        # Pre-allocate buffers for optimization
        self.rgb_buffer = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        self.depth_buffer = np.zeros((input_size[1], input_size[0]), dtype=np.float32)

        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models (simplified - in practice you'd load actual models)
        self.object_detection_model = None
        self.segmentation_model = None
        self.grasp_detection_model = None

    @timing_decorator
    def preprocess_frame(self, color_image: np.ndarray, depth_image: Optional[np.ndarray] = None):
        """Optimized preprocessing of input frames"""
        # Resize color image to model input size
        color_resized = cv2.resize(color_image, self.model_input_size)

        # Convert BGR to RGB
        color_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        color_tensor = torch.from_numpy(color_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        color_tensor = color_tensor.to(self.device)

        if depth_image is not None:
            # Resize depth image
            depth_resized = cv2.resize(depth_image, self.model_input_size)

            # Normalize depth
            depth_normalized = (depth_resized - np.min(depth_resized)) / (np.max(depth_resized) - np.min(depth_resized) + 1e-6)

            # Convert to tensor
            depth_tensor = torch.from_numpy(depth_normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            depth_tensor = depth_tensor.to(self.device)

            return color_tensor, depth_tensor
        else:
            return color_tensor, None

    @timing_decorator
    def batch_process(self, frames: list):
        """Process multiple frames in a batch for efficiency"""
        if not frames:
            return []

        # Preprocess all frames
        batch_color = []
        batch_depth = []

        for frame in frames:
            color_tensor, depth_tensor = self.preprocess_frame(frame['color'], frame.get('depth'))
            batch_color.append(color_tensor)

            if depth_tensor is not None:
                batch_depth.append(depth_tensor)

        # Stack into batch tensors
        batch_color_tensor = torch.cat(batch_color, dim=0)

        if batch_depth:
            batch_depth_tensor = torch.cat(batch_depth, dim=0)
        else:
            batch_depth_tensor = None

        # Process batch (simplified)
        results = self.process_batch(batch_color_tensor, batch_depth_tensor)

        return results

    def process_batch(self, batch_color, batch_depth):
        """Process batch of frames"""
        # This would run the actual models
        # For now, return mock results
        return [{'objects': [], 'grasps': []} for _ in range(batch_color.size(0))]

    def optimize_memory(self):
        """Optimize memory usage"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

class MultiScaleDetector:
    """Multi-scale object detection for better performance"""
    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales

    def detect_multi_scale(self, image, model):
        """Detect objects at multiple scales"""
        all_detections = []

        for scale in self.scales:
            # Resize image
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            resized_img = cv2.resize(image, new_size)

            # Run detection
            detections = self.run_detection(resized_img, model)

            # Scale back to original coordinates
            for det in detections:
                det['bbox'] = [
                    det['bbox'][0] / scale,
                    det['bbox'][1] / scale,
                    det['bbox'][2] / scale,
                    det['bbox'][3] / scale
                ]

            all_detections.extend(detections)

        # Apply non-maximum suppression to remove duplicates
        final_detections = self.non_max_suppression(all_detections)

        return final_detections

    def run_detection(self, image, model):
        """Run object detection on image"""
        # This would call the actual detection model
        # Return mock detections for now
        return []

    def non_max_suppression(self, detections, iou_threshold=0.5):
        """Apply non-maximum suppression to remove duplicate detections"""
        if len(detections) == 0:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        for i, det in enumerate(detections):
            suppress = False
            for kept_det in keep:
                iou = self.calculate_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    suppress = True
                    break

            if not suppress:
                keep.append(det)

        return keep

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1, w2, h2 = bbox2

        x1_2, y1_2 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x2_1 + w2, y2_1 + h2

        # Calculate intersection
        xi1, yi1 = max(x1_1, x2_1), max(y1_1, y2_1)
        xi2, yi2 = min(x1_2, x2_2), min(y1_2, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area

        return inter_area / union_area if union_area > 0 else 0
```

## Best Practices and Guidelines

### Best Practices for Computer Vision in Manipulation

1. **Multi-Modal Fusion**:
   - Combine RGB, depth, and other sensor data
   - Use temporal consistency for robust tracking
   - Implement sensor fusion algorithms

2. **Real-time Performance**:
   - Optimize models for inference speed
   - Use appropriate input resolutions
   - Implement efficient preprocessing pipelines

3. **Robustness**:
   - Handle varying lighting conditions
   - Account for occlusions and clutter
   - Implement fallback mechanisms

4. **Calibration**:
   - Maintain accurate camera calibration
   - Regularly verify extrinsic calibration
   - Account for camera movement

5. **Safety**:
   - Validate grasp points before execution
   - Implement collision avoidance
   - Plan safe approach trajectories

## Troubleshooting Common Issues

### Common Computer Vision Issues and Solutions

1. **Poor Detection Accuracy**:
   - Improve lighting conditions
   - Retrain models on domain-specific data
   - Use data augmentation techniques

2. **Real-time Performance**:
   - Use smaller, optimized models
   - Implement model quantization
   - Use hardware acceleration (GPU/TPU)

3. **Depth Inaccuracy**:
   - Calibrate depth sensors properly
   - Use filtering techniques for noisy data
   - Implement multi-view fusion

4. **Occlusion Handling**:
   - Use temporal tracking
   - Implement partial object recognition
   - Use multiple camera viewpoints

## Summary

In this chapter, you learned:
- How to implement 3D vision and depth perception for manipulation
- Techniques for object detection and recognition in robotics
- Grasp detection and planning algorithms
- Semantic segmentation for object manipulation
- Integration of vision systems with manipulation planning
- Performance optimization strategies for real-time applications
- Best practices for robust computer vision in robotics
- Troubleshooting techniques for common vision issues

Computer vision forms the foundation of perception for humanoid robot manipulation. By combining depth sensing, object detection, segmentation, and grasp planning, robots can effectively perceive and interact with objects in their environment.

---
**Continue to [Chapter 4: Capstone Project Implementation](/docs/module-4-vla-humanoids/chapter-4-capstone-project)**