---
sidebar_label: 'Chapter 4: Capstone Project Implementation'
title: 'Chapter 4: Capstone Project Implementation'
description: 'Implementing a complete Vision-Language-Action system for humanoid robotics'
slug: '/module-4-vla-humanoids/chapter-4-capstone-project'
difficulty: 'advanced'
requiredHardware: ['computer', 'nvidia_gpu', 'camera', 'microphone']
recommendedHardware: ['jetson_orin', 'realsense_camera', 'respeaker']
---

# Chapter 4: Capstone Project Implementation

This capstone project integrates all the Vision-Language-Action (VLA) systems covered in this module into a complete humanoid robotics application. You'll implement a system that can understand natural language commands, perceive objects in the environment, and execute complex manipulation tasks.

## Project Overview: Autonomous Object Retrieval System

The capstone project implements an autonomous object retrieval system for a humanoid robot that can:
- Understand natural language commands like "Bring me the red cup from the kitchen"
- Perceive and locate objects in the environment using computer vision
- Plan and execute manipulation tasks to retrieve and deliver objects
- Navigate to different locations to find requested objects
- Handle dynamic environments and unexpected situations

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Input   │    │   LLM Cognitive │    │   Computer      │
│   Processing    │───▶│   Planner       │───▶│   Vision        │
│   (Whisper)     │    │   (GPT/Claude)  │    │   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Natural       │    │   Task          │    │   Object        │
│   Language      │    │   Decomposition │    │   Detection &   │
│   Understanding │    │   & Planning    │    │   Localization  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Action Execution      │
                    │   (Navigation &        │
                    │   Manipulation)         │
                    └─────────────────────────┘
```

## Implementation Phase 1: Voice Command Processing

### Voice Command Processing Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import whisper
import torch
import pyaudio
import numpy as np
import threading
import queue
import time
import json

class VoiceCommandProcessor(Node):
    def __init__(self):
        super().__init__('voice_command_processor')

        # Publishers
        self.interpretation_pub = self.create_publisher(
            String, '/command_interpretation', 10)

        # Internal state
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.command_buffer = []

        # Initialize Whisper model
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info('Whisper model loaded successfully')

        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.voice_threshold = 0.01
        self.silence_duration_threshold = 1.0

        # Start audio processing
        self.start_audio_recording()

        self.get_logger().info('Voice Command Processor initialized')

    def start_audio_recording(self):
        """Start audio recording in a separate thread"""
        import pyaudio

        def audio_callback(in_data, frame_count, time_info, status):
            audio_array = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_array)
            return (in_data, pyaudio.paContinue)

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )

        self.is_listening = True
        self.stream.start_stream()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

    def calculate_audio_energy(self, audio_data):
        """Calculate audio energy for voice activity detection"""
        return np.mean(np.abs(audio_data))

    def is_voice_present(self, audio_data):
        """Simple voice activity detection"""
        energy = self.calculate_audio_energy(audio_data)
        return energy > self.voice_threshold

    def process_audio(self):
        """Process audio chunks and perform transcription"""
        audio_buffer = np.array([])
        min_speech_duration = 0.5
        max_silence_duration = 0.8

        while self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.1)

                voice_present = self.is_voice_present(chunk)

                if voice_present:
                    audio_buffer = np.concatenate([audio_buffer, chunk])
                    last_voice_time = time.time()
                else:
                    if len(audio_buffer) > 0:
                        time_since_voice = time.time() - last_voice_time if 'last_voice_time' in locals() else 0

                        min_samples = int(min_speech_duration * self.sample_rate)
                        max_silence_samples = int(max_silence_duration * self.sample_rate)

                        if (len(audio_buffer) >= min_samples and
                            time_since_voice >= max_silence_duration):

                            # Transcribe the accumulated audio
                            self.transcribe_and_publish(audio_buffer)

                            # Reset buffer
                            audio_buffer = np.array([])

                # Limit buffer size
                max_buffer_duration = 5.0
                max_samples = int(max_buffer_duration * self.sample_rate)
                if len(audio_buffer) > max_samples:
                    audio_buffer = audio_buffer[-max_samples:]

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in audio processing: {e}')
                continue

    def transcribe_and_publish(self, audio_data):
        """Transcribe audio and publish result"""
        try:
            audio_tensor = torch.from_numpy(audio_data).to(self.model.device)
            result = self.model.transcribe(audio_tensor)
            text = result["text"].strip()

            if text and len(text) > 2:
                # Publish the transcribed text
                transcript_msg = String()
                transcript_msg.data = text
                self.interpretation_pub.publish(transcript_msg)

                self.get_logger().info(f'Transcribed: {text}')

        except Exception as e:
            self.get_logger().error(f'Error in transcription: {e}')

    def destroy_node(self):
        """Clean up resources"""
        self.is_listening = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        if hasattr(self, 'audio'):
            self.audio.terminate()

        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    processor = VoiceCommandProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()
```

## Implementation Phase 2: LLM-Based Command Interpretation

### Command Interpretation Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class CommandInterpreter(Node):
    def __init__(self):
        super().__init__('command_interpreter')

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, '/command_interpretation', self.voice_callback, 10)

        # Publishers
        self.task_plan_pub = self.create_publisher(
            String, '/task_plan', 10)
        self.navigation_goal_pub = self.create_publisher(
            PoseStamped, '/navigation_goal', 10)

        # Internal state
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.get_logger().info('Command Interpreter initialized')

    def voice_callback(self, msg):
        """Process transcribed voice command"""
        command_text = msg.data
        self.get_logger().info(f'Processing command: {command_text}')

        # Interpret the command using LLM
        interpretation = self.interpret_command(command_text)

        if interpretation:
            # Publish the interpreted task
            interpretation_msg = String()
            interpretation_msg.data = json.dumps(interpretation)
            self.task_plan_pub.publish(interpretation_msg)

            self.get_logger().info(f'Published interpretation: {interpretation}')

    def interpret_command(self, command_text: str) -> dict:
        """Interpret natural language command using LLM"""
        system_prompt = """
        You are a command interpreter for a humanoid robot. Parse the user's command and extract:
        - action: The main action (e.g., "bring", "fetch", "go_to", "follow")
        - object: The object to manipulate (e.g., "red cup", "water bottle")
        - location: The location where the object is or should be taken (e.g., "kitchen", "living room")
        - recipient: Who should receive the object (if applicable)
        - priority: Task priority (1-5)

        Respond in JSON format with keys: action, object, location, recipient, priority.
        If any information is missing, use null or reasonable defaults.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command_text}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            interpretation = json.loads(response.choices[0].message.content)

            # Add timestamp
            interpretation['timestamp'] = self.get_clock().now().nanoseconds
            interpretation['original_command'] = command_text

            return interpretation

        except Exception as e:
            self.get_logger().error(f'Error interpreting command: {e}')
            return {
                'action': 'unknown',
                'object': None,
                'location': None,
                'recipient': None,
                'priority': 3,
                'timestamp': self.get_clock().now().nanoseconds,
                'original_command': command_text
            }

def main(args=None):
    rclpy.init(args=args)
    interpreter = CommandInterpreter()

    try:
        rclpy.spin(interpreter)
    except KeyboardInterrupt:
        pass
    finally:
        interpreter.destroy_node()
        rclpy.shutdown()
```

## Implementation Phase 3: Object Detection and Localization

### Integrated Object Detection Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Pose
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import json
from scipy.spatial.transform import Rotation as R

class IntegratedObjectDetector(Node):
    def __init__(self):
        super().__init__('integrated_object_detector')

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        self.task_sub = self.create_subscription(
            String, '/task_plan', self.task_callback, 10)

        # Publishers
        self.detection_pub = self.create_publisher(
            String, '/object_detections', 10)
        self.target_pose_pub = self.create_publisher(
            PointStamped, '/target_object_pose', 10)

        # Internal state
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.current_task = None
        self.depth_scale = 0.001

        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load detection model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # COCO class names
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

        self.get_logger().info('Integrated Object Detector initialized')

    def info_callback(self, msg):
        """Process camera calibration info"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def task_callback(self, msg):
        """Update current task from command interpreter"""
        try:
            task_data = json.loads(msg.data)
            self.current_task = task_data
            self.get_logger().info(f'Updated task: {task_data.get("action")} for {task_data.get("object")}')
        except Exception as e:
            self.get_logger().error(f'Error parsing task: {e}')

    def image_callback(self, msg):
        """Process incoming images for object detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run object detection
            detections = self.detect_objects(cv_image)

            # Filter detections based on current task if available
            if self.current_task and 'object' in self.current_task:
                target_object = self.current_task['object']
                if target_object:
                    detections = self.filter_detections_for_task(detections, target_object)

            # Convert 2D detections to 3D positions if possible
            detections_3d = self.add_3d_positions(detections, cv_image)

            # Publish detections
            self.publish_detections(detections_3d)

            # If we found the target object, publish its pose
            if self.current_task and 'object' in self.current_task:
                target_obj = self.find_target_object(detections_3d, self.current_task['object'])
                if target_obj:
                    self.publish_target_pose(target_obj)

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

    def filter_detections_for_task(self, detections, target_object):
        """Filter detections to prioritize objects matching the task"""
        if not target_object:
            return detections

        # Convert target object to lowercase for matching
        target_lower = target_object.lower()

        # Score each detection based on how well it matches the target
        scored_detections = []
        for det in detections:
            score = 0

            # Exact match
            if det['class_name'].lower() == target_lower:
                score = 10
            # Partial match
            elif target_lower in det['class_name'].lower() or det['class_name'].lower() in target_lower:
                score = 5
            # Color in name (simplified)
            elif 'red' in target_lower and 'red' in det['class_name'].lower():
                score = 3
            elif 'blue' in target_lower and 'blue' in det['class_name'].lower():
                score = 3
            elif 'green' in target_lower and 'green' in det['class_name'].lower():
                score = 3

            scored_detections.append((det, score))

        # Sort by score and return top detections
        scored_detections.sort(key=lambda x: x[1], reverse=True)
        return [det for det, score in scored_detections if score > 0]

    def add_3d_positions(self, detections, image):
        """Add 3D positions to detections (simplified - would need depth in practice)"""
        # In a real system, this would use depth information
        # For this example, we'll add placeholder 3D positions
        detections_3d = []

        for det in detections:
            # Calculate 3D position from 2D position and camera parameters
            center_x, center_y = det['center']

            # Convert to 3D (simplified - in reality you'd use depth information)
            if self.camera_matrix is not None:
                # This is a simplified calculation - real depth would be needed
                # For now, we'll assume a fixed distance of 1 meter
                fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
                cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

                # Convert 2D pixel coordinates to 3D world coordinates (simplified)
                z = 1.0  # Fixed depth for example
                x = (center_x - cx) * z / fx
                y = (center_y - cy) * z / fy

                det['position_3d'] = [float(x), float(y), float(z)]

            detections_3d.append(det)

        return detections_3d

    def find_target_object(self, detections, target_object):
        """Find the most relevant object for the current task"""
        if not target_object:
            return None

        target_lower = target_object.lower()

        for det in detections:
            if (det['class_name'].lower() == target_lower or
                target_lower in det['class_name'].lower() or
                det['class_name'].lower() in target_lower):
                return det

        # If exact match not found, return the highest confidence detection
        if detections:
            return max(detections, key=lambda x: x['confidence'])

        return None

    def publish_detections(self, detections):
        """Publish object detections"""
        detection_msg = String()
        detection_msg.data = json.dumps({
            'timestamp': self.get_clock().now().nanoseconds,
            'detections': detections
        })
        self.detection_pub.publish(detection_msg)

    def publish_target_pose(self, target_obj):
        """Publish the pose of the target object"""
        if 'position_3d' in target_obj:
            x, y, z = target_obj['position_3d']

            pose_msg = PointStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'camera_link'
            pose_msg.point.x = x
            pose_msg.point.y = y
            pose_msg.point.z = z

            self.target_pose_pub.publish(pose_msg)

            self.get_logger().info(f'Published target pose: ({x:.2f}, {y:.2f}, {z:.2f})')

def main(args=None):
    rclpy.init(args=args)
    detector = IntegratedObjectDetector()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()
```

## Implementation Phase 4: Manipulation Planning and Execution

### Manipulation Planner Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Vector3
from builtin_interfaces.msg import Duration
import json
import numpy as np
from typing import Dict, List, Optional

class ManipulationPlanner(Node):
    def __init__(self):
        super().__init__('manipulation_planner')

        # Subscribers
        self.task_sub = self.create_subscription(
            String, '/task_plan', self.task_callback, 10)
        self.detection_sub = self.create_subscription(
            String, '/object_detections', self.detection_callback, 10)
        self.target_pose_sub = self.create_subscription(
            String, '/target_object_pose', self.target_pose_callback, 10)

        # Publishers
        self.manipulation_plan_pub = self.create_publisher(
            String, '/manipulation_plan', 10)
        self.navigation_goal_pub = self.create_publisher(
            String, '/navigation_goal', 10)

        # Internal state
        self.current_task = None
        self.current_detections = []
        self.target_object_pose = None

        # Robot parameters
        self.robot_reach = 1.0  # meters
        self.gripper_width = 0.1  # meters
        self.safety_margin = 0.05  # meters

        self.get_logger().info('Manipulation Planner initialized')

    def task_callback(self, msg):
        """Update current task"""
        try:
            self.current_task = json.loads(msg.data)
            self.get_logger().info(f'Updated task: {self.current_task.get("action")}')

            # Plan manipulation if we have all necessary information
            self.plan_manipulation_if_ready()
        except Exception as e:
            self.get_logger().error(f'Error parsing task: {e}')

    def detection_callback(self, msg):
        """Update object detections"""
        try:
            data = json.loads(msg.data)
            self.current_detections = data.get('detections', [])
            self.get_logger().debug(f'Updated detections: {len(self.current_detections)} objects')

            # Plan manipulation if we have all necessary information
            self.plan_manipulation_if_ready()
        except Exception as e:
            self.get_logger().error(f'Error parsing detections: {e}')

    def target_pose_callback(self, msg):
        """Update target object pose"""
        try:
            # In a real system, this would come from a PoseStamped message
            # For this example, we'll simulate the pose
            pose_data = json.loads(msg.data)
            self.target_object_pose = pose_data
            self.get_logger().info(f'Updated target pose: {pose_data}')

            # Plan manipulation if we have all necessary information
            self.plan_manipulation_if_ready()
        except Exception as e:
            self.get_logger().error(f'Error parsing target pose: {e}')

    def plan_manipulation_if_ready(self):
        """Plan manipulation if we have all necessary information"""
        if (self.current_task and
            self.current_detections and
            self.target_object_pose):

            manipulation_plan = self.create_manipulation_plan()
            if manipulation_plan:
                self.publish_manipulation_plan(manipulation_plan)

    def create_manipulation_plan(self) -> Optional[Dict]:
        """Create a manipulation plan based on current state"""
        if not self.current_task:
            return None

        task = self.current_task
        action = task.get('action', '').lower()

        # Determine what needs to be done
        if action in ['bring', 'fetch', 'get']:
            return self.plan_fetch_object(task)
        elif action == 'go_to':
            return self.plan_navigation(task)
        else:
            self.get_logger().warn(f'Unknown action: {action}')
            return None

    def plan_fetch_object(self, task: Dict) -> Dict:
        """Plan for fetching an object"""
        target_object = task.get('object', 'unknown')
        target_location = task.get('location')

        # Find the target object in detections
        target_obj = self.find_target_object(target_object)
        if not target_obj:
            self.get_logger().warn(f'Could not find {target_object} for fetching')
            return self.create_search_plan(task)

        # Check if object is reachable
        if not self.is_object_reachable(target_obj):
            self.get_logger().info('Object not reachable, planning navigation')
            return self.create_navigation_plan(target_obj, task)

        # Create manipulation plan
        plan = {
            'task_id': task.get('timestamp', 0),
            'action_sequence': [
                {
                    'action': 'approach_object',
                    'description': f'Approach the {target_object}',
                    'target': {
                        'position': self.get_approach_position(target_obj),
                        'orientation': self.calculate_approach_orientation(target_obj)
                    },
                    'parameters': {
                        'approach_distance': 0.3,  # 30cm from object
                        'approach_height': 0.2   # 20cm above object initially
                    }
                },
                {
                    'action': 'align_with_object',
                    'description': 'Align gripper with object',
                    'target': {
                        'position': self.get_grasp_position(target_obj),
                        'orientation': self.calculate_grasp_orientation(target_obj)
                    },
                    'parameters': {
                        'precision_approach': True
                    }
                },
                {
                    'action': 'grasp_object',
                    'description': f'Grasp the {target_object}',
                    'target': {
                        'position': self.get_grasp_position(target_obj),
                        'orientation': self.calculate_grasp_orientation(target_obj)
                    },
                    'parameters': {
                        'gripper_width': self.calculate_gripper_width(target_obj),
                        'grasp_force': 50  # 50% force
                    }
                },
                {
                    'action': 'lift_object',
                    'description': 'Lift object to safe height',
                    'target': {
                        'position': self.get_lift_position(target_obj),
                        'orientation': self.calculate_grasp_orientation(target_obj)
                    },
                    'parameters': {
                        'lift_height': 0.2  # Lift 20cm above grasp point
                    }
                }
            ],
            'completion_criteria': {
                'object_grasped': True,
                'object_lifted': True
            }
        }

        # Add delivery step if location is specified
        if target_location:
            delivery_plan = {
                'action': 'deliver_object',
                'description': f'Deliver {target_object} to {target_location}',
                'target_location': target_location,
                'parameters': {
                    'delivery_height': 0.8  # Deliver at table height
                }
            }
            plan['action_sequence'].append(delivery_plan)

        return plan

    def plan_navigation(self, task: Dict) -> Dict:
        """Plan for navigation to a location"""
        target_location = task.get('location', 'unknown')

        # In a real system, this would use a map to find the location
        # For this example, we'll use placeholder coordinates
        location_coordinates = self.get_location_coordinates(target_location)

        plan = {
            'task_id': task.get('timestamp', 0),
            'action_sequence': [
                {
                    'action': 'navigate_to_location',
                    'description': f'Navigate to {target_location}',
                    'target': {
                        'position': location_coordinates,
                        'orientation': [0, 0, 0, 1]  # Identity quaternion
                    },
                    'parameters': {
                        'planning_algorithm': 'navfn',
                        'collision_avoidance': True
                    }
                },
                {
                    'action': 'arrive_at_location',
                    'description': f'Arrived at {target_location}',
                    'target': {
                        'position': location_coordinates,
                        'orientation': [0, 0, 0, 1]
                    }
                }
            ],
            'completion_criteria': {
                'navigation_completed': True
            }
        }

        return plan

    def create_search_plan(self, task: Dict) -> Dict:
        """Create a search plan when object is not immediately visible"""
        target_object = task.get('object', 'unknown')
        target_location = task.get('location')

        plan = {
            'task_id': task.get('timestamp', 0),
            'action_sequence': [
                {
                    'action': 'navigate_to_search_area',
                    'description': f'Navigate to area where {target_object} might be',
                    'target': {
                        'position': self.get_location_coordinates(target_location or 'known_areas'),
                        'orientation': [0, 0, 0, 1]
                    }
                },
                {
                    'action': 'search_for_object',
                    'description': f'Search for {target_object} in the area',
                    'parameters': {
                        'search_pattern': 'spiral',
                        'search_height': 1.0,
                        'search_radius': 2.0
                    }
                },
                {
                    'action': 'object_detection_retry',
                    'description': 'Retry object detection after moving',
                    'parameters': {
                        'detection_timeout': 10.0
                    }
                }
            ],
            'completion_criteria': {
                'object_found': True
            }
        }

        return plan

    def create_navigation_plan(self, target_obj, task) -> Dict:
        """Create navigation plan to reach a distant object"""
        # Calculate navigation target (a position near the object)
        nav_target = self.get_navigation_approach_position(target_obj)

        plan = {
            'task_id': task.get('timestamp', 0),
            'action_sequence': [
                {
                    'action': 'navigate_to_object_area',
                    'description': 'Navigate closer to object location',
                    'target': {
                        'position': nav_target,
                        'orientation': [0, 0, 0, 1]
                    }
                },
                {
                    'action': 'relocalize_object',
                    'description': 'Re-detect object from closer range',
                    'parameters': {
                        'relocalization_timeout': 5.0
                    }
                }
            ],
            'completion_criteria': {
                'navigation_completed': True,
                'object_relocalized': True
            }
        }

        return plan

    def find_target_object(self, target_name: str) -> Optional[Dict]:
        """Find target object in current detections"""
        target_lower = target_name.lower()

        for obj in self.current_detections:
            if (obj['class_name'].lower() == target_lower or
                target_lower in obj['class_name'].lower() or
                obj['class_name'].lower() in target_lower):
                return obj

        # If not found exactly, return highest confidence object
        if self.current_detections:
            return max(self.current_detections, key=lambda x: x['confidence'])

        return None

    def is_object_reachable(self, obj: Dict) -> bool:
        """Check if object is within robot's reach"""
        if 'position_3d' not in obj:
            return False

        x, y, z = obj['position_3d']
        distance = np.sqrt(x**2 + y**2 + z**2)
        return distance <= self.robot_reach

    def get_approach_position(self, obj: Dict) -> List[float]:
        """Calculate approach position for object"""
        if 'position_3d' in obj:
            x, y, z = obj['position_3d']
            # Approach from a safe distance
            return [x * 0.7, y * 0.7, z + 0.2]  # 30% closer, 20cm higher
        else:
            # Default approach position
            return [0.5, 0.0, 0.8]

    def get_grasp_position(self, obj: Dict) -> List[float]:
        """Calculate grasp position for object"""
        if 'position_3d' in obj:
            x, y, z = obj['position_3d']
            # Grasp at object position but slightly above initially
            return [x, y, z + 0.05]  # 5cm above object center
        else:
            return [0.6, 0.0, 0.7]

    def get_lift_position(self, obj: Dict) -> List[float]:
        """Calculate lift position after grasping"""
        if 'position_3d' in obj:
            x, y, z = obj['position_3d']
            # Lift to safe height
            return [x, y, z + 0.2]  # 20cm above object
        else:
            return [0.6, 0.0, 0.9]

    def calculate_approach_orientation(self, obj: Dict) -> List[float]:
        """Calculate approach orientation"""
        # Default approach orientation (looking down at object)
        return [0.0, 0.0, 0.0, 1.0]  # Identity quaternion

    def calculate_grasp_orientation(self, obj: Dict) -> List[float]:
        """Calculate grasp orientation"""
        # For now, use default orientation
        # In practice, this would depend on object shape and grasp type
        return [0.0, 0.0, 0.0, 1.0]

    def calculate_gripper_width(self, obj: Dict) -> float:
        """Calculate appropriate gripper width"""
        # Simplified calculation based on object size
        # In practice, this would use object dimensions from detection
        return min(0.08, self.gripper_width)  # Maximum 8cm gripper width

    def get_navigation_approach_position(self, obj: Dict) -> List[float]:
        """Get position for navigation to approach object"""
        if 'position_3d' in obj:
            x, y, z = obj['position_3d']
            # Navigate to a position 1 meter away from object
            distance = max(np.sqrt(x**2 + y**2), 1.0)
            scale = 0.8 / distance  # Scale to 80cm away
            return [x * scale, y * scale, 0.8]  # Fixed height
        else:
            return [1.0, 0.0, 0.8]

    def get_location_coordinates(self, location_name: str) -> List[float]:
        """Get coordinates for a named location (placeholder)"""
        # In a real system, this would come from a map
        # For this example, we'll use placeholder coordinates
        location_map = {
            'kitchen': [2.0, 1.0, 0.0],
            'living_room': [0.0, 0.0, 0.0],
            'bedroom': [-2.0, 1.0, 0.0],
            'office': [1.0, -2.0, 0.0]
        }

        return location_map.get(location_name.lower(), [0.0, 0.0, 0.0])

    def publish_manipulation_plan(self, plan: Dict):
        """Publish the manipulation plan"""
        plan_msg = String()
        plan_msg.data = json.dumps(plan)
        self.manipulation_plan_pub.publish(plan_msg)

        self.get_logger().info(f'Published manipulation plan with {len(plan["action_sequence"])} steps')

def main(args=None):
    rclpy.init(args=args)
    planner = ManipulationPlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()
```

## Implementation Phase 5: System Integration

### Main Integration Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import threading
import time
from typing import Dict, Any

class VLAIntegrationNode(Node):
    def __init__(self):
        super().__init__('vla_integration_node')

        # Subscribers
        self.task_sub = self.create_subscription(
            String, '/task_plan', self.task_callback, 10)
        self.manipulation_plan_sub = self.create_subscription(
            String, '/manipulation_plan', self.manipulation_plan_callback, 10)
        self.navigation_goal_sub = self.create_subscription(
            String, '/navigation_goal', self.navigation_goal_callback, 10)

        # Publishers
        self.system_status_pub = self.create_publisher(
            String, '/system_status', 10)
        self.execution_command_pub = self.create_publisher(
            String, '/execution_commands', 10)

        # Internal state
        self.current_task = None
        self.current_manipulation_plan = None
        self.system_status = {
            'state': 'idle',
            'active_task': None,
            'progress': 0.0,
            'last_update': 0
        }

        # Execution thread
        self.execution_thread = None
        self.execution_lock = threading.Lock()

        self.get_logger().info('VLA Integration Node initialized')

    def task_callback(self, msg):
        """Handle new task from command interpreter"""
        try:
            task_data = json.loads(msg.data)
            self.current_task = task_data

            self.get_logger().info(f'New task received: {task_data.get("action")} for {task_data.get("object")}')

            # Update system status
            self.system_status['state'] = 'planning'
            self.system_status['active_task'] = task_data.get('original_command', 'unknown')
            self.system_status['progress'] = 0.1
            self.publish_system_status()

        except Exception as e:
            self.get_logger().error(f'Error processing task: {e}')

    def manipulation_plan_callback(self, msg):
        """Handle new manipulation plan"""
        try:
            plan_data = json.loads(msg.data)
            self.current_manipulation_plan = plan_data

            self.get_logger().info(f'New manipulation plan received with {len(plan_data["action_sequence"])} steps')

            # Start execution if we have a plan and task
            if self.current_task and self.current_manipulation_plan:
                self.execute_plan_async()

        except Exception as e:
            self.get_logger().error(f'Error processing manipulation plan: {e}')

    def navigation_goal_callback(self, msg):
        """Handle navigation goals"""
        try:
            goal_data = json.loads(msg.data)
            self.get_logger().info(f'Navigation goal received: {goal_data}')

            # Publish to navigation system
            self.publish_execution_command({
                'command': 'navigate',
                'goal': goal_data,
                'task_id': self.system_status.get('active_task_id', 0)
            })

        except Exception as e:
            self.get_logger().error(f'Error processing navigation goal: {e}')

    def execute_plan_async(self):
        """Execute the plan in a separate thread"""
        with self.execution_lock:
            if self.execution_thread and self.execution_thread.is_alive():
                self.get_logger().warn('Plan execution already in progress, skipping')
                return

            self.execution_thread = threading.Thread(target=self.execute_plan)
            self.execution_thread.start()

    def execute_plan(self):
        """Execute the manipulation plan step by step"""
        if not self.current_manipulation_plan:
            return

        plan = self.current_manipulation_plan
        steps = plan.get('action_sequence', [])

        self.system_status['state'] = 'executing'
        self.system_status['progress'] = 0.0
        self.publish_system_status()

        for i, step in enumerate(steps):
            self.get_logger().info(f'Executing step {i+1}/{len(steps)}: {step["action"]}')

            # Update progress
            self.system_status['progress'] = (i + 1) / len(steps)
            self.publish_system_status()

            # Execute the step
            success = self.execute_step(step)

            if not success:
                self.get_logger().error(f'Step {i+1} failed: {step["action"]}')
                self.system_status['state'] = 'error'
                self.publish_system_status()
                return

            # Small delay between steps for safety
            time.sleep(0.5)

        # Plan completed successfully
        self.system_status['state'] = 'completed'
        self.system_status['progress'] = 1.0
        self.publish_system_status()

        self.get_logger().info('Manipulation plan completed successfully')

    def execute_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single step of the plan"""
        action = step.get('action', '')

        # Create execution command
        command = {
            'command': action,
            'parameters': step.get('parameters', {}),
            'target': step.get('target', {}),
            'description': step.get('description', ''),
            'timestamp': self.get_clock().now().nanoseconds
        }

        # Publish command to execution system
        command_msg = String()
        command_msg.data = json.dumps(command)
        self.execution_command_pub.publish(command_msg)

        self.get_logger().info(f'Published command: {action}')

        # In a real system, you would wait for feedback from the execution system
        # For this example, we'll simulate execution time
        time.sleep(2.0)  # Simulate execution time

        # For this example, assume all steps succeed
        # In practice, you would check execution feedback
        return True

    def publish_system_status(self):
        """Publish current system status"""
        status_msg = String()
        self.system_status['last_update'] = self.get_clock().now().nanoseconds
        status_msg.data = json.dumps(self.system_status)
        self.system_status_pub.publish(status_msg)

    def publish_execution_command(self, command: Dict[str, Any]):
        """Publish execution command"""
        command_msg = String()
        command_msg.data = json.dumps(command)
        self.execution_command_pub.publish(command_msg)

def main(args=None):
    rclpy.init(args=args)
    integration_node = VLAIntegrationNode()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        pass
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()
```

## Testing and Validation

### Integration Test Script

```python
#!/usr/bin/env python3
"""
Integration test script for the complete VLA system
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import time

class VLATestNode(Node):
    def __init__(self):
        super().__init__('vla_test_node')

        # Publishers for testing
        self.command_pub = self.create_publisher(
            String, '/command_interpretation', 10)
        self.voice_pub = self.create_publisher(
            String, '/command_interpretation', 10)

        # Subscribers for monitoring
        self.status_sub = self.create_subscription(
            String, '/system_status', self.status_callback, 10)
        self.detection_sub = self.create_subscription(
            String, '/object_detections', self.detection_callback, 10)

        self.current_status = None
        self.detection_count = 0

        self.get_logger().info('VLA Test Node initialized')

    def status_callback(self, msg):
        """Monitor system status"""
        try:
            status = json.loads(msg.data)
            self.current_status = status
            self.get_logger().info(f'System status: {status["state"]}, Progress: {status["progress"]:.2f}')
        except Exception as e:
            self.get_logger().error(f'Error parsing status: {e}')

    def detection_callback(self, msg):
        """Monitor object detections"""
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])
            self.detection_count = len(detections)
            self.get_logger().info(f'Detected {len(detections)} objects')
        except Exception as e:
            self.get_logger().error(f'Error parsing detections: {e}')

    def run_test_sequence(self):
        """Run a complete test sequence"""
        self.get_logger().info('Starting VLA system test sequence...')

        # Test 1: Simple object fetch command
        self.get_logger().info('Test 1: Fetching a cup')
        command_msg = String()
        command_msg.data = "Bring me the cup from the kitchen"
        self.command_pub.publish(command_msg)

        # Wait for execution
        start_time = time.time()
        while time.time() - start_time < 30:  # Wait up to 30 seconds
            if (self.current_status and
                self.current_status.get('state') == 'completed'):
                self.get_logger().info('Test 1 completed successfully!')
                break
            time.sleep(1.0)

        # Test 2: Navigation command
        self.get_logger().info('Test 2: Going to living room')
        command_msg.data = "Go to the living room"
        self.command_pub.publish(command_msg)

        # Wait for execution
        start_time = time.time()
        while time.time() - start_time < 30:
            if (self.current_status and
                self.current_status.get('state') == 'completed'):
                self.get_logger().info('Test 2 completed successfully!')
                break
            time.sleep(1.0)

        # Test 3: Object search command
        self.get_logger().info('Test 3: Finding a book')
        command_msg.data = "Find my book"
        self.command_pub.publish(command_msg)

        # Wait for execution
        start_time = time.time()
        while time.time() - start_time < 45:  # Longer for search
            if (self.current_status and
                self.current_status.get('state') == 'completed'):
                self.get_logger().info('Test 3 completed successfully!')
                break
            time.sleep(1.0)

        self.get_logger().info('All tests completed!')

def main(args=None):
    rclpy.init(args=args)
    test_node = VLATestNode()

    # Run tests after a short delay
    def run_tests():
        time.sleep(5.0)  # Wait for system to initialize
        test_node.run_test_sequence()

    # Schedule test execution
    timer = test_node.create_timer(1.0, run_tests)

    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()
```

## Performance Optimization and Monitoring

### Performance Monitor Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import time
import psutil
import torch
import threading

class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        # Publishers
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.cpu_pub = self.create_publisher(Float64, '/cpu_usage', 10)
        self.gpu_pub = self.create_publisher(Float64, '/gpu_usage', 10)
        self.memory_pub = self.create_publisher(Float64, '/memory_usage', 10)

        # Timer for monitoring
        self.monitor_timer = self.create_timer(1.0, self.monitor_system)

        self.get_logger().info('Performance Monitor initialized')

    def monitor_system(self):
        """Monitor system performance"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        # Get GPU metrics if available
        gpu_percent = 0.0
        gpu_memory_percent = 0.0

        if torch.cuda.is_available():
            gpu_percent = torch.cuda.utilization()
            gpu_memory_allocated = torch.cuda.memory_allocated()
            gpu_memory_reserved = torch.cuda.memory_reserved()
            gpu_memory_percent = (gpu_memory_allocated / gpu_memory_reserved) * 100 if gpu_memory_reserved > 0 else 0.0

        # Publish metrics
        cpu_msg = Float64()
        cpu_msg.data = float(cpu_percent)
        self.cpu_pub.publish(cpu_msg)

        gpu_msg = Float64()
        gpu_msg.data = float(gpu_percent)
        self.gpu_pub.publish(gpu_msg)

        memory_msg = Float64()
        memory_msg.data = float(memory_percent)
        self.memory_pub.publish(memory_msg)

        # Publish diagnostics
        self.publish_diagnostics(cpu_percent, gpu_percent, memory_percent)

    def publish_diagnostics(self, cpu, gpu, memory):
        """Publish diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # CPU diagnostic
        cpu_diag = DiagnosticStatus()
        cpu_diag.name = 'CPU Usage'
        cpu_diag.level = DiagnosticStatus.OK if cpu < 80 else DiagnosticStatus.WARN if cpu < 90 else DiagnosticStatus.ERROR
        cpu_diag.message = f'CPU usage: {cpu:.1f}%'
        cpu_diag.values = [KeyValue(key='usage_percent', value=f'{cpu:.1f}')]
        diag_array.status.append(cpu_diag)

        # GPU diagnostic
        gpu_diag = DiagnosticStatus()
        gpu_diag.name = 'GPU Usage'
        gpu_diag.level = DiagnosticStatus.OK if gpu < 80 else DiagnosticStatus.WARN if gpu < 90 else DiagnosticStatus.ERROR
        gpu_diag.message = f'GPU usage: {gpu:.1f}%'
        gpu_diag.values = [KeyValue(key='usage_percent', value=f'{gpu:.1f}')]
        diag_array.status.append(gpu_diag)

        # Memory diagnostic
        memory_diag = DiagnosticStatus()
        memory_diag.name = 'Memory Usage'
        memory_diag.level = DiagnosticStatus.OK if memory < 80 else DiagnosticStatus.WARN if memory < 90 else DiagnosticStatus.ERROR
        memory_diag.message = f'Memory usage: {memory:.1f}%'
        memory_diag.values = [KeyValue(key='usage_percent', value=f'{memory:.1f}')]
        diag_array.status.append(memory_diag)

        self.diag_pub.publish(diag_array)

def main(args=None):
    rclpy.init(args=args)
    monitor = PerformanceMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()
```

## Launch File for Complete System

### VLA System Launch File

```python
# launch/vla_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def launch_setup(context, *args, **kwargs):
    # Get launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    model_size = LaunchConfiguration('model_size').perform(context)

    # Voice command processor
    voice_processor = Node(
        package='vla_system',
        executable='voice_command_processor',
        name='voice_command_processor',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    # Command interpreter
    command_interpreter = Node(
        package='vla_system',
        executable='command_interpreter',
        name='command_interpreter',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    # Object detector
    object_detector = Node(
        package='vla_system',
        executable='integrated_object_detector',
        name='integrated_object_detector',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    # Manipulation planner
    manipulation_planner = Node(
        package='vla_system',
        executable='manipulation_planner',
        name='manipulation_planner',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    # System integration
    vla_integration = Node(
        package='vla_system',
        executable='vla_integration_node',
        name='vla_integration_node',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    # Performance monitor
    performance_monitor = Node(
        package='vla_system',
        executable='performance_monitor',
        name='performance_monitor',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    # Test node (optional)
    test_node = Node(
        package='vla_system',
        executable='vla_test_node',
        name='vla_test_node',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    return [
        voice_processor,
        command_interpreter,
        object_detector,
        manipulation_planner,
        vla_integration,
        performance_monitor,
        test_node
    ]

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'model_size',
            default_value='base',
            description='Whisper model size (tiny, base, small, medium, large)'
        ),

        # Opaque function to handle the launch
        OpaqueFunction(function=launch_setup)
    ])
```

## Best Practices and Guidelines

### Best Practices for VLA System Development

1. **Modular Design**:
   - Keep each component focused on a single responsibility
   - Use ROS 2 topics and services for loose coupling
   - Implement proper error handling and recovery

2. **Performance Optimization**:
   - Use appropriate model sizes for your hardware
   - Implement caching for repeated operations
   - Optimize image processing pipelines

3. **Safety Considerations**:
   - Validate all commands before execution
   - Implement safety limits for robot movements
   - Include human-in-the-loop for critical operations

4. **Robustness**:
   - Handle sensor failures gracefully
   - Implement fallback strategies
   - Include comprehensive error reporting

5. **Testing**:
   - Test each component individually
   - Validate system integration thoroughly
   - Include edge case testing

## Troubleshooting Common Issues

### Common Issues and Solutions

1. **High Latency**:
   - Use smaller models for real-time applications
   - Optimize network connectivity for cloud APIs
   - Implement asynchronous processing

2. **Poor Recognition Accuracy**:
   - Improve audio quality and reduce noise
   - Use domain-specific training data
   - Implement confidence-based filtering

3. **Object Detection Failures**:
   - Ensure proper lighting conditions
   - Calibrate cameras regularly
   - Use multiple detection approaches

4. **Manipulation Failures**:
   - Implement robust grasp planning
   - Include force feedback for grasping
   - Plan safe approach trajectories

## Summary

In this capstone project, you implemented a complete Vision-Language-Action system for humanoid robotics that:

1. **Integrated multiple technologies**: Combined speech recognition, LLMs, computer vision, and robotics control
2. **Created a complete pipeline**: From voice commands to action execution
3. **Implemented robust error handling**: With fallback strategies and safety measures
4. **Optimized for performance**: With efficient processing and monitoring
5. **Validated the system**: With comprehensive testing procedures

The system demonstrates the power of integrating advanced AI technologies with robotics to create intuitive, capable humanoid robots that can understand natural language commands and perform complex manipulation tasks.

This capstone project represents the culmination of the Vision-Language-Action module, combining all the concepts learned into a practical, working system that showcases the potential of modern AI-powered robotics.

---
**Congratulations! You have completed the Vision-Language-Action Systems module. You now have the knowledge and skills to build sophisticated humanoid robots that can perceive, understand, and interact with the world around them.**