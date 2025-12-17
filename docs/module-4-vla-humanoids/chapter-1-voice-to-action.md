---
sidebar_label: 'Chapter 1: Voice-to-Action with OpenAI Whisper'
title: 'Chapter 1: Voice-to-Action with OpenAI Whisper'
description: 'Understanding voice-to-action systems using OpenAI Whisper for humanoid robotics'
slug: '/module-4-vla-humanoids/chapter-1-voice-to-action'
difficulty: 'advanced'
requiredHardware: ['computer', 'microphone', 'audio_interface']
recommendedHardware: ['respeaker', 'jetson_orin', 'nvidia_gpu']
---

# Chapter 1: Voice-to-Action with OpenAI Whisper

Voice-to-action systems are crucial for natural human-robot interaction in humanoid robotics. OpenAI Whisper provides state-of-the-art speech recognition capabilities that can be leveraged to create intuitive voice interfaces for humanoid robots. This chapter explores how to implement voice-to-action systems using OpenAI Whisper and integrate them with humanoid robot control systems.

## Introduction to Voice-to-Action Systems

Voice-to-action systems enable humanoid robots to understand spoken commands and translate them into executable actions. The key components include:

- **Speech Recognition**: Converting audio to text (OpenAI Whisper)
- **Natural Language Understanding**: Interpreting the meaning of commands
- **Action Mapping**: Translating commands to robot actions
- **Execution**: Performing the requested actions

### Why OpenAI Whisper for Robotics?

OpenAI Whisper offers several advantages for humanoid robotics:

- **Multilingual Support**: Supports multiple languages for diverse user interactions
- **Robustness**: Handles various accents, background noise, and speech patterns
- **Real-time Processing**: Can be optimized for real-time applications
- **Open Source**: Allows customization and adaptation for specific use cases
- **High Accuracy**: State-of-the-art performance in speech recognition

## Installing and Setting Up OpenAI Whisper

### Prerequisites

Before installing OpenAI Whisper, ensure your system meets the requirements:

- **Python**: 3.8 or higher
- **Hardware**: GPU recommended for real-time processing (CUDA 11.6+)
- **Memory**: 8GB+ RAM (16GB recommended for real-time applications)
- **Audio**: Microphone or audio interface for voice input

### Installation

#### Option 1: Basic Installation

```bash
# Install Whisper from OpenAI
pip install git+https://github.com/openai/whisper.git

# Install additional dependencies
pip install torch torchvision torchaudio
```

#### Option 2: GPU-Accelerated Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Whisper
pip install git+https://github.com/openai/whisper.git

# Install additional dependencies for audio processing
pip install sounddevice pyaudio
```

#### Option 3: Docker Installation

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuda118
RUN pip3 install git+https://github.com/openai/whisper.git
RUN pip3 install sounddevice pyaudio

# Set up your application
WORKDIR /app
COPY . .
CMD ["python3", "voice_to_action.py"]
```

## Basic Whisper Implementation

### Simple Speech Recognition

```python
import whisper
import torch
import numpy as np
import pyaudio
import wave
import threading
import queue
import time

class WhisperSpeechRecognizer:
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Whisper speech recognizer
        :param model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        :param device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_size = model_size
        self.model = whisper.load_model(model_size, device=device)

        # Audio parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_size = 1024
        self.audio_format = pyaudio.paFloat32
        self.channels = 1

        # Audio recording
        self.audio = pyaudio.PyAudio()
        self.is_listening = False
        self.audio_queue = queue.Queue()

        print(f"Whisper model loaded on {device}")
        print(f"Model size: {model_size}")

    def record_audio(self, duration=5):
        """
        Record audio for specified duration
        :param duration: Recording duration in seconds
        :return: Audio data as numpy array
        """
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        frames = []
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        return audio_array

    def transcribe_audio(self, audio_data):
        """
        Transcribe audio data using Whisper
        :param audio_data: Audio data as numpy array
        :return: Transcribed text
        """
        # Convert to tensor and move to device
        audio_tensor = torch.from_numpy(audio_data).to(self.device)

        # Transcribe
        result = self.model.transcribe(audio_tensor)

        return result["text"]

    def transcribe_file(self, audio_file_path):
        """
        Transcribe audio from file
        :param audio_file_path: Path to audio file
        :return: Transcribed text
        """
        result = self.model.transcribe(audio_file_path)
        return result["text"]

    def continuous_listening(self, callback_func):
        """
        Continuously listen for speech and process it
        :param callback_func: Function to call with transcribed text
        """
        def audio_callback(in_data, frame_count, time_info, status):
            audio_array = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_array)
            return (in_data, pyaudio.paContinue)

        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )

        self.is_listening = True
        stream.start_stream()

        # Process audio chunks in a separate thread
        def process_audio():
            while self.is_listening:
                try:
                    # Collect audio for 2 seconds (adjust as needed)
                    audio_chunks = []
                    start_time = time.time()

                    while time.time() - start_time < 2.0 and not self.audio_queue.empty():
                        chunk = self.audio_queue.get_nowait()
                        audio_chunks.append(chunk)

                    if audio_chunks:
                        # Combine audio chunks
                        combined_audio = np.concatenate(audio_chunks)

                        # Only process if audio has sufficient energy (simple VAD)
                        if np.max(np.abs(combined_audio)) > 0.01:  # Threshold
                            text = self.transcribe_audio(combined_audio)
                            if text.strip():  # Only call back if there's text
                                callback_func(text)

                except queue.Empty:
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error processing audio: {e}")

        processing_thread = threading.Thread(target=process_audio)
        processing_thread.start()

        return stream, processing_thread

# Example usage
def main():
    recognizer = WhisperSpeechRecognizer(model_size="base")

    # Record and transcribe
    print("Recording for 5 seconds...")
    audio_data = recognizer.record_audio(duration=5)
    text = recognizer.transcribe_audio(audio_data)
    print(f"Transcribed text: {text}")

if __name__ == "__main__":
    main()
```

## Advanced Whisper Configuration for Robotics

### Optimized Configuration for Real-time Processing

```python
import whisper
import torch
import numpy as np
import pyaudio
import threading
import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class WhisperConfig:
    model_size: str = "base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate: int = 16000
    language: Optional[str] = None  # None for auto-detection
    task: str = "transcribe"  # "transcribe" or "translate"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

class OptimizedWhisperRecognizer:
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.model = whisper.load_model(
            config.model_size,
            device=config.device,
            download_root="./models"  # Cache models locally
        )

        # Audio parameters
        self.sample_rate = config.sample_rate
        self.chunk_size = 1024
        self.audio_format = pyaudio.paFloat32
        self.channels = 1

        # Audio recording
        self.audio = pyaudio.PyAudio()
        self.is_listening = False
        self.audio_queue = queue.Queue()

        # Processing state
        self.processing_lock = threading.Lock()

        print(f"Optimized Whisper model loaded on {config.device}")
        print(f"Model size: {config.model_size}, Language: {config.language}")

    def transcribe_with_options(self, audio_data, prompt: str = ""):
        """
        Transcribe audio with custom options
        :param audio_data: Audio data as numpy array
        :param prompt: Optional prompt to guide transcription
        :return: Transcribed text
        """
        with self.processing_lock:
            # Convert to tensor and move to device
            audio_tensor = torch.from_numpy(audio_data).to(self.config.device)

            # Prepare options for transcription
            options = {
                "task": self.config.task,
                "language": self.config.language,
                "beam_size": self.config.beam_size,
                "best_of": self.config.best_of,
                "patience": self.config.patience,
                "length_penalty": self.config.length_penalty,
                "temperature": self.config.temperature,
                "compression_ratio_threshold": self.config.compression_ratio_threshold,
                "logprob_threshold": self.config.logprob_threshold,
                "no_speech_threshold": self.config.no_speech_threshold,
            }

            # Add prompt if provided
            if prompt:
                options["initial_prompt"] = prompt

            # Transcribe with options
            result = self.model.transcribe(audio_tensor, **options)

            return result["text"]

    def transcribe_in_chunks(self, audio_data, chunk_duration=30.0):
        """
        Transcribe long audio by splitting into chunks
        :param audio_data: Long audio data as numpy array
        :param chunk_duration: Duration of each chunk in seconds
        :return: Complete transcribed text
        """
        sample_rate = self.sample_rate
        chunk_size = int(sample_rate * chunk_duration)

        full_text = ""
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            chunk_text = self.transcribe_audio(chunk)
            full_text += chunk_text + " "

        return full_text.strip()

    def get_audio_energy(self, audio_data):
        """Calculate audio energy for voice activity detection"""
        return np.mean(np.abs(audio_data))

    def is_speech_present(self, audio_data, threshold=0.01):
        """Simple voice activity detection"""
        return self.get_audio_energy(audio_data) > threshold
```

## Voice Command Processing Pipeline

### Natural Language Understanding for Robotics

```python
import re
from typing import Dict, List, Tuple, Optional
import json

class VoiceCommandProcessor:
    def __init__(self):
        # Define command patterns for humanoid robot actions
        self.command_patterns = {
            # Movement commands
            "move_forward": [r"move forward", r"go forward", r"step forward", r"walk forward"],
            "move_backward": [r"move backward", r"go backward", r"step backward", r"walk backward"],
            "turn_left": [r"turn left", r"rotate left", r"turn anticlockwise", r"pivot left"],
            "turn_right": [r"turn right", r"rotate right", r"turn clockwise", r"pivot right"],
            "stop": [r"stop", r"halt", r"freeze", r"pause"],

            # Navigation commands
            "go_to_location": [r"go to (.+)", r"move to (.+)", r"navigate to (.+)", r"walk to (.+)"],
            "follow_me": [r"follow me", r"follow", r"come with me"],
            "wait_here": [r"wait here", r"stay here", r"wait", r"stay"],

            # Interaction commands
            "greet": [r"say hello", r"hello", r"greet", r"introduce yourself"],
            "wave": [r"wave", r"wave hello", r"raise your hand"],
            "nod": [r"nod", r"nod your head", r"agree"],
            "shake_head": [r"shake your head", r"disagree", r"no"],

            # Object manipulation
            "pick_up": [r"pick up (.+)", r"grab (.+)", r"take (.+)", r"lift (.+)"],
            "put_down": [r"put down (.+)", r"release (.+)", r"drop (.+)"],
            "point_to": [r"point to (.+)", r"show (.+)", r"indicate (.+)"],

            # Information requests
            "tell_time": [r"what time is it", r"tell me the time", r"what's the time"],
            "tell_date": [r"what date is it", r"tell me the date", r"what's the date"],
            "weather": [r"what's the weather", r"tell me the weather", r"weather forecast"],
        }

        # Location aliases
        self.location_aliases = {
            "kitchen": ["kitchen", "cooking area", "food area"],
            "living_room": ["living room", "livingroom", "sitting room", "lounge"],
            "bedroom": ["bedroom", "sleeping area", "bed area"],
            "bathroom": ["bathroom", "toilet", "restroom", "washroom"],
            "office": ["office", "study", "workroom", "desk area"],
        }

    def parse_command(self, text: str) -> Dict:
        """
        Parse voice command and extract action and parameters
        :param text: Transcribed text from speech
        :return: Dictionary with action and parameters
        """
        text_lower = text.lower().strip()

        # Check each command pattern
        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    result = {"action": action}

                    # Extract parameters if present
                    if match.groups():
                        result["parameters"] = list(match.groups())

                    # Special handling for location commands
                    if action == "go_to_location" and "parameters" in result:
                        location = self.normalize_location(result["parameters"][0])
                        result["parameters"] = [location]

                    return result

        # If no pattern matches, return unknown command
        return {"action": "unknown", "text": text_lower}

    def normalize_location(self, location: str) -> str:
        """Normalize location names to standard format"""
        location_lower = location.lower().strip()

        # Check for location aliases
        for standard_name, aliases in self.location_aliases.items():
            for alias in aliases:
                if alias in location_lower:
                    return standard_name

        # If not found, return as is
        return location_lower

    def validate_command(self, command: Dict) -> bool:
        """Validate if command is safe and executable"""
        action = command.get("action", "")

        # Define safe actions (whitelist approach)
        safe_actions = [
            "move_forward", "move_backward", "turn_left", "turn_right", "stop",
            "go_to_location", "follow_me", "wait_here", "greet", "wave", "nod",
            "shake_head", "pick_up", "put_down", "point_to", "tell_time",
            "tell_date", "weather", "unknown"
        ]

        return action in safe_actions

    def extract_entities(self, text: str) -> Dict:
        """Extract named entities from text"""
        entities = {}

        # Extract numbers
        numbers = re.findall(r'\d+', text)
        if numbers:
            entities["numbers"] = [int(n) for n in numbers]

        # Extract object names (simple approach)
        object_patterns = [
            r"pick up the (\w+)",
            r"grab the (\w+)",
            r"take the (\w+)",
            r"point to the (\w+)",
        ]

        for pattern in object_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                entities["objects"] = matches

        return entities

# Example usage of command processor
def test_command_processor():
    processor = VoiceCommandProcessor()

    test_commands = [
        "Move forward 2 meters",
        "Go to the kitchen",
        "Turn left and wait",
        "Pick up the red ball",
        "Tell me the time",
        "Wave hello to everyone"
    ]

    for command in test_commands:
        parsed = processor.parse_command(command)
        print(f"Command: '{command}' -> {parsed}")
```

## Integration with Humanoid Robot Control

### Voice-to-Action Bridge

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import threading
import queue
import time
import numpy as np

class VoiceToActionBridge(Node):
    def __init__(self):
        super().__init__('voice_to_action_bridge')

        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)

        # Subscribers
        self.voice_sub = self.create_subscription(
            String, '/transcribed_voice', self.voice_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Internal state
        self.current_joint_positions = np.zeros(28)  # Example: 28 DOF humanoid
        self.command_queue = queue.Queue()
        self.action_lock = threading.Lock()

        # Command processor
        self.command_processor = VoiceCommandProcessor()

        # Timer for processing commands
        self.command_timer = self.create_timer(0.1, self.process_commands)

        self.get_logger().info('Voice-to-Action Bridge initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        if len(msg.position) == len(self.current_joint_positions):
            self.current_joint_positions = np.array(msg.position)

    def voice_callback(self, msg):
        """Process transcribed voice command"""
        text = msg.data
        self.get_logger().info(f'Received voice command: {text}')

        # Parse the command
        command = self.command_processor.parse_command(text)

        if self.command_processor.validate_command(command):
            # Add to command queue for processing
            self.command_queue.put(command)
            self.get_logger().info(f'Command added to queue: {command["action"]}')
        else:
            self.get_logger().warn(f'Invalid command: {command}')

    def process_commands(self):
        """Process commands from the queue"""
        while not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                self.execute_command(command)
            except queue.Empty:
                break

    def execute_command(self, command):
        """Execute the parsed command on the robot"""
        action = command.get("action")

        with self.action_lock:
            if action == "move_forward":
                self.move_forward()
            elif action == "move_backward":
                self.move_backward()
            elif action == "turn_left":
                self.turn_left()
            elif action == "turn_right":
                self.turn_right()
            elif action == "stop":
                self.stop_robot()
            elif action == "go_to_location":
                if "parameters" in command:
                    location = command["parameters"][0]
                    self.navigate_to_location(location)
            elif action == "greet":
                self.greet()
            elif action == "wave":
                self.wave()
            elif action == "nod":
                self.nod()
            elif action == "shake_head":
                self.shake_head()
            elif action == "tell_time":
                self.tell_time()
            elif action == "unknown":
                self.acknowledge_unknown_command(command.get("text", ""))
            else:
                self.get_logger().warn(f'Unknown action: {action}')

    def move_forward(self):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = 0.3  # 0.3 m/s forward
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Moving forward')

    def move_backward(self):
        """Move robot backward"""
        cmd = Twist()
        cmd.linear.x = -0.3  # 0.3 m/s backward
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Moving backward')

    def turn_left(self):
        """Turn robot left"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # 0.5 rad/s counter-clockwise
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Turning left')

    def turn_right(self):
        """Turn robot right"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5  # 0.5 rad/s clockwise
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Turning right')

    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.get_logger().info('Robot stopped')

    def navigate_to_location(self, location):
        """Navigate to specified location"""
        # This would integrate with navigation stack
        # For now, just log the intent
        self.get_logger().info(f'Navigating to {location}')

        # Example: Publish goal to navigation system
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'

        # Set goal based on location (would need location mapping)
        if location == 'kitchen':
            goal_msg.pose.position.x = 5.0
            goal_msg.pose.position.y = 3.0
        elif location == 'living_room':
            goal_msg.pose.position.x = 2.0
            goal_msg.pose.position.y = 1.0
        # Add more locations as needed

        # In a real implementation, you would publish to navigation goal topic
        # self.nav_goal_pub.publish(goal_msg)

    def greet(self):
        """Perform greeting action"""
        # Publish speech
        speech_msg = String()
        speech_msg.data = "Hello! It's nice to meet you."
        self.speech_pub.publish(speech_msg)

        # Perform greeting motion (simplified)
        self.perform_greeting_motion()

    def wave(self):
        """Perform waving motion"""
        # This would send specific joint commands for waving
        # For demonstration, we'll just log
        self.get_logger().info('Performing waving motion')

        # Example joint commands for waving (simplified)
        joint_cmd = Float64MultiArray()
        # In a real implementation, this would contain specific joint angles
        # for the waving motion
        joint_cmd.data = self.current_joint_positions.tolist()
        self.joint_cmd_pub.publish(joint_cmd)

    def nod(self):
        """Perform nodding motion"""
        self.get_logger().info('Performing nodding motion')
        # Send joint commands for nodding motion

    def shake_head(self):
        """Perform head shaking motion"""
        self.get_logger().info('Performing head shaking motion')
        # Send joint commands for head shaking motion

    def tell_time(self):
        """Tell the current time"""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M")
        speech_msg = String()
        speech_msg.data = f"The current time is {current_time}"
        self.speech_pub.publish(speech_msg)

    def acknowledge_unknown_command(self, text):
        """Acknowledge unknown command"""
        speech_msg = String()
        speech_msg.data = f"I'm sorry, I didn't understand '{text}'. Could you please repeat that?"
        self.speech_pub.publish(speech_msg)

def main(args=None):
    rclpy.init(args=args)
    voice_bridge = VoiceToActionBridge()

    try:
        rclpy.spin(voice_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        voice_bridge.destroy_node()
        rclpy.shutdown()
```

## Real-time Voice Recognition Node

### ROS 2 Node for Real-time Voice Recognition

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import torch
import pyaudio
import numpy as np
import threading
import queue
import time

class RealTimeWhisperNode(Node):
    def __init__(self):
        super().__init__('real_time_whisper')

        # Publisher for transcribed text
        self.transcript_pub = self.create_publisher(String, '/transcribed_voice', 10)

        # Configuration
        self.model_size = "base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_format = pyaudio.paFloat32
        self.channels = 1

        # Initialize Whisper model
        self.get_logger().info(f'Loading Whisper model ({self.model_size}) on {self.device}...')
        self.model = whisper.load_model(self.model_size, device=self.device)
        self.get_logger().info('Whisper model loaded successfully')

        # Audio recording setup
        self.audio = pyaudio.PyAudio()
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()

        # Voice activity detection
        self.voice_threshold = 0.01
        self.silence_duration_threshold = 1.0  # seconds
        self.last_voice_time = time.time()

        # Start audio processing
        self.start_audio_recording()

        self.get_logger().info('Real-time Whisper node initialized')

    def start_audio_recording(self):
        """Start audio recording in a separate thread"""
        def audio_callback(in_data, frame_count, time_info, status):
            audio_array = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_array)
            return (in_data, pyaudio.paContinue)

        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
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
        # Buffer for collecting audio segments
        audio_buffer = np.array([])
        min_speech_duration = 0.5  # Minimum speech duration in seconds
        max_silence_duration = 0.8  # Maximum silence before processing

        while self.is_listening:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)

                # Check if voice is present in this chunk
                voice_present = self.is_voice_present(chunk)

                if voice_present:
                    # Add to buffer
                    audio_buffer = np.concatenate([audio_buffer, chunk])
                    self.last_voice_time = time.time()
                else:
                    # Check if we have accumulated enough audio and silence duration
                    if len(audio_buffer) > 0:
                        time_since_voice = time.time() - self.last_voice_time
                        min_samples = int(min_speech_duration * self.sample_rate)
                        max_silence_samples = int(max_silence_duration * self.sample_rate)

                        # Process if we have enough speech or silence duration exceeded
                        if (len(audio_buffer) >= min_samples and
                            time_since_voice >= max_silence_duration):

                            # Transcribe the accumulated audio
                            self.transcribe_and_publish(audio_buffer)

                            # Reset buffer
                            audio_buffer = np.array([])

                # Limit buffer size to prevent excessive memory usage
                max_buffer_duration = 5.0  # seconds
                max_samples = int(max_buffer_duration * self.sample_rate)
                if len(audio_buffer) > max_samples:
                    # Keep only the last portion of audio
                    audio_buffer = audio_buffer[-max_samples:]

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in audio processing: {e}')
                continue

    def transcribe_and_publish(self, audio_data):
        """Transcribe audio and publish result"""
        try:
            # Convert to tensor and move to device
            audio_tensor = torch.from_numpy(audio_data).to(self.device)

            # Transcribe
            result = self.model.transcribe(audio_tensor)
            text = result["text"].strip()

            # Only publish if there's meaningful text
            if text and len(text) > 2:  # At least 2 characters
                # Publish the transcribed text
                transcript_msg = String()
                transcript_msg.data = text
                self.transcript_pub.publish(transcript_msg)

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
    whisper_node = RealTimeWhisperNode()

    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_node.destroy_node()
        rclpy.shutdown()
```

## Performance Optimization

### Optimized Whisper Processing for Edge Devices

```python
import whisper
import torch
import numpy as np
import threading
import queue
import time
from collections import deque
import gc

class OptimizedWhisperEdge:
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_size = model_size

        # Load model with optimizations
        self.model = whisper.load_model(
            model_size,
            device=device,
            in_memory=True
        ).eval()  # Set to evaluation mode

        # Use half precision if using CUDA for faster inference
        if device == "cuda":
            self.model = self.model.half()

        # Audio parameters
        self.sample_rate = 16000
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Performance optimization parameters
        self.batch_size = 1  # Process one at a time for real-time
        self.max_buffer_duration = 5.0  # Maximum seconds to buffer
        self.min_speech_duration = 0.5  # Minimum speech to process

        # Threading
        self.processing_thread = None
        self.is_running = False

        # Memory management
        self.energy_history = deque(maxlen=10)  # Track recent energy levels

        print(f"Optimized Whisper model loaded on {device} with {model_size} size")

    def process_audio_batch(self, audio_segments):
        """Process a batch of audio segments"""
        results = []

        with torch.no_grad():  # Disable gradient computation for inference
            for audio_data in audio_segments:
                try:
                    # Convert to tensor
                    audio_tensor = torch.from_numpy(audio_data).to(self.device)

                    # Ensure tensor is in correct format
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)

                    # Transcribe
                    result = self.model.transcribe(audio_tensor.float() if self.device == "cuda" else audio_tensor)
                    results.append(result["text"])

                except Exception as e:
                    print(f"Error processing audio segment: {e}")
                    results.append("")

        return results

    def calculate_audio_energy(self, audio_data):
        """Calculate audio energy with smoothing"""
        energy = np.mean(np.abs(audio_data))

        # Add to history for smoothing
        self.energy_history.append(energy)

        # Return smoothed energy
        if len(self.energy_history) > 0:
            return sum(self.energy_history) / len(self.energy_history)
        return energy

    def optimize_memory(self):
        """Optimize memory usage"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def set_model_offload(self, offload_size):
        """Set model offloading to save memory"""
        # This is a simplified approach
        # In practice, you'd use more sophisticated memory management
        pass

# Example usage for resource-constrained environments
def run_optimized_whisper():
    # Use smaller model for edge devices
    recognizer = OptimizedWhisperEdge(model_size="tiny")  # Use "tiny" for edge

    # Process audio with memory optimization
    # Implementation would continue with real-time processing
    pass
```

## Integration with ROS 2 Launch System

### Launch File for Voice-to-Action System

```python
# launch/voice_to_action.launch.py
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

    # Real-time Whisper node
    whisper_node = Node(
        package='voice_to_action',
        executable='real_time_whisper',
        name='real_time_whisper',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'},
            {'model_size': model_size}
        ],
        output='screen'
    )

    # Voice-to-action bridge
    voice_action_bridge = Node(
        package='voice_to_action',
        executable='voice_to_action_bridge',
        name='voice_to_action_bridge',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    # Audio input node (if needed)
    audio_input_node = Node(
        package='audio_common',
        executable='audio_input_node',
        name='audio_input',
        parameters=[
            {'use_sim_time': use_sim_time == 'true'}
        ],
        output='screen'
    )

    return [whisper_node, voice_action_bridge, audio_input_node]

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

## Best Practices and Considerations

### Voice Recognition in Noisy Environments

```python
class RobustVoiceRecognizer:
    def __init__(self):
        # Noise reduction parameters
        self.noise_threshold = 0.005
        self.snr_threshold = 10  # Signal-to-noise ratio
        self.background_noise_samples = []
        self.max_background_samples = 100

    def estimate_background_noise(self, audio_data):
        """Estimate background noise level"""
        # Simple approach: take minimum energy from recent samples
        energy = np.mean(np.abs(audio_data))
        self.background_noise_samples.append(energy)

        if len(self.background_noise_samples) > self.max_background_samples:
            self.background_noise_samples.pop(0)

        if self.background_noise_samples:
            return min(self.background_noise_samples)
        return 0.0

    def apply_noise_reduction(self, audio_data):
        """Apply simple noise reduction"""
        background_level = self.estimate_background_noise(audio_data)

        # Simple spectral subtraction approach
        audio_energy = np.abs(audio_data)
        reduced_energy = np.maximum(audio_energy - background_level, 0)

        # Preserve original sign
        sign = np.sign(audio_data)
        denoised_audio = sign * reduced_energy

        return denoised_audio

    def validate_recognition_confidence(self, transcription_result):
        """Validate recognition confidence"""
        # Check if transcription seems reasonable
        text = transcription_result.get("text", "")

        # Basic validation: check for common speech patterns
        if len(text.strip()) < 2:
            return False, "Too short"

        # Check for common non-speech results
        non_speech_patterns = ["thank you for watching", "this video", "subscribe"]
        for pattern in non_speech_patterns:
            if pattern in text.lower():
                return False, "Non-speech content detected"

        return True, "Valid transcription"
```

## Troubleshooting and Common Issues

### Common Voice Recognition Issues and Solutions

1. **High Latency**:
   - Use smaller Whisper models for real-time applications
   - Optimize audio processing pipeline
   - Use GPU acceleration when available

2. **Poor Recognition Accuracy**:
   - Ensure good audio quality and minimal background noise
   - Use directional microphones
   - Calibrate the voice activity detection threshold

3. **Memory Issues**:
   - Use smaller model sizes on resource-constrained devices
   - Implement proper memory management and garbage collection
   - Process audio in smaller chunks

4. **Real-time Performance**:
   - Use threading for audio capture and processing
   - Implement proper buffering strategies
   - Optimize the audio pipeline for minimal latency

## Summary

In this chapter, you learned:
- How to install and configure OpenAI Whisper for robotics applications
- Techniques for real-time voice recognition and processing
- How to parse voice commands and map them to robot actions
- Integration with ROS 2 for humanoid robot control
- Performance optimization strategies for edge devices
- Best practices for robust voice recognition in noisy environments
- Troubleshooting techniques for common issues

Voice-to-action systems enable natural human-robot interaction, making humanoid robots more accessible and intuitive to use. By leveraging OpenAI Whisper's advanced speech recognition capabilities, you can create sophisticated voice interfaces for your humanoid robotics applications.

---
**Continue to [Chapter 2: Cognitive Planning with LLMs](/docs/module-4-vla-humanoids/chapter-2-cognitive-planning)**