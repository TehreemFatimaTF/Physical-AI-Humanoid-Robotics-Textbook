---
sidebar_label: 'Chapter 2: Cognitive Planning with LLMs'
title: 'Chapter 2: Cognitive Planning with LLMs'
description: 'Understanding cognitive planning using Large Language Models for humanoid robotics'
slug: '/module-4-vla-humanoids/chapter-2-cognitive-planning'
difficulty: 'advanced'
requiredHardware: ['computer', 'nvidia_gpu', 'high_memory']
recommendedHardware: ['jetson_orin', 'rtx_4090', '32gb_ram']
---

# Chapter 2: Cognitive Planning with LLMs

Large Language Models (LLMs) have revolutionized artificial intelligence by providing sophisticated reasoning, planning, and decision-making capabilities. For humanoid robots, LLMs enable high-level cognitive planning that bridges the gap between natural language commands and complex robotic behaviors. This chapter explores how to integrate LLMs for cognitive planning in humanoid robotics applications.

## Introduction to Cognitive Planning with LLMs

Cognitive planning involves high-level reasoning to achieve complex goals. In humanoid robotics, this includes:

- **Task Decomposition**: Breaking complex tasks into manageable subtasks
- **Spatial Reasoning**: Understanding and navigating 3D environments
- **Temporal Planning**: Sequencing actions over time
- **Multi-step Reasoning**: Planning complex sequences of actions
- **Context Awareness**: Understanding the environment and situation
- **Adaptive Planning**: Adjusting plans based on feedback and changing conditions

### Why LLMs for Cognitive Planning?

LLMs offer several advantages for cognitive planning in robotics:

- **Natural Language Understanding**: Direct interpretation of human commands
- **Common Sense Reasoning**: Knowledge about the physical world
- **Symbolic Reasoning**: Logical inference and planning
- **Contextual Understanding**: Awareness of situational context
- **Generalization**: Ability to handle novel situations
- **Multi-modal Integration**: Combining text, vision, and action

## LLM Integration Architecture

### Cognitive Planning Pipeline

```
Human Command (Natural Language)
         ↓
    LLM Parser (Intent Recognition)
         ↓
    Task Decomposition
         ↓
    Subtask Planning
         ↓
    Action Selection
         ↓
    Execution Layer
         ↓
    Feedback Integration
```

### LLM Selection for Robotics

Different LLMs have different strengths for robotics applications:

1. **OpenAI GPT-4**: High reasoning capability, good for complex planning
2. **Anthropic Claude**: Strong reasoning with good safety features
3. **Open Source Models**: Hugging Face transformers (e.g., Llama, Mistral)
4. **Specialized Models**: Function-calling optimized models

## Setting Up LLM Integration

### Prerequisites and Installation

```bash
# Install required packages
pip install openai anthropic transformers torch accelerate
pip install python-dotenv openai-function-calling
pip install langchain langchain-community langchain-openai

# For local models
pip install llama-cpp-python
pip install vllm  # For optimized inference
```

### Environment Configuration

```python
# .env file
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

### Basic LLM Integration

```python
import os
from dotenv import load_dotenv
import openai
import anthropic
from typing import Dict, List, Optional, Any
import json
import asyncio

load_dotenv()

class LLMPlanner:
    def __init__(self, model_type: str = "openai", model_name: str = "gpt-4"):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif model_type == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif model_type == "local":
            # Setup for local models (e.g., using transformers)
            from transformers import pipeline
            self.pipeline = pipeline("text-generation", model=model_name)

    def generate_plan(self, task_description: str, context: Dict = None) -> Dict:
        """
        Generate a plan for the given task using LLM
        :param task_description: Natural language description of the task
        :param context: Additional context information (environment, capabilities, etc.)
        :return: Structured plan with steps and parameters
        """
        if self.model_type == "openai":
            return self._generate_plan_openai(task_description, context)
        elif self.model_type == "anthropic":
            return self._generate_plan_anthropic(task_description, context)
        elif self.model_type == "local":
            return self._generate_plan_local(task_description, context)

    def _generate_plan_openai(self, task_description: str, context: Dict) -> Dict:
        """Generate plan using OpenAI API"""
        system_prompt = """
        You are a cognitive planning assistant for a humanoid robot. Your role is to break down complex tasks into executable steps that the robot can perform.

        Consider the following capabilities of the humanoid robot:
        - Navigation: Can move in 2D space, avoid obstacles
        - Manipulation: Can pick up, move, and place objects
        - Interaction: Can speak, gesture, recognize faces
        - Perception: Can see, hear, and understand its environment

        Respond with a JSON object containing:
        - "task": The original task
        - "steps": Array of steps, each with "action", "parameters", "description"
        - "estimated_duration": Estimated time in seconds
        - "potential_issues": List of potential problems
        """

        user_prompt = f"Task: {task_description}\nContext: {json.dumps(context)}"

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def _generate_plan_anthropic(self, task_description: str, context: Dict) -> Dict:
        """Generate plan using Anthropic API"""
        system_prompt = """
        You are a cognitive planning assistant for a humanoid robot. Your role is to break down complex tasks into executable steps that the robot can perform.

        Consider the following capabilities of the humanoid robot:
        - Navigation: Can move in 2D space, avoid obstacles
        - Manipulation: Can pick up, move, and place objects
        - Interaction: Can speak, gesture, recognize faces
        - Perception: Can see, hear, and understand its environment
        """

        user_prompt = f"Task: {task_description}\nContext: {json.dumps(context)}"

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return json.loads(response.content[0].text)

    def _generate_plan_local(self, task_description: str, context: Dict) -> Dict:
        """Generate plan using local model"""
        prompt = f"""
        [INST] <<SYS>>
        You are a cognitive planning assistant for a humanoid robot. Your role is to break down complex tasks into executable steps that the robot can perform.

        Consider the following capabilities of the humanoid robot:
        - Navigation: Can move in 2D space, avoid obstacles
        - Manipulation: Can pick up, move, and place objects
        - Interaction: Can speak, gesture, recognize faces
        - Perception: Can see, hear, and understand its environment
        <</SYS>>

        Task: {task_description}
        Context: {json.dumps(context)}

        Respond with a JSON object containing:
        - "task": The original task
        - "steps": Array of steps, each with "action", "parameters", "description"
        - "estimated_duration": Estimated time in seconds
        - "potential_issues": List of potential problems
        [/INST]
        """

        result = self.pipeline(
            prompt,
            max_length=1000,
            temperature=0.3,
            do_sample=True
        )

        # Extract JSON from response (this is simplified)
        response_text = result[0]['generated_text']
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]

        return json.loads(json_str)
```

## Advanced Planning with Function Calling

### LLM Function Calling for Robotics

```python
import json
from typing import Dict, List, Any
import openai
from dataclasses import dataclass
from enum import Enum

class RobotAction(Enum):
    MOVE_TO = "move_to"
    GRAB_OBJECT = "grab_object"
    PLACE_OBJECT = "place_object"
    SPEAK = "speak"
    GREET = "greet"
    WAIT = "wait"
    NAVIGATE_TO = "navigate_to"
    DETECT_OBJECT = "detect_object"
    FOLLOW_PERSON = "follow_person"

@dataclass
class ActionStep:
    action: RobotAction
    parameters: Dict[str, Any]
    description: str
    estimated_duration: float

class FunctionCallingPlanner:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.available_functions = {
            "move_to_location": {
                "name": "move_to_location",
                "description": "Move the robot to a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The target location"},
                        "x": {"type": "number", "description": "X coordinate"},
                        "y": {"type": "number", "description": "Y coordinate"},
                        "orientation": {"type": "number", "description": "Orientation in radians"}
                    },
                    "required": ["location"]
                }
            },
            "grab_object": {
                "name": "grab_object",
                "description": "Grab an object with the robot's manipulator",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {"type": "string", "description": "Name of the object to grab"},
                        "object_type": {"type": "string", "description": "Type of object"},
                        "location": {"type": "string", "description": "Location of the object"}
                    },
                    "required": ["object_name"]
                }
            },
            "place_object": {
                "name": "place_object",
                "description": "Place an object at a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {"type": "string", "description": "Name of the object to place"},
                        "location": {"type": "string", "description": "Target location for placing"},
                        "x": {"type": "number", "description": "X coordinate"},
                        "y": {"type": "number", "description": "Y coordinate"}
                    },
                    "required": ["object_name", "location"]
                }
            },
            "speak_text": {
                "name": "speak_text",
                "description": "Make the robot speak a text message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The message to speak"},
                        "voice_type": {"type": "string", "description": "Type of voice (friendly, formal, etc.)"}
                    },
                    "required": ["message"]
                }
            },
            "detect_object": {
                "name": "detect_object",
                "description": "Detect objects in the robot's environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_type": {"type": "string", "description": "Type of object to detect"},
                        "search_area": {"type": "string", "description": "Area to search in"}
                    }
                }
            }
        }

    def generate_plan_with_functions(self, task_description: str, context: Dict = None) -> List[ActionStep]:
        """
        Generate a plan using LLM function calling
        """
        messages = [
            {
                "role": "system",
                "content": """You are a cognitive planning assistant for a humanoid robot.
                Your role is to break down complex tasks into a sequence of specific actions
                that the robot can execute. Use the available functions to create a detailed plan."""
            },
            {
                "role": "user",
                "content": f"Task: {task_description}\nContext: {json.dumps(context)}"
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            functions=list(self.available_functions.values()),
            function_call="auto",
            temperature=0.3
        )

        # Process the response
        plan = []
        response_message = response.choices[0].message

        if response_message.get("function_calls"):
            for call in response_message["function_calls"]:
                function_name = call["name"]
                function_args = json.loads(call["arguments"])

                action_step = self._convert_function_call_to_action(function_name, function_args)
                plan.append(action_step)
        elif response_message.get("function_call"):
            # Handle single function call
            call = response_message["function_call"]
            function_name = call["name"]
            function_args = json.loads(call["arguments"])

            action_step = self._convert_function_call_to_action(function_name, function_args)
            plan.append(action_step)

        return plan

    def _convert_function_call_to_action(self, function_name: str, function_args: Dict) -> ActionStep:
        """Convert function call to robot action"""
        if function_name == "move_to_location":
            return ActionStep(
                action=RobotAction.NAVIGATE_TO,
                parameters=function_args,
                description=f"Navigate to {function_args.get('location', 'target location')}",
                estimated_duration=10.0
            )
        elif function_name == "grab_object":
            return ActionStep(
                action=RobotAction.GRAB_OBJECT,
                parameters=function_args,
                description=f"Grab {function_args.get('object_name')}",
                estimated_duration=5.0
            )
        elif function_name == "place_object":
            return ActionStep(
                action=RobotAction.PLACE_OBJECT,
                parameters=function_args,
                description=f"Place {function_args.get('object_name')} at {function_args.get('location')}",
                estimated_duration=5.0
            )
        elif function_name == "speak_text":
            return ActionStep(
                action=RobotAction.SPEAK,
                parameters=function_args,
                description=f"Speak: {function_args.get('message', '')}",
                estimated_duration=2.0
            )
        elif function_name == "detect_object":
            return ActionStep(
                action=RobotAction.DETECT_OBJECT,
                parameters=function_args,
                description=f"Detect {function_args.get('object_type', 'object')}",
                estimated_duration=3.0
            )
        else:
            return ActionStep(
                action=RobotAction.WAIT,
                parameters={},
                description="Unknown action",
                estimated_duration=1.0
            )
```

## Context-Aware Planning

### Environmental Context Integration

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class RobotCapabilities:
    navigation: bool = True
    manipulation: bool = True
    speech: bool = True
    vision: bool = True
    grasping_range: float = 1.0  # meters
    max_speed: float = 0.5  # m/s
    battery_level: float = 1.0  # 0-1

@dataclass
class EnvironmentState:
    locations: Dict[str, Dict]  # location_name: {x, y, description}
    objects: Dict[str, Dict]  # object_name: {location, type, properties}
    people: List[Dict]  # List of people with positions
    obstacles: List[Dict]  # Obstacle locations and properties
    current_robot_position: Dict[str, float]  # {x, y, orientation}

class ContextAwarePlanner:
    def __init__(self):
        self.capabilities = RobotCapabilities()
        self.environment = None
        self.llm_planner = LLMPlanner()

    def update_environment_state(self, env_state: EnvironmentState):
        """Update the current environment state"""
        self.environment = env_state

    def generate_context_aware_plan(self, task_description: str) -> Dict:
        """
        Generate a plan considering current environmental context
        """
        if not self.environment:
            raise ValueError("Environment state not initialized")

        # Create context for the LLM
        context = {
            "robot_capabilities": self._serialize_capabilities(),
            "environment": self._serialize_environment(),
            "current_time": "2024-01-01T12:00:00Z",  # In practice, use current time
            "battery_level": self.capabilities.battery_level
        }

        # Generate plan with context
        plan = self.llm_planner.generate_plan(task_description, context)

        # Validate and optimize the plan based on context
        optimized_plan = self._validate_and_optimize_plan(plan)

        return optimized_plan

    def _serialize_capabilities(self) -> Dict:
        """Serialize robot capabilities for LLM context"""
        return {
            "navigation": self.capabilities.navigation,
            "manipulation": self.capabilities.manipulation,
            "speech": self.capabilities.speech,
            "vision": self.capabilities.vision,
            "grasping_range_meters": self.capabilities.grasping_range,
            "max_speed_mps": self.capabilities.max_speed,
            "battery_level": self.capabilities.battery_level
        }

    def _serialize_environment(self) -> Dict:
        """Serialize environment state for LLM context"""
        if not self.environment:
            return {}

        return {
            "locations": self.environment.locations,
            "objects": self.environment.objects,
            "people_count": len(self.environment.people),
            "obstacle_count": len(self.environment.obstacles),
            "robot_position": self.environment.current_robot_position
        }

    def _validate_and_optimize_plan(self, plan: Dict) -> Dict:
        """Validate and optimize the plan based on current context"""
        validated_steps = []

        for step in plan.get("steps", []):
            # Validate that the robot can perform this action
            if self._is_action_valid(step):
                # Optimize based on current position
                optimized_step = self._optimize_step_for_context(step)
                validated_steps.append(optimized_step)

        plan["steps"] = validated_steps
        return plan

    def _is_action_valid(self, step: Dict) -> bool:
        """Check if the robot can perform the given action"""
        action = step.get("action", "")

        # Check if robot has required capabilities
        if "navigate" in action and not self.capabilities.navigation:
            return False
        if "grab" in action and not self.capabilities.manipulation:
            return False
        if "speak" in action and not self.capabilities.speech:
            return False

        return True

    def _optimize_step_for_context(self, step: Dict) -> Dict:
        """Optimize a step based on current context"""
        # Add estimated time based on distance if it's a navigation step
        if step.get("action") == "navigate_to":
            target_location = step.get("parameters", {}).get("location")
            if target_location and self.environment:
                distance = self._calculate_distance_to_location(target_location)
                estimated_time = distance / self.capabilities.max_speed
                step["estimated_duration"] = estimated_time

        return step

    def _calculate_distance_to_location(self, location_name: str) -> float:
        """Calculate distance from current position to target location"""
        if not self.environment or location_name not in self.environment.locations:
            return float('inf')

        target_pos = self.environment.locations[location_name]
        current_pos = self.environment.current_robot_position

        # Calculate Euclidean distance
        dx = target_pos.get('x', 0) - current_pos.get('x', 0)
        dy = target_pos.get('y', 0) - current_pos.get('y', 0)

        return (dx**2 + dy**2)**0.5
```

## ROS 2 Integration for Cognitive Planning

### Cognitive Planning Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.qos import QoSProfile, qos_profile_services_default
import json
import asyncio
from typing import Optional

class CognitivePlanningActionServer(Node):
    def __init__(self):
        super().__init__('cognitive_planning_server')

        # Action server for high-level tasks
        self._action_server = ActionServer(
            self,
            ExecuteTask,  # Custom action type
            'execute_task',
            self.execute_task_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers and subscribers
        self.task_plan_pub = self.create_publisher(String, '/task_plan', 10)
        self.current_pose_sub = self.create_subscription(
            PoseStamped, '/current_pose', self.pose_callback, 10)
        self.object_detection_sub = self.create_subscription(
            String, '/detected_objects', self.object_detection_callback, 10)

        # Internal state
        self.current_pose = None
        self.detected_objects = {}
        self.active_task = None
        self.planner = FunctionCallingPlanner()

        # Context manager
        self.context_manager = ContextAwarePlanner()

        self.get_logger().info('Cognitive Planning Server initialized')

    def pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y,
            'z': msg.pose.position.z,
            'orientation': msg.pose.orientation  # In practice, extract yaw
        }

    def object_detection_callback(self, msg):
        """Update detected objects"""
        try:
            objects = json.loads(msg.data)
            self.detected_objects.update(objects)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid JSON in object detection message')

    def goal_callback(self, goal_request):
        """Accept or reject task goals"""
        self.get_logger().info(f'Received task goal: {goal_request.task_description}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject task cancellation"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_task_callback(self, goal_handle):
        """Execute the high-level task"""
        self.get_logger().info('Executing task...')

        feedback_msg = ExecuteTask.Feedback()
        result = ExecuteTask.Result()

        try:
            # Generate plan using LLM
            task_description = goal_handle.request.task_description
            context = self._get_current_context()

            plan = self.planner.generate_plan_with_functions(task_description, context)

            # Publish the plan
            plan_msg = String()
            plan_msg.data = json.dumps([{
                'action': step.action.value,
                'parameters': step.parameters,
                'description': step.description,
                'estimated_duration': step.estimated_duration
            } for step in plan])
            self.task_plan_pub.publish(plan_msg)

            # Execute the plan step by step
            for i, step in enumerate(plan):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = "Task canceled"
                    return result

                # Update feedback
                feedback_msg.current_action = step.description
                feedback_msg.progress = (i + 1) / len(plan)
                goal_handle.publish_feedback(feedback_msg)

                # Execute the step
                success = await self._execute_step(step)

                if not success:
                    result.success = False
                    result.message = f"Failed to execute step: {step.description}"
                    goal_handle.abort()
                    return result

            # Task completed successfully
            result.success = True
            result.message = "Task completed successfully"
            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f'Error executing task: {e}')
            result.success = False
            result.message = f"Error executing task: {str(e)}"
            goal_handle.abort()
            return result

    def _get_current_context(self):
        """Get current context for planning"""
        context = {
            'current_pose': self.current_pose,
            'detected_objects': self.detected_objects,
            'battery_level': 0.8,  # In practice, get from battery topic
            'robot_capabilities': {
                'navigation': True,
                'manipulation': True,
                'speech': True
            }
        }
        return context

    async def _execute_step(self, step):
        """Execute a single step of the plan"""
        try:
            # This would call specific robot action services
            # For example, if step.action is RobotAction.NAVIGATE_TO:
            if step.action == RobotAction.NAVIGATE_TO:
                return await self._navigate_to_location(step.parameters)
            elif step.action == RobotAction.GRAB_OBJECT:
                return await self._grab_object(step.parameters)
            elif step.action == RobotAction.SPEAK:
                return await self._speak_text(step.parameters)
            else:
                self.get_logger().warn(f'Unknown action: {step.action}')
                return False

        except Exception as e:
            self.get_logger().error(f'Error executing step: {e}')
            return False

    async def _navigate_to_location(self, params):
        """Navigate to a specific location"""
        # This would call navigation action server
        # Implementation depends on navigation stack
        self.get_logger().info(f'Navigating to: {params}')
        await asyncio.sleep(2)  # Simulate navigation
        return True

    async def _grab_object(self, params):
        """Grab an object"""
        # This would call manipulation action server
        self.get_logger().info(f'Grabbing object: {params}')
        await asyncio.sleep(3)  # Simulate grasping
        return True

    async def _speak_text(self, params):
        """Make the robot speak"""
        # This would call speech synthesis service
        self.get_logger().info(f'Speaking: {params}')
        await asyncio.sleep(1)  # Simulate speaking
        return True

def main(args=None):
    rclpy.init(args=args)
    cognitive_planner = CognitivePlanningActionServer()

    try:
        rclpy.spin(cognitive_planner)
    except KeyboardInterrupt:
        pass
    finally:
        cognitive_planner.destroy_node()
        rclpy.shutdown()
```

## Memory and Learning Integration

### Experience-Based Planning

```python
import pickle
import os
from datetime import datetime
from typing import List, Dict, Any
import hashlib

class ExperienceMemory:
    def __init__(self, memory_file: str = "robot_experience.pkl"):
        self.memory_file = memory_file
        self.experiences = self._load_memory()
        self.max_memory_size = 1000  # Maximum number of experiences to store

    def _load_memory(self) -> List[Dict]:
        """Load experiences from memory file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading memory: {e}")
                return []
        return []

    def _save_memory(self):
        """Save experiences to memory file"""
        try:
            # Keep only recent experiences if memory is too large
            if len(self.experiences) > self.max_memory_size:
                self.experiences = self.experiences[-self.max_memory_size:]

            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.experiences, f)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def add_experience(self, task: str, plan: List[Dict], outcome: str, context: Dict = None):
        """Add a new experience to memory"""
        experience = {
            'task': task,
            'plan': plan,
            'outcome': outcome,
            'context': context or {},
            'timestamp': datetime.now().isoformat(),
            'success_rate': 1.0 if outcome == 'success' else 0.0
        }

        self.experiences.append(experience)
        self._save_memory()

    def find_similar_experiences(self, task_description: str, top_k: int = 5) -> List[Dict]:
        """Find similar past experiences to the current task"""
        # Simple similarity based on task description
        # In practice, you might use embeddings or more sophisticated similarity measures
        similar_experiences = []

        for exp in self.experiences:
            similarity = self._calculate_task_similarity(task_description, exp['task'])
            if similarity > 0.5:  # Threshold for similarity
                similar_experiences.append((exp, similarity))

        # Sort by similarity and return top_k
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp[0] for exp in similar_experiences[:top_k]]

    def _calculate_task_similarity(self, task1: str, task2: str) -> float:
        """Calculate similarity between two task descriptions"""
        # Simple word overlap similarity
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def get_successful_plan_for_task(self, task_description: str) -> Optional[List[Dict]]:
        """Get a successful plan for a similar task"""
        similar_experiences = self.find_similar_experiences(task_description)

        for exp in similar_experiences:
            if exp['outcome'] == 'success':
                return exp['plan']

        return None

class LearningPlanner:
    def __init__(self):
        self.experience_memory = ExperienceMemory()
        self.llm_planner = FunctionCallingPlanner()

    def generate_adaptive_plan(self, task_description: str, context: Dict = None) -> List[ActionStep]:
        """Generate a plan that adapts based on past experiences"""
        # First, check if we have a successful plan for a similar task
        similar_plan = self.experience_memory.get_successful_plan_for_task(task_description)

        if similar_plan:
            # Adapt the similar plan to current context
            adapted_plan = self._adapt_plan_to_context(similar_plan, context)
            return adapted_plan
        else:
            # Generate new plan using LLM
            plan = self.llm_planner.generate_plan_with_functions(task_description, context)

            # Store the new plan in memory as pending (until executed)
            self.experience_memory.add_experience(
                task=task_description,
                plan=[{
                    'action': step.action.value,
                    'parameters': step.parameters,
                    'description': step.description
                } for step in plan],
                outcome='pending',
                context=context
            )

            return plan

    def _adapt_plan_to_context(self, plan: List[Dict], context: Dict) -> List[ActionStep]:
        """Adapt a plan to the current context"""
        adapted_plan = []

        for step in plan:
            # Adapt step parameters based on context
            adapted_parameters = self._adapt_parameters(step['parameters'], context)

            adapted_step = ActionStep(
                action=RobotAction(step['action']),
                parameters=adapted_parameters,
                description=step['description'],
                estimated_duration=step.get('estimated_duration', 5.0)
            )

            adapted_plan.append(adapted_step)

        return adapted_plan

    def _adapt_parameters(self, parameters: Dict, context: Dict) -> Dict:
        """Adapt action parameters based on context"""
        adapted_params = parameters.copy()

        # Example adaptations:
        # - Adjust navigation coordinates based on current position
        # - Modify object locations based on detection
        # - Update timing based on battery level

        if context and 'current_pose' in context:
            # If this is a navigation task, adjust coordinates relative to current position
            if 'x' in adapted_params and 'y' in adapted_params:
                current_x = context['current_pose'].get('x', 0)
                current_y = context['current_pose'].get('y', 0)

                # In a real system, you'd have more sophisticated coordinate transformation
                adapted_params['x'] += current_x
                adapted_params['y'] += current_y

        return adapted_params

    def update_experience_outcome(self, task_description: str, outcome: str):
        """Update the outcome of a previously stored experience"""
        # Find the most recent experience with this task
        for exp in reversed(self.experiences):
            if exp['task'] == task_description and exp['outcome'] == 'pending':
                exp['outcome'] = outcome
                exp['success_rate'] = 1.0 if outcome == 'success' else 0.0
                self._save_memory()
                break
```

## Safety and Validation

### Safe Planning with LLMs

```python
from enum import Enum
from typing import Dict, List, Tuple
import re

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"

class SafePlanner:
    def __init__(self):
        self.safety_rules = self._load_safety_rules()

    def _load_safety_rules(self) -> List[Dict]:
        """Load safety rules for validation"""
        return [
            {
                "name": "navigation_boundaries",
                "description": "Check if navigation is within safe boundaries",
                "pattern": r"navigate_to|move_to|go_to",
                "validator": self._validate_navigation_boundaries
            },
            {
                "name": "object_interaction",
                "description": "Validate object interaction safety",
                "pattern": r"grab|pick_up|lift|place|put_down",
                "validator": self._validate_object_interaction
            },
            {
                "name": "human_interaction",
                "description": "Validate human interaction safety",
                "pattern": r"greet|follow|approach|help",
                "validator": self._validate_human_interaction
            }
        ]

    def validate_plan(self, plan: List[ActionStep], context: Dict = None) -> Tuple[SafetyLevel, List[str]]:
        """Validate a plan for safety"""
        safety_issues = []
        overall_safety = SafetyLevel.SAFE

        for step in plan:
            step_safety, step_issues = self._validate_step(step, context)

            if step_safety == SafetyLevel.UNSAFE:
                overall_safety = SafetyLevel.UNSAFE
            elif step_safety == SafetyLevel.WARNING and overall_safety == SafetyLevel.SAFE:
                overall_safety = SafetyLevel.WARNING

            safety_issues.extend(step_issues)

        return overall_safety, safety_issues

    def _validate_step(self, step: ActionStep, context: Dict) -> Tuple[SafetyLevel, List[str]]:
        """Validate a single step"""
        issues = []

        # Check against safety rules
        for rule in self.safety_rules:
            if re.search(rule["pattern"], step.action.value, re.IGNORECASE):
                rule_safety, rule_issues = rule["validator"](step, context)

                if rule_safety == SafetyLevel.UNSAFE:
                    return SafetyLevel.UNSAFE, rule_issues
                elif rule_safety == SafetyLevel.WARNING:
                    issues.extend(rule_issues)

        return SafetyLevel.SAFE if not issues else SafetyLevel.WARNING, issues

    def _validate_navigation_boundaries(self, step: ActionStep, context: Dict) -> Tuple[SafetyLevel, List[str]]:
        """Validate navigation safety"""
        issues = []

        # Check if target location is in safe area
        if context and 'safe_zones' in context:
            target_x = step.parameters.get('x')
            target_y = step.parameters.get('y')

            if target_x is not None and target_y is not None:
                if not self._is_in_safe_zone(target_x, target_y, context['safe_zones']):
                    issues.append(f"Navigation target ({target_x}, {target_y}) is outside safe zones")

        # Check if navigation distance is reasonable
        if context and 'current_pose' in context:
            current_x = context['current_pose'].get('x', 0)
            current_y = context['current_pose'].get('y', 0)
            target_x = step.parameters.get('x', current_x)
            target_y = step.parameters.get('y', current_y)

            distance = ((target_x - current_x)**2 + (target_y - current_y)**2)**0.5

            max_distance = context.get('max_navigation_distance', 20.0)  # Default 20m
            if distance > max_distance:
                issues.append(f"Navigation distance ({distance:.2f}m) exceeds maximum ({max_distance}m)")

        return (SafetyLevel.UNSAFE if issues else SafetyLevel.SAFE, issues)

    def _validate_object_interaction(self, step: ActionStep, context: Dict) -> Tuple[SafetyLevel, List[str]]:
        """Validate object interaction safety"""
        issues = []

        object_name = step.parameters.get('object_name', '').lower()

        # Check for dangerous objects
        dangerous_objects = ['knife', 'blade', 'sharp', 'hot', 'fire', 'chemical', 'toxic']
        for dangerous in dangerous_objects:
            if dangerous in object_name:
                issues.append(f"Interaction with potentially dangerous object: {object_name}")

        # Check object weight if available
        if 'objects' in context and object_name in context['objects']:
            obj_info = context['objects'][object_name]
            weight = obj_info.get('weight', 0)
            max_weight = context.get('max_grasp_weight', 5.0)  # Default 5kg

            if weight > max_weight:
                issues.append(f"Object weight ({weight}kg) exceeds maximum grasp weight ({max_weight}kg)")

        return (SafetyLevel.UNSAFE if issues else SafetyLevel.SAFE, issues)

    def _validate_human_interaction(self, step: ActionStep, context: Dict) -> Tuple[SafetyLevel, List[str]]:
        """Validate human interaction safety"""
        issues = []

        # Check personal space violations
        if step.action == RobotAction.NAVIGATE_TO:
            target_x = step.parameters.get('x')
            target_y = step.parameters.get('y')

            if target_x is not None and target_y is not None:
                # Check if approaching too close to people
                if context and 'people' in context:
                    personal_space_radius = 0.5  # 50cm personal space
                    for person in context['people']:
                        person_x = person.get('x', 0)
                        person_y = person.get('y', 0)

                        distance = ((target_x - person_x)**2 + (target_y - person_y)**2)**0.5

                        if distance < personal_space_radius:
                            issues.append(f"Navigation target too close to person (distance: {distance:.2f}m)")

        return (SafetyLevel.UNSAFE if issues else SafetyLevel.SAFE, issues)

    def _is_in_safe_zone(self, x: float, y: float, safe_zones: List[Dict]) -> bool:
        """Check if coordinates are within safe zones"""
        for zone in safe_zones:
            if zone['type'] == 'circle':
                center_x, center_y = zone['center']
                radius = zone['radius']
                distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
                if distance <= radius:
                    return True
            elif zone['type'] == 'rectangle':
                min_x, min_y = zone['min']
                max_x, max_y = zone['max']
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    return True

        return False
```

## Performance Optimization

### Optimized LLM Planning for Real-time Applications

```python
import asyncio
import concurrent.futures
from threading import Lock
import time
from functools import lru_cache

class OptimizedLLMPlanner:
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.cache_lock = Lock()
        self.plan_cache = {}
        self.max_cache_size = 100

    @lru_cache(maxsize=50)
    def _cached_plan_generation(self, task_hash: str, context_hash: str) -> str:
        """Cached plan generation for identical tasks"""
        # This would call the actual LLM
        # Implementation depends on specific LLM provider
        pass

    def generate_plan_async(self, task_description: str, context: Dict = None) -> asyncio.Future:
        """Generate plan asynchronously"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            self._generate_plan_sync,
            task_description,
            context
        )
        return future

    def _generate_plan_sync(self, task_description: str, context: Dict) -> Dict:
        """Synchronous plan generation"""
        # Create hash of task and context for caching
        task_hash = hash(task_description)
        context_hash = hash(str(sorted(context.items())) if context else "")
        cache_key = (task_hash, context_hash)

        # Check cache first
        with self.cache_lock:
            if cache_key in self.plan_cache:
                return self.plan_cache[cache_key]

        # Generate plan using LLM
        start_time = time.time()
        plan = self._call_llm_for_planning(task_description, context)
        generation_time = time.time() - start_time

        # Add to cache if generation was slow (indicating it was complex)
        if generation_time > 1.0:  # Cache if it took more than 1 second
            with self.cache_lock:
                if len(self.plan_cache) >= self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.plan_cache))
                    del self.plan_cache[oldest_key]

                self.plan_cache[cache_key] = plan

        return plan

    def _call_llm_for_planning(self, task_description: str, context: Dict) -> Dict:
        """Call LLM for planning (simplified)"""
        # This would implement the actual LLM call
        # For now, return a mock response
        return {
            "task": task_description,
            "steps": [
                {"action": "speak", "parameters": {"message": "Hello"}, "description": "Greet user"}
            ],
            "estimated_duration": 5.0,
            "potential_issues": []
        }

    def warm_up_model(self):
        """Warm up the model to reduce first-call latency"""
        # Run a simple task to initialize the model
        dummy_task = "Say hello"
        dummy_context = {}
        try:
            self._call_llm_for_planning(dummy_task, dummy_context)
            print("Model warmed up successfully")
        except Exception as e:
            print(f"Error warming up model: {e}")
```

## Best Practices and Guidelines

### Best Practices for LLM-Based Cognitive Planning

1. **Context Management**:
   - Always provide rich environmental context
   - Include robot capabilities and limitations
   - Update context in real-time as environment changes

2. **Safety First**:
   - Implement multiple layers of safety validation
   - Use conservative parameters by default
   - Plan for failure scenarios

3. **Performance Optimization**:
   - Use caching for common tasks
   - Implement asynchronous processing
   - Optimize for edge deployment when needed

4. **Error Handling**:
   - Plan for LLM failures gracefully
   - Implement fallback strategies
   - Log and learn from failures

5. **Human-in-the-Loop**:
   - Allow human override of dangerous plans
   - Provide plan explanations to users
   - Enable plan modification during execution

## Troubleshooting Common Issues

### Common Issues and Solutions

1. **High Latency**:
   - Use smaller models for real-time applications
   - Implement caching for common tasks
   - Optimize network connectivity

2. **Context Window Limitations**:
   - Summarize complex environments
   - Use hierarchical planning
   - Implement memory mechanisms

3. **Safety Concerns**:
   - Implement comprehensive validation
   - Use conservative planning approaches
   - Provide human oversight capabilities

4. **Resource Constraints**:
   - Use quantized models for edge deployment
   - Implement model offloading strategies
   - Optimize for specific hardware

## Summary

In this chapter, you learned:
- How to integrate LLMs for cognitive planning in humanoid robotics
- Architecture patterns for LLM-based planning systems
- Function calling techniques for robotics-specific actions
- Context-aware planning considering environment and capabilities
- ROS 2 integration for cognitive planning systems
- Experience-based learning and adaptation
- Safety validation and risk mitigation strategies
- Performance optimization techniques for real-time applications
- Best practices for deploying LLM-based planning in robotics

Cognitive planning with LLMs enables humanoid robots to understand complex natural language commands and translate them into sophisticated action sequences. By properly integrating LLMs with robotic systems, you can create more intuitive and capable robots that can adapt to novel situations and learn from experience.

---
**Continue to [Chapter 3: Computer Vision for Object Manipulation](/docs/module-4-vla-humanoids/chapter-3-computer-vision)**