---
sidebar_label: 'Chapter 1: Introduction to ROS 2'
title: 'Chapter 1: Introduction to ROS 2'
description: 'Introduction to ROS 2 concepts, architecture, and basic setup for humanoid robotics applications'
slug: '/module-1-ros2/chapter-1-intro-ros2'
difficulty: 'beginner'
requiredHardware: ['ros_system']
recommendedHardware: ['computer']
---

# Chapter 1: Introduction to ROS 2

Welcome to the first chapter of Module 1: The Robotic Nervous System. In this chapter, we'll introduce you to the Robot Operating System 2 (ROS 2), which serves as the middleware for robot control in the Physical AI ecosystem.

## What is ROS 2?

ROS 2 (Robot Operating System 2) is not an actual operating system but rather a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms and environments.

### Key Features of ROS 2

- **Distributed computing**: ROS 2 allows multiple processes (potentially on different machines) to communicate with each other
- **Language independence**: Write nodes in different programming languages (C++, Python, etc.) that can communicate seamlessly
- **Platform independence**: Run on various operating systems and hardware platforms
- **Package management**: Organize code into reusable packages with dependencies
- **Real-time capabilities**: Support for real-time systems (in certain configurations)

## Why ROS 2 for Physical AI?

In the context of Physical AI and humanoid robotics, ROS 2 serves as the "nervous system" of the robot. It enables:

- Communication between different robot components (sensors, actuators, processing units)
- Integration of AI algorithms with physical robot control
- Simulation-to-reality transfer through consistent interfaces
- Collaboration between multiple robots or between robots and humans

## ROS 2 Architecture

The architecture of ROS 2 is built around the concept of nodes, which are individual processes that perform computation. Nodes are organized into packages, which are collections of related functionality.

### The DDS Layer

ROS 2 uses Data Distribution Service (DDS) as its communication layer. DDS is a middleware standard that provides:

- **Publisher/Subscriber model**: Asynchronous communication between nodes
- **Request/Reply model**: Synchronous communication for services
- **Discovery**: Automatic discovery of other nodes on the network
- **Quality of Service (QoS) settings**: Configurable reliability, durability, and performance parameters

## Getting Started with ROS 2

### Installation

For this course, we recommend installing the latest LTS version of ROS 2 (currently Humble Hawksbill for Ubuntu 22.04). If you're using a Jetson platform, you'll need to follow the ARM64 installation instructions.

### Basic Commands

Here are some essential ROS 2 commands you'll use throughout this module:

```bash
# Source the ROS 2 environment
source /opt/ros/humble/setup.bash

# List all active nodes
ros2 node list

# List all active topics
ros2 topic list

# List all active services
ros2 service list
```

## ROS 2 vs. ROS 1

ROS 2 was designed to address several limitations of ROS 1:

- **Real-time support**: ROS 2 has better support for real-time systems
- **Multi-robot systems**: Improved support for multiple robots
- **Security**: Built-in security features
- **DDS-based communication**: More robust communication layer
- **OS platform support**: Broader OS support beyond Linux

## Next Steps

In the next chapter, we'll dive deeper into ROS 2 nodes, topics, and services - the fundamental building blocks of ROS 2 communication.

## Summary

In this chapter, you learned:
- What ROS 2 is and why it's important for Physical AI
- The key features and architecture of ROS 2
- How ROS 2 serves as the "nervous system" for humanoid robots
- Basic ROS 2 commands to get started

---
**Continue to [Chapter 2: ROS 2 Nodes and Topics](/docs/module-1-ros2/chapter-2-nodes-topics)**