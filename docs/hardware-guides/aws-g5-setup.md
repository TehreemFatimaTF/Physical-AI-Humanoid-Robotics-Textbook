---
sidebar_label: 'AWS G5 Instance Setup Guide'
title: 'AWS G5 Instance Setup for Robotics Development'
description: 'Step-by-step guide to setting up AWS G5 instances for humanoid robotics development'
slug: '/hardware-guides/aws-g5-setup'
---

# AWS G5 Instance Setup Guide for Robotics Development

## Overview

AWS G5 instances are purpose-built for graphics-intensive applications, gaming, and remote workstations. With NVIDIA A10G Tensor Core GPUs, they provide excellent performance for robotics simulation, computer vision, and AI training workloads. This guide provides step-by-step instructions for setting up G5 instances for humanoid robotics development.

## G5 Instance Specifications

### Hardware Configuration
- **GPU**: NVIDIA A10G Tensor Core GPU (24GB VRAM)
- **CPU**: Intel Xeon Scalable processors (up to 96 vCPUs)
- **Memory**: Up to 384 GiB RAM
- **Storage**: NVMe SSD storage for high I/O performance
- **Network**: Up to 100 Gbps networking

### Robotics Development Capabilities
- Real-time physics simulation (Gazebo, Unity)
- Computer vision processing
- Deep learning model training and inference
- 3D rendering and visualization
- High-performance computing for robotics algorithms

## Prerequisites

### AWS Account Requirements
- Active AWS account with billing configured
- IAM permissions for EC2 and related services
- Sufficient service quotas for G5 instances
- AWS CLI configured on your local machine

### Local Environment
- AWS CLI installed and configured
- SSH key pair for secure access
- Basic knowledge of Linux command line

## Step 1: Launch G5 Instance

### Using AWS Console

1. **Navigate to EC2 Dashboard**
   - Go to AWS Management Console
   - Navigate to EC2 service
   - Click "Launch Instance"

2. **Configure Instance Details**
   - **Name**: `robotics-g5-dev` (or your preferred name)
   - **Application and OS Images**: Ubuntu Server 22.04 LTS (HVM)
   - **Instance Type**: Select `g5.xlarge` or appropriate size
   - **Key Pair**: Select your existing key pair or create new one

3. **Configure Storage**
   - **Size**: Minimum 100GB (recommended 200GB+ for robotics workloads)
   - **Type**: General Purpose SSD (gp3) or Provisioned IOPS (io2)

4. **Configure Security Group**
   ```bash
   # Required ports for robotics development
   SSH: 22 (TCP, your IP)
   VNC: 5900-5910 (TCP, your IP) - for remote desktop
   Jupyter: 8888 (TCP, your IP) - for development
   ROS: 11311 (TCP, your IP) - for ROS master
   ```

5. **Launch Instance**
   - Review configuration
   - Click "Launch Instance"

### Using AWS CLI

```bash
# Launch G5 instance with recommended configuration
aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \
    --count 1 \
    --instance-type g5.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxx \
    --subnet-id subnet-xxxxxxxx \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=robotics-g5-dev}]' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]'
```

## Step 2: Initial Instance Setup

### Connect to Instance

```bash
# SSH into the instance
ssh -i "your-key.pem" ubuntu@your-instance-public-ip

# Update system packages
sudo apt update && sudo apt upgrade -y
```

### Install Essential Development Tools

```bash
# Install basic development tools
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y git curl wget vim htop
sudo apt install -y nvidia-driver-535 nvidia-utils-535
```

## Step 3: Install NVIDIA GPU Drivers and CUDA

### Verify GPU Detection

```bash
# Check if GPU is detected
lspci | grep -i nvidia
nvidia-smi
```

### Install CUDA Toolkit

```bash
# Download and install CUDA
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run
```

### Configure CUDA Environment

```bash
# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 4: Install Robotics Frameworks

### Install ROS 2 Humble Hawksbill

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-argcomplete
sudo apt install ros-humble-vision-msgs ros-humble-geometry-msgs
sudo apt install ros-humble-sensor-msgs ros-humble-nav-msgs
```

### Source ROS 2 Environment

```bash
# Add to bashrc
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

## Step 5: Install Simulation Environments

### Install Gazebo Garden

```bash
# Add Gazebo repository
sudo curl -sSL http://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install gazebo-garden
```

### Install Unity Robotics Hub (Optional)

```bash
# For Unity simulation integration
# Download Unity Hub from Unity website
# Install Unity with robotics packages
```

## Step 6: Set Up Python Environment for Robotics

### Create Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools
```

### Install Robotics Libraries

```bash
# Core robotics libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python
pip install numpy scipy scikit-learn
pip install openai anthropic
pip install python-dotenv
pip install pyaudio sounddevice
pip install open3d
pip install pycuda

# ROS 2 Python client
pip install rclpy

# Computer vision and perception
pip install Pillow albumentations
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.1/index.html
pip install supervision
```

## Step 7: Install Vision-Language-Action Components

### OpenAI Whisper for Speech Recognition

```bash
# Install Whisper with GPU support
pip install git+https://github.com/openai/whisper.git
pip install soundfile librosa
pip install pydub
```

### Large Language Model Integration

```bash
# Install OpenAI and Anthropic clients
pip install openai anthropic

# Install Hugging Face transformers for local models
pip install transformers accelerate bitsandbytes
pip install auto-gptq optimum
```

## Step 8: Configure Development Environment

### Install Development Tools

```bash
# Code editors and IDEs
sudo snap install code --classic  # VS Code
pip install jupyter jupyterlab

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Set Up Workspace

```bash
# Create robotics workspace
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws

# Source ROS 2 and build workspace
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## Step 9: Configure Remote Desktop Access (Optional)

### Install VNC Server

```bash
# Install TigerVNC server
sudo apt install tigervnc-standalone-server tigervnc-common

# Set VNC password
vncpasswd

# Create VNC startup script
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF

chmod +x ~/.vnc/xstartup
```

### Start VNC Server

```bash
# Start VNC server
vncserver :1 -geometry 1920x1080 -depth 24

# Connect using VNC client to: your-instance-ip:5901
```

## Step 10: Performance Optimization

### GPU Memory Management

```python
# Python script for GPU memory optimization
import torch
import gc

def optimize_gpu_memory():
    """Optimize GPU memory usage for robotics applications"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Limit GPU memory fraction if needed
def limit_gpu_memory(fraction=0.8):
    """Limit GPU memory usage to prevent out-of-memory errors"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
```

### System Optimization

```bash
# Increase swap space for memory-intensive operations
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Optimize I/O scheduler for SSD storage
echo 'deadline' | sudo tee /sys/block/nvme0n1/queue/scheduler
```

## Step 11: Testing the Setup

### Test GPU Acceleration

```python
# Test GPU functionality
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device name: {torch.cuda.get_device_name()}")

# Test basic GPU computation
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("GPU matrix multiplication successful!")
```

### Test ROS 2 Installation

```bash
# Test ROS 2
ros2 run demo_nodes_cpp talker
# In another terminal: ros2 run demo_nodes_py listener
```

## Step 12: Cost Optimization

### Spot Instances for Development

```bash
# Launch spot instance to reduce costs
aws ec2 request-spot-instances \
    --spot-price "0.50" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification \
    --image-id ami-0abcdef1234567890 \
    --instance-type g5.xlarge \
    --key-name your-key-pair
```

### Auto-Stop Scripts

```bash
# Create auto-stop script to save costs
cat > ~/auto_stop.sh << 'EOF'
#!/bin/bash
# Auto-stop script for development instances
# Stops instance after 8 hours of inactivity
sudo shutdown -h +480  # 8 hours
EOF

chmod +x ~/auto_stop.sh
```

## Troubleshooting

### Common Issues and Solutions

#### GPU Not Detected
```bash
# Reinstall NVIDIA drivers
sudo apt remove --purge nvidia-*
sudo apt autoremove
sudo apt update
sudo apt install nvidia-driver-535 nvidia-utils-535
sudo reboot
```

#### CUDA Installation Issues
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall CUDA if needed
sudo apt remove --purge cuda*
sudo apt autoremove
# Then reinstall using the official installer
```

#### ROS 2 Installation Issues
```bash
# Verify ROS 2 installation
echo $ROS_DISTRO
source /opt/ros/humble/setup.bash
ros2 --version
```

## Security Best Practices

### SSH Hardening

```bash
# Configure SSH security
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no, PasswordAuthentication no, AllowUsers ubuntu
sudo systemctl restart ssh
```

### Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 8888/tcp  # Jupyter
sudo ufw allow 11311/tcp  # ROS
```

## Monitoring and Maintenance

### GPU Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Log GPU usage to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > gpu_usage.log
```

### System Monitoring

```bash
# Monitor system resources
htop
iotop  # Install with: sudo apt install iotop
```

## Conclusion

Your AWS G5 instance is now configured for humanoid robotics development with:
- NVIDIA GPU acceleration for AI and simulation workloads
- ROS 2 framework for robotics communication
- Simulation environments (Gazebo)
- Development tools and libraries
- Optimized performance settings

This setup provides a powerful cloud-based development environment for creating and testing humanoid robotics applications. Remember to monitor costs and consider using spot instances for development work to optimize expenses.

For production deployment, consider setting up additional security measures and implementing proper backup strategies for your development work.