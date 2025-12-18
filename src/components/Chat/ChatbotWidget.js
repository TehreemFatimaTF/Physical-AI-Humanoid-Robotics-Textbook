/**
 * ChatbotWidget Component
 *
 * An embedded chat widget for the Physical AI textbook that allows users to ask
 * questions about the content and receive responses restricted to the curriculum.
 */

import React, { useState, useRef, useEffect } from 'react';
import clsx from 'clsx';
import styles from './ChatbotWidget.module.css';

const ChatbotWidget = ({ initialOpen = false }) => {
  const [isOpen, setIsOpen] = useState(initialOpen);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Predefined responses to demonstrate the chatbot functionality
  const getPredefinedResponse = (userMessage) => {
    const lowerMessage = userMessage.toLowerCase();

    if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
      return "Hello! I'm your Physical AI & Humanoid Robotics assistant. I can help answer questions about the curriculum content. What would you like to know?";
    } else if (lowerMessage.includes('ros') && (lowerMessage.includes('what') || lowerMessage.includes('explain'))) {
      return "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. In our curriculum, you'll learn about ROS 2 Nodes, Topics, Services, and how to bridge Python AI agents to ROS controllers using rclpy.";
    } else if (lowerMessage.includes('urdf')) {
      return "URDF (Unified Robot Description Format) is an XML format used in ROS for representing a robot model. It can describe joints, links, inertias, visual and collision properties. For humanoid robots, URDF is essential for defining the robot's physical structure, which is crucial for simulation and control.";
    } else if (lowerMessage.includes('isaac') && lowerMessage.includes('nvidia')) {
      return "NVIDIA Isaac is a comprehensive robotics platform that includes Isaac Sim for photorealistic simulation and synthetic data generation, and Isaac ROS for hardware-accelerated perception and navigation. Isaac Sim allows you to create complex virtual environments for testing your robots, while Isaac ROS provides GPU-accelerated processing for perception tasks like VSLAM (Visual Simultaneous Localization and Mapping).";
    } else if (lowerMessage.includes('vla') || (lowerMessage.includes('vision') && lowerMessage.includes('language'))) {
      return "Vision-Language-Action (VLA) systems represent the convergence of LLMs and robotics. They enable robots to understand natural language commands and translate them into physical actions. For example, a VLA system could receive a command like 'Clean the room' and translate it into a sequence of ROS 2 actions that navigate to objects, identify them using computer vision, and manipulate them.";
    } else if (lowerMessage.includes('simulation') || lowerMessage.includes('gazebo')) {
      return "Simulation is critical in robotics for safe and cost-effective development. Gazebo provides physics simulation with realistic gravity, collisions, and sensor simulation (LiDAR, depth cameras, IMUs). Unity provides high-fidelity rendering and human-robot interaction. The sim-to-real transfer technique helps apply knowledge learned in simulation to real robots.";
    } else {
      return "I can help with questions about Physical AI & Humanoid Robotics curriculum content. This includes topics like ROS 2, Digital Twin simulation (Gazebo/Unity), NVIDIA Isaac, and Vision-Language-Action systems. Could you ask a more specific question about these topics?";
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate API call delay
    setTimeout(() => {
      const responseText = getPredefinedResponse(inputValue);
      const botMessage = {
        id: Date.now() + 1,
        text: responseText,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        sources: ['Physical AI Curriculum'] // In a real implementation, this would come from the RAG system
      };

      setMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
    }, 1000);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={styles.chatContainer}>
      {isOpen ? (
        <div className={styles.chatWidget}>
          <div className={styles.chatHeader}>
            <div className={styles.chatTitle}>Physical AI Assistant</div>
            <div className={styles.chatSubtitle}>Ask about the curriculum</div>
            <button
              className={styles.closeButton}
              onClick={toggleChat}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>

          <div className={styles.chatMessages}>
            {messages.length === 0 ? (
              <div className={styles.welcomeMessage}>
                <p>Hello! I'm your Physical AI & Humanoid Robotics assistant.</p>
                <p>Ask me questions about:</p>
                <ul>
                  <li>ROS 2 (Robot Operating System 2)</li>
                  <li>Digital Twin simulation (Gazebo & Unity)</li>
                  <li>NVIDIA Isaac platform</li>
                  <li>Vision-Language-Action systems</li>
                </ul>
                <p>I only respond based on the textbook content.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={clsx(
                    styles.message,
                    styles[message.sender]
                  )}
                >
                  <div className={styles.messageContent}>
                    <div className={styles.messageText}>{message.text}</div>
                    <div className={styles.messageMeta}>
                      <span className={styles.timestamp}>{message.timestamp}</span>
                      {message.sender === 'bot' && message.sources && (
                        <span className={styles.sources}>
                          Sources: {message.sources.join(', ')}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className={clsx(styles.message, styles.bot)}>
                <div className={styles.messageContent}>
                  <div className={styles.typingIndicator}>
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className={styles.chatInputArea}>
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about Physical AI & Robotics..."
              className={styles.chatInput}
              rows="1"
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className={clsx(
                styles.sendButton,
                (!inputValue.trim() || isLoading) && styles.sendButtonDisabled
              )}
            >
              Send
            </button>
          </div>

          <div className={styles.chatFooter}>
            <small>
              Responses are restricted to Physical AI curriculum content only
            </small>
          </div>
        </div>
      ) : (
        <button
          className={styles.chatToggleButton}
          onClick={toggleChat}
          aria-label="Open chat"
        >
          <span className={styles.chatIcon}>💬</span>
          <span>AI Assistant</span>
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;