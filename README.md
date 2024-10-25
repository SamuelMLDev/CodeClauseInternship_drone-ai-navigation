# Drone AI Autonomous Navigation System

![Drone AI](https://img.shields.io/badge/Drone_AI-Autonomous_Navigation-brightgreen)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)


## Overview

The **Drone AI Autonomous Navigation System** is a comprehensive project aimed at enabling drones to navigate autonomously using computer vision and Simultaneous Localization and Mapping (SLAM). This system integrates object detection to identify obstacles and a rule-based decision-making mechanism to determine navigation actions, ensuring safe and efficient movement towards a target destination.

## Features

- **Computer Vision:** Utilizes YOLOv5 for real-time object detection, enabling the drone to recognize and avoid obstacles such as people, cars, and bicycles.
- **Simultaneous Localization and Mapping (SLAM):** Implements a simplified SLAM system to map the environment and determine the drone's current position.
- **Rule-Based Navigation:** Employs a deterministic rule-based system to make navigation decisions based on detected objects and the drone's pose relative to the target.

## Installation

Follow the steps below to set up the project on your local machine.

### Prerequisites

- **Python 3.8 or higher**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **Git**: To clone the repository. Download it from [git-scm.com](https://git-scm.com/downloads).

### Steps

1. **Clone the Repository**

   Open your terminal or command prompt and run:

   ```bash
   git clone https://github.com/SamuelMLDev/CodeClause_Internship_drone-ai-navigation.git
   ```


2. **Navigate to the Project Directory**

    ```bash
    cd CodeClause_Internship_drone-ai-navigation
    ```

3. **Create a Virtual Environment**
It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    ```

4. **Activate the Virtual Environment**
Windows:

    ```bash
    venv\Scripts\activate
    ```

macOS/Linux:

    ```bash
    source venv/bin/activate
    ```

5. **Upgrade Pip**
Ensure you have the latest version of pip.

    ```bash
    pip install --upgrade pip
    ```

6. **Install Dependencies**
    
    ```bash
    pip install -r requirements.txt
    ```
Note: If you encounter any dependency issues, ensure that the versions specified in requirements.txt are compatible with your system.

### Usage

After completing the installation steps, you can run the main application to start the drone's autonomous navigation.

1. Ensure the Virtual Environment is Activated. If not already activated, activate it as shown in the Installation section.

2. Run the Application

    ```bash
    python src/main.py
    ```

### Technologies Used

Python, OpenCV, YOLOv5, NLTK, Flask, TensorFlow, PyTorch, Gym, Matplotlib, NumPy, Scikit-learn, Git, Visual Studio Code, SLAM.

This will start the video capture from your webcam, perform object detection, update SLAM, and execute navigation actions based on the detected objects and drone's pose.

3. Controls
Exit the Application: Press the q key in the video window to terminate the application.
