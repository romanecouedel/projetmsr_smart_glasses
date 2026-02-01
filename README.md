# Project Launch Guide

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Arduino programmed with `smartglassesvf.ino`
- Raspberry Pi with configured camera

## Installing Dependencies

### On the PC (Computer directory)

```bash
pip install -r requirements.txt
```

## Running the Code

### Step 1: Start the video receiver (PC)

```bash
python pc_cam_com.py
```

This starts listening on UDP port 5000 to receive the video stream from the Raspberry Pi.

### Step 2: Start the Raspberry Pi camera

On the Raspberry Pi:

```bash
python com_pc.py
```

This starts capturing and sending the video stream to the PC.

### Step 3: Launch the main launcher (PC)

```bash
python launcher.py
```

This script:
- Listens for commands from Arduino via serial port
- Launches/stops detection scripts based on received modes:
  - Mode 1: `emotion_detection2_pneu.py` (emotion detection)
  - Mode 2: `objet.py` (obstacle detection with YOLO)
  - Mode 3: Neutral (no script active)

## Project Structure

- **Computer/** - Main code (launcher, emotion detection, YOLO)
- **RPI_CAM/** - Raspberry Pi camera / PC communication
- **Arduino/** - Microcontroller code (smartglassesvf.ino)
- **depth_anything_v2/** - Depth detection model (optional)
- **CAO/** - CAD files for connected glasses
