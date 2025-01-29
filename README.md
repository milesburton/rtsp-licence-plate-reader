# 🎥 RTSP Licence Plate Reader

⚠️ Experimental - Work in Progress

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js Version](https://img.shields.io/badge/bun-%3E%3D1.2.0-brightgreen)](https://bun.sh)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/milesburton/rtsp-licence-plate-reader/issues)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)
[![Code Style](https://img.shields.io/badge/code_style-prettier-ff69b4.svg)](https://prettier.io/)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/milesburton/rtsp-licence-plate-reader/test.yml)

A real-time licence plate detection system that processes RTSP video streams to identify vehicles and their licence plates. This application was developed with assistance from Anthropic's Claude AI.

## ✨ Features
- Real-time RTSP stream processing
- Vehicle and person detection
- Licence plate OCR with confidence scoring
- Support for UK, US, and EU plate formats
- Person detection with visual tracking
- Debug mode with visual output
- Robust error handling and logging
- Performance monitoring and metrics
- Memory usage tracking
- Configurable frame queue size

## 🔧 Requirements
- [Bun](https://bun.sh/) runtime v1.2.0+
- FFmpeg
  - Linux/WSL2: `sudo apt update && sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: [FFmpeg Downloads](https://ffmpeg.org/download.html)
- Build essentials (for TensorFlow.js native modules)
  - Linux/WSL2: `sudo apt install build-essential pkg-config`
  - macOS: `xcode-select --install`
  - Windows: Visual Studio Build Tools

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/milesburton/rtsp-licence-plate-reader

# Install dependencies
bun install
```

## ⚙️ Configuration
Create a `.env` file:
```env
# RTSP stream URL (required)
RTSP_URL=rtsp://username:password@camera-ip:554/stream

# Frame capture settings
FPS=15                    # Recommended: 10-15 for processing efficiency
FRAME_WIDTH=1920         # Standard HD resolution
FRAME_HEIGHT=1080        # Standard HD resolution
FRAME_QUEUE_SIZE=30      # Buffer size for frame processing

# Object detection parameters
MIN_VEHICLE_AREA=5000    # Minimum pixel area for vehicle detection
MAX_VEHICLE_AREA=120000  # Maximum pixel area for vehicle detection
MIN_PERSON_AREA=5000     # Minimum pixel area for person detection
MAX_PERSON_AREA=50000    # Maximum pixel area for person detection

# Application settings
DEBUG_MODE=false         # Enable for debug images and verbose logging
MAX_RETRIES=3           # Connection retry attempts
RETRY_DELAY=5000        # Delay between retries (ms)
```

## 📖 Usage
```bash
bun run index.ts
```

## 📤 Output
The application generates:
- Real-time console logs of detected vehicles, people, and licence plates
- Debug images of detected objects (when DEBUG_MODE=true)
- Confidence scores for vehicle, person, and plate detection
- Processing time metrics for each detection stage
- TensorFlow memory usage statistics
- Application heartbeat status

### Example Console Output
```
🚀 [2025-01-29 16:35:10] INFO: TensorFlow initialized
🚀 [2025-01-29 16:35:11] INFO: OCR Worker initialized
🚀 [2025-01-29 16:35:12] INFO: COCO-SSD Object Detection model loaded
🚀 [2025-01-29 16:35:12] INFO: Application configuration:
🚀 [2025-01-29 16:35:12] INFO: ----------------------------------------
🚀 [2025-01-29 16:35:12] INFO: FPS: 15
🚀 [2025-01-29 16:35:12] INFO: Frame Width: 1920
🚀 [2025-01-29 16:35:12] INFO: Frame Height: 1080
🚀 [2025-01-29 16:35:12] INFO: Max Retries: 3
🚀 [2025-01-29 16:35:12] INFO: Retry Delay: 5000 ms
🚀 [2025-01-29 16:35:12] INFO: Debug Mode: Enabled
🚀 [2025-01-29 16:35:12] INFO: Frame Queue Size: 30
🚀 [2025-01-29 16:35:12] INFO: ----------------------------------------
📂 [2025-01-29 16:35:12] INFO: Directory checks passed
📡 [2025-01-29 16:35:12] INFO: Attempting to connect to RTSP stream (Attempt 1/3)...
🎥 [2025-01-29 16:35:12] INFO: FFmpeg command: ffmpeg -rtsp_transport tcp -stimeout 5000000 -fflags nobuffer -flags low_delay -i rtsp://*****:*****@camera-ip:554/stream -r 15 -f image2pipe -vcodec mjpeg -pix_fmt yuvj420p pipe:1
🔍 [2025-01-29 16:35:13] INFO: Processing frame (Queue size: 2)
👤 [2025-01-29 16:35:13] INFO: Detected 2 people in frame:
👤 Person at (450, 280): 92.5% confidence
👤 Person at (820, 310): 88.7% confidence
🚗 [2025-01-29 16:35:14] INFO: Detected 1 vehicle in frame:
🚗 Car at (650, 400): 96.3% confidence
🚘 [2025-01-29 16:35:14] INFO: Detected license plates:
🚘 Detected plate: AB12CDE (89.5% confidence)
🐞 [2025-01-29 16:35:14] DEBUG: Vehicle region file deleted: /temp/vehicle_1706543714123.jpg
🚀 [2025-01-29 16:35:20] INFO: Heartbeat: Application is running...
🐞 [2025-01-29 16:35:20] INFO: TensorFlow Memory Stats:
        Active Tensors: 24
        Data Buffers: 18
        Memory Used: 156.82 MB
        Memory State: Reliable
```

## 🔍 Monitoring
The application provides real-time monitoring through:
- Heartbeat logs every 10 seconds
- TensorFlow memory statistics
- Frame processing queue size
- Processing time metrics for detection stages
- Detailed error logging with stack traces

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgements
- Developed with assistance from Anthropic's Claude AI
- Uses Tesseract.js for OCR
- Uses TensorFlow.js for object detection