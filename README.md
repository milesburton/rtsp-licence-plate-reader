⚠️ This is an experimental library

# 🎥 RTSP Licence Plate Reader
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