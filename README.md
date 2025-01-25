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
RTSP_URL=rtsp://username:password@camera-ip:554/stream
FPS=30
FRAME_WIDTH=2592
FRAME_HEIGHT=1944
DEBUG_MODE=true
MIN_VEHICLE_AREA=20000
MAX_VEHICLE_AREA=160000
MAX_RETRIES=3
RETRY_DELAY=5000
MIN_PERSON_AREA=10000
MAX_PERSON_AREA=100000
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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgements

- Developed with assistance from Anthropic's Claude AI
- Uses Tesseract.js for OCR
- Uses TensorFlow.js for object detection