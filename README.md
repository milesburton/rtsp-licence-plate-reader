# 🎥 RTSP License Plate Reader

A real-time license plate detection system that processes RTSP video streams to identify vehicles and their license plates. This application was developed with assistance from Anthropic's Claude AI.

## ✨ Features

- Real-time RTSP stream processing
- GPU-accelerated vehicle and person detection
- License plate OCR with confidence scoring
- Support for UK, US, and EU plate formats
- Person detection with visual tracking
- Debug mode with visual output
- Robust error handling and logging

## 🔧 Requirements

- [Bun](https://bun.sh/) runtime v1.2.0+
- FFmpeg
- Node.js build tools (for TensorFlow.js)

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
FPS=5
FRAME_WIDTH=1280
FRAME_HEIGHT=720
DEBUG_MODE=true
USE_GPU=true
MIN_VEHICLE_AREA=10000
MAX_VEHICLE_AREA=80000
```

## 📖 Usage

```bash
bun run index.ts
```

## 📤 Output

The application generates:
- Real-time console logs of detected vehicles, people, and license plates
- Debug images of detected objects (when DEBUG_MODE=true)
- Confidence scores for vehicle, person, and plate detection

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgements

- Developed with assistance from Anthropic's Claude AI
- Uses Tesseract.js for OCR
- Uses TensorFlow.js for GPU-accelerated processing