# VISIONVOICE 2.0
> **Intelligent Voice-Controlled Assistive Reading Device**  
> Designed for macOS development and Raspberry Pi 4 physical deployment.

---

## 1. Project Overview

**VISIONVOICE** is a production-ready, hands-free assistive reading appliance designed for visually impaired and elderly users. It continuously monitors physical book alignment under a camera, automatically captures still pages when stable, executes high-accuracy multilingual OCR (English, Tamil, Kannada, Telugu, Malayalam, Hindi), translates to preferred languages if requested, and reads the text aloud sentence-by-sentence with full conversational voice control (pause, resume, repeat, next page, volume control).

### Core Highlights:
- **Clean Hardware Abstraction**: Complete decoupling between platform implementations (Mac OpenCV & native TTS vs. Raspberry Pi Picamera2, eSpeak-ng, and physical GPIO buttons/LED).
- **High-Accuracy OCR Engine**: Evaluates 8 specialized OpenCV preprocessing variants across 4 Tesseract Page Segmentation Modes (PSMs 3, 4, 6, 11) using composite lexical and confidence quality scoring.
- **Multilingual Support**: High-accuracy recognition for English (`eng`), Tamil (`tam`), Kannada (`kan`), Telugu (`tel`), Malayalam (`mal`), and Devanagari Hindi (`hin`).
- **Page Alignment & Stability Detection**: Real-time quad contour detection and temporal stability tracking to trigger captures without requiring physical buttons.
- **Natural Voice Control**: Single controlled microphone pipeline listening for wake words ("*Hey VisionVoice*") and conversational commands ("*Start reading*", "*Pause*", "*Repeat*", "*Next page*", "*Stop*").
- **Local Memory & Anti-Duplicate System**: SQLite-backed session persistence that hashes captures to prevent re-reading the same page accidentally.

---

## 2. Architecture

```
visionvoice-2.0/
├── main.py                  # Main CLI entry point (Run, Dev, OCR test, Camera test)
├── requirements.txt         # Core lightweight dependencies
├── .env.example             # Configuration environment template
├── .gitignore
├── README.md
│
├── visionvoice/
│   ├── core/
│   │   ├── models.py        # Data structures (OCRResult, ReadingSession, Intents)
│   │   ├── state_machine.py # Thread-safe State Machine with clean transitions
│   │   └── controller.py    # Master coordinator integrating all subsystems
│   │
│   ├── camera/
│   │   ├── base.py          # BaseCamera abstract interface & factory
│   │   ├── mac_camera.py    # MacBook OpenCV VideoCapture driver
│   │   └── pi_camera.py     # Raspberry Pi Camera Module 3 (Picamera2)
│   │
│   ├── ocr/
│   │   ├── engine.py        # Orchestrator running variants, PSMs & candidate selection
│   │   ├── preprocess.py    # 8 OpenCV illumination & contrast pipelines
│   │   ├── page_detector.py # Quad contour detection, aspect ratio, stability tracker
│   │   ├── scorer.py        # Multi-metric composite quality scoring
│   │   └── text_cleaner.py  # Conservative Indic-preserving text cleaner
│   │
│   ├── speech/
│   │   ├── base.py          # BaseTTS abstract interface & factory
│   │   ├── mac_tts.py       # macOS native 'say' speech driver
│   │   └── pi_tts.py        # Raspberry Pi eSpeak-ng + ALSA USB driver
│   │
│   ├── voice/
│   │   ├── controller.py    # Controlled microphone input & background listener
│   │   └── commands.py      # Spoken utterance intent parser & regex matchers
│   │
│   ├── hardware/
│   │   ├── base.py          # BaseHardware interface (Buttons & LED)
│   │   ├── buttons.py       # Button event dispatcher
│   │   ├── led.py           # LED pattern manager
│   │   ├── mac_mock.py      # macOS software simulation & logger
│   │   └── pi_gpio.py       # Raspberry Pi GPIO interrupt driver
│   │
│   ├── translation/
│   │   └── translator.py    # deep-translator wrapper with in-memory caching
│   │
│   ├── memory/
│   │   └── database.py      # SQLite session persistence & duplicate avoidance
│   │
│   └── utils/
│       ├── config.py        # Environment-driven configuration (.env)
│       ├── logging.py       # ANSI color console & rotating file logger
│       └── text.py          # Indic sentence segmenter & Unicode script detector
│
├── data/
│   ├── captures/            # Original captured book page images
│   ├── processed/           # Best preprocessed / dewarped page variants
│   ├── ocr/                 # Extracted text & JSON metadata
│   └── logs/                # visionvoice.log
│
└── tests/                   # Automated unit & integration tests
```

---

## 3. Installation & Setup on macOS

### 3.1 Prerequisites
1. **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.14).
2. **Homebrew** (for Tesseract OCR and PortAudio).

### 3.2 Install System Dependencies via Homebrew
```bash
# Install Tesseract OCR and Multilingual Indian Language packs
brew install tesseract tesseract-lang

# Install PortAudio (required for live microphone streaming)
brew install portaudio
```

### 3.3 Set Up Python Environment
```bash
# Navigate to project directory
cd /path/to/visionvoice-2.0

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Install PyAudio for live microphone input
pip install pyaudio
```

### 3.4 Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration is already set to `PLATFORM=mac` and `CAMERA_BACKEND=mac`.

---

## 4. Running on macOS

### 4.1 Primary Voice-Controlled Assistive Reading Flow
```bash
python main.py
```
**Workflow:**
1. Device starts and speaks: *"VisionVoice is ready. Say 'Hey VisionVoice' to begin."*
2. User says: *"Hey VisionVoice"*
3. Device asks for language and reading mode (original or translated).
4. User responds: *"English"* or *"Original"*.
5. Device prompts: *"Please place your book under the camera and say 'Start reading'."*
6. User says: *"Start reading"*.
7. The system continuously inspects camera frames until the page is aligned and stable.
8. Once stable, the high-accuracy OCR pipeline runs across preprocessing variants and PSMs.
9. TTS reads the text sentence-by-sentence.
10. Say *"Pause"*, *"Resume"*, *"Repeat"*, or *"Next page"* at any time!

---

### 4.2 Development Mode with Live HUD Preview
```bash
python main.py --dev
```
- Opens a live camera window showing page quadrilateral tracking (Green = Stable, Yellow = Detected, Red = Searching).
- Press `Spacebar` in the preview window to force an instant page capture.
- Press `q` to quit.

---

### 4.3 Standalone High-Accuracy OCR on an Image File
Run the complete multi-variant OCR engine on any book page image and print diagnostic confidence scores, selected preprocessing method, detected language, and segmented sentences:
```bash
python main.py --ocr path/to/book_page.jpg
```
*Tip: Add `--fast` to evaluate the top 4 fastest variants.*

---

### 4.4 Diagnostic Hardware & Audio Tests
```bash
# Test Camera capture and live stream
python main.py --camera-test

# Test Multilingual Text-To-Speech audio
python main.py --test-tts

# Test Microphone & Voice Command Intent Recognizer
python main.py --test-voice

# Test Page Contour Detection on a static image
python main.py --test-page path/to/page.jpg
```

---

## 5. Automated Testing

Run the full automated test suite (OCR scorer, Indic sentence segmenter, conservative text cleaner, page detector, state machine transitions, SQLite persistence):
```bash
python -m unittest discover tests -v
```

---

## 6. Raspberry Pi 4 Deployment Guide

When moving from macOS to Raspberry Pi 4, **the exact same codebase runs without modifications** simply by adjusting environment variables.

### 6.1 Hardware Bill of Materials (BOM)
- **Raspberry Pi 4 Model B** (4GB or 8GB recommended).
- **Raspberry Pi Camera Module 3** (Supports continuous hardware autofocus).
- **USB Speaker / USB DAC** (e.g., USB Sound Card or 3.5mm amplifier).
- **USB Microphone** (or I2S MEMS microphone).
- **Physical Push Buttons (3x)**:
  - Button 1: Power / Start / Pause
  - Button 2: Volume Up / Speed Up
  - Button 3: Volume Down / Speed Down
- **Status LED (1x)** with a 220Ω resistor.

---

### 6.2 Physical GPIO Pin Mapping

All push buttons use Raspberry Pi **internal pull-up resistors** (`PUD_UP`). Connect one side of each button to the GPIO pin and the other side to any **GND** pin (Active LOW).

| Function | Raspberry Pi Physical Pin | BCM GPIO Number | Wiring Notes |
| :--- | :--- | :--- | :--- |
| **Main / Power Button** | Pin 11 | **GPIO17** | To Button $\to$ GND |
| **Volume Up (+)** | Pin 13 | **GPIO27** | To Button $\to$ GND |
| **Volume Down (-)** | Pin 15 | **GPIO22** | To Button $\to$ GND |
| **Status LED (+)** | Pin 18 | **GPIO24** | Through 220Ω resistor to Anode (+) $\to$ Cathode (-) to GND |
| **Ground (GND)** | Pin 6, 9, 14, 20, 25, 30, 34, or 39 | **GND** | Common ground for buttons & LED |

#### Status LED Patterns:
- **Solid ON**: Device Ready / Active Reading.
- **Slow Pulse**: Waiting for Wake Word / Aligning Page.
- **Fast Blink**: Capturing Image & Executing OCR Pipeline.
- **Rapid Error Blink**: Error recovery mode.

---

### 6.3 Raspberry Pi OS System Setup

1. **Install Raspberry Pi OS 64-bit (Bookworm or Bullseye)** using Raspberry Pi Imager.
2. **Install System Dependencies**:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv \
    tesseract-ocr tesseract-ocr-tam tesseract-ocr-kan tesseract-ocr-tel tesseract-ocr-mal tesseract-ocr-hin \
    libcamera-tools python3-picamera2 \
    espeak-ng alsa-utils libportaudio2 portaudio19-dev
```

3. **Clone / Copy Project to Raspberry Pi**:
```bash
cd ~
git clone <YOUR_REPO_URL> visionvoice
cd visionvoice
```

4. **Set Up Python Virtual Environment on Pi**:
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
pip install pyaudio RPi.GPIO
```

5. **Configure `.env` for Raspberry Pi**:
```ini
PLATFORM=pi
CAMERA_BACKEND=pi
AUDIO_BACKEND=pi
PI_AUDIO_DEVICE=plughw:3,0
SPEECH_SPEED=175
OCR_LANGUAGES=eng+tam+kan+tel+mal+hin
SHOW_CAMERA_PREVIEW=false
DEBUG_MODE=false
```

---

### 6.4 USB Audio Configuration (`aplay`)

1. Find the card number of your USB Speaker/DAC:
```bash
aplay -l
```
Example output:
```
card 3: Device [USB Audio Device], device 0: USB Audio [USB Audio]
```
2. If your device is card 3, device 0, the ALSA identifier is `plughw:3,0`. Update `PI_AUDIO_DEVICE=plughw:3,0` in `.env`.
3. Test sound output:
```bash
speaker-test -D plughw:3,0 -t wav -c 2
```

---

### 6.5 Auto-Start on Boot via Systemd Service

Create a systemd unit file to automatically start VisionVoice on power-up:
```bash
sudo nano /etc/systemd/system/visionvoice.service
```

Paste the following configuration (replace `<USER>` with your Linux username, e.g. `pi` or `admin`):
```ini
[Unit]
Description=VisionVoice Assistive Reading Device
After=network.target sound.target

[Service]
Type=simple
User=<USER>
WorkingDirectory=/home/<USER>/visionvoice
ExecStart=/home/<USER>/visionvoice/venv/bin/python main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable visionvoice.service
sudo systemctl start visionvoice.service
```

Check live logs:
```bash
journalctl -u visionvoice.service -f
```

---

## 7. Configuration Reference (`.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PLATFORM` | `mac` | Host OS: `mac` or `pi`. |
| `CAMERA_BACKEND` | `mac` | `mac` (OpenCV VideoCapture) or `pi` (Picamera2). |
| `CAMERA_INDEX` | `0` | Camera device index on Mac. |
| `CAMERA_WIDTH` | `1920` | Capture resolution width. |
| `CAMERA_HEIGHT` | `1080` | Capture resolution height. |
| `AUDIO_BACKEND` | `mac` | `mac` (`say`) or `pi` (`espeak-ng` + `aplay`). |
| `PI_AUDIO_DEVICE` | `plughw:3,0` | ALSA audio device name for USB DAC on Raspberry Pi. |
| `SPEECH_SPEED` | `175` | Text-To-Speech speed in words per minute. |
| `SUPPORTED_LANGUAGES` | `eng,tam,kan,tel,mal,hin` | Supported OCR languages probed dynamically via Stage 1. |
| `OCR_CONFIDENCE_THRESHOLD` | `40.0` | Minimum composite score to accept recognized text. |
| `OCR_FAST_MODE` | `false` | Restricts preprocessing variants for lower latency on Raspberry Pi. |
| `OCR_MAX_VARIANTS` | `4` | Max preprocessing variants evaluated in Stage 2. |
| `OCR_EARLY_STOP_SCORE` | `85.0` | Quality threshold to stop Stage 2 OCR immediately upon high accuracy. |
| `PAGE_STABILITY_FRAMES` | `4` | Consecutive stable frames before auto-capturing. |
| `PAGE_MIN_AREA_RATIO` | `0.15` | Minimum bounding page area ratio relative to camera frame. |
| `TRANSLATION_ENABLED` | `false` | Translate text before speaking (`true`/`false`). |
| `DEFAULT_LANGUAGE` | `en` | Default reading/translation language (`en`, `ta`, `hi`, `kn`, `te`, `ml`). |

---

## 8. Troubleshooting

### 1. `tesseract not found` Error
- **macOS**: Ensure you ran `brew install tesseract tesseract-lang`. If installed in a non-standard path, set `TESSERACT_CMD=/opt/homebrew/bin/tesseract` in `.env`.
- **Raspberry Pi**: Run `sudo apt install -y tesseract-ocr tesseract-ocr-all`.

### 2. Camera Preview Fails on macOS
- Grant camera permissions to your terminal or IDE in **macOS System Settings > Privacy & Security > Camera**.
- Check if another application (FaceTime, Zoom) is locking the built-in camera.

### 3. Microphone Recognition Not Hearing Commands
- Ensure `pyaudio` is installed (`pip install pyaudio`).
- If running in a headless or quiet environment, use physical buttons or inject simulated commands in `--dev` mode.

### 4. High Latency on Raspberry Pi
- Set `OCR_FAST_MODE=true` in `.env` to restrict OCR evaluation to the top 4 most effective preprocessing variants.

---

## 9. License

This project is licensed under the MIT License for open assistive technology research and development.
