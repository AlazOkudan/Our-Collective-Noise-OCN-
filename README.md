# Our Collective Noise (OCN) v3.2

[![License](https://img.shields.io/badge/license-open--source-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)

## Overview

**Our Collective Noise (OCN) v3.2** is a research-based tactical media project that transforms top-down surveillance technologies into a bottom-up tactical tool. As an offline system, OCN uses live webcam feeds, machine learning (ML), and computer vision (CV) to detect people and simultaneously converts them into coarse pixels—replacing the common aim of precise identification in surveillance technologies with anonymity.

The project explores concepts of **noise**, **de-identification**, **accidental aesthetics**, and **human-machine collaboration**. Coarse pixels are constantly stitched together to create collective abstract human-machine interaction patterns that people are collectively and unidentifiably part of. In a world where thriving ML, CV, and AI technologies increasingly rely on cleaner datasets, higher processing capacities, and precise labels, OCN turns the technology against itself in pursuit of revealing the latent potential in noise, anonymity, and collective action.

## Academic Reference

This project was developed as part of published research on tactical media and surveillance resistance. If you use this work in your research, please cite:

Okudan, Alaz. (2025). Our Collective Noise (OCN): A Tactical Response to Computer Vision, Surveillance, and Noise. xCoAx 2025, School of X. DOI: 10.34626/2025_xcoax_x_02. Available from: https://classof25.xcoax.org/paper02.html

**Abstract**: *Our Collective Noise (OCN) is a research-based tactical media project. As an attempt to transform the top-down pervasive qualities of machine learning (ML), computer vision (CV), and surveillance technologies into a bottom-up tactical tool, it plays around the concepts of noise, de-identification, accidental aesthetics, and human-machine collaboration. OCN, as an offline system, uses live webcam feed, ML, and CV to detect people and simultaneously turn them into coarse pixels to replace the common aim of precise identification in surveillance technologies with anonymity. Coarse pixels are constantly stitched together to create collective abstract human-machine interaction patterns that people are collectively and unidentifiably part of. In a world where thriving ML, CV, and AI (artificial intelligence) technologies increasingly rely on cleaner datasets, higher processing capacities, precise labels and categories, OCN turns the technology against itself, in pursuit of revealing the latent potential in noise, anonymity, and collective action.*

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Features

- 🎯 **Tactical Media Tool** - Subverts surveillance technology for collective anonymity
- 🔍 **Real-time Person Detection** - Uses YOLOv3 for human presence identification
- 🎨 **Coarse Pixelation** - Transforms precise identification into anonymous noise
- 🖼️ **Collective Collage Creation** - Stitches pixelated humans into abstract patterns  
- 📹 **Dual Recording Modes** - Document both process and output
- 🔒 **Privacy-First Design** - Prioritizes anonymity over identification accuracy
- 🎛️ **Human-Machine Collaboration** - Interactive controls for collective creation
- ⚡ **Offline Operation** - No data transmission, fully local processing
- 🌐 **Accidental Aesthetics** - Embraces imperfection and noise as creative elements

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV
- YOLO dependencies

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlazOkudan/Our-Collective-Noise-OCN-.git
   cd Our-Collective-Noise-OCN-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required YOLO files**
   
   Download these files and place them in your project directory:
   
   | File | Description | Download Link |
   |------|-------------|---------------|
   | `yolov3.weights` | Pre-trained YOLO weights | [Download (248MB)](https://pjreddie.com/media/files/yolov3.weights) |
   | `yolov3.cfg` | YOLO configuration | [Download](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) |
   | `coco.names` | COCO class names | [Download](https://github.com/pjreddie/darknet/blob/master/data/coco.names) |

4. **Run the application**
   ```bash
   python "OCN v.3.2_Published.py"
   ```

## Usage

### Quick Start

1. Connect your webcam
2. Run the application: `python "OCN v.3.2_Published.py"`
3. The application will open showing your webcam feed and an empty collage grid
4. Press `R` to start recording detected persons to the collage
5. Detected individuals will automatically be added to the grid with pixelation effects

### Keyboard Controls

| Key | Function |
|-----|----------|
| `R` | Toggle person recording to collage |
| `V` | Start/stop combined video recording |
| `Y` | Start/stop webcam-only video recording |
| `P` | Toggle pixelation on/off |
| `+/-` | Increase/decrease pixelation intensity |
| `W/S` | Increase/decrease grid rows |
| `A/D` | Decrease/increase grid columns |
| `O` | Toggle color/black & white mode |
| `K` | Toggle contrast enhancement |
| `[/]` | Decrease/increase contrast level |
| `L` | Switch between vertical/horizontal layout |
| `B` | Toggle collage background (black/white) |
| `T` | Toggle random/linear placement mode |
| `F` | Toggle fullscreen |
| `X` | Reset current collage |
| `C` | Choose save directory |
| `Esc` | Exit application |

### How It Works

OCN operates as a **tactical reversal** of surveillance technology:

1. **Detection Without Identification**: YOLOv3 detects human presence but immediately obscures identity through coarse pixelation
2. **Noise as Resistance**: Instead of pursuing cleaner data, OCN embraces and amplifies noise as a form of technological resistance  
3. **Collective Pattern Creation**: Individual pixelated forms are stitched into collective abstract patterns where people become unidentifiably part of a larger whole
4. **Human-Machine Collaboration**: Users actively participate in the creation process through real-time controls, making each session a unique collaborative artwork
5. **Accidental Aesthetics**: The system celebrates imperfection, glitches, and unintended visual outcomes as meaningful creative elements

### Conceptual Framework

**Noise vs. Signal**: OCN deliberately introduces noise where surveillance systems seek signal clarity, using technological "failure" as a creative and political strategy.

**De-identification**: Rather than improving recognition accuracy, OCN purposefully degrades visual information to protect individual privacy while maintaining collective presence.

**Bottom-up Reclamation**: The project transforms top-down surveillance tools into bottom-up instruments for collective expression and anonymity.

## File Structure

```
Our-Collective-Noise-OCN-/
├── OCN v.3.2_Published.py   # Main application script
├── yolov3.weights          # YOLOv3 pre-trained weights (download required)
├── yolov3.cfg             # YOLOv3 configuration file (download required)
├── coco.names             # COCO dataset class names (download required)
├── README.md              # This documentation
├── LICENSE                # License file
├── detected_persons/      # Auto-created: Saved collage images
├── video_recordings/      # Auto-created: Recorded video files
└── requirements.txt       # Python dependencies
```

## Requirements

```txt
opencv-python>=4.5.0
numpy>=1.19.0
tkinter  # Usually included with Python
```

**System Requirements:**
- Python 3.7+
- Webcam/Camera device
- ~250MB free space for YOLO weights file
- Optional: CUDA-compatible GPU for acceleration

## Configuration Options

The application includes several configurable parameters at the top of the script:

```python
FRAME_WIDTH = 640          # Webcam resolution width
FRAME_HEIGHT = 480         # Webcam resolution height
CONFIDENCE_THRESHOLD = 0.5 # Person detection confidence (0.0-1.0)
INPUT_SIZE = (320, 320)    # YOLO input size for processing
PIXELATION_BLOCKS = 5      # Default pixelation intensity
CAPTURE_INTERVAL = 5       # Seconds between collage captures
THUMBNAIL_WIDTH = 64       # Collage grid cell width
THUMBNAIL_HEIGHT = 48      # Collage grid cell height
```

## Output Files

- **Collage Images**: Saved as `collage_YYYYMMDD_HHMMSS.jpg` in the `detected_persons/` folder
- **Combined Videos**: Saved as `recording_YYYYMMDD_HHMMSS.avi` in the `video_recordings/` folder
- **Webcam Videos**: Saved as `webcam_recording_YYYYMMDD_HHMMSS.avi` in the `video_recordings/` folder

## Troubleshooting

**Common Issues:**

- **"yolov3.weights not found"**: Download the weights file from the link above
- **Low detection accuracy**: Ensure good lighting and clear camera view
- **Performance issues**: Try reducing frame size or enable GPU acceleration
- **Camera not detected**: Check camera permissions and that no other app is using it

**GPU Acceleration:**
- The application automatically attempts to use CUDA if available
- Install CUDA toolkit and OpenCV with GPU support for better performance

## Contributing

Contributions are welcome! If you make changes to this code, please publish them as open-source as per the project's philosophy.

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request with clear documentation

## Ethical Use & Important Notice

⚠️ **This code should only be used to subvert and divert top-down practices of surveillance and detection.**

**Project Philosophy:**
- **Tactical Media**: OCN is designed as a tool for technological resistance and creative subversion
- **Collective Anonymity**: Prioritizes group identity over individual identification
- **Noise as Liberation**: Embraces technological imperfection as a form of resistance

**Intended Use:**
- Tactical media art projects
- Privacy protection and awareness demonstrations  
- Research on surveillance resistance
- Exploring human-machine collaboration
- Digital rights advocacy and education

**Prohibited Use:**
- Traditional surveillance applications
- Individual identification or tracking
- Commercial surveillance systems
- Any use that contradicts the project's anti-surveillance philosophy

Users are responsible for ensuring their use aligns with the project's tactical media principles and complies with local laws and regulations.

## License

This project is licensed under the [License Name] - see the [LICENSE](LICENSE) file for details.


**A tactical media project for collective anonymity and technological resistance** 🔒✊🎨
