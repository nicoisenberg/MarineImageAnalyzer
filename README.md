# MarineImageAnalyzer

MarineImageAnalyzer is a Python and OpenCV project for batch processing and analysis of marine images using classical computer-vision techniques.

The project explores image preprocessing, edge detection, morphological operations, contour analysis, and configurable experiments on underwater imagery. It also demonstrates the limitations of handcrafted computer-vision pipelines when trying to identify semantically meaningful objects such as marine animals.

## Features

- Batch processing of `.jpg` images
- Image resizing and grayscale conversion
- Optional CLAHE contrast enhancement
- Gaussian blur for noise reduction
- Canny edge detection
- Morphological closing
- Contour detection and area-based filtering
- Bounding-box visualization
- Per-image contour statistics
- CSV summary across all processed images
- Configurable processing parameters through a command-line interface
- Automated tests with pytest

## Processing Pipeline

Each input image is processed using the following pipeline:

```text
Input image
    ↓
Resize
    ↓
Grayscale
    ↓
Optional CLAHE
    ↓
Gaussian blur
    ↓
Canny edge detection
    ↓
Morphological closing
    ↓
Contour detection
    ↓
Area filtering
    ↓
Bounding boxes
    ↓
Contour statistics
```

The same pipeline is applied independently to every `.jpg` image in the `images/` directory.

## Project Structure

```text
MarineImageAnalyzer/
├── images/
│   └── ...
├── output/
│   └── ...
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── detection.py
│   ├── image_loader.py
│   ├── image_saver.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   └── report.py
├── tests/
│   ├── test_analysis.py
│   ├── test_cli.py
│   ├── test_detection.py
│   ├── test_image_loader.py
│   ├── test_image_saver.py
│   ├── test_preprocessing.py
│   └── test_report.py
├── main.py
├── README.md
└── requirements.txt
```

## Installation

Clone the repository and enter the project directory.

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

## Usage

Place one or more `.jpg` images inside the `images/` directory.

Run the analyzer with the default parameters:

```powershell
python main.py
```

The program processes every `.jpg` file found in the directory.

### Example with Custom Parameters

```powershell
python main.py --use-clahe --min-area 300 --canny-low 50 --canny-high 150
```

To see all available command-line options:

```powershell
python main.py --help
```

## Command-Line Parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `--use-clahe` | disabled | Enables CLAHE contrast enhancement |
| `--clahe-clip` | `2.0` | CLAHE clip limit |
| `--clahe-grid X Y` | `8 8` | CLAHE tile grid size |
| `--blur-kernel` | `5` | Gaussian blur kernel size; must be a positive odd integer |
| `--canny-low` | `100` | Lower Canny edge-detection threshold |
| `--canny-high` | `200` | Upper Canny edge-detection threshold |
| `--morph-kernel` | `3` | Kernel size for morphological closing |
| `--min-area` | `100` | Minimum contour area used for filtering |

Command-line inputs are validated before image processing starts.

## Output

A separate output directory is created for every processed image.

For example:

```text
images/
├── shark.jpg
└── reef.jpg

output/
├── shark/
│   ├── resized_image.jpg
│   ├── grayscale_image.jpg
│   ├── blurred_image.jpg
│   ├── edge_image.jpg
│   ├── closed_edge_image.jpg
│   ├── image_contours.jpg
│   └── bounding_box_image.jpg
├── reef/
│   └── ...
└── summary.csv
```

If CLAHE is enabled, an additional `enhanced_image.jpg` is saved.

### Batch Summary

After all images have been processed, the program creates:

```text
output/summary.csv
```

The summary contains:

- input image name
- number of original contours
- number of contours after morphological closing
- number of contours remaining after area filtering
- total area of relevant contours
- average area of relevant contours
- largest relevant contour area

This allows results from several images to be compared without inspecting every output manually.

## Testing

The project contains automated tests for image loading, saving, preprocessing, detection, analysis, reporting, and command-line validation.

Run the complete test suite with:

```powershell
python -m pytest
```

## Limitations

This project uses classical computer-vision techniques and does not perform semantic object detection.

The generated bounding boxes represent connected contour regions that satisfy the configured filtering criteria. The algorithm does not understand whether a region contains a shark, turtle, coral, reflection, or another object.

As a result:

- strong reflections and background structures may create detections
- low-contrast animals may not form complete contours
- parameters that work well for one underwater image may perform poorly on another
- contour-based detection is highly dependent on lighting, contrast, background, and image composition

These limitations are an important outcome of the project and motivate the transition from handcrafted image-processing rules to learned computer-vision methods.

## Project Status

The classical computer-vision implementation is complete.