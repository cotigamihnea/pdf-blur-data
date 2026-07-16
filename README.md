# PDF Data Blocker

PDF Data Blocker is an enterprise-grade, 100% offline desktop application designed to automatically detect and anonymize Personally Identifiable Information (PII), handwritten signatures, and official engineering/architectural stamps from scanned PDF documents and technical blueprints.

By combining Natural Language Processing (NLP), Advanced Computer Vision (YOLOv8), and Traditional Image Processing (OpenCV), the application guarantees GDPR compliance while protecting the geometric integrity of CAD/cadastral drawings.

---

## Key Features

- **Dual-Engine Page Classification:** Automatically analyzes text density (via Tesseract OCR) to instantly distinguish between standard text documents and complex technical schematics/blueprints.
- **Advanced Text Redaction (PII Engine):** Utilizes Microsoft Presidio and a fine-tuned Romanian language model (`spaCy ro_core_news_lg`) to detect and redact Names, CNPs (Romanian National Identification Numbers), Phone Numbers, Emails, and Locations.
- **High-Sensitivity Signature Redaction:** Employs a highly sensitive YOLOv8 computer vision model to track down and obliterate fine ink lines and faint signatures hidden inside regular text pages.
- **Custom-Trained Stamp Detection (99% Accuracy):** Deploys a custom-trained YOLOv8 model specialized in isolating architectural, cadastral, and institutional stamps on blueprint pages, ignoring background noise.
- **Sequential OpenCV Fallback (Circle Sniper):** Implements a color-filtered Hough Circles routine to capture circular stamps that are heavily intersected by black grid tables or handwriting, avoiding false positives on CAD grid lines.
- **Modern User Interface:** Built with `CustomTkinter` and enhanced with seamless native **Drag & Drop** capabilities (`tkinterdnd2`).
- **100% Offline & Secure:** Processing is done entirely local. No data ever leaves your computer, ensuring total privacy for sensitive legal or engineering documents.

---

## System Requirements

- **Operating System:** Windows 10 / 11 (64-bit)
- **Hardware:** - Minimum: Modern Intel Core i5 / AMD Ryzen 5 CPU, 8 GB RAM
  - Recommended for Training/Fast Batch processing: Dedicated NVIDIA GPU (CUDA compatible, e.g., RTX 3060/4060 Laptop or Desktop)
- **External Dependencies (Packaged):** Poppler PDF utilities and Tesseract OCR engine binary.

---

## Development Installation

If you wish to run or modify the source code locally, follow these steps:

1. Clone or download the project repository.
2. Install Python 3.11 (64-bit) side-by-side if you are using newer incompatible versions globally.
3. Create a local virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: .\venv\Scripts\activate
   ```
4. Install PyTorch with CUDA 12.1 support (Required for GPU acceleration):
   ```bash
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
   ```
5. Install the remaining requirements:
   ```bash
   pip install ultralytics customtkinter pytesseract opencv-python presidio-analyzer spacy tkinterdnd2
   python -m spacy download ro_core_news_lg
   ```

---

## How to Run

Simply execute the main script from your virtual environment:
```bash
python app.py
```

---

## Production Compilation

To bundle the entire framework, including both AI models, NLP engines, Tesseract, and Poppler into a standalone Windows directory, run the following standalone `PyInstaller` pipeline command:

```powershell
pyinstaller --noconfirm --onedir --windowed --icon="icon.ico" --name "PDFDataBlocker" --add-data "icon.ico;." --add-data "signature_detector.pt;." --add-data "stamp_detector.pt;." --add-data "tesseract;tesseract" --add-data "poppler;poppler" --collect-all ultralytics --collect-all presidio_analyzer --collect-all spacy --collect-all ro_core_news_lg --hidden-import pytesseract --hidden-import cv2 --hidden-import pdf2image app.py
```

Once compiled, navigate to the `dist/PDFDataBlocker` directory and double-click `PDFDataBlocker.exe` to run the application natively on any Windows computer without requiring Python or terminal configurations.
