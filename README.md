<div align="center">

# 🔬 ThresYm
### Advanced Image Thresholding Workbench

*Formerly known as **BinariX***

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A powerful Computer Vision tool for binary image segmentation using advanced thresholding algorithms — with real-time parameter tuning and side-by-side comparison.

</div>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **6 Segmentation Methods** | Simple, Otsu, Adaptive (Mean/Gaussian), Niblack, and Sauvola |
| ⚡ **Real-time Tuning** | Adjust parameters via sliders and see results instantly |
| 🖼️ **Side-by-Side Comparison** | Compare original image with all processed results |
| 📊 **Detailed Analytics** | White vs Black pixel percentage breakdown per method |
| 💾 **High-Res Export** | Download processed images in PNG format |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (Custom Cyberpunk UI)
- **Image Processing:** OpenCV, Scikit-Image, NumPy
- **Language:** Python

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure Python is installed, then install the required libraries:

```bash
pip install streamlit opencv-python numpy Pillow scikit-image
```

### 2. Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ThresYm.git
cd ThresYm
```

### 3. Run the App

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` 🚀

---

## 📖 Algorithms Included

| Method | Type | Best For |
|---|---|---|
| **Simple Threshold** | Global | Uniform lighting images |
| **Otsu's Method** | Auto | Bimodal histogram images |
| **Adaptive Mean** | Local | Varying illumination |
| **Adaptive Gaussian** | Local | Noisy images with local variation |
| **Niblack** | Advanced | Low-contrast document images |
| **Sauvola** | Advanced | Document & Text scanning |

---

## 🗂️ Project Structure

```
ThresYm/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── assets/              # Sample images & screenshots
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

## 👨‍💻 Author

**Mayank Kumar Mishra**


⭐ *If you found this project helpful, please give it a star!* ⭐

</div>
