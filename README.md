# Detector de Postura Humana con YOLOv8 🧘‍♂️🚀

Este proyecto utiliza Inteligencia Artificial para detectar la pose de una persona en tiempo real a través de la cámara web, calculando específicamente el ángulo de la cadera para determinar si el usuario está sentado o de pie.

## 🛠️ Tecnologías y Librerías
* **Python 3**
* **YOLOv8 (Ultralytics)** - Modelo `yolov8n-pose.pt`
* **OpenCV** - Procesamiento de video y manejo de la webcam
* **NumPy** - Cálculos matemáticos y trigonometría de ángulos

## 🚀 Cómo Ejecutar el Proyecto

1. **Activar el entorno virtual:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instalar las dependencias:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Correr la aplicación:**
   ```powershell
   python app.py
   ```