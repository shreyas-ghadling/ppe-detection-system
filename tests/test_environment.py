import torch
import cv2
import ultralytics
import pandas
import numpy
import matplotlib
import fastapi
import streamlit

print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("Ultralytics:", ultralytics.__version__)
print("Pandas:", pandas.__version__)
print("NumPy:", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
print("FastAPI:", fastapi.__version__)
print("Streamlit:", streamlit.__version__)

print("\nCUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))