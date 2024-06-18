# Python In-built packages
from pathlib import Path
import PIL
import pywt
from PIL import Image,ImageEnhance
from SSIM_PIL import compare_ssim
import cv2
import numpy as np
# External packages
import streamlit as st
import subprocess

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Pedestrian detection  using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Pedestrian Detection using YOLOv8")

# Sidebar
st.sidebar.header("Configuration")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# st.sidebar.header("Configuration")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image.", type=("jpg", "jpeg", "png"))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                
                default_image = PIL.Image.open(default_image_path)

                st.image(default_image_path, caption="Input Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                uploaded_image.save("uploaded_image.png")
                img=ImageEnhance.Brightness(uploaded_image)
                img=img.enhance(15.0) 
                img.save("Output_Image.png")
                st.image(uploaded_image, caption="Uploaded Image",
                         use_column_width=True)                

        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image, caption='Output Image after Detection',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Pedstrain'):
                uploaded_image1 = PIL.Image.open("Output_Image.png")
                res = model.predict(uploaded_image1,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    
    subprocess.run(["streamlit", "run", "frame.py"])


else:
    st.error("Please select a valid source type!")


img1 = cv2.imread("uploaded_image.png")
img2 = cv2.imread("Output_Image.png")
psnr = cv2.PSNR(img1, img2)
mse = np.mean((img1 - img2) ** 2) 
#SSIM
acc=mse/1.2
image1 = Image.open("uploaded_image.png")
image2 = Image.open("Output_Image.png")
value = compare_ssim(image1, image2)
print("### Cover and Embedded Image ###")
print(f"The MSE Value : {mse}")
print(f"The PSNR Value : {psnr} db")
print(f"The SSIM Value : {value} ")
print(f"The Accuracy Value : {acc} ")

print("\n")
