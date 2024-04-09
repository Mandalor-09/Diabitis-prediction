import streamlit as st
import google.generativeai as genai
import pathlib
import textwrap
from IPython.display import display
from IPython.display import Markdown
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def save_uploaded_image_and_ret_img_obj(uploaded_file):
    os.makedirs("Uploads", exist_ok=True)
    image_path = f'Uploads/{uploaded_file.name}'
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    img = Image.open(image_path)
    return img
    

photo = st.file_uploader('upload image',type=['jpg','png','jpeg'])
if photo is not None:
    st.image(photo, width=100)
    photo = save_uploaded_image_and_ret_img_obj(photo)
    response = model.generate_content(["You are a helpful assistant with good knowledge regarding diet planning and healthy eating analysis. You will be given an image of a person's food and data indicating whether they are diabetic or not diabetic. Analyze the image and suggest a diet that will help them be fit and manage their diabetes (if applicable).The Person is Diabatic",        photo], stream=True)
    response.resolve()
    data = to_markdown(response.text)
    st.write(data.data)