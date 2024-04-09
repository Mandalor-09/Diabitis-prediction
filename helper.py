import joblib
import pathlib
import textwrap
from IPython.display import display
from IPython.display import Markdown
import google.generativeai as genai
import streamlit as st
import os
from PIL import Image
<<<<<<< HEAD
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

from llama_index.core import TreeIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
=======
>>>>>>> origin/main

model_filename = 'model/diabaties_model_og.joblib'
model = joblib.load(model_filename)

def prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Convert input values to numeric
    a = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    a = [float(val) for val in a]  # Convert to float
    
    # Make prediction
    prediction = model.predict([a])[0] 

    if int(prediction) == 1:
        print('Patient is Diabetic')
        return 'Diabetic', a
    else:
        print('Patient is non Diabetic')
        return 'Non Diabetic', a

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


def analysis_diet(gemini_model,photo,result):
    if result is not None:
        photo = st.file_uploader('upload image',type=['jpg','png','jpeg'])
        if photo is not None:
            st.image(photo, width=100)
            photo = save_uploaded_image_and_ret_img_obj(photo)
            response = gemini_model.generate_content([f"You are a helpful assistant with good knowledge regarding diet planning and healthy eating analysis. You will be given an image of a person's food and data indicating whether they are diabetic or not diabetic. Analyze the image and suggest a diet that will help them be fit and manage their diabetes (if applicable).The Person is {result}",photo], stream=True)
            response.resolve()
            data = to_markdown(response.text)
            st.write(data.data)
    else:
        st.error('Please check your Prediction')
<<<<<<< HEAD


def load_index():
    client = QdrantClient(
        url="https://4be798a7-e853-471c-b326-964ea14cd209.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="BWA-fIGL31hpFlewoYzDALpRVYOzIHw0XzmvdLm19E6R0MApBFITFQ",
    )
    vector_store = QdrantVectorStore(client=client, collection_name="collection")  # replace with your collection name
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.embed_model = GeminiEmbedding(
        model_name="models/embedding-001", api_key="AIzaSyCrMHJ51_2MLEKOykxITqxhW-c3RU37waU"
    )
    Settings.llm = Gemini(model = "models/gemini-pro-vision",api_key="AIzaSyCGWec3bDfEVfXRhD2pyb3w1hK8sjbclqU")

    loaded_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return loaded_index

def initialize_agent(loaded_index):
    tool1 = QueryEngineTool.from_defaults(
    query_engine=loaded_index.as_query_engine(),
    description="Use this query engine to if the user want to know about dishes , it's recipe & its origin",
    )
    #gemini_llm = Gemini(model="models/gemini-pro-vision",api_key="AIzaSyCGWec3bDfEVfXRhD2pyb3w1hK8sjbclqU")
    openai_llm = OpenAI()
    agent = ReActAgent.from_tools([tool1], llm=openai_llm, verbose=True)
    return agent
=======
>>>>>>> origin/main
