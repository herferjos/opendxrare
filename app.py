# IMPORTAMOS TODOS LOS MÓDULOS NECESARIOS
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from hugchat import hugchat
import numpy as np
import requests
import pandas as pd
import json
import os
import glob
import tabulate
import time
from st_paywall import add_auth
from streamlit_option_menu import option_menu
from modules import chatbot, extractor, search_database, selector, jsoner, get_ranked_list, orchest

# INICIAMOS TODAS LAS VARIABLES ESTÁTICAS NECESARIAS

if 'model' not in st.session_state:
    st.session_state['model'] = SentenceTransformer('joseluhf11/symptom_encoder')

if 'index_database' not in st.session_state:

    with open("index.faiss", 'wb') as f:
        for root, _, files in os.walk("vector_database/"):
            for file_name in sorted(files):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'rb') as chunk_file:
                    chunk = chunk_file.read()
                    f.write(chunk)

    st.session_state['index_database'] = faiss.read_index("index.faiss")

if 'texts_database' not in st.session_state:
    with open('vector_database/texts.pkl', 'rb') as f:
        st.session_state['texts_database'] = pickle.load(f)

if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = hugchat.ChatBot(cookie_path='hugchat_cookies.json')


# ----------------------------------------------- FRONTED STREAMLIT APP ------------------------------------------------------------------------------------

st.set_page_config(page_title="OpenDxRare", page_icon="🧬", layout="wide")

with st.sidebar:
   selectec = option_menu(
      menu_title = "DxRare",
      options=["Home", "Diagnose", "Terms and conditions"],
      icons=["house", "clipboard", "file-text-fill"],
      menu_icon = "cast",
      default_index = 0,
   )

st.markdown(
  """
  <div style='text-align: center;'>
      <h1>🧬 DxRare 🧬</h1>
      <h4>Empowering clinicians in the diagnostic process</h4>
  </div>
  """,
    unsafe_allow_html=True
)
st.write("---")

add_auth(required=True)

descripcion = st.text_area(label = "Clinical Description")

if st.button(label = "Extract symptoms", type = "primary"):
    st.session_state['df_sintomas'] = orchest(descripcion)

if 'df_sintomas' in st.session_state:
    st.data_editor(st.session_state.df_sintomas, use_container_width=True, num_rows="dynamic", disabled=False)
    
    if st.button(label = "Diagnose symptoms", type = "primary"):
        lista_codigos = st.session_state.df_sintomas["ID"].to_list()
        tabla, lista_ids = get_ranked_list(lista_codigos)
        st.session_state['tabla'] = tabla

if 'tabla' in st.session_state:
    st.write("---")
    st.markdown(st.session_state.tabla.to_markdown(index=False), unsafe_allow_html=True)
