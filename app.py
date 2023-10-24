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
from streamlit_option_menu import option_menu
from modules import chatbot, extractor, search_database, selector, jsoner, get_ranked_list, orchest, reconstruir_faiss

# INICIAMOS TODAS LAS VARIABLES ESTÁTICAS NECESARIAS

if 'index_database' not in st.session_state:
    reconstruir_faiss()
    st.session_state['index_database'] = faiss.read_index("index.faiss")

if 'texts_database' not in st.session_state:
    with open('vector_database/texts.pkl', 'rb') as f:
        st.session_state['texts_database'] = pickle.load(f)

if 'model' not in st.session_state:
    st.session_state['model'] = SentenceTransformer('joseluhf11/symptom_encoder_v10')

if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = hugchat.ChatBot(cookie_path='hugchat_cookies.json')


# ----------------------------------------------- FRONTED STREAMLIT APP ------------------------------------------------------------------------------------

st.set_page_config(page_title="OpenDxRare", page_icon="🧬", layout="wide")

st.markdown(
  """
  <div style='text-align: center;'>
      <h1>🧬 DxRare 🧬</h1>
      <h4>Mejorando el proceso de diagnóstico</h4>
  </div>
  """,
    unsafe_allow_html=True
)
st.write("---")

st.markdown("<h3 style='text-align: center;'>¡Bienvenidos a la plataforma DxRare!</h4>", unsafe_allow_html=True)
st.markdown("""<h5 style='text-align: justify;'>
      Esta iniciativa pretende <b>revolucionar la forma en que se diagnostican las enfermedades genéticas.</b> 
  </h5>""", unsafe_allow_html=True)
st.markdown("""<h5 style='text-align: justify;'>
      Sabemos lo difícil que es para los profesionales de la salud ofrecer un diagnóstico preciso y rápido a pacientes con patologías genéticas. 
  </h5>""", unsafe_allow_html=True)

with st.expander("¿Cómo funciona la plataforma?"):
    st.video("https://youtu.be/6owq8uIESqA")
    
st.write("---")

descripcion = st.text_area(label = "Descripción Clínica")

if st.button(label = "Extraer Síntomas", type = "primary"):
    st.session_state['df_sintomas'] = orchest(descripcion)

if 'df_sintomas' in st.session_state:
    st.data_editor(st.session_state.df_sintomas, use_container_width=True, num_rows="dynamic", disabled=False)
    
    if st.button(label = "Diagnosticar Síntomas", type = "primary"):
        lista_codigos = st.session_state.df_sintomas["ID"].to_list()
        tabla, lista_ids = get_ranked_list(lista_codigos)
        st.session_state['tabla'] = tabla

if 'tabla' in st.session_state:
    st.write("---")
    st.markdown(st.session_state.tabla.to_markdown(index=False), unsafe_allow_html=True)
