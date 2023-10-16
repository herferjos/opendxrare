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


# DEFINIMOS TODAS LAS FUNCIONES NECESARIAS
@st.cache_data(show_spinner=False, persist = True)
def extractor(caso_clinico):
    max_intentos = 3
    intentos = 0
    while intentos < max_intentos:
        try:
            prompt = f"""Esta es la descripción clínica proporcionada por el usuario: '{caso_clinico}'
            """
            prompt = prompt + '''
            CONDICIONES
            Usted es un asistente médico para ayudar a extraer síntomas y fenotipos de un caso clínico.
            Sea preciso y no alucine con la información.
            MISIÓN
            Generar un diccionario en python que recoja los síntomas clínicos mencionados.
            FORMATO RESPUESTA:
            python dictionary -> {"symptoms":[]}
            ¡Recuerda extraer los síntomas médicos de la descripcion clínica proporcionada anteriormente y SOLO contestar con el diccionario en python para los síntomas, nada más!
            '''
            id = st.session_state.chatbot.new_conversation()
            st.session_state.chatbot.change_conversation(id)
            respuesta = st.session_state.chatbot.query(prompt)['text']
            return respuesta
        except StopIteration:
            # Manejo del error StopIteration
            intentos += 1
            if intentos < max_intentos:
                # Espera unos segundos antes de intentar de nuevo
                time.sleep(2)
            else:
                st.error("Se alcanzó el máximo número de intentos. No se pudo obtener una respuesta válida.")
                return None

@st.cache_data(show_spinner=False, persist = True)
def search_database(query):
    k = 5
    query_vector = st.session_state.model.encode(query)

    # Buscar los vectores más similares al vector de consulta usando faiss como antes
    distances, indices = st.session_state.index_database.search(np.array([query_vector]), k)

    # Obtener los ID y Texts correspondientes a los vectores encontrados con mayor similaridad al texto de input usando ids_texts como antes
    results = []
    for i in range(k):
        result = {"ID": st.session_state.texts_database[indices[0][i]]["id"], "Text": st.session_state.texts_database[indices[0][i]]["text"]}
        results.append(result)

    return results
    
@st.cache_data(show_spinner=False, persist = True)
def selector(respuesta_database, sintoma):
    max_intentos = 3
    intentos = 0
    while intentos < max_intentos:
        try:
            prompt = """
            CONDICIONES
            Usted es un asistente médico para ayudar a elegir el síntoma correcto para cada caso.
            Sea preciso y no alucine con la información.
            MISIÓN
            Voy a hacer una búsqueda rápida de los síntomas posibles asociados a la descripción. Responde únicamente con el ID que mejor se ajuste al síntoma descrito
            """
            prompt = prompt + f"""Esta es la descripción del síntoma proporcionada: '{sintoma}'
            Esta son las posibilidades que he encontrado: {respuesta_database}"""

            id = st.session_state.chatbot.new_conversation()
            st.session_state.chatbot.change_conversation(id)
            respuesta = st.session_state.chatbot.query(prompt)['text']
            prompt2= """
            Contestame con el siguiente formato a la pregunta que te hice:
            FORMATO RESPUESTA:
            {"ID": <HPO_ID>, "Name": <HPO_NAME>"}
            ¡Recuerda SOLO elegir el síntoma más adecuado y contestar con el FORMATO RESPUESTA DADO para que pueda ser cargado como json en python!
            """
            respuesta2 = st.session_state.chatbot.query(prompt2)['text']
            return respuesta2
        except StopIteration:
            # Manejo del error StopIteration
            intentos += 1
            if intentos < max_intentos:
                # Espera unos segundos antes de intentar de nuevo
                time.sleep(2)
            else:
                st.error("Se alcanzó el máximo número de intentos. No se pudo obtener una respuesta válida.")
                return None

@st.cache_data(show_spinner=False, persist = True)
def get_ranked_list(hpo_ids):
    omim_url = "https://pubcasefinder.dbcls.jp/api/get_ranked_list?target=omim&format=json&hpo_id={}".format(",".join(hpo_ids))
    orpha_url = "https://pubcasefinder.dbcls.jp/api/get_ranked_list?target=orphanet&format=json&hpo_id={}".format(",".join(hpo_ids))
    
    omim_response = requests.get(omim_url).json()
    orpha_response = requests.get(orpha_url).json()
    
    omim_data = []
    orpha_data = []
    
    for i in range(10):
        if i < len(omim_response):
            omim_item = omim_response[i]
            omim_id = omim_item.get("id")
            omim_id_string = f"[{omim_id}](https://omim.org/entry/{omim_id.split(':')[1]})"
            omim_name = omim_item.get("omim_disease_name_en").capitalize()
            omim_mondo_id = ", ".join(omim_item.get("mondo_id", []))
            omim_mondo_id_string = f"[{omim_mondo_id}](https://monarchinitiative.org/disease/{omim_mondo_id})"
            omim_score = omim_item.get("score")
            omim_description = omim_item.get("description")
            omim_matched_hpo_id = omim_item.get("matched_hpo_id")
            omim_matched_hpo_id_string = ", ".join([f"[{codigo}](https://monarchinitiative.org/phenotype/{codigo})" for codigo in omim_matched_hpo_id.split(',')])
            ncbi_genes_omim = omim_item.get("ncbi_gene_id")
            if ncbi_genes_omim is not None:
                ncbi_genes_omim_string = ", ".join([f"[NCBIGene:{ncbi_id.split(':')[1]}](https://www.ncbi.nlm.nih.gov/gene/{ncbi_id.split(':')[1]})" for ncbi_id in ncbi_genes_omim])
            else:
                ncbi_genes_omim_string = "" 
            omim_inheritance = omim_item.get("inheritance_en")
            if omim_inheritance:
                omim_inheritance = ", ".join([v for k, v in omim_inheritance.items()])
            omim_data.append([omim_id_string, omim_mondo_id_string, omim_name, omim_score, omim_description, omim_matched_hpo_id_string, ncbi_genes_omim_string, omim_inheritance])
        
        if i < len(orpha_response):
            orpha_item = orpha_response[i]
            orpha_id = orpha_item.get("id")
            orpha_id_string = f"[{orpha_id}](https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Lng=EN&Expert={orpha_id.split(':')[1]})"
            orpha_name = orpha_item.get("orpha_disease_name_en").capitalize()
            orpha_mondo_id = ", ".join(orpha_item.get("mondo_id", []))
            orpha_mondo_id_string = f"[{orpha_mondo_id}](https://monarchinitiative.org/disease/{orpha_mondo_id})"
            orpha_score = orpha_item.get("score")
            orpha_description = orpha_item.get("description")
            orpha_matched_hpo_id = orpha_item.get("matched_hpo_id")
            orpha_matched_hpo_id_string = ", ".join([f"[{codigo}](https://monarchinitiative.org/phenotype/{codigo})" for codigo in orpha_matched_hpo_id.split(',')])
            ncbi_genes_orpha = orpha_item.get("ncbi_gene_id")
            if ncbi_genes_orpha is not None:
              ncbi_genes_orpha_string = ", ".join([f"[NCBIGene:{ncbi_id.split(':')[1]}](https://www.ncbi.nlm.nih.gov/gene/{ncbi_id.split(':')[1]})" for ncbi_id in ncbi_genes_orpha])
            else:
              ncbi_genes_orpha_string = ""         
            orpha_inheritance = orpha_item.get("inheritance_en")
            if orpha_inheritance:
                orpha_inheritance = ", ".join([v for k, v in orpha_inheritance.items()])
            orpha_data.append([orpha_id_string, orpha_mondo_id_string, orpha_name, orpha_score, orpha_description, orpha_matched_hpo_id_string, ncbi_genes_orpha_string, orpha_inheritance])
    
    omim_df = pd.DataFrame(omim_data, columns=["ID", "MONDO ID", "Disease", "Score", "Description", "Shared Phenotypes", "Associated Genes", "Inheritance"])
    orpha_df = pd.DataFrame(orpha_data, columns=["ID", "MONDO ID", "Disease", "Score", "Description", "Shared Phenotypes", "Associated Genes", "Inheritance"])
    
    df = pd.concat([omim_df, orpha_df], axis=0, ignore_index=True)
    df['Score'] = df['Score'].astype(float)

    # Remove duplicates based on MONDO ID and keep the row with the highest score
    df = df.sort_values(by=["MONDO ID", "Score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["MONDO ID"], keep="first")
    
    # Get the top 10 rows based on score
    df = df.nlargest(5, "Score")
    df = df.reset_index(drop=True)


    lista_diseases_id = df.iloc[:, 1].tolist()

    return df, lista_diseases_id


@st.cache_data(show_spinner=False, persist = True)
def jsoner(respuesta, max_intentos=3):
    intentos = 0
    while intentos < max_intentos:
        try:
            diccionario = json.loads(respuesta)
            return diccionario
        except json.JSONDecodeError:
            if intentos < max_intentos - 1:
                prompt = """Formatea la respuesta correctamente a un diccionario en python:
                """
                prompt = prompt + f"Respuesta mal formateada: {respuesta}"
                respuesta = st.session_state.chatbot.query(prompt)['text']
            else:
                print("Se alcanzó el máximo número de intentos. La respuesta no se pudo convertir a JSON.")
                return None
        intentos += 1

@st.cache_data(show_spinner=False, persist = True)
def orchest(description):
    respuesta = extractor(description)
    diccionario = jsoner(respuesta)
    lista_sintomas = diccionario['symptoms']

    lista_codigo_sintomas = []
    lista_nombre_sintomas = []

    for sintoma in lista_sintomas:
        respuesta2 = selector(search_database(sintoma), sintoma)
        diccionario_sintoma = jsoner(respuesta2)
        codigo_sintoma = diccionario_sintoma["ID"]
        nombre_sintoma = diccionario_sintoma["Name"]
        lista_codigo_sintomas.append(codigo_sintoma)
        lista_nombre_sintomas.append(nombre_sintoma)

    df = pd.DataFrame({"Original Symptom": lista_sintomas, "ID": lista_codigo_sintomas, "Name HPO ID": lista_nombre_sintomas})

    return df



# ----------------------------------------------- FRONTED STREAMLIT APP ------------------------------------------------------------------------------------

st.set_page_config(page_title="OpenDxRare", page_icon="🧬", layout="wide")

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

descripcion = st.text_area(label = "Clinical Description")

if st.button(label = "Extract symptoms", type = "primary"):
    df_sintomas = orchest(descripcion)
    st.write(df_sintomas)
    st.write("---")
    if st.button(label = "Diagnose symptoms", type = "primary"):
        lista_codigos = df_sintomas["ID"].to_list()
        tabla, lista_ids = get_ranked_list(lista_codigos)
        st.markdown(tabla.to_markdown(index=False), unsafe_allow_html=True)
