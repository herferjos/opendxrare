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

def extractor(caso_clinico):
    prompt = '''
    CONDICIONES

    Usted es un asistente médico para ayudar a extraer síntomas y fenotipos de un caso clínico.
    Sea preciso y no alucine con la información.

    MISIÓN

    Generar un diccionario en python que recoja los síntomas clínicos mencionados.

    FORMATO RESPUESTA:

    python dictionary -> {"symptoms":[]}

    '''

    prompt = prompt + f"""Esta es la descripción clínica proporcionada por el usuario: '{caso_clinico}'

    Recuerda SOLO contestar con el diccionario, nada más
    """
    respuesta = st.session_state.chatbot.query(prompt)['text']
    return respuesta


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

def selector(respuesta_database, sintoma):

    prompt = """
    CONDICIONES

    Usted es un asistente médico para ayudar a elegir el síntoma correcto para cada caso.
    Sea preciso y no alucine con la información.

    MISIÓN

    Voy a hacer una búsqueda rápida de los síntomas posibles asociados a la descripción. Responde únicamente con el ID que mejor se ajuste al síntoma descrito

    FORMATO RESPUESTA:

    SOLO EL HPO_ID CORRECTO, NADA MÁS

    """

    prompt = prompt + f"""Esta es la descripción del síntoma proporcionada: '{sintoma}'

    Esta son las posibilidades que he encontrado: {respuesta_database}
    ¡Recuerda SOLO contestar con el HPO_ID, nada más!
    """
    respuesta = st.session_state.chatbot.query(prompt)['text']
    return respuesta

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
            omim_id_string = f"[{omim_id}](https://omim.org/entry/{omim_id.split(':')})"
            omim_name = omim_item.get("omim_disease_name_en").capitalize()
            omim_mondo_id = omim_item.get("mondo_id")
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
            omim_gene_reviews_url = f"[Gene Review]({omim_item.get('gene_reviews_url')})"   
            omim_inheritance = omim_item.get("inheritance_en")
            if omim_inheritance:
                omim_inheritance = ", ".join([v for k, v in omim_inheritance.items()])
            omim_data.append([omim_id_string, omim_mondo_id_string, omim_name, omim_score, omim_description, omim_matched_hpo_id_string, ncbi_genes_omim_string, omim_gene_reviews_url, omim_inheritance])
        
        if i < len(orpha_response):
            orpha_item = orpha_response[i]
            orpha_id = orpha_item.get("id")
            orpha_id_string = f"[{orpha_id}](https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Lng=EN&Expert={orpha_id.split(':')})"
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
            orpha_gene_reviews_url = f"[Gene Review]({orpha_item.get('gene_reviews_url')})"               
            orpha_inheritance = orpha_item.get("inheritance_en")
            if orpha_inheritance:
                orpha_inheritance = ", ".join([v for k, v in orpha_inheritance.items()])
            orpha_data.append([orpha_id_string, orpha_mondo_id_string, orpha_name, orpha_score, orpha_description, orpha_matched_hpo_id_string, ncbi_genes_orpha_string, orpha_gene_reviews_url, orpha_inheritance])
    
    omim_df = pd.DataFrame(omim_data, columns=["ID", "MONDO ID", "Disease", "Score", "Description", "Shared Phenotypes", "Associated Genes", "Gene Reviews URL", "Inheritance"])
    orpha_df = pd.DataFrame(orpha_data, columns=["ID", "MONDO ID", "Disease", "Score", "Description", "Shared Phenotypes", "Associated Genes", "Gene Reviews URL", "Inheritance"])
    
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


def orchest(description):
    respuesta = extractor(description)
    diccionario = json.loads(respuesta)
    lista_sintomas = diccionario['symptoms']

    lista_codigos = []

    for sintoma in lista_sintomas:
        codigo_sintoma = selector(search_database(sintoma), sintoma)
        lista_codigos.append(codigo_sintoma)

    tabla, lista_ids = get_ranked_list(lista_codigos)

    return tabla, lista_ids



# ----------------------------------------------- FRONTED STREAMLIT APP ------------------------------------------------------------------------------------


st.title("Prueba")

descripcion = st.text_input(label = "Descripcion")

if st.button(label = "Enviar"):
    respuesta = orchest(descripcion)
    
    st.markdown(respuesta[0].to_markdown(index=False), unsafe_allow_html=True)

# prompt_prueba = """

# CONDICIONES

# Usted es un asistente médico para ayudar a extraer síntomas y fenotipos de un caso clínico.
# Sea preciso y no alucine con la información.

# MISIÓN

# Generar un diccionario en python que recoja los síntomas clínicos mencionados.

# FORMATO RESPUESTA:

# python dictionary -> {'symptoms':[]}

# Esta es la descripción clínica proporcionada por el usuario: '
# Nombre del paciente: María Pérez

# Edad: 35 años

# Género: Femenino

# Ocupación: Maestra de primaria

# Antecedentes médicos: No hay antecedentes médicos significativos.

# Síntomas:

# Fiebre: María ha experimentado fiebre intermitente durante los últimos 5 días, con una temperatura que oscila entre 38°C y 39.5°C.

# Tos seca: Presenta una tos persistente y seca que ha empeorado en los últimos días. A menudo siente irritación en la garganta.

# Dificultad para respirar: María ha notado dificultad para respirar, especialmente al realizar actividades físicas leves, como subir escaleras o caminar distancias cortas.

# Fatiga extrema: Se siente agotada constantemente y ha perdido interés en las actividades diarias debido a la fatiga.

# Dolor de cabeza: María ha tenido dolores de cabeza intensos y persistentes, principalmente en la frente y en la parte posterior de la cabeza.

# Dolor muscular y articular: Ha experimentado dolores musculares y articulares en todo el cuerpo, especialmente en los hombros y las piernas.

# Pérdida del sentido del gusto y del olfato: María ha notado que ha perdido la capacidad de saborear y oler los alimentos, lo que ha afectado su apetito.

# Congestión nasal: Presenta congestión nasal intermitente, lo que dificulta su capacidad para respirar por la nariz.

# Pérdida de peso involuntaria: Durante las últimas semanas, María ha perdido aproximadamente 3 kilogramos sin intentar hacerlo.

# Malestar general: María se siente mal en general, con un estado de ánimo bajo y falta de energía.
# '

# Recuerda SOLO contestar con el diccionario, nada más
# """

# respuesta = st.session_state.chatbot.query(prompt_prueba)['text']

# st.write(respuesta)
