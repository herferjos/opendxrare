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
import shutil
import re

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from io import BytesIO
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import inch

# DEFINIMOS TODAS LAS FUNCIONES NECESARIAS

def generar_informe(df_list, names, header, text1):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
    elements = []

    # Add header as a larger header at the beginning of the PDF
    header_style = getSampleStyleSheet()['Heading1']
    header_style.fontSize = 24  # Bigger font size
    header_style.alignment = 1  # center alignment
    elements.append(Paragraph(header, header_style))
    elements.append(Spacer(0.1 * inch, 0.4 * inch))  # Add space after the header

    # Add name of text1 as a smaller header after the header
    text1_name_style = getSampleStyleSheet()['Heading2']
    text1_name_style.fontSize = 16  # Smaller font size
    text1_name_style.alignment = 0  # left alignment
    elements.append(Paragraph(names[0], text1_name_style))
    elements.append(Spacer(0.2 * inch, 0.2 * inch))  # Add space after the text1 name

    # Add text1 as a paragraph after the text1 name
    text1_style = getSampleStyleSheet()['Normal']
    text1_style.fontSize = 12
    elements.append(Paragraph(text1, text1_style))
    elements.append(Spacer(0.1 * inch, 0.4 * inch))  # Add space after text1

    for i, df in enumerate(df_list):
        if df.empty:
            continue
        else:
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, (df.applymap(lambda x: bool(str(x).strip())).any(axis=0))]

            style = getSampleStyleSheet()['Heading2']
            style.alignment = 0
            elements.append(Paragraph(names[i + 1], style))

            available_page_width = landscape(A4)[0] - 2 * inch  # Reducir el ancho disponible por un pequeño margen
            num_cols = len(df.columns)
            col_width = available_page_width / num_cols  # Ancho igual para todas las columnas

            body_text_style = ParagraphStyle(name='BodyText', fontName='Helvetica', fontSize=12)

            def create_hyperlink(cell_text, style):
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', cell_text)

                # Si se encontraron enlaces, crear un párrafo con todos los enlaces
                if links:
                    paragraphs = []
                    for label, url in links:
                        link = f'<a href="{url}">{label}</a>'
                        paragraphs.append(Paragraph(link, style))
                    return paragraphs
                else:
                    return [Paragraph(cell_text, style)]

            # Crear una lista de columnas con el ancho igual para todas
            column_widths = [col_width] * num_cols
            table_data = [[create_hyperlink(f'{col}', body_text_style) for col in df.columns]] + [
                [create_hyperlink(f'{col}', body_text_style) for col in row] for row in df.values]

            # Crear un estilo de tabla con bordes
            table_style = TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('BOTTOMPADDING', (0, -1), (-1, -1), 12),
            ])

            table = Table(table_data, colWidths=column_widths)
            table.setStyle(table_style)

            elements.append(table)
            elements.append(Spacer(0.1 * inch, 0.4 * inch))

    doc.build(elements)

    buffer.seek(0)
    return buffer

def enviar_informe_diagnostico(caso_clinico, destinatario_final, archivo_pdf_data):
    # Configurar los detalles del servidor SMTP de Gmail
    smtp_host = 'smtp.hostinger.com'
    smtp_port = 465  # Use port 465 for SMTP_SSL
    smtp_username = 'contacto@dxrare.com'
    smtp_password = st.secrets["email_pass"]

    # Configurar los detalles del mensaje
    sender = 'contacto@dxrare.com'
    recipients = [destinatario_final]  # Lista de destinatarios sin el usuario SMTP
    subject = 'Tu informe diagnóstico disponible'
    message = f"""Gracias por confiar en DxRare para tus diagnósticos clínicos. Adjunto en el propio email, podrás encontrar el informe con toda la información sobre el caso clínico que describiste: {caso_clinico}\nEl archivo pdf contiene tablas interactivas con enlaces a las bases de datos más utilizadas.\n\nPor otro lado, el archivo de texto (.txt) contiene información encriptada que nos permitirá retomar el proceso diagnóstico en futuras ocasiones.\n\nSimplemente si quieres continuar con dicho paciente, entra en analisisgenetico.es y sube el archivo .txt en el apartado 'Diagnóstico' y todo volverá como estaba.\n\n¡Te esperamos pronto en la plataforma DxRare!\n\nSaludos,\nEquipo DxRare"""

    # Crear el objeto MIME para el correo electrónico
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)  # Convertir la lista de destinatarios en una cadena separada por comas
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Obtener los bytes del objeto BytesIO
    archivo_pdf_bytes = archivo_pdf_data.getvalue()
    
    # Crear el objeto MIME para el archivo PDF
    attachment_pdf = MIMEApplication(archivo_pdf_bytes, _subtype="pdf")
    attachment_pdf.add_header('Content-Disposition', 'attachment', filename=f'informe_{caso_clinico}.pdf')
    msg.attach(attachment_pdf)

    # Iniciar la conexión SMTP con SMTP_SSL
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        # Iniciar sesión en la cuenta de correo
        server.login(smtp_username, smtp_password)

        # Enviar el correo electrónico
        server.send_message(msg)


def enviar_info_usuario(email):
    # Configurar los detalles del servidor SMTP de Gmail
    smtp_host = 'smtp.hostinger.com'
    smtp_port = 465  # Use port 465 for SMTP_SSL
    smtp_username = 'contacto@dxrare.com'
    smtp_password = st.secrets["email_pass"]

    # Configurar los detalles del mensaje
    sender = 'contacto@dxrare.com'
    recipients = ['contacto@dxrare.com', "jluis.hernandezfernandez@gmail.com"]  # Lista de destinatarios
    subject = 'Nuevo Registro en DxRare'
    message = f"""
    Nuevo registro en DxRare:
    email = {email}
    """

    # Crear el objeto MIME para el correo electrónico
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)  # Convertir la lista de destinatarios en una cadena separada por comas
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Iniciar la conexión SMTP con SMTP_SSL
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        # Iniciar sesión en la cuenta de correo
        server.login(smtp_username, smtp_password)

        # Enviar el correo electrónico
        server.send_message(msg)

def reconstruir_faiss():
    nombres_partes = ['vector_database/index.faiss.part1',
     'vector_database/index.faiss.part2',
     'vector_database/index.faiss.part3',
     'vector_database/index.faiss.part4',
     'vector_database/index.faiss.part5',
     'vector_database/index.faiss.part6',
     'vector_database/index.faiss.part7']
    
    # Abre el archivo faiss en modo escritura binaria
    with open("index.faiss", "wb") as f_out:
        # Para cada nombre de parte en la lista de nombres de las partes
        for nombre_parte in nombres_partes:
            # Abre el archivo de la parte en modo lectura binaria
            with open(nombre_parte, "rb") as f_in:
                # Copia el contenido del archivo de la parte al archivo faiss
                shutil.copyfileobj(f_in, f_out)
    
    return 


def chatbot(prompt):
    max_intentos = 3
    intentos = 0
    while intentos < max_intentos:
        try:
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


def extractor(caso_clinico):

    prompt = f"""Esta es la descripción clínica proporcionada por el usuario: '{caso_clinico}'
    """

    prompt = prompt + '''
    CONDICIONES

    Usted es un asistente médico para ayudar a extraer síntomas y fenotipos de un caso clínico.
    Sea preciso y no alucine con la información.

    MISIÓN

    Generar un diccionario en python que recoja los síntomas clínicos mencionados.

    FORMATO RESPUESTA:

    python dictionary -> {"original_symptoms": [], "symptoms_english":[]}

    ¡Recuerda extraer los síntomas médicos de la descripcion clínica proporcionada anteriormente y SOLO contestar con el diccionario en python para lo síntomas, nada más! Ten en cuenta que la descripción clínica puede estar en varios idiomas pero tu debes siempre responder con un listado en inglés y en el idioma original
    '''
    
    return chatbot(prompt)


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

    {"ID": ..., "Name": ...}

    """

    prompt = prompt + f"""Esta es la descripción del síntoma proporcionada: '{sintoma}'

    Esta son las posibilidades que he encontrado: {respuesta_database}
    ¡Recuerda SOLO contestar con el FORMATO de JSON en python, nada más! Recuerda contestar la columna "Name" en el idioma original del síntoma proporcionado.
    """
    return chatbot(prompt)
    

def jsoner(respuesta, instrucciones):
    max_intentos=3
    intentos = 0
    while intentos < max_intentos:
        try:
            respuesta_ok = respuesta.strip().replace(r'\_', '_'). replace('`', '')
            diccionario = json.loads(respuesta_ok)
            return diccionario
        except json.JSONDecodeError:
            if intentos < max_intentos - 1:
                prompt = """Responde únicamente con un diccionario json de python con la siguiente estructura:
                """
                prompt = prompt + f"""
                Instrucciones del JSON:
                {instrucciones}
                Respuesta mal formateada: {respuesta}"""
                respuesta = st.session_state.chatbot.query(prompt)['text']
            else:
                print("Se alcanzó el máximo número de intentos. La respuesta no se pudo convertir a JSON.")
                return None
        intentos += 1


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
            omim_data.append([omim_id_string, omim_mondo_id_string, omim_name, omim_score, omim_matched_hpo_id_string, ncbi_genes_omim_string, omim_inheritance])
        
        if i < len(orpha_response):
            orpha_item = orpha_response[i]
            orpha_id = orpha_item.get("id")
            orpha_id_string = f"[{orpha_id}](https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Lng=EN&Expert={orpha_id.split(':')[1]})"
            orpha_name = orpha_item.get("orpha_disease_name_en").capitalize()
            orpha_mondo_id = ", ".join(orpha_item.get("mondo_id", []))
            orpha_mondo_id_string = f"[{orpha_mondo_id}](https://monarchinitiative.org/disease/{orpha_mondo_id})"
            orpha_score = orpha_item.get("score")
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
            orpha_data.append([orpha_id_string, orpha_mondo_id_string, orpha_name, orpha_score, orpha_matched_hpo_id_string, ncbi_genes_orpha_string, orpha_inheritance])
    
    omim_df = pd.DataFrame(omim_data, columns=["ID", "MONDO ID", "Enfermedad", "Puntuación", "Síntomas en común", "Genes asociados", "Herencia"])
    orpha_df = pd.DataFrame(orpha_data, columns=["ID", "MONDO ID", "Enfermedad", "Puntuación", "Síntomas en común", "Genes asociados", "Herencia"])
    
    df = pd.concat([omim_df, orpha_df], axis=0, ignore_index=True)
    df['Score'] = df['Score'].astype(float)

    # Remove duplicates based on MONDO ID and keep the row with the highest score
    df = df.sort_values(by=["MONDO ID", "Puntuación"], ascending=[True, False])
    df = df.drop_duplicates(subset=["MONDO ID"], keep="first")
    
    # Get the top 10 rows based on score
    df = df.nlargest(5, "Puntuación")
    df = df.reset_index(drop=True)


    lista_diseases_id = df.iloc[:, 1].tolist()

    return df, lista_diseases_id


def orchest(description):
    respuesta = extractor(description)
    diccionario = jsoner(respuesta,'{"original_symptoms": [], "symptoms_english":[]}')
    lista_sintomas_english = diccionario['symptoms_english']
    lista_sintomas_original = diccionario['original_symptoms']

    lista_codigo_sintomas = []
    lista_nombre_sintomas = []

    for sintoma_en, sintoma_original in zip(lista_sintomas_english, lista_sintomas_original):
        respuesta2 = selector(search_database(sintoma_en), sintoma_original)
        diccionario_sintoma = jsoner(respuesta2, '{"ID": ..., "Name": ...}')
        codigo_sintoma = diccionario_sintoma["ID"]
        nombre_sintoma = diccionario_sintoma["Name"]
        lista_codigo_sintomas.append(codigo_sintoma)
        lista_nombre_sintomas.append(nombre_sintoma)

    df = pd.DataFrame({"Síntoma Original": lista_sintomas_original, "ID": lista_codigo_sintomas, "Nombre del ID": lista_nombre_sintomas})
    df['Original Symptom'] = df['Original Symptom'].str.capitalize()
    return df
