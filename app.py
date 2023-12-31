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
import re
from modules import chatbot, extractor, search_database, selector, jsoner, get_ranked_list, orchest, reconstruir_faiss, enviar_informe_diagnostico, enviar_info_usuario, generar_informe, enviar_email_seguimiento

# INICIAMOS TODAS LAS VARIABLES ESTÁTICAS NECESARIAS

if 'index_database' not in st.session_state:
    # reconstruir_faiss()
    st.session_state['index_database'] = faiss.read_index("index.faiss")

if 'texts_database' not in st.session_state:
    with open('texts.pkl', 'rb') as f:
        st.session_state['texts_database'] = pickle.load(f)

if 'model' not in st.session_state:
    st.session_state['model'] = SentenceTransformer('joseluhf11/symptom_encoder_v9')

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

if 'email' in st.session_state:
    st.markdown("<h3 style='text-align: center;'>🎉¡Ya puedes empezar a navegar por la plataforma!🎉</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>¿Quieres probar con un caso clínico ficticio para empezar?</h3>", unsafe_allow_html=True)      

    columnas1, columnas2 = st.tabs(["Demo paciente 1", "Demo paciente 2"])
    with columnas1:
      st.markdown("[Han J, Young JW, Frausto RF, Isenberg SJ, Aldave AJ. X-linked Megalocornea Associated with the Novel CHRDL1 Gene Mutation p.(Pro56Leu*8). Ophthalmic Genet. 2015 Jun;36(2):145-8. doi: 10.3109/13816810.2013.837187. Epub 2013 Sep 27. PMID: 24073597; PMCID: PMC3968246.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3968246/)")
      if st.button(label = "Demo paciente 1", type = "primary"):
        with open('demo_1.txt', "r",encoding="utf-8") as archivo:
          content = archivo.read()
        exec(content)
        st.rerun()
        st.success("Información clínica cargada correctamente")

    with columnas2:
      st.markdown("[Brizola E, Gnoli M, Tremosini M, Nucci P, Bargiacchi S, La Barbera A, Giglio S, Sangiorgi L. Variable clinical expression of Stickler Syndrome: A case report of a novel COL11A1 mutation. Mol Genet Genomic Med. 2020 Sep;8(9):e1353. doi: 10.1002/mgg3.1353. Epub 2020 Jun 17. PMID: 32558342; PMCID: PMC7507508.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7507508/)")
      if st.button(label = "Demo paciente 2", type = "primary"):
        st.session_state['uploaded_file'] = True
        with open('demo_2.txt', "r", encoding="utf-8") as archivo:
          content = archivo.read()
        exec(content)
        st.rerun()
        st.success("Información clínica cargada correctamente")
else:
    st.write("### Comienza a diagnosticar")
    st.info("¡Gracias a ti podemos seguir avanzando, déjanos tu email antes de empezar a usar la plataforma para concerte y que nos aportes tu opinión al final de la sesión!")
    with st.form(key="formulario"):
        email = st.text_input(label='Email', placeholder="Escribe un email con el que poder contactar")
        with st.expander("Términos y Condiciones"):
              st.write("""
              Términos y Condiciones de DxRare - Plataforma de Procesamiento de Información Clínica
            
              Última actualización: 18 de mayo de 2023
            
              Por favor, lee detenidamente los siguientes términos y condiciones ("Términos") antes de utilizar la plataforma DxRare ("nosotros", "nuestro" o "nosotros"). Al acceder y utilizar nuestra plataforma, aceptas cumplir con estos Términos. Si no estás de acuerdo con estos Términos, no utilices nuestra plataforma.
            
              Uso y Consentimiento
              1.1. DxRare es una plataforma en línea que procesa información clínica del paciente y realiza búsquedas en bases de datos para ayudar a los médicos en el diagnóstico. Al proporcionar información en la plataforma, se entiende que el paciente ha otorgado su consentimiento para el procesamiento de su información clínica y la realización de búsquedas relacionadas.
              1.2. Es responsabilidad del médico utilizar y evaluar la información proporcionada por DxRare de manera profesional y ética. El médico es el único responsable de cualquier decisión médica basada en la información obtenida a través de la plataforma.
            
              Responsabilidad y Limitaciones
              2.1. DxRare no almacena ninguna información clínica proporcionada por los pacientes. Toda la información se borra automáticamente después de cerrar la sesión en la plataforma. Sin embargo, no podemos garantizar la seguridad absoluta de la transmisión de datos a través de internet, y no nos hacemos responsables de cualquier acceso no autorizado o divulgación de información que esté fuera de nuestro control.
              2.2. DxRare no se hace responsable de la exactitud, integridad o confiabilidad de los datos proporcionados por las bases de datos a las que accede. Los resultados obtenidos de las búsquedas en bases de datos deben ser evaluados por el médico a su propia discreción y juicio profesional.
              2.3. En ningún caso, DxRare o sus afiliados serán responsables por cualquier daño directo, indirecto, incidental, especial o consecuencial derivado del uso de la plataforma, incluso si se ha informado de la posibilidad de dichos daños.
            
              Privacidad y Confidencialidad
              3.1. DxRare se compromete a proteger la privacidad y confidencialidad de la información del paciente. No compartiremos ni venderemos información personal identificable a terceros sin el consentimiento explícito del paciente, a menos que así lo exija la ley o se requiera para el adecuado funcionamiento de la plataforma.
              3.2. La información recopilada y procesada por DxRare se utilizará únicamente con fines de diagnóstico médico y mejora de la plataforma. Consulta nuestra Política de Privacidad para obtener más información sobre cómo manejamos tus datos.
            
              Propiedad Intelectual
              4.1. Todos los derechos de propiedad intelectual relacionados con DxRare, incluidos, entre otros, los derechos de autor, marcas comerciales, nombres comerciales y diseños, son propiedad de DxRare o se utilizan con permiso de los propietarios correspondientes. No se otorga ninguna licencia o derecho de uso sobre la propiedad intelectual sin el consentimiento previo por escrito de DxRare.
            
              Modificaciones de los Términos
              5.1. Nos reservamos el derecho de modificar o actualizar estos Términos en cualquier momento y a nuestra discreción. Cualquier modificación de los Términos será efectiva a partir de su publicación en la plataforma. Se te recomienda revisar periódicamente los Términos para estar al tanto de cualquier cambio.
            
              Al utilizar la plataforma DxRare, aceptas cumplir con estos Términos y cualquier otro aviso legal o pautas operativas publicadas en la plataforma. Si no estás de acuerdo con estos Términos, te rogamos que no utilices nuestra plataforma.
            
              Si tienes alguna pregunta o duda sobre estos Términos, contáctanos a través de los canales proporcionados en la plataforma.
            
              ¡Gracias por confiar en DxRare para el procesamiento de información clínica y diagnóstico médico!
              """)
        consentimiento = st.checkbox("Acepto los términos y condiciones de la plataforma DxRare")
        aceptar_boton = st.form_submit_button(label="¡Empezar!")
        
        if consentimiento and aceptar_boton:
          if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            st.session_state['consentimiento'] = True
            st.session_state['email'] = email
            if email == "prueba@gmail.com":
                st.rerun()
            else:
                enviar_info_usuario(email)
                enviar_email_seguimiento(email)
                st.rerun()
          else:
              st.warning("Por favor, introduce un email válido y acepta los términos y condiciones")

#---------------------------------------------------------------------------------------------------------------------------------------------------
st.write("### Demo de la plataforma")   
with st.expander("⬇️ Vídeo ⬇️"):
    st.video("https://youtu.be/9XC86eTTWDA")
    
st.write("---")

if 'email' in st.session_state:
    st.markdown("<h2 style='text-align: center;'>¡Bienvenido a la plataforma DxRare!</h2>", unsafe_allow_html=True)
    st.write("### 1) Introduce la descripción clínica de tu paciente")
    st.info("Copia y pega en la cajita la situación de tu paciente o bien copia y pega el texto de la historia clínica, como prefieras")
    descripcion = st.text_area(label=":blue[Descripción clínica]", placeholder="Escribe aquí...")
    
    if st.button(label = "Extraer Síntomas", type = "primary"):
        st.session_state['description'] = descripcion
        with st.spinner("Estamos procesando tu petición, puede tardar unos minutos..."):
            st.session_state['df_sintomas'] = orchest(descripcion)

    st.write("---")
    if 'description' in st.session_state:
        st.write("#### Descripción Clínica:")
        st.write(st.session_state.description)
        
    if 'df_sintomas' in st.session_state:
        st.write('### 2) Fenotipos encontrados')
        st.info("Selecciona de la lista proporcionada los fenotipos que deseas añadir al proceso de diagnóstico")
        st.data_editor(st.session_state.df_sintomas, use_container_width=True, num_rows="dynamic", disabled=False, hide_index = True)
        
        if st.button(label = "Diagnosticar Síntomas", type = "primary"):
            with st.spinner("Estamos procesando tu petición, puede tardar unos minutos..."):
                lista_codigos = st.session_state.df_sintomas["ID"].to_list()
                tabla, lista_ids = get_ranked_list(lista_codigos)
                st.session_state['tabla'] = tabla
    
    if 'tabla' in st.session_state:
        st.write("---")
        st.write("### 3) Proceso diagnóstico finalizado")
        st.info("Aquí encontrarás un listado de posibles enfermedades ordenadas por su puntuación junto a otra información de interés y enlaces.")
        st.markdown(st.session_state.tabla.to_markdown(index=False), unsafe_allow_html=True)

        st.write("### 4) Descargar la información obtenida")
        st.info("Si deseas guardar el fenotipado realizado mediante inteligencia artificial del caso clínico descrito para posteriores ocasiones, introduce un identificador útil para ti")
        
        nombre_caso = st.text_input(label='Identificador del caso clínico',placeholder="Escribe un identificador para el caso...")
        if st.button(label = "Generar informe", type = "primary") and nombre_caso:
            caso_clinico_txt = f"st.session_state['description'] = '''{st.session_state.description}'''\n" + f"st.session_state['df_sintomas'] = pd.DataFrame({st.session_state.df_sintomas.to_dict()})\n" + f"st.session_state['tabla'] = pd.DataFrame({st.session_state.tabla.to_dict()})\n"
            informe_pdf = generar_informe([st.session_state.df_sintomas,st.session_state.tabla],["Descripción clínica", "Síntomas encontrados", "Diagnósticos posibles"], f"Caso clínico {nombre_caso}", st.session_state.description)
            enviar_informe_diagnostico(nombre_caso, st.session_state.email, informe_pdf, caso_clinico_txt)
            st.download_button(
                label="Descargar informe",
                data=informe_pdf,
                file_name=f'informe_{nombre_caso}.pdf',
                mime="application/pdf"
            )

else:
    st.markdown("<h3 style='text-align: center;'>⛔Acceso Denegado⛔</h3>", unsafe_allow_html=True)
    st.error("Debes Iniciar Sesión en la primera página para poder continuar...")
    
st.write("---")
