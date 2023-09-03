import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import streamlit as st
import matplotlib.pyplot as plt

# Path del modelo preentrenado
MODEL_PATH = '/mount/src/tpinicial/web/models/kmeans_model.pkl'


data = pd.read_csv('/mount/src/tpinicial/data_ref.csv')


# Se recibe los datos del usuario y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1, -1)
    preds = model.predict(x)
    return preds

# Crea un gráfico de distribución de riesgo de suicidio en cada Cluster
def scatter_plot_cases_suic(data_ref):
    unique_clusters = np.unique(data_ref['Cluster'])
    for cluster in unique_clusters:
        cluster_data = data_ref[data_ref['Cluster'] == cluster]
        st.write(f'Distribución de riesgo de suicidio en Cluster {cluster}')
        st.pyplot(plt.hist(cluster_data['SUIC RISK'], bins=20, color='blue', alpha=0.7))


def main():

    model = ''
    dataset = ''

    # Se carga el modelo
    if model == '':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    

    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">SISTEMA DE CLASIFICACION DE RIESGO DE SUICIDIO</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Lecctura de datos
    N = st.text_input("Riesgo de suicidio:")
    P = st.text_input("Promedio de las variables restantes:")

    # El botón clasificar se usa para iniciar el procesamiento
    if st.button("Clasificar :"):
        x_in = [np.float_(N.title()),
                np.float_(P.title()),
                ]
        predictS = model_prediction(x_in, model)
        st.success('EL GRUPO DE RIESGO DE SUICIDIO AL QUE PERTENECE ES ES: {}'.format(
            predictS[0]).upper())

        # Define un diccionario de mapeo de valores de predicción a rutas de imágenes
        imagen_por_prediccion = {
            0: "/mount/src/tpinicial/web/assets/termoModificadoCluster1.png",
            1: "/mount/src/tpinicial/web/assets/termoModificadoCluster0.png",
            2: "/mount/src/tpinicial/web/assets/termoModificadoCluster2.png",
        }

        cluster = predictS[0]

        if cluster in imagen_por_prediccion:
            # Obtiene la ruta de la imagen correspondiente
            ruta_imagen = imagen_por_prediccion[cluster]
            # Muestra la imagen en Streamlit
            st.image(ruta_imagen)
        else:
            st.write("No se encontró una imagen para la predicción.")

        scatter_plot_cases_suic(data)

        


if __name__ == '__main__':
    main()
