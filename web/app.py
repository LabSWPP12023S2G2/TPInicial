import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Path del modelo preentrenado
MODEL_PATH = '/mount/src/tpinicial/web/models/kmeans_model.pkl'


data = pd.read_csv('/mount/src/tpinicial/data.csv')
data_ref = pd.read_csv('/mount/src/tpinicial/data_ref.csv')


# Se recibe los datos del usuario y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1, -1)
    preds = model.predict(x)
    return preds


# Histograma por clusters de riesgo de suicidio
def hist_suic_clusters(data_ref):
    unique_clusters = np.unique(data_ref['Cluster'])
    for cluster in unique_clusters:
        cluster_data = data_ref[data_ref['Cluster'] == cluster]

        # Crear una figura de Matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.hist(cluster_data['SUIC RISK'], bins=20, color='blue', alpha=0.7)

        # Configurar etiquetas y título
        plt.title(f'Distribución de riesgo de suicidio en Cluster {cluster}')
        plt.xlabel('Riesgo de suicidio')
        plt.ylabel('Cantidad de casos')
        plt.grid(True)

        # Mostrar la figura en Streamlit
        st.write(f'Distribución de riesgo de suicidio en Cluster {cluster}')
        st.pyplot(fig)


# Histograma de provincias para cada cluster
def hist_suic_clusters_regions(data_ref):
    unique_clusters = np.unique(data_ref['Cluster'])
    for cluster in unique_clusters:
        cluster_data = data_ref[data_ref['Cluster'] == cluster]
        province_counts = cluster_data['REGION'].value_counts()

        plt.figure(figsize=(10, 6))
        province_counts.plot(kind='bar', color='blue', alpha=0.7)

        plt.title(f'Distribución de regiones y provincias en Cluster {cluster}')
        plt.xlabel('Region-Provincia')
        plt.ylabel('Cantidad de casos')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)

        # Mostrar la figura en Streamlit
        st.write(f'Distribución de regiones y provincias en Cluster {cluster}')
        st.pyplot()


# Diccionario para mapear los nombres de los clusters
cluster_names = {
    0: 'Riesgo medio',
    1: 'Riesgo bajo',
    2: 'Riesgo alto',
}


# Scatter plot de clusters y tsne
def scatter_plot_clusters(data_ref, kmeans, cluster_names):
    # Crear un scatter plot
    plt.figure(figsize=(12, 8))
    unique_clusters = np.unique(kmeans.labels_)
    for cluster in unique_clusters:
        cluster_data = data_ref[data_ref['Cluster'] == cluster]
        cluster_name = cluster_names.get(cluster, f'Cluster {cluster}')

        plt.scatter(cluster_data['tsne_x'], cluster_data['tsne_y'],
                    label=cluster_name, alpha=0.7, s=50)

    # Crear leyenda personalizada
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=f'C{cluster}', markersize=10, label=cluster_names.get(cluster, f'Cluster {cluster}')) for cluster in unique_clusters]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1))

    # Configurar título y etiquetas de ejes
    plt.title('Clustering con K-Means (3 clústers) y t-SNE (80 de perplejidad) de regiones y provincias')
    plt.xlabel('t-SNE x')
    plt.ylabel('t-SNE y')

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)


def main():

    model = ''
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
        st.success('El grupo de riesgo al que pertenecen estos valores es: {}'.format(
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

        title_hist_suic_clusters_regions = """
        <h1 style="color:#181082;text-align:center;">Distribución de clusters</h1>
        </div>
        """
        st.markdown(title_hist_suic_clusters_regions, unsafe_allow_html=True)

        scatter_plot_clusters(data, model, cluster_names)

        title_hist_suic_clusters = """
        <h1 style="color:#181082;text-align:center;">Histograma cantidad de casos según riesgo</h1>
        </div>
        """
        st.markdown(title_hist_suic_clusters, unsafe_allow_html=True)

        hist_suic_clusters(data_ref)

        title_hist_suic_clusters_regions = """
        <h1 style="color:#181082;text-align:center;">Histograma cantidad de casos según riesgo por regiones-provincias</h1>
        </div>
        """
        st.markdown(title_hist_suic_clusters_regions, unsafe_allow_html=True)

        hist_suic_clusters_regions(data_ref)

        
if __name__ == '__main__':
    main()
