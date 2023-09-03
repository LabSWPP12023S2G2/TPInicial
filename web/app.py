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

'''
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
'''

# Función para asignar colores gradualmente en función de los valores del eje x
def assign_colors(x_values):
    # Define una paleta de colores gradual de verde (x=0) a rojo (x=100)
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=0, vmax=100)
    colors = cmap(norm(x_values))
    return colors


# Histograma por clusters de riesgo de suicidio
def hist_suic_clusters(data_ref):
    unique_clusters = np.unique(data_ref['Cluster'])
    for cluster in unique_clusters:
        cluster_data = data_ref[data_ref['Cluster'] == cluster]

        # Crear una figura de Matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        hist, bins, _ = plt.hist(cluster_data['SUIC RISK'], bins=20, alpha=0.7)

        # Obtener colores para las barras del histograma
        colors = assign_colors(bins[:-1])

        # Dibujar las barras del histograma con colores graduales
        plt.bar(bins[:-1], hist, width=bins[1] - bins[0], color=colors, edgecolor='black', alpha=0.7)

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

        fig, ax = plt.subplots(figsize=(10, 6))  # Definir la figura y los ejes
        province_counts.plot(kind='bar', color='blue', alpha=0.7, ax=ax)

        plt.title(f'Distribución de regiones y provincias en Cluster {cluster}')
        plt.xlabel('Region-Provincia')
        plt.ylabel('Cantidad de casos')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)

        # Mostrar la figura en Streamlit
        st.write(f'Distribución de regiones y provincias en Cluster {cluster}')
        st.pyplot(fig)


# Scatter plot de clusters y tsne
def scatter_plot_clusters(data_ref, kmeans, cluster_names):
    colors = ['orange', 'green', 'red']
    plt.figure(figsize=(12, 8))
    unique_clusters = np.unique(kmeans.labels_)
    for i, cluster in enumerate(unique_clusters):
        cluster_data = data_ref[data_ref['Cluster'] == cluster]
        cluster_name = cluster_names.get(cluster, f'Cluster {cluster}')
        # Asignar color a cada cluster
        color = colors[i % len(colors)]
        plt.scatter(cluster_data['tsne_x'], cluster_data['tsne_y'],
                    label=cluster_name, alpha=0.7, s=50, c=color)
    plt.xlabel('t-SNE x')
    plt.ylabel('t-SNE y')
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
    if st.button("Clasificar:"):
        x_in = [np.float_(N.title()),
                np.float_(P.title()),
                ]
        predictS = model_prediction(x_in, model)  # Supongamos que predictS es 0, 1 o 2
        cluster_names_for_pred = {
            0: 'riesgo medio',
            1: 'riesgo bajo',
            2: 'riesgo alto',
        }

        # Obtener el nombre del cluster correspondiente
        predicted_cluster = cluster_names_for_pred.get(predictS[0], 'Riesgo Desconocido')

        # Mostrar el resultado en Streamlit
        st.success(f'El grupo de riesgo al que pertenecen estos valores es: {predicted_cluster}'.upper())


        with st.expander("Termómetro de riesgo"):
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
        
        how = """
        <div>
        <h1 style="color:#181082;text-align:center;">Visualizaciones del modelo entrenado</h1>
        </div>
        """
        st.markdown(how, unsafe_allow_html=True)
        
        with st.expander("Distribución de clusters"):
            title_hist_suic_clusters_regions = """
            <div>
            <h1 style="color:#181082;text-align:center;">Distribución de clusters</h1>
            </div>
            <div>
            <p style="color:#181082;text-align:center;">
            Así es como se ve gracias a la reducción de dimensionalidad t-SNE (80 de perplejidad) 
            la distribución de los 3 clusters con el entrenamiento de K-Means. El color verde 
            nos indica los casos de bajo riesgo, el naranja de riesgo medio y el rojo de riesgo alto.
            </p>
            </div>
            """
            st.markdown(title_hist_suic_clusters_regions, unsafe_allow_html=True)
            scatter_plot_clusters(data, model, cluster_names_for_pred)


        with st.expander("Histograma cantidad de casos según riesgo"):
            title_hist_suic_clusters = """
            <h1 style="color:#181082;text-align:center;">Histograma cantidad de casos según riesgo</h1>
            </div>
            """
            st.markdown(title_hist_suic_clusters, unsafe_allow_html=True)
            hist_suic_clusters(data_ref)

        
        with st.expander("Histograma cantidad de casos según riesgo por regiones-provincias"):
            title_hist_suic_clusters_regions = """
            <h1 style="color:#181082;text-align:center;">Histograma cantidad de casos según riesgo por regiones-provincias</h1>
            </div>
            """
            st.markdown(title_hist_suic_clusters_regions, unsafe_allow_html=True)
            hist_suic_clusters_regions(data_ref)

        
if __name__ == '__main__':
    main()
