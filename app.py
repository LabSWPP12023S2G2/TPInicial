from src.data_processing import load_and_clean_data
from src.clustering import perform_clustering
import streamlit as st

# Realizar clustering
# data = perform_clustering(data)

def main():
    st.title("Hello, World! EDA Streamlit App")

    # URL del dataset
    url = 'https://raw.githubusercontent.com/LabSWPP12023S2G2/TPInicial/main/datasetUNC.csv'

    if url is not None:
        # Cargar y limpiar datos
        data = load_and_clean_data(url)
        st.write("Data overview:")
        st.write(data.head())

        st.sidebar.header("Visualizations")
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", data.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Select a column", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()