import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from sklearn.metrics import silhouette_score
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import plotly.express as px
from folium import plugins
from sklearn.preprocessing import StandardScaler


# Set Streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set Streamlit page configuration
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

# Read CSS file for styling
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Read the dataset
data = pd.read_csv('Jumlah-2021 - 2023 -Lengkap-Dataset_Longsor - PROV JABAR.csv')  # Change path to your new dataset

# Fungsi untuk melakukan Pengelompokan Hirarki Aglomeratif berdasarkan fitur dan tautan yang dipilih
def ahc_clustering(data, n_clusters, selected_features, linkage_method):
    features = data[selected_features + ['LATITUDE', 'LONGITUDE']]
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    data['cluster'] = clusterer.fit_predict(features)

    # Hitung centroid untuk setiap klaster
    centroids = data.groupby('cluster')[['JUMLAH_LONGSOR']].mean()

    # Tentukan nilai ambang batas untuk kategori tanah longsor (sesuaikan dengan analisis )
    threshold_low = 10  # Contoh ambang batas rendah
    threshold_high = 101  # Contoh ambang batas tinggi

    # Tambahkan kolom Kategori Tanah Longsor berdasarkan nilai kuartil
    data['Landslide Category'] = data.apply(lambda row: 'Tingkat Rawan Rendah' if row['JUMLAH_LONGSOR'] < threshold_low else ('Tingkat Rawan Sedang' if row['JUMLAH_LONGSOR'] < threshold_high else 'Tingkat Rawan Tinggi'), axis=1)
    
    return data


# Fungsi untuk menghitung skor siluet untuk berbagai nomor klaster
def calculate_silhouette_scores(data, max_clusters=10, linkage_method='ward'):
    selected_features = ['JUMLAH_LONGSOR', 'JIWA_TERDAMPAK', 'JIWA_MENINGGAL', 'RUSAK_TERDAMPAK', 'RUSAK_RINGAN', 'RUSAK_SEDANG', 'RUSAK_BERAT', 'TERTIMBUN', 'LATITUDE', 'LONGITUDE']
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        features = data[selected_features]
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = clusterer.fit_predict(features)
        silhouette_avg = silhouette_score(features, labels)
        silhouette_scores.append(silhouette_avg)
    return pd.DataFrame({'num_clusters': range(2, max_clusters + 1), 'silhouette_score': silhouette_scores})

# Fungsi untuk menambahkan Google Maps ke peta Folium
def add_google_maps(m):
    tiles = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    attr = "Google Digital Satellite"
    folium.TileLayer(tiles=tiles, attr=attr, name=attr, overlay=True, control=True).add_to(m)

    # Menambahkan label untuk jalan dan objek
    label_tiles = "https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}"
    label_attr = "Google Labels"
    folium.TileLayer(tiles=label_tiles, attr=label_attr, name=label_attr, overlay=True, control=True).add_to(m)

    return m

# Fungsi untuk membuat peta Folium dengan penanda berkelompok
def create_marker_map(df_clustered, selected_kabupaten, scaled_data):
    # Atur lebar dan tinggi secara langsung saat membuat peta Folium
    m = folium.Map(location=[df_clustered['LATITUDE'].mean(), df_clustered['LONGITUDE'].mean()], zoom_start=8, width=1240, height=600)

    # Tambahkan penanda untuk setiap titik data
    for i, row in df_clustered.iterrows():
        # Menyesuaikan konten popup
        popup_content = f"""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <div style='width:400px; height:300px;'>
            <ul class="list-group">
                <li class="list-group-item active" aria-current="true">
                    <h3 class="mb-0">Cluster Information</h3>
                </li>
                <li class="list-group-item">Cluster Number: {row['cluster']}</li>
                <li class="list-group-item">KABUPATEN: {row['KABUPATEN']}</li>
                <li class="list-group-item">JUMLAH_LONGSOR: {row['JUMLAH_LONGSOR']}</li>
                <li class="list-group-item">JIWA_TERDAMPAK: {row['JIWA_TERDAMPAK']}</li>
                <li class="list-group-item">JIWA_MENINGGAL: {row['JIWA_MENINGGAL']}</li>
                <li class="list-group-item">RUSAK_TERDAMPAK: {row['RUSAK_TERDAMPAK']}</li>
                <li class="list-group-item">RUSAK_RINGAN: {row['RUSAK_RINGAN']}</li>
                <li class="list-group-item">RUSAK_SEDANG: {row['RUSAK_SEDANG']}</li>
                <li class="list-group-item">RUSAK_BERAT: {row['RUSAK_BERAT']}</li>
                <li class="list-group-item">TERTIMBUN: {row['TERTIMBUN']}</li>
            </ul>
        </div>
        """

        if row['KABUPATEN'] == selected_kabupaten:
            icon = folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
        else:
            icon = folium.Icon(color='orange', icon='exclamation-triangle', prefix='fa')
        
        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            tooltip=row['KABUPATEN'],
            icon=icon,  # Gunakan variabel ikon yang ditentukan
        ).add_to(m).add_child(folium.Popup(popup_content, max_width=1240))

    # Heatmap Layer
    heat_data = [[row['LATITUDE'], row['LONGITUDE']] for _, row in df_clustered.iterrows()]
    HeatMap(heat_data).add_to(m)

    # Drawing Tools
    draw = plugins.Draw()
    draw.add_to(m)

    # Add Google Maps
    add_google_maps(m)

    return m


# Fungsi untuk menangani halaman Pengelompokan Hirarki Aglomeratif
def ahc_page():
    center = True
    st.header("Agglomerative Hierarchical Clustering Page", anchor='center' if center else 'left')

    # Sidebar: Memilih jumlah cluster
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

    # Sidebar: Pilih metode penautan
    linkage_method = st.sidebar.selectbox('Select Linkage Method', ['single', 'average', 'complete'])

    # Read the dataset
    data = pd.read_csv('Jumlah-2021 - 2023 -Lengkap-Dataset_Longsor - PROV JABAR.csv')  # Change path to your new dataset

    # Dropdown untuk memilih KABUPATEN
    selected_kabupaten = st.sidebar.selectbox('Select Kabupaten', data['KABUPATEN'].unique())

    # Hitung Skor Siluet untuk berbagai kelompok
    silhouette_scores_df = calculate_silhouette_scores(data, max_clusters=10, linkage_method=linkage_method)
    
    # Lakukan Pengelompokan Hirarki Aglomeratif berdasarkan fitur yang dipilih dan metode tautan
    if len(data) >= 2:
        df_clustered = ahc_clustering(data, num_clusters, [], linkage_method)  # Empty list for selected_features
    else:
        # Menangani kasus ketika tidak ada cukup titik data untuk pengelompokan
        st.warning("Not enough data points for clustering. Please select different criteria.")
        return

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['JUMLAH_LONGSOR', 'JIWA_TERDAMPAK', 'JIWA_MENINGGAL', 'RUSAK_TERDAMPAK', 'RUSAK_RINGAN', 'RUSAK_SEDANG', 'RUSAK_BERAT', 'TERTIMBUN', 'LATITUDE', 'LONGITUDE']])
    
    # Simpan fitur yang diskalakan di session_state
    st.session_state.scaled_features = scaled_features
    st.session_state.df_clustered = df_clustered
    st.session_state.selected_kabupaten = selected_kabupaten

    tab1, tab2, tab3 = st.columns([1, 1, 1])

    tab1, tab2, tab3 = st.tabs(["DATASET", "VISUALISASI MAP", "SILHOUETTE SCORE"])

    with tab1:
        # Menampilkan metrik untuk setiap klaster
       
        for cluster_num in range(num_clusters):
            landslide_category = df_clustered.loc[df_clustered['cluster'] == cluster_num, 'Landslide Category'].iloc[0]
            
           # Menambahkan kolom baru untuk indeks
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_num].reset_index(drop=True)
            cluster_data.insert(1, "Index", cluster_data.index)  # Add index column
            
            # Hitung jumlah anggota klaster
            num_members = cluster_data.shape[0]

            with st.expander(f"Cluster {cluster_num + 0} Data Table - {landslide_category} ({num_members} Kabupaten/Kota)", expanded=True):
                st.dataframe(cluster_data,
                            column_order=("Index", "KABUPATEN", "JUMLAH_LONGSOR", "cluster"),
                            hide_index=True,
                            width=500,
                            use_container_width=True,
                            column_config={
                                "Index": st.column_config.TextColumn(
                                    "Index",
                                ),
                                "KABUPATEN": st.column_config.TextColumn(
                                    "Area",
                                )}
                            )



    with tab2:
        with st.expander('Kabupaten/Kota Maps View Analitycs Clustering', expanded=True):
            # Gunakan folium_static untuk menampilkan peta Folium
            folium_map = create_marker_map(st.session_state.df_clustered, st.session_state.selected_kabupaten, st.session_state.scaled_features)
            folium_static(folium_map, width=1240, height=600)
            
        with st.expander("SELECT DATA"):
            selected_city = st.selectbox("Select ", df_clustered['KABUPATEN'])
            selected_row = df_clustered[df_clustered['KABUPATEN'] == selected_city].squeeze()
            # Menampilkan informasi tambahan dalam tabel
            st.table(selected_row)
            
            # Graphs
        col1, col2, col3= st.columns(3)
        with col1:
            with st.container(border=True):
                # Bar chart
                st.write("Cluster Bart Chart:")
                st.bar_chart(df_clustered.groupby('cluster').size(), use_container_width=True)
        with col2:
            with st.container(border=True):
                # Pie chart
                st.write("Cluster Distribution:")
                st.plotly_chart(px.pie(df_clustered, names='cluster', title='Cluster Distribution'), use_container_width=True)

        with col3:
             with st.container(border=True):
                st.write("Scatter Plot:")
                # Asumsikan 'LATITUDE' dan 'LONGITUDE' adalah kolom yang ingin Anda gunakan untuk scatter plot
                scatter_fig = px.scatter(st.session_state.df_clustered, x='LATITUDE', y='LONGITUDE', color='cluster', title='Scatter Plot')
                st.plotly_chart(scatter_fig, use_container_width=True)

        with st.expander('Informasi', expanded=True):
            st.write('''
                - Data: [BARATA BADAN PENANGGULANGAN BENCANA DAERAH PROVINSI JAWA BARAT](http://barata.jabarprov.go.id/).
                - :orange[**Area Prioritas Berdasarkan Tingkat Rawan Bencana**]: Area dengan tingkat rawan bencana tertinggi untuk tahun yang dipilih.
                - :orange[**Perubahan Kejadian Bencana yang Signifikan**]: Area dengan peningkatan atau penurunan kejadian bencana terbesar dari tahun sebelumnya.
                - :information_source: **Rata-rata Kejadian Bencana:** Rata-rata jumlah kejadian bencana untuk tahun yang dipilih.
                - :information_source: **Rata-rata Kejadian Bencana (Area Prioritas):** Rata-rata jumlah kejadian bencana di area prioritas.
                - :information_source: **Modus Kejadian Bencana (Area Prioritas):** Modus jumlah kejadian bencana di area prioritas.
                - :bar_chart: **Visualisasi Kejadian Bencana:** Peta korelasi dan peta panas menampilkan total kejadian bencana di berbagai area.
                - :chart_with_upwards_trend: **Tren Kejadian Bencana:** Grafik menampilkan tren kenaikan/penurunan kejadian bencana, serta area prioritas dengan tingkat rawan bencana tertinggi dan terendah, serta perubahan kejadian bencana yang signifikan.
                ''')

    with tab3:
        col = st.columns((5, 1.5), gap='medium')
        with col[0]:
            st.expander('Kabupaten/Kota Maps View Silhouette Score Clustering', expanded=True)
            # Line plot for silhouette scores
            silhouette_line_plot = px.line(silhouette_scores_df, x='num_clusters', y='silhouette_score',
                                           markers=True, labels={'num_clusters': 'Number of Clusters', 'silhouette_score': 'Silhouette Score'})
            st.plotly_chart(silhouette_line_plot, use_container_width=True)
        with col[1]:
            st.write(silhouette_scores_df)

        with st.expander('Informasi', expanded=True):
            st.info('''
            - Data: [BARATA BADAN PENANGGULANGAN BENCANA DAERAH PROVINSI JAWA BARAT](http://barata.jabarprov.go.id/).
            - :orange[**Area Prioritas Berdasarkan Tingkat Rawan Bencana**]: Area dengan tingkat rawan bencana tertinggi untuk tahun yang dipilih.
            - :orange[**Perubahan Kejadian Bencana yang Signifikan**]: Area dengan peningkatan atau penurunan kejadian bencana terbesar dari tahun sebelumnya.
            - :information_source: **Rata-rata Kejadian Bencana:** Rata-rata jumlah kejadian bencana untuk tahun yang dipilih.
            - :information_source: **Rata-rata Kejadian Bencana (Area Prioritas):** Rata-rata jumlah kejadian bencana di area prioritas.
            - :information_source: **Modus Kejadian Bencana (Area Prioritas):** Modus jumlah kejadian bencana di area prioritas.
            - :bar_chart: **Visualisasi Kejadian Bencana:** Peta korelasi dan peta panas menampilkan total kejadian bencana di berbagai area.
            - :chart_with_upwards_trend: **Tren Kejadian Bencana:** Grafik menampilkan tren kenaikan/penurunan kejadian bencana, serta area prioritas dengan tingkat rawan bencana tertinggi dan terendah, serta perubahan kejadian bencana yang signifikan.
            ''')

if __name__ == "__main__":
    # Call the ahc_page function
    ahc_page()
