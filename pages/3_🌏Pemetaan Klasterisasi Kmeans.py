import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import folium
from folium import plugins
from folium.plugins import HeatMap
from sklearn.metrics import silhouette_score
import plotly.express as px
from streamlit_folium import folium_static

# Mengatur konfigurasi halaman
st.set_page_config(
    page_title="KMeans Clustering and Visualization",
    layout="wide",  # Atur tata letak ke lebar untuk konten dengan lebar penuh
    initial_sidebar_state="collapsed",  # Menciutkan bilah samping secara default
)

# Fungsi untuk memuat data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv("Jumlah-2021 - 2023 -Lengkap-Dataset_Longsor - PROV JABAR.csv")
    return data

# Fungsi untuk melakukan pengelompokan KMeans dan menghitung Skor Siluet
def kmeans_clustering(data, num_clusters):
    features = data[['JUMLAH_LONGSOR', 'JIWA_TERDAMPAK', 'JIWA_MENINGGAL', 'RUSAK_TERDAMPAK', 'RUSAK_RINGAN', 'RUSAK_SEDANG', 'RUSAK_BERAT', 'TERTIMBUN', 'LATITUDE', 'LONGITUDE']]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)
    
    # Hitung centroid untuk setiap klaster
    centroids = data.groupby('cluster')[['JUMLAH_LONGSOR']].mean()

    # Tentukan nilai ambang batas untuk kategori tanah longsor (sesuaikan dengan analisis Anda)
    threshold_low = 10  # Contoh ambang batas untuk “tidak padat”
    threshold_high = 101  # Contoh ambang batas untuk “padat”

    # Tambahkan kolom Kategori Tanah Longsor berdasarkan nilai centroid
    data['Landslide Category'] = data['cluster'].map(lambda cluster: 'Tingkat Rawan Rendah' if centroids.loc[cluster].mean() < threshold_low else ('Tingkat Rawan Sedang' if centroids.loc[cluster].mean() < threshold_high else 'Tingkat Rawan Tinggi'))
    
    # Elbow Method data
    elbow_data = pd.DataFrame({'num_clusters': range(1, 11),
                               'inertia': [KMeans(n_clusters=i, random_state=42).fit(features).inertia_ for i in range(1, 11)]})
    
    return data, elbow_data

# Fungsi untuk menghitung skor siluet untuk berbagai nomor klaster
def calculate_silhouette_scores(data, max_clusters=10):
    features = data[['JUMLAH_LONGSOR', 'JIWA_TERDAMPAK', 'JIWA_MENINGGAL', 'RUSAK_TERDAMPAK', 'RUSAK_RINGAN', 'RUSAK_SEDANG', 'RUSAK_BERAT', 'TERTIMBUN', 'LATITUDE', 'LONGITUDE']]
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
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

def create_marker_map(df_clustered):
    # Atur lebar dan tinggi secara langsung saat membuat peta Folium
    m = folium.Map(location=[df_clustered['LATITUDE'].mean(), df_clustered['LONGITUDE'].mean()], zoom_start=10, width=1240, height=600)

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

        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            tooltip=row['KABUPATEN'],
            icon=folium.Icon(color='red', icon='home', prefix='fa'),
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


def display_clustering_summary(df_clustered, num_clusters):
    with st.expander('Informasi', expanded=True):
        summary = f'''
        Hasil clustering menggunakan KMeans menghasilkan beberapa klaster berdasarkan data bencana longsor di Provinsi Jawa Barat. Berikut adalah kesimpulan dan informasi penting dari hasil clustering:

        1. **Jumlah Klaster**: Data dikelompokkan menjadi {num_clusters} klaster. Setiap klaster memiliki karakteristik yang berbeda-beda terkait jumlah kejadian longsor dan dampaknya.

        2. **Kategori Tingkat Rawan Bencana**: 
           - **Tingkat Rawan Rendah**: Klaster yang memiliki rata-rata kejadian longsor yang rendah.
           - **Tingkat Rawan Sedang**: Klaster dengan rata-rata kejadian longsor yang sedang.
           - **Tingkat Rawan Tinggi**: Klaster dengan rata-rata kejadian longsor yang tinggi.

        3. **Klaster Prioritas**: Klaster {', '.join(map(str, df_clustered[df_clustered['Landslide Category'] == 'Tingkat Rawan Tinggi']['cluster'].unique()))} termasuk dalam kategori Tingkat Rawan Tinggi dan perlu mendapatkan perhatian lebih dalam mitigasi dan penanganan bencana.

        4. **Perubahan Signifikan**: Beberapa area menunjukkan perubahan signifikan dalam jumlah kejadian longsor dari tahun sebelumnya, baik peningkatan maupun penurunan.

        5. **Visualisasi Peta dan Grafik**:
           - **Peta Korelasi dan Peta Panas**: Menampilkan distribusi kejadian longsor di berbagai area.
           - **Grafik Tren**: Menampilkan tren kenaikan atau penurunan kejadian longsor di setiap klaster, membantu dalam mengidentifikasi area dengan perubahan signifikan.

        6. **Rata-rata dan Modus Kejadian Longsor**:
           - **Rata-rata Kejadian Longsor**: Menunjukkan rata-rata jumlah kejadian longsor untuk tahun yang dipilih adalah {df_clustered['JUMLAH_LONGSOR'].mean():.2f}.
           - **Rata-rata Kejadian Longsor di Area Prioritas**: Rata-rata jumlah kejadian longsor di area yang termasuk dalam klaster prioritas.
           - **Modus Kejadian Longsor di Area Prioritas**: Menampilkan jumlah kejadian longsor yang paling sering terjadi di area prioritas adalah {df_clustered['JUMLAH_LONGSOR'].mode()[0]}.

        Dengan hasil clustering ini, diharapkan dapat memberikan gambaran yang lebih jelas mengenai distribusi dan karakteristik kejadian longsor di Provinsi Jawa Barat, sehingga dapat digunakan sebagai dasar dalam perencanaan mitigasi dan penanggulangan bencana yang lebih efektif.
        '''
        
        st.info(summary)

# Fungsi untuk menangani halaman KMeans
def kmeans_page():
    st.header("KMeans Clustering Page", anchor='center')


    # Bilah samping: Memilih jumlah cluster
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=2)

    # Memuat data dari halaman beranda
    data_from_homepage = load_data('Jumlah-2021 - 2023 -Lengkap-Dataset_Longsor - PROV JABAR.csv')  # Replace with your actual data file path

    # Lakukan pengelompokan KMeans
    df_clustered, elbow_data = kmeans_clustering(data_from_homepage, num_clusters)
   
    # Hitung Skor Siluet untuk berbagai kelompok
    silhouette_scores_df = calculate_silhouette_scores(data_from_homepage)

    # Simpan data yang dikelompokkan di session_state
    st.session_state.df_clustered = df_clustered
    st.session_state.elbow_data = elbow_data
    st.session_state.silhouette_scores_df = silhouette_scores_df

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
        with st.expander('Kabupaten/Kota View Analitycs Clustering', expanded=True):
            # Gunakan folium_static untuk menampilkan peta Folium
            folium_map = create_marker_map(st.session_state.df_clustered)
            folium_static(folium_map, width=1240, height=600)

        with st.expander("SELECT DATA"):
            selected_city = st.selectbox("Select ", df_clustered['KABUPATEN'])
            selected_row = df_clustered[df_clustered['KABUPATEN'] == selected_city].squeeze()
               # Menampilkan informasi tambahan dalam tabel
            st.table(selected_row)
        # Graphs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.expander("⬇ HISTOGRAM:"):
                # Bar chart
                st.bar_chart(df_clustered.groupby('cluster').size(), use_container_width=True)

        with col2:
            with st.expander("⬇ DONUT CHART:"):
                # Donut chart
                fig = px.pie(df_clustered, names='cluster', hole=0.3)
                st.plotly_chart(fig, use_container_width=True)

        with col3:
            with st.expander("⬇ SCATTERPLOT:"):
                scatter_plot = px.scatter(df_clustered, x='LATITUDE', y='LONGITUDE', color='cluster',
                                          hover_data=['KABUPATEN'])
                st.plotly_chart(scatter_plot, use_container_width=True)
                
        with col4:
            with st.expander("⬇ ELBOW METHOD:"):
                fig_elbow = px.line(elbow_data, x='num_clusters', y='inertia', markers=True, title='Elbow Method',
                                    labels={'num_clusters': 'Number of Clusters', 'inertia': 'Inertia'})
                fig_elbow.update_layout(
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis=dict(showgrid=True, gridcolor='#cecdcd'),
                    yaxis=dict(showgrid=True, gridcolor='#cecdcd'),
                    font=dict(color='#cecdcd'),
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
                
            # Panggil fungsi untuk menampilkan ringkasan setelah pengelompokan
        display_clustering_summary(df_clustered, num_clusters)

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

        with st.expander('Informasi Skor Siluet', expanded=True):
            st.write('''
                - **Ikhtisar Skor Siluet**: Skor Siluet adalah ukuran seberapa mirip suatu objek dengan klasternya sendiri (koherensi) dibandingkan dengan klaster lain (pemisahan). Skor tersebut berkisar dari -1 hingga 1, di mana nilai tinggi menunjukkan bahwa objek tersebut cocok dengan klasternya sendiri dan tidak cocok dengan klaster tetangga.
                - **Interpretasi Skor Siluet**:
                - Skor mendekati 1 menunjukkan bahwa objek terklasifikasi dengan baik.
                - Skor mendekati 0 menunjukkan adanya tumpang tindih antar klaster.
                - Skor mendekati -1 menunjukkan bahwa objek salah terklasifikasi.
                - **Jumlah Optimal Klaster**: Cari puncak atau titik tinggi dalam plot Skor Siluet untuk mengidentifikasi jumlah optimal klaster. Skor yang lebih tinggi menunjukkan pemisahan dan koherensi klaster yang lebih baik.
                - **Penggunaan**: Gunakan Skor Siluet untuk mengevaluasi kualitas algoritma pengelompokan Anda dan menentukan jumlah klaster yang sesuai untuk dataset Anda.
                - **Sumber Data**: [BARATA BADAN PENANGGULANGAN BENCANA DAERAH PROVINSI JAWA BARAT](https://barata.jabarprov.go.id/).
                ''')

# Run the Streamlit app
if __name__ == "__main__":
    kmeans_page()
