import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from streamlit_folium import folium_static
import folium
from folium import plugins
from folium.plugins import HeatMap
import plotly_express as px

# Mengatur konfigurasi halaman
st.set_page_config(
    page_title="Dashboard",
    page_icon="📈",
    layout="wide"
)

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.header("ALGORITMA AGGLOMERATIVE HIERARCHICAL CLUSTERING (AHC)")


# Penjelasan tentang AHC
st.markdown(
"""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<hr>

<div class="card mb-3">
    <div class="card">
        <div class="card-body">
            <h3 class="card-title" style="color:#007710;"><strong> PEMAHAMAN </strong></h3>
            <p class="card-text">Agglomerative Hierarchical Clustering (AHC) adalah algoritma klasterisasi hierarkis yang secara bertahap menggabungkan klaster terdekat tanpa memerlukan jumlah klaster awal.</p>
            <p class="card-text">AHC menggunakan matriks linkage (Z) untuk merepresentasikan penggabungan klaster, dan metode linkage seperti 'single', 'complete', 'average', mempengaruhi perhitungan jarak antara klaster.</p>
            <p class="card-text">Keunggulan AHC adalah kemampuannya menghasilkan dendrogram, yang memudahkan pemahaman struktur hierarkis data. Jumlah klaster dapat ditentukan dengan menetapkan ambang batas jarak atau memotong dendrogram pada ketinggian tertentu, memberikan fleksibilitas terhadap struktur data yang beragam.</p>
    </div>
</div>
<style>
    [data-testid=stSidebar] {
         color: white;
         text-size:24px;
    }
</style>
""", unsafe_allow_html=True)
    
# Read data
df = pd.read_csv("Jumlah-2021 - 2023 -Lengkap-Dataset_Longsor - PROV JABAR.csv")

# Memilih fitur untuk pengelompokan
features_ahc = df[
    [
        'JUMLAH_LONGSOR',
        'JIWA_TERDAMPAK',
        'JIWA_MENINGGAL',
        'RUSAK_TERDAMPAK',
        'RUSAK_RINGAN',
        'RUSAK_SEDANG',
        'RUSAK_BERAT',
        'TERTIMBUN',
        'LATITUDE',
        'LONGITUDE'
    ]
]

# Agglomerative Hierarchical Clustering method
linkage_matrix = linkage(features_ahc, method='single')

# Ekspander untuk menampilkan data
with st.expander("⬇ DATA UNDERSTANDING FOR AGGLOMERATIVE HIERARCHICAL CLUSTERING :"):
    # Display summary statistics
    st.write("Summary statistic adalah ringkasan statistik deskriptif yang mencakup ukuran-ukuran seperti mean, median, mode, range, varians, standar deviasi, serta quartiles, yang memberikan gambaran singkat mengenai distribusi dan tendensi data dalam sebuah dataset.")

    st.write("Tujuannya memberikan gambaran singkat dan ringkas tentang karakteristik utama dari suatu dataset, termasuk distribusi dan tendensi data, sehingga memudahkan pemahaman awal tentang pola dan tren dalam data sebelum melakukan analisis yang lebih mendalam.")

   
    st.write("### Summary Statistics:")
    st.write(df.describe())

# Choose the column for the line chart
selected_column = 'JUMLAH_LONGSOR'

# Calculate quartiles
quartiles = df[selected_column].quantile([0.25, 0.5, 0.75])

# Create columns for expanders
c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("⬇ QUARTILE Q1"):
        # Display quartile values
        st.write(f"**Quartile Information for {selected_column}:**")
        st.write(f"- 0.25% Percentile (Q1): {quartiles[0.25]}")

        # Line chart for the 25th percentile
        fig = px.line(df, x=df.index, y=selected_column, title="Line Chart - 25th Percentile (Q1)")
        fig.update_layout(height=300, width=400)  # Adjust the size
        st.plotly_chart(fig)

with c2:
    with st.expander("⬇ QUARTILE Q2"):
        # Display quartile values
        st.write(f"**Quartile Information for {selected_column}:**")
        st.write(f"- 50th Percentile (Q2): {quartiles[0.5]}")

        # Line chart for the 50th percentile
        fig = px.line(df, x=df.index, y=selected_column, title="Line Chart - 50th Percentile (Q2)")
        fig.update_layout(height=300, width=400)  # Adjust the size
        st.plotly_chart(fig)

with c3:
    with st.expander("⬇ QUARTILE Q3"):
        # Display quartile values
        st.write(f"**Quartile Information for {selected_column}:**")
        st.write(f"- 75th Percentile (Q3): {quartiles[0.75]}")

        # Line chart for the 75th percentile
        fig = px.line(df, x=df.index, y=selected_column, title="Line Chart - 75th Percentile (Q3)")
        fig.update_layout(height=300, width=400)  # Adjust the size
        st.plotly_chart(fig)

# Exploring variables
with st.expander("⬇ EKSPLORASI VARIABEL:"):
    st.subheader("Korelasi antara Variabel")
    selected_features = [
        'JUMLAH_LONGSOR',
        'JIWA_TERDAMPAK',
        'JIWA_MENINGGAL',
        'RUSAK_TERDAMPAK',
        'RUSAK_RINGAN',
        'RUSAK_SEDANG',
        'RUSAK_BERAT',
        'TERTIMBUN',
    ]

    # Calculate correlation matrix
    correlation_matrix = df[selected_features].corr()

    # Plot heatmap using Plotly Express
    fig = px.imshow(correlation_matrix,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=selected_features,
                    y=selected_features,
                    color_continuous_scale="viridis",
                    title="Heatmap Korelasi")

    st.plotly_chart(fig)

    st.write("Visualisasi ini memberikan representasi visual dari data dalam bentuk matriks, di mana intensitas warna pada setiap sel matriks menggambarkan nilai variabel yang bersangkutan, memudahkan identifikasi pola, keterkaitan, atau perbedaan dalam data.")

# checking null value
with st.expander("⬇ NULL VALUES, TENDENCY & VARIABLE DISPERSION"):
    a1, a2 = st.columns(2)
    a1.write("Jumlah nilai yang tidak ada (NaN atau None) dalam setiap kolom DataFrame.")
    a1.dataframe(df.isnull().sum(), use_container_width=True)

    a2.write("Informasi tentang tendensi pusat, dispersi, dan distribusi data.")
    a2.dataframe(df.describe().T, use_container_width=True)


st.header("METODE LINKAGE")

with st.expander("⬇ LINKAGE INFORMATION"):
    st.write("Single Linkage: Menggunakan jarak minimum antara anggota klaster.")
    st.write("Complete Linkage: Menggunakan jarak maksimum antara anggota klaster.")
    st.write("Average Linkage: Menggunakan rata-rata jarak antara semua pasangan anggota klaster.")

# Convert 'KABUPATEN' column to numeric
df['KABUPATEN'] = pd.factorize(df['KABUPATEN'])[0]

# Select numerical columns for clustering
numeric_columns = ['JUMLAH_LONGSOR', 'JIWA_TERDAMPAK', 'JIWA_MENINGGAL', 'RUSAK_TERDAMPAK', 'RUSAK_RINGAN', 'RUSAK_SEDANG', 'RUSAK_BERAT', 'TERTIMBUN', 'LATITUDE', 'LONGITUDE']
X_ahc = df[numeric_columns]

# Handle NaN values by filling with column means
X_ahc.fillna(X_ahc.mean(), inplace=True)

# Perform Agglomerative Hierarchical Clustering (AHC)
ahc = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
df['Cluster_AHC'] = ahc.fit_predict(X_ahc)

# Calculate CCC for single linkage
linkage_matrix_single = linkage(X_ahc, method='single')
cophenet_matrix_single, _ = cophenet(linkage_matrix_single, pdist(X_ahc))
ccc_single = cophenet_matrix_single.mean()

# Calculate CCC for Complete linkage
linkage_matrix_complete = linkage(X_ahc, method='complete')
cophenet_matrix_complete, _ = cophenet(linkage_matrix_complete, pdist(X_ahc))
ccc_complete = cophenet_matrix_complete.mean()

# Calculate CCC for Average linkage
linkage_matrix_average = linkage(X_ahc, method='average')
cophenet_matrix_average, _ = cophenet(linkage_matrix_average, pdist(X_ahc))
ccc_average = cophenet_matrix_average.mean()

# Definisikan tinggi pemotongan untuk setiap metode linkage
cut_height_single = 25  # Sesuaikan dengan visualisasi dendrogram single
cut_height_complete = 10.0  # Sesuaikan dengan visualisasi dendrogram Complete
cut_height_average = 5.0

# Menggunakan fcluster untuk mendapatkan label klaster
labels_single = fcluster(linkage_matrix_single, t=cut_height_single, criterion='distance')
labels_complete = fcluster(linkage_matrix_complete, t=cut_height_complete, criterion='distance')
labels_average = fcluster(linkage_matrix_average, t=cut_height_average, criterion='distance')

# Menampilkan kesimpulan
c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("⬇ DENDROGRAM SINGLE"):
        # Visualisasi Dendrogram untuk single
        plt.figure(figsize=(8, 6))
        dendrogram(linkage_matrix_single)
        plt.title('Dendrogram AHC (Single)')
        plt.xlabel('Indeks Data')
        plt.ylabel('Jarak')
        st.pyplot()

        st.write(f"Cophenetic Correlation Coefficient (CCC) untuk Dendrogram AHC (single): {ccc_single:.4f}")
        
with c2:
    with st.expander("⬇ DENDROGRAM COMPLETE"):
        # Visualisasi Dendrogram untuk Complete
        plt.figure(figsize=(8, 6))
        dendrogram(linkage_matrix_complete)
        plt.title('Dendrogram AHC (Complete)')
        plt.xlabel('Indeks Data')
        plt.ylabel('Jarak')
        st.pyplot()

        st.write(f"Cophenetic Correlation Coefficient (CCC) untuk Dendrogram AHC (Complete): {ccc_complete:.4f}")
        
with c3:
    with st.expander("⬇ DENDROGRAM AVERAGE"):
        # Visualisasi Dendrogram untuk Average
        plt.figure(figsize=(8, 6))
        dendrogram(linkage_matrix_average)
        plt.title('Dendrogram AHC (Average)')
        plt.xlabel('Indeks Data')
        plt.ylabel('Jarak')
        st.pyplot()

        st.write(f"Cophenetic Correlation Coefficient (CCC) untuk Dendrogram AHC (Average): {ccc_average:.4f}")
      
   

# Dengan asumsi X_ahc sudah didefinisikan
# Kode penghitungan skor pengelompokan dan siluet Anda
silhouette_scores_single = []
silhouette_scores_complete = []
silhouette_scores_average = []

n_clusters_range = range(2, 11)

for n_clusters in n_clusters_range:
    # Metode 'single'
    clusterer_single = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    cluster_labels_single = clusterer_single.fit_predict(X_ahc)
    silhouette_avg_single = silhouette_score(X_ahc, cluster_labels_single)
    silhouette_scores_single.append(silhouette_avg_single)
    
    
    # Metode 'complete'
    clusterer_complete = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    cluster_labels_complete = clusterer_complete.fit_predict(X_ahc)
    silhouette_avg_complete = silhouette_score(X_ahc, cluster_labels_complete)
    silhouette_scores_complete.append(silhouette_avg_complete)

     # Metode 'average'
    clusterer_average = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    cluster_labels_average = clusterer_average.fit_predict(X_ahc)
    silhouette_avg_average = silhouette_score(X_ahc, cluster_labels_average)
    silhouette_scores_average.append(silhouette_avg_average)

# Create dataframes for silhouette scores
silhouette_scores_single_df = pd.DataFrame({'Number of Clusters': n_clusters_range, 'Silhouette Score': silhouette_scores_single})
silhouette_scores_complete_df = pd.DataFrame({'Number of Clusters': n_clusters_range, 'Silhouette Score': silhouette_scores_complete})
silhouette_scores_average_df = pd.DataFrame({'Number of Clusters': n_clusters_range, 'Silhouette Score': silhouette_scores_average})

# Display the line charts and optimal cluster information using Plotly Express
c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("⬇ SILLHOUTE SCORE SINGLE"):
        st.write("Silhouette Scores for Single Linkage:")
        st.table(silhouette_scores_single_df)

with c2:
     with st.expander("⬇ SILLHOUTE SCORE COMPLETE"):
        st.write("Silhouette Scores for Complete Linkage:")
        st.table(silhouette_scores_complete_df)

with c3:
    with st.expander("⬇ SILLHOUTE SCORE AVERAGE"):
        st.write("Silhouette Scores for Average Linkage:")
        st.table(silhouette_scores_average_df)

# Membuat DataFrame untuk perbandingan CCC
ccc_comparison = {
    'Metode': ['Single', 'Complete', 'Average'],
    'CCC': [ccc_single, ccc_complete, ccc_average ]
}
ccc_comparison_df = pd.DataFrame(ccc_comparison)

# Menampilkan plot Silhouette Score terhadap jumlah cluster untuk masing-masing metode
silhouette_df = pd.DataFrame({
    'Jumlah Cluster': list(n_clusters_range),
    'Single Linkage': silhouette_scores_single,
    'Complete Linkage': silhouette_scores_complete,
    'Average Linkage': silhouette_scores_average
   
})

c1,c2 = st.columns(2)
with c1:
    with st.expander("⬇ PERBANDINGAN METODE SINGLE, COMPLETE DAN AVERAGE DENGAN COPHENETIC CORRELATION COEFFICIENT"):
        # Membuat plot perbandingan CCC
        fig_ccc = px.bar(ccc_comparison_df, x='Metode', y='CCC', 
                        title='Perbandingan Cophenetic Correlation Coefficient (CCC)',
                        labels={'CCC': 'Cophenetic Correlation Coefficient', 'Metode': 'Metode Clustering'},
                        color='Metode',
                        color_discrete_map={
                            'Single': 'green',
                            'Complete': 'red',
                            'Average': 'blue'
                            
                        })

        # Menampilkan plot
        st.plotly_chart(fig_ccc)

with c2:
    with st.expander("⬇ PERBANDINGAN METODE SINGLE, COMPLETE DAN AVERAGE DENGAN SILLHOUTE SCORE"):
        fig = px.line(silhouette_df, x='Jumlah Cluster', y=['Single Linkage', 'Complete Linkage', 'Average Linkage'],
                    labels={'value': 'Silhouette Score', 'variable': 'Metode'},
                    title='Silhouette Score untuk Berbagai Jumlah Cluster',
                    color_discrete_map={
                        'Single Linkage': 'green',
                        'Complete Linkage': 'red',
                        'Average Linkage': 'blue'
                        
                    })

        # Tampilkan plot
        st.plotly_chart(fig)
