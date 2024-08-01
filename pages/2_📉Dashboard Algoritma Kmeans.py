import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from streamlit_extras.metric_cards import style_metric_cards
import plotly_express as px
#from query import *
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
df = pd.read_csv("Jumlah-2021 - 2023 -Lengkap-Dataset_Longsor - PROV JABAR.csv")

#navicon and header
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")  

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

 
st.header(" ALGORITMA KMEANS ")
st.markdown(
 """
 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
 <hr>

<div class="card mb-3">
<div class="card">
  <div class="card-body">
    <h3 class="card-title" style="color:#007710;"><strong> PEMAHAMAN  </strong></h3>
    <p class="card-text">Algoritma KMeans mengelompokkan data dengan memisahkan sampel ke dalam n kelompok, meminimalkan varians dalam setiap kelompok. Algoritma ini bekerja dengan mengurangi inersia, yaitu jumlah kuadrat jarak antara sampel dalam cluster dengan centroidnya, melalui iterasi.</p>
    <p class="card-text">Setiap cluster diwakili oleh centroid, yang merupakan mean dari sampel dalam cluster tersebut. Proses ini melibatkan penempatan sampel dalam cluster berdasarkan jarak Euclidean ke centroid terdekat dan memperbarui centroid hingga konvergensi.</p>
    <p class="card-text">KMeans memerlukan jumlah cluster sebagai input, dan untuk menentukan jumlah cluster yang optimal, metode seperti Elbow Method dapat digunakan. Algoritma ini efisien untuk sejumlah besar sampel dan memiliki berbagai aplikasi di banyak bidang.</p>
</div>
</div>
 <style>
    [data-testid=stSidebar] {
         color: white;
         text-size:24px;
    }
</style>
""",unsafe_allow_html=True
)



# Pilih fitur yang ingin digunakan untuk klasterisasi
features_kmeans = df[['KABUPATEN', 'JUMLAH_LONGSOR', 'JIWA_TERDAMPAK', 'JIWA_MENINGGAL', 'RUSAK_TERDAMPAK', 'RUSAK_RINGAN', 'RUSAK_SEDANG', 'RUSAK_BERAT', 'TERTIMBUN', 'LATITUDE', 'LONGITUDE']]

# Sample DataFrame
df_sample = df.sample(n=10)  # Ambil sampel 10 desa

# Ekspander untuk menampilkan data
with st.expander("⬇ DATA UNDERSTANDING FOR KMEANS :"):
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

# Membuat ekspander untuk menampilkan korelasi
with st.expander("⬇ EKSPLORASI VARIABEL:"):
    st.subheader("Korelasi antara Variabel")
    
    # Ganti df_selection dengan dataframe yang ingin digunakan
    selected_features = ['JUMLAH_LONGSOR', 'JIWA_TERDAMPAK', 'JIWA_MENINGGAL', 'RUSAK_TERDAMPAK', 'RUSAK_RINGAN', 'RUSAK_SEDANG', 'RUSAK_BERAT','TERTIMBUN']
    
    # Hitung matriks korelasi
    correlation_matrix = df[selected_features].corr()

    # Plot heatmap using Plotly Express
    fig = px.imshow(correlation_matrix,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=selected_features,
                    y=selected_features,
                    color_continuous_scale="viridis",  # Use 'viridis' instead of 'coolwarm'
                    title="Heatmap Korelasi")

    # Show the plot
    st.plotly_chart(fig)

    st.write("Visualisasi ini memberikan representasi visual dari data dalam bentuk matriks, di mana intensitas warna pada setiap sel matriks menggambarkan nilai variabel yang bersangkutan, memudahkan identifikasi pola, keterkaitan, atau perbedaan dalam data.")

# checking null value
with st.expander("⬇ NULL VALUES, TENDENCY & VARIABLE DISPERSION"):
    a1, a2 = st.columns(2)
    a1.write("Jumlah nilai yang tidak ada (NaN atau None) dalam setiap kolom DataFrame.")
    a1.dataframe(df.isnull().sum(), use_container_width=True)

    a2.write("Wawasan tentang tendensi pusat, dispersi, dan distribusi data.")
    a2.dataframe(df.describe().T, use_container_width=True)

# Pilih kolom numerik untuk klasterisasi KMeans
kolom_numerik = ['JUMLAH_LONGSOR', 'JIWA_TERDAMPAK', 'JIWA_MENINGGAL', 'RUSAK_TERDAMPAK', 'RUSAK_RINGAN', 'RUSAK_SEDANG', 'RUSAK_BERAT', 'LATITUDE', 'LONGITUDE']
X = df[kolom_numerik]

# Isi nilai NaN dengan rata-rata kolom numerik
X.fillna(X.mean(), inplace=True)

# Terapkan klasterisasi KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)


# Metode Elbow untuk menentukan jumlah klaster optimal
distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Menghitung Silhouette Score untuk berbagai jumlah klaster
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))

c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("⬇ ELBOW METHOD"):
        st.write("Metode Elbow digunakan untuk membantu penentuan jumlah cluster yang optimal, dengan mengidentifikasi titik di mana penurunan inersia menjadi lebih lambat, memberikan panduan dalam memilih jumlah cluster yang sesuai untuk data yang dianalisis")
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), distortions, marker='o')
        plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
        plt.xlabel('Jumlah Klaster')
        plt.ylabel('Distorsi')
        st.pyplot()

# Visualisasi Silhouette Score
with c2:
    with st.expander("⬇ SILHOUETTE SCORE"):
        st.write("Silhouette Score digunakan untuk mengevaluasi seberapa baik setiap titik data sesuai dengan klaster tempat berada, memberikan pengukuran yang membantu menilai kualitas klasterisasi secara keseluruhan.")
        
        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.title('Silhouette Score untuk Menentukan Jumlah Klaster Optimal')
        plt.xlabel('Jumlah Klaster')
        plt.ylabel('Silhouette Score')
        st.pyplot()

# Kesimpulan
with c3:
    with st.expander("⬇ KESIMPULAN"):
        st.write("Dari analisis yang melibatkan Elbow Method dan Silhouette Score, dapat di simpulkan::")
        
        # Menentukan jumlah klaster optimal dari Elbow Method
        optimal_clusters = 2  # Ganti dengan hasil analisis Elbow Method
        st.write(f"Jumlah klaster optimal berdasarkan Elbow Method: {optimal_clusters}")
        
        # Menampilkan Silhouette Score tertinggi
        best_silhouette_score = max(silhouette_scores)
        st.write(f"Silhouette Score tertinggi: {best_silhouette_score}")

# Menampilkan plot sebaran menggunakan Plotly Express
with st.expander("⬇ CLUSTER VISUALIZATION"):
    fig = px.scatter(df, x='LONGITUDE', y='LATITUDE', color='Cluster',
                 title="Clusters of Regions", labels={'LONGITUDE': 'LONGITUDE', 'LATITUDE': 'LATITUDE'},
                 color_continuous_scale='viridis', size_max=10)
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig)

