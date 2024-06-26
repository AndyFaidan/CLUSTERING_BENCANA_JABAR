import streamlit as st
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

# Fungsi untuk tooltip (jika diperlukan)
def tooltip(image_url, text):
    return f'<img src="{image_url}" title="{text}" style="border: 1px solid blue; border-radius: 4px; background-color: green; color: white; padding: 2px;">'

# Konten HTML yang diperbarui
html_content = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <style>
    iframe {{
        margin-bottom: 0px;
    }}
    html {{
        margin-top: 20px;
        margin-bottom: 0px;
        border: 1px solid #de3f53;
        padding: 0px 4px;
        font-family: "Source Sans Pro", sans-serif;
        font-weight: 400;
        line-height: 1.6;
        color: rgb(49, 51, 63);
        background-color: rgb(255, 255, 255);
        text-size-adjust: 100%;
        -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
        -webkit-font-smoothing: auto;
    }}
    h2 {{
        color: #ffffff;
        margin-top: 0px;
        border-bottom: solid 5px;
        text-align: center;
    }}
    .highlight-text {{
        font-weight: bold;
    }}
    p {{
        text-align: justify;
    }}
    </style>

    <h2>Pengelompokan Daerah Rawan Bencana Tanah Longsor Kabupaten/Kota di Provinsi Jawa Barat</h2>

    
    <p> Berdasarkan data Badan Nasional Penanggulangan Bencana (BNPB) melaporkan ada 3.531 bencana alam. Berikut daftar lengkap 10 provinsi yang paling banyak mengalami bencana alam di Indonesia pada 2022 diantaranya, Jawa Barat 823 kejadian.</p>
    <p> Berdasarkan data Badan Nasional Penanggulangan Bencana (BNPB), Jawa Barat  menjadi provinsi dengan bencana tanah longsor terbanyak dalam sedekade terakhir. Kejadian tanah longsor selama periode 2014-2023. Diantaranya Jawa Barat 2.440 Kejadian.</p>
    <p> Dalam menghadapi bencana alam, kesiapsiagaan menjadi kunci untuk meminimalisir dampak. Masyarakat di Jawa Barat harus siap dan mampu bertindak saat bencana terjadi. Kecepatan dan ketepatan penanggulangan sangat penting untuk mengurangi kerentanan fisik dan sosial ekonomi. </p> 
    <p> Melalui analisis komprehensif dan pemetaan akurat, perencanaan mitigasi dan pengembangan kebijakan yang efektif dapat dicapai. Untuk mengidentifikasi daerah rawan bencana tanah longsor di Jawa Barat, algoritma K-means dan Agglomerative Hierarchical Clustering (AHC) digunakan untuk mengelompokkan data berdasarkan karakteristiknya.</p>
"""

# Tampilkan konten HTML
st.markdown(html_content, unsafe_allow_html=True)

# Adding a divider line
st.divider()

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
        Berikut adalah video YouTube yang menjelaskan tentang teknik analisis clustering menggunakan algoritma k-means.
    """)

    # Embed YouTube video
    video_url_kmeans = "https://www.youtube.com/embed/BMzXuG1p3lQ?t=887s"
    video_html_kmeans = f"""
        <div style="display: flex; justify-content: center;">
            <iframe width="1000" height="450" src="{video_url_kmeans}" 
            frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
            </iframe>
        </div>
    """
    st.markdown(video_html_kmeans, unsafe_allow_html=True)

with c2:
    st.markdown("""
    Berikut adalah video YouTube yang menjelaskan tentang teknik analisis clustering menggunakan algoritma AHC (Agglomerative Hierarchical Clustering).
    """)

    # Embed YouTube video
    video_url_ahc = "https://www.youtube.com/embed/s8K0lO9OFOA?start=1067"
    video_html_ahc = f"""
        <div style="display: flex; justify-content: center;">
            <iframe width="1000" height="450" src="{video_url_ahc}" 
            frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
            </iframe>
        </div>
    """
    components.html(video_html_ahc, height=450)

# Ekspander untuk informasi tentang clustering bencana longsor
with st.expander("**Tentang Analisis Klaster Bencana Tanah Longsor di Kabupaten dan Kota di Provinsi Jawa Barat**"):
    st.markdown('''
    
    **Metode Klaster:🌐**
    - Metode clustering digunakan untuk mengelompokkan daerah rawan bencana 
      di kabupaten dan kota Provinsi Jawa Barat, berdasarkan atribut:
      - Jumlah kejadian bencana longsor
      - Jiwa terdampak
      - Jiwa meninggal
      - Rusak terdampak
      - Rusak ringan
      - Rusak sedang
      - Rusak berat
      - Tertimbun

    **Sumber Data: 📈**
    - Data yang digunakan dalam analisis ini mencakup data jumlah menurut Kabupaten/Kota yang mengalami kejadian bencana tanah longsor di Provinsi Jawa Barat pada tahun 2021-2023.
    - Bersumber dari website BARATA yang dikelola oleh Badan Penanggulangan Bencana Daerah Provinsi Jawa Barat.

    **Tujuan: 🎯**
    - Tujuan utama adalah mengetahui pemetaan dan pengelompokan daerah rawan bencana longsor di kabupaten dan kota di Provinsi Jawa Barat. 
    - Metode ini memberikan wawasan yang berharga untuk mencapai perencanaan mitigasi optimal dan pengembangan kebijakan efektif tingkat provinsi, infrastruktur, serta upaya peningkatan kesiapsiagaan, kecepatan, dan ketepatan penanggulangan antisipasi bencana di Provinsi Jawa Barat.
    ''')

