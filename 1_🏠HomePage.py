import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import folium_static
from folium import plugins
import altair as alt


# Set page configuration
st.set_page_config(
    page_title="Analisis Clustering Daerah Rawan Bencana Tanah Longsor Provinsi Jawa Barat",
    layout="wide",  # Set layout to wide for full-width content
    initial_sidebar_state="collapsed",  # Collapse the sidebar by default
)

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Title of the application
st.title('ANALISIS PENGELOMPOKKAN DAERAH RAWAN BENCANA TANAH LONGSOR KABUPATEN/KOTA DI PROVINSI JAWA BARAT')

# Adding a divider line
st.divider()


# Muat DataFrame Anda dari DATA_JABAR.csv
file_path = 'UPDATE-Selection-Dataset_Longsor 2021 - 2023 - PROV JABAR.csv'
df = pd.read_csv(file_path)

# Sidebar untuk memilih tahun
selected_year = st.sidebar.slider('Select Year', min_value=df['TAHUN'].min(), max_value=df['TAHUN'].max(), value=df['TAHUN'].max(), step=1)

# Dropdown untuk memilih KABUPATEN
selected_kabupaten = st.sidebar.selectbox('Select Kabupaten', df['KABUPATEN'].unique())

# Filter bingkai data untuk tahun yang dipilih dan KABUPATEN
df_filtered_year = df[df['TAHUN'] == selected_year]
df_filtered_kabupaten = df_filtered_year[df_filtered_year['KABUPATEN'] == selected_kabupaten]

# Membuat peta dengan kunci unik berdasarkan tahun yang dipilih
m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=8, key=f"map-{selected_year}", width='100%')


# Tambahkan penanda untuk setiap titik data
for i, row in df[df['TAHUN'] == selected_year].iterrows():
    popup_content = f"""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <div style='width:400px; height:300px;'>
        <ul class="list-group">
            <li class="list-group-item active" aria-current="true">
                <h3 class="mb-0">Information of {row['KABUPATEN']}</h3>
            </li>
            <li class="list-group-item">
                <b>Number of Landslides:</b> {row['JUMLAH_LONGSOR']}
            </li>
            <li class="list-group-item">
                <b>Affected Population:</b> {row['JIWA_TERDAMPAK']}
            </li>
            <li class="list-group-item">
                <b>Deaths:</b> {row['JIWA_MENINGGAL']}
            </li>
            <li class="list-group-item">
                <b>Infrastructure Affected:</b> {row['RUSAK_TERDAMPAK']}
            </li>
            <li class="list-group-item">
                <b>Minor Damage:</b> {row['RUSAK_RINGAN']}
            </li>
            <li class="list-group-item">
                <b>Moderate Damage:</b> {row['RUSAK_SEDANG']}
            </li>
            <li class="list-group-item">
                <b>Severe Damage:</b> {row['RUSAK_BERAT']}
            </li>
            <li class="list-group-item">
                <b>Buried:</b> {row['TERTIMBUN']}
            </li>
            <li class="list-group-item">
                <h4>Latitude: {row['LATITUDE']}</h4>
            </li>
            <li class="list-group-item">
                <h4>Longitude: {row['LONGITUDE']}</h4>
            </li>
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
        icon=icon,
    ).add_to(m).add_child(folium.Popup(popup_content, max_width=600))

# Heatmap Layer
heat_data = [[row['LATITUDE'], row['LONGITUDE']] for _, row in df[df['TAHUN'] == selected_year].iterrows()]
HeatMap(heat_data).add_to(m)

# Fullscreen Control
plugins.Fullscreen(position='topright', title='Fullscreen', title_cancel='Exit Fullscreen').add_to(m)

# Drawing Tools
draw = plugins.Draw()
draw.add_to(m)

# Adding Google Maps tiles
def add_google_maps(m):
    tiles = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    attr = "Google Digital Satellite"
    folium.TileLayer(tiles=tiles, attr=attr, name=attr, overlay=True, control=True).add_to(m)
    # Add labels for streets and objects
    label_tiles = "https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}"
    label_attr = "Google Labels"
    folium.TileLayer(tiles=label_tiles, attr=label_attr, name=label_attr, overlay=True, control=True).add_to(m)

    return m


# Fungsi untuk membuat peta panas dengan pemilihan tema warna
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="TAHUN", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'{input_color}:Q',
                        legend=None,
                        scale=alt.Scale(scheme=input_color_theme)),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=500  # Adjust width as needed
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )
    return heatmap

# Fungsi format_number untuk memformat angka dengan koma sebagai pemisah ribuan
def format_number(num):
    return "{:,}".format(num)

# Fungsi perhitungan bencana tanah longsor tahun ke tahun
def calculate_population_difference(input_df, input_year):
    selected_year_data = input_df[input_df['TAHUN'] == input_year].reset_index()
    previous_year_data = input_df[input_df['TAHUN'] == input_year - 1].reset_index()
    selected_year_data['population_difference'] = selected_year_data['JUMLAH_LONGSOR'].sub(previous_year_data['JUMLAH_LONGSOR'], fill_value=0)
    return pd.concat([selected_year_data['KABUPATEN'], selected_year_data['TAHUN'], selected_year_data['JUMLAH_LONGSOR'], selected_year_data['population_difference']], axis=1).sort_values(by="population_difference", ascending=False)

a1, a2, a3 = st.columns(3)

with a1:
    df_population_difference_sorted = calculate_population_difference(df, selected_year)
    if selected_year > 2020:
        first_state_name = df_population_difference_sorted['KABUPATEN'].iloc[0]
        first_state_population = format_number(int(df_population_difference_sorted['JUMLAH_LONGSOR'].iloc[0]))
        first_state_delta = format_number(int(df_population_difference_sorted['population_difference'].iloc[0]))
    else:
        first_state_name = '-'
        first_state_population = '-'
        first_state_delta = ''
    st.metric(label=first_state_name, value=first_state_population, delta=first_state_delta)
with a2:
    if selected_year > 2020:
        last_state_name = df_population_difference_sorted['KABUPATEN'].iloc[-1]
        last_state_population = format_number(int(df_population_difference_sorted['JUMLAH_LONGSOR'].iloc[-1]))   
        last_state_delta = format_number(int(df_population_difference_sorted['population_difference'].iloc[-1]))   
    else:
        last_state_name = '-'
        last_state_population = '-'
        last_state_delta = ''
    st.metric(label=last_state_name, value=last_state_population, delta=last_state_delta)
with a3:
    if selected_year > 2020:
        total_landslides_selected_year = df[df['TAHUN'] == selected_year]['JUMLAH_LONGSOR'].sum()
        total_landslides_last_year = df[df['TAHUN'] == selected_year - 1]['JUMLAH_LONGSOR'].sum()
        landslide_difference = total_landslides_selected_year - total_landslides_last_year

        total_landslides_name = "Total Landslides"
        total_landslides_value = format_number(total_landslides_selected_year)
        total_landslides_delta = format_number(landslide_difference) if landslide_difference != 0 else "0"

        st.metric(label=total_landslides_name, value=total_landslides_value, delta=total_landslides_delta)
    else:
        st.metric(label="-", value="-", delta="")

# Panel Dasbor Utama
# Membuat kolom dengan rasio lebar tertentu
col1, col2 = st.columns((5, 2), gap='medium')

with col1:
        m = add_google_maps(m)
        m.add_child(folium.LayerControl(collapsed=False))
        folium_static(m, width=850, height=520)


with col2:
    # Memfilter bingkai data berdasarkan tahun yang dipilih
    df_filtered = df[df['TAHUN'] == selected_year]
    
    # Urutkan bingkai data yang telah difilter berdasarkan jumlah tanah longsor
    df_landslide_sorted = df_filtered.sort_values(by="JUMLAH_LONGSOR", ascending=False)
    
    # Hapus kolom LATITUDE dan LONGITUDE untuk tujuan tampilan
    df_landslide_sorted_no_geom = df_landslide_sorted.drop(columns=['LATITUDE', 'LONGITUDE'])

    # Display the dataframe
    st.dataframe(df_landslide_sorted_no_geom,
                 column_order=("KABUPATEN", "JUMLAH_LONGSOR"),
                 hide_index=True,
                 width=500,
                 height=529,
                 column_config={
                     "KABUPATEN": st.column_config.TextColumn(
                         "Area",
                     ),
                     "JUMLAH_LONGSOR": st.column_config.ProgressColumn(
                         "Number of Landslides",
                         format="%f",
                         min_value=0,
                         max_value=max(df_landslide_sorted.JUMLAH_LONGSOR),
                     )}
                 )


with st.expander("HeatMap", expanded=False):
    st.markdown('#### HeatMap')

    heatmap_chart = make_heatmap(df, 'TAHUN', 'KABUPATEN', 'JUMLAH_LONGSOR', 'viridis') # Assuming 'viridis' as the color theme

    st.altair_chart(heatmap_chart, use_container_width=True)

with st.expander('Information', expanded=True):
    st.info('''
        - **Data Sumber**: [BARATA BADAN PENANGGULANGAN BENCANA DAERAH PROVINSI JAWA BARAT](https://barata.jabarprov.go.id/).
        - :orange[**Top Areas by Landslides**]: Kabupaten dengan jumlah longsor terbanyak pada tahun yang dipilih.
        - :orange[**Extreme Landslide Changes**]: Daerah dengan peningkatan dan penurunan jumlah longsor terbesar dari tahun sebelumnya.
        - :information_source: **Total Landslides**: Jumlah total longsor pada tahun yang dipilih.
        - :information_source: **Affected Population**: Jumlah total populasi yang terdampak oleh longsor pada tahun yang dipilih.
        - :information_source: **Infrastructure Damage**: Rincian kerusakan infrastruktur akibat longsor (ringan, sedang, berat) pada tahun yang dipilih.
        - :bar_chart: **Heatmap Visualization**: Peta panas yang menunjukkan distribusi longsor di berbagai daerah.
        - :chart_with_upwards_trend: **Landslide Trends**: Dinamika dan tren longsor, termasuk peningkatan/penurunan serta daerah dengan jumlah longsor tertinggi dan terendah dari tahun ke tahun.
    ''')
