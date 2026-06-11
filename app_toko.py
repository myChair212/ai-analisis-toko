import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# --- CONFIG HALAMAN ---
st.set_page_config(page_title="AI Analisis Toko Pro", layout="wide")

st.title("🚀 Platform AI Analisis Konsumen (Sapu Jagat)")
st.markdown("Unggah file apa saja, AI akan otomatis melatih diri atau menebak sentimen secara masal.")

# --- DATA SETTINGAN BAWAAN (Untuk Modal Otak AI) ---
# Ini data 85 baris Anda kemarin agar AI punya modal kepintaran dasar jika file baru tidak punya label
@st.cache_resource
def buat_otak_palsu():
    data_dasar = {
        'Ulasan': [
            'barangnya bagus banget saya suka', 'kecewa banget barang hancur pas sampai',
            'respons penjual cepat dan ramah', 'kurir lambat pengiriman telat seminggu',
            'kualitas oke sesuai harga', 'nyesel beli di sini pelayanan buruk'
        ],
        'Label': ['Membeli', 'Tidak Membeli', 'Membeli', 'Tidak Membeli', 'Membeli', 'Tidak Membeli']
    }
    df_dasar = pd.DataFrame(data_dasar)
    vec = TfidfVectorizer()
    X_base = vec.fit_transform(df_dasar['Ulasan'])
    mdl = LogisticRegression()
    mdl.fit(X_base, df_dasar['Label'])
    return vec, mdl

vectorizer_base, model_base = buat_otak_palsu()

# --- SIDEBAR: UNGGAH FILE ---
st.sidebar.header("Pusat Unggah Data")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file (Excel, CSV, JSON, atau TXT)", 
    type=['xlsx', 'xls', 'csv', 'json', 'txt']
)

@st.cache_data
def load_data(file):
    name = file.name
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(file)
    elif name.endswith('.csv'):
        return pd.read_csv(file)
    elif name.endswith('.json'):
        return pd.read_json(file)
    elif name.endswith('.txt'):
        text = file.read().decode("utf-8").splitlines()
        return pd.DataFrame(text, columns=['Ulasan'])
    return None

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"Berhasil memuat data: {uploaded_file.name}")
        
        # Cari nama kolom yang mirip dengan 'Ulasan' (mengantisipasi huruf kecil/besar)
        kolom_ulasan = None
        for col in df.columns:
            if 'ulasan' in str(col).lower() or 'text' in str(col).lower() or 'komentar' in str(col).lower():
                kolom_ulasan = col
                break
        
        if kolom_ulasan is None and len(df.columns) > 0:
            # Jika tidak ketemu, paksa pakai kolom pertama sebagai kolom ulasan
            kolom_ulasan = df.columns[0]
            df = df.rename(columns={kolom_ulasan: 'Ulasan'})
            kolom_ulasan = 'Ulasan'

        if kolom_ulasan:
            # --- PROSES PREDIKSI OTOMATIS ---
            # Tebak sentimen untuk setiap baris di file baru menggunakan otak AI dasar
            X_new = vectorizer_base.transform(df[kolom_ulasan].astype(str))
            df['Prediksi_AI'] = model_base.predict(X_new)
            
            # --- DASHBOARD VISUAL ---
            st.header("📊 Dashboard Analitik Hasil Tebakan AI")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribusi Hasil Analisis")
                fig_pie = px.pie(df, names='Prediksi_AI', hole=0.3, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                st.subheader("Analisis Kata Kunci (Word Cloud)")
                text_combined = " ".join(df[kolom_ulasan].astype(str))
                if text_combined.strip():
                    wc = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)
                else:
                    st.text("Teks tidak cukup untuk membuat Word Cloud.")

            # --- TAMPILKAN TABEL HASIL ---
            st.divider()
            st.header("📋 Data Hasil Analisis Otomatis")
            st.dataframe(df[[kolom_ulasan, 'Prediksi_AI']].head(20))

            # --- FITUR EXPORT DATA ---
            st.divider()
            st.header("📥 Ekspor Laporan Bisnis")
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil_Analisis')
            
            st.download_button(
                label="Unduh Laporan Hasil Tebakan AI (.xlsx)",
                data=output.getvalue(),
                file_name="Hasil_Analisis_Masal_AI.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            st.error("File kosong atau tidak memiliki kolom teks yang bisa dibaca.")
else:
    st.info("Silakan unggah file ulasan Anda di sidebar untuk memulai analisis otomatis.")