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

st.title("🚀 Platform AI Analisis Konsumen (Premium)")
st.markdown("Analisis ulasan dari berbagai format file dan dapatkan insight bisnis instan.")

# --- SIDEBAR: UNGGAH FILE ---
st.sidebar.header("Pusat Unggah Data")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file (Excel, CSV, JSON, atau TXT)", 
    type=['xlsx', 'xls', 'csv', 'json', 'txt']
)

# Fungsi untuk membaca berbagai format
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
        
        # --- LOGIKA AI (TRAINING OTOMATIS) ---
        if 'Ulasan' in df.columns and 'Label' in df.columns:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df['Ulasan'].astype(str))
            y = df['Label']
            
            model = LogisticRegression()
            model.fit(X, y)
            
            # --- DASHBOARD VISUAL ---
            st.header("📊 Dashboard Analitik")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribusi Sentimen")
                fig_pie = px.pie(df, names='Label', hole=0.3, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                st.subheader("Analisis Kata Kunci (Word Cloud)")
                text_combined = " ".join(df['Ulasan'].astype(str))
                wc = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)

            # --- FITUR PREDIKSI INDIVIDUAL ---
            st.divider()
            st.header("🧠 Uji Prediksi Real-Time")
            input_text = st.text_input("Masukkan ulasan baru untuk diuji oleh AI:")
            
            if input_text:
                vec_input = vectorizer.transform([input_text])
                pred = model.predict(vec_input)[0]
                warna = "green" if pred == "Membeli" else "red"
                st.markdown(f"### Hasil Analisis: :{warna}[{pred}]")

            # --- FITUR EXPORT DATA ---
            st.divider()
            st.header("📥 Ekspor Laporan Bisnis")
            
            # Tambahkan kolom prediksi ke seluruh data original
            df['Prediksi_AI'] = model.predict(X)
            
            # Tombol Download Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil_Analisis')
            
            st.download_button(
                label="Unduh Laporan Excel Lengkap",
                data=output.getvalue(),
                file_name="Laporan_Analisis_AI.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            st.error("Format kolom tidak sesuai. Pastikan file memiliki kolom 'Ulasan' dan 'Label' untuk melatih AI.")
    else:
        st.error("Gagal memproses file.")
else:
    st.info("Silakan unggah file di sidebar untuk memulai analisis.")