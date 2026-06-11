import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import os

# --- 1. PENGATURAN HALAMAN ---
st.set_page_config(page_title="AI Analisis Toko Publik", layout="centered")

st.title("🚀 Platform AI Analisis Konsumen (Pure Analytics)")
st.markdown("""
👋 Selamat datang! Sistem ini dikonfigurasi untuk menganalisis data ulasan konsumen yang **polosan (tanpa label)**.
AI akan membaca teks ulasan Anda, menganalisis bobot katanya secara mandiri, dan merakit laporan analisis lengkap untuk Anda.
""")
st.divider()

# --- 2. SIDEBAR: PUSAT UNGGAH DATA POLOS ---
st.sidebar.header("📁 Unggah Berkas Anda")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file Excel ulasan kosong Anda", 
    type=['xlsx', 'xls', 'csv']
)

# --- 3. OTAK UTAMA AI (INTELLIGENT KNOWLEDGE BASE) ---
@st.cache_resource
def inisialisasi_otak_ai():
    # Ini adalah pola hafalan dasar N-Gram agar AI tahu kata mana yang bermakna positif/negatif
    data_pengetahuan = {
        'Ulasan': [
            'bagus banget suka pas mantap', 
            'rusak hancur kecewa jelek buruk', 
            'cepat ramah kurir baik', 
            'lambat telat parah kapok', 
            'oke produk original sesuai', 
            'menyesal salah warna robek'
        ],
        'Label': ['Membeli', 'Tidak Membeli', 'Membeli', 'Tidak Membeli', 'Membeli', 'Tidak Membeli']
    }
    df_knowledge = pd.DataFrame(data_pengetahuan)
    
    # Menggunakan kombinasi N-gram (1 sampai 3 kata) seperti versi sukses kemarin
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
    X_train = vectorizer.fit_transform(df_knowledge['Ulasan'])
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, df_knowledge['Label'])
    
    return vectorizer, model

# Aktifkan otak AI di latar belakang
vectorizer, model = inisialisasi_otak_ai()

# --- 4. PEMROSESAN & EKSEKUSI PREDIKSI MANDIRI ---
if uploaded_file:
    # Membaca file mentah uploader
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
        
    if df is not None:
        jumlah_baris = len(df)
        st.success(f"✓ Berhasil memuat: **{uploaded_file.name}** ({jumlah_baris} baris ulasan terdeteksi)")
        
        # Deteksi otomatis letak kolom teks ulasan
        kolom_ulasan = None
        for col in df.columns:
            if 'ulasan' in str(col).lower() or 'text' in str(col).lower() or 'komentar' in str(col).lower():
                kolom_ulasan = col
                break
        
        if kolom_ulasan is None and len(df.columns) > 0:
            kolom_ulasan = df.columns[0]
            df = df.rename(columns={kolom_ulasan: 'Ulasan'})
            kolom_ulasan = 'Ulasan'
            
        # 🏃‍♂️ PROSES INTI: AI MENGANALISA SENDIRI SECARA MANDIRI
        st.info("🧠 AI sedang menyisir teks dan melakukan analisis sentimen mandiri...")
        
        # Mengubah teks ulasan baru menjadi vektor matematika
        X_new = vectorizer.transform(df[kolom_ulasan].astype(str))
        
        # AI menebak hasilnya secara mandiri (Output: 'Membeli' atau 'Tidak Membeli')
        df['Hasil_Analisis_AI'] = model.predict(X_new)
        
        # Merapikan teks label untuk keperluan visualisasi dashboard publik
        df['Status_Teks'] = df['Hasil_Analisis_AI'].apply(
            lambda x: "Membeli (Positif)" if str(x).lower() == 'membeli' else "Tidak Membeli (Negatif)"
        )
        
        # --- TAMPILAN DASHBOARD HASIL ANALISIS AI ---
        
        # A. Grafik Distribusi Hasil Prediksi AI
        st.header("📊 Ringkasan Distribusi Sentimen Hasil Analisis AI")
        fig_pie = px.pie(
            df, 
            names='Status_Teks', 
            hole=0.4, 
            color='Status_Teks', 
            color_discrete_map={'Membeli (Positif)': '#10b981', 'Tidak Membeli (Negatif)': '#ef4444'}
        )
        fig_pie.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.divider() 
        
        # B. Word Cloud dari Dokumen Publik
        st.header("🔍 Kata Kunci Terpopuler")
        text_combined = " ".join(df[kolom_ulasan].astype(str)).lower()
        if text_combined.strip():
            wc = WordCloud(width=1000, height=450, background_color="white", colormap="viridis").generate(text_combined)
            fig_wc, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True)
        st.divider()

        # C. Tabel Detil Analisis
        st.header("📋 Hasil Detil Analisis Per Baris")
        st.dataframe(df[[kolom_ulasan, 'Status_Teks']], use_container_width=True, height=300)
        st.divider()

        # D. PUSAT UNDUH: MEMASUKKAN HASIL PREDIKSI AI KE EXCEL BARU
        st.header("📥 Pusat Unduh")
        st.markdown("Unduh file berkas Anda yang kini telah dilengkapi dengan kolom hasil analisis otomatis oleh AI:")
        
        # Merakit file excel baru di RAM memori
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # File hasil analisis disimpan bersih ke file excel baru
            df[[kolom_ulasan, 'Status_Teks']].to_excel(writer, index=False, sheet_name='Hasil_Analisis_AI')
            
        st.download_button(
            label="📥 Unduh Hasil Analisis Lengkap (.xlsx)", 
            data=output.getvalue(), 
            file_name="Laporan_Analisis_Sentimen_AI.xlsx", 
            mime="application/vnd.ms-excel", 
            use_container_width=True
        )
        st.divider()

        # E. Kotak Uji Cepat Real-time
        st.header("🧠 Kotak Uji Coba Cepat")
        input_text = st.text_input("Ketik sampel kalimat baru untuk diuji langsung:")
        if input_text:
            vec_input = vectorizer.transform([input_text])
            pred = model.predict(vec_input)[0]
            if str(pred).lower() == 'membeli':
                st.markdown("### Hasil Analisis: :green[Membeli (Positif)]")
            else:
                st.markdown("### Hasil Analisis: :red[Tidak Membeli (Negatif)]")
                
else:
    st.info("👋 Sistem Pure Analytics Siap! Silakan unggah file Excel ulasan polosan Anda (tanpa kolom status) di sidebar untuk melihat kehebatan AI menganalisis data Anda secara mandiri.")