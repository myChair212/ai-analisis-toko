import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import json
import sqlite3
import os
import pickle # Pustaka khusus untuk membungkus otak AI menjadi file fisik

# --- 1. PENGATURAN HALAMAN ---
st.set_page_config(page_title="AI Analisis Toko Ultimate", layout="centered")

st.title("🚀 Platform AI Analisis Konsumen (AutoML + Export Mode)")
st.markdown("Sistem otomatis yang mendukung ekspor file fisik Otak AI (.pkl) untuk portabilitas tinggi.")
st.divider()

# --- 2. SIDEBAR: PUSAT UNGGAH DATA ---
st.sidebar.header("📁 Pusat Data Multi-Format")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file data Anda", 
    type=['xlsx', 'xls', 'csv', 'txt', 'json', 'sql', 'db', 'sqlite']
)

def load_data_multiformat(file):
    name = file.name
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(file)
        elif ext == '.csv':
            return pd.read_csv(file)
        elif ext == '.txt':
            text = file.read().decode("utf-8").splitlines()
            return pd.DataFrame(text, columns=['Ulasan'])
        elif ext == '.json':
            return pd.read_json(file)
        elif ext in ['.sql', '.db', '.sqlite']:
            with open("temp_db.db", "wb") as f:
                f.write(file.getbuffer())
            conn = sqlite3.connect("temp_db.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if tables:
                nama_tabel_pertama = tables[0][0]
                df_sql = pd.read_sql_query(f"SELECT * FROM {nama_tabel_pertama}", conn)
                conn.close()
                return df_sql
            else:
                st.error("Database SQL tidak memiliki tabel data.")
                conn.close()
                return None
    except Exception as e:
        st.error(f"Gagal membaca format {ext}: {str(e)}")
        return None
    return None

# --- 3. PEMROSESAN DATA & AUTO-TRAINING DYNAMIC ---
if uploaded_file:
    df = load_data_multiformat(uploaded_file)
    
    if df is not None:
        jumlah_baris = len(df)
        st.success(f"✓ Berhasil memuat data: **{uploaded_file.name}** ({jumlah_baris} baris data terdeteksi)")
        
        kolom_ulasan = None
        kolom_label = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'ulasan' in col_lower or 'text' in col_lower or 'komentar' in col_lower or 'review' in col_lower:
                kolom_ulasan = col
            if 'label' in col_lower or 'sentimen' in col_lower or 'status' in col_lower:
                kolom_label = col

        if kolom_ulasan is None and len(df.columns) > 0:
            kolom_ulasan = df.columns[0]
            df = df.rename(columns={kolom_ulasan: 'Ulasan'})
            kolom_ulasan = 'Ulasan'

        # --- PROSES MACHINE LEARNING + EKSPOR OTAK AI ---
        if kolom_label and df[kolom_label].notna().any():
            st.info(f"🧠 Memulai pelatihan ulang (Retraining) otomatis dari {uploaded_file.name}...")
            
            # Setup N-Gram
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
            X_train = vectorizer.fit_transform(df[kolom_ulasan].astype(str))
            y_train = df[kolom_label].astype(str)
            
            model = LogisticRegression(class_weight='balanced')
            model.fit(X_train, y_train)
            
            df['Prediksi_AI'] = model.predict(X_train)
            label_display = 'Prediksi_AI'
            
            st.success("✨ Sukses! Otak AI telah diperbarui otomatis.")
            
            # ---------------------------------------------------------
            # FITUR BARU: TOMBOL UNDUH OTAK AI (.PKL)
            # ---------------------------------------------------------
            st.subheader("💾 Ekspor Portabilitas Otak AI")
            st.markdown("Unduh file fisik kecerdasan AI ini untuk ditanam di aplikasi lain (WhatsApp Bot, Web Kasir, dll):")
            
            # Kita bungkus Vectorizer dan Model menjadi satu paket di dalam sebuah dictionary
            otak_siap_pakai = {
                'vectorizer': vectorizer,
                'model': model
            }
            
            # Ubah objek python menjadi bytes data
            otak_bytes = pickle.dumps(otak_siap_pakai)
            
            st.download_button(
                label="🧠 Unduh File Fisik Otak AI (.pkl)",
                data=otak_bytes,
                file_name="otak_cerdas_ai.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
            st.divider()
        else:
            st.warning("⚠️ File tidak memiliki kolom 'Label' untuk melatih AI.")
            st.stop()
            
        # --- BAGIAN A: GRAFIK PAI ---
        st.header("📊 Ringkasan Distribusi Sentimen")
        warna_custom = {'Membeli': '#10b981', 'Tidak Membeli': '#ef4444'}
        fig_pie = px.pie(df, names=label_display, hole=0.4, color=label_display, color_discrete_map=warna_custom)
        fig_pie.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.divider() 
        
        # --- BAGIAN B: WORD CLOUD ---
        st.header("🔍 Kata Kunci yang Sering Muncul")
        text_combined = " ".join(df[kolom_ulasan].astype(str)).lower()
        if text_combined.strip():
            wc = WordCloud(width=1000, height=450, background_color="white", colormap="viridis", prefer_horizontal=0.85, max_words=100).generate(text_combined)
            fig_wc, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True)
        st.divider()

        # --- BAGIAN C: DATA DETIL TABEL ---
        st.header("📋 Hasil Detil Analisis")
        st.dataframe(df[[kolom_ulasan, label_display]], use_container_width=True, height=350)
        st.divider()

        # --- BAGIAN D: TOMBOL UNDUH LAPORAN EXCEL ---
        st.header("📥 Pusat Unduh Laporan")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Analisis_AI_Otomatis')
        st.download_button(label="📥 Unduh Laporan Excel Lengkap (.xlsx)", data=output.getvalue(), file_name="Laporan_AutoML.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
        st.divider()

        # --- BAGIAN E: UJI COBA REAL-TIME (VERSI PINTAR MULTI-LABEL) ---
        st.header("🧠 Kotak Uji Coba Cepat")
        input_text = st.text_input("Ketik ulasan baru untuk diuji oleh otak AI hasil training:")
        
        if input_text:
            vec_input = vectorizer.transform([input_text])
            pred = model.predict(vec_input)[0]
            
            # KANAL PENERJEMAH: Ubah angka 0/1 menjadi teks agar enak dibaca manusia
            if str(pred) == "1" or str(pred).lower() == "membeli":
                hasil_teks = "Membeli (Positif)"
                warna = "green"
            elif str(pred) == "0" or str(pred).lower() == "tidak membeli":
                hasil_teks = "Tidak Membeli (Negatif)"
                warna = "red"
            else:
                # Jaga-jaga jika file Anda menggunakan label format lain (A, B, C, dll)
                hasil_teks = str(pred)
                warna = "blue"
                
            st.markdown(f"### Hasil Analisis: :{warna}[{hasil_teks}]")
            
    else:
        st.error("Gagal memproses file.")
else:
    st.info("👋 Sistem Super AutoML + Export Aktif! Silakan unggah file data Anda di menu sidebar untuk melatih AI dan mengunduh file fisiknya secara instan.")