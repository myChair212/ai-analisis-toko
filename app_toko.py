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

st.title("🚀 Platform AI Analisis Konsumen (Dynamic Engine)")
st.markdown("""
👋 Selamat datang! Sistem ini berjalan dalam mode **Dynamic Machine Learning**. 
Otak AI tidak ditanam di dalam kode, melainkan akan dirakit secara otomatis langsung dari file data latihan yang Anda unggah.
""")
st.divider()

# --- 2. SIDEBAR: PUSAT UNGGAH DATA ---
st.sidebar.header("📁 Pusat Unggah Berkas")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file Excel/CSV data latihan Anda", 
    type=['xlsx', 'xls', 'csv']
)

# --- 3. PROSES PEMBACAAN DAN GENERATE OTAK AI SECARA OTOMATIS ---
if uploaded_file:
    # Membaca berkas yang diunggah
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        df = None

    if df is not None:
        jumlah_baris = len(df)
        st.success(f"✓ Berhasil memuat berkas: **{uploaded_file.name}** ({jumlah_baris} baris data)")
        
        # Deteksi otomatis letak kolom Ulasan dan Label di dalam file yang diupload
        kolom_ulasan = None
        kolom_label = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'ulasan' in col_lower or 'text' in col_lower or 'komentar' in col_lower or 'review' in col_lower:
                kolom_ulasan = col
            if 'label' in col_lower or 'sentimen' in col_lower or 'status' in col_lower:
                kolom_label = col

        # Proteksi jika nama kolom tidak standar
        if kolom_ulasan is None and len(df.columns) > 0:
            kolom_ulasan = df.columns[0]
        
        # SYARAT UTAMA: File harus punya kolom Ulasan dan kolom Label agar AI bisa belajar
        if kolom_ulasan and kolom_label:
            st.info("🧠 Men-generate Otak AI baru secara otomatis langsung dari data Anda...")
            
            # 🚀 PROSES GENERATE OTAK AI LANGSUNG DARI FILE UPLOAD
            # AI membaca kombinasi kata n-gram (1-3 kata) dari file yang Anda upload
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
            X_train = vectorizer.fit_transform(df[kolom_ulasan].astype(str))
            y_train = df[kolom_label].astype(str)
            
            # Melatih model secara instan di memori RAM server
            model = LogisticRegression(class_weight='balanced')
            model.fit(X_train, y_train)
            
            st.success("✨ Sukses! Otak AI telah berhasil dirakit dan disesuaikan dengan pola data Anda.")
            st.divider()
            
            # Menghitung hasil prediksi untuk visualisasi dashboard
            df['Prediksi_AI'] = model.predict(X_train)
            
            # Standardisasi teks label agar rapi di grafik
            def standarkan_output(x):
                if str(x) == "1" or "membeli" in str(x).lower() or "positif" in str(x).lower():
                    return "Membeli (Positif)"
                return "Tidak Membeli (Negatif)"
            
            df['Status_Teks'] = df['Prediksi_AI'].apply(standarkan_output)

            # --- 4. TAMPILAN DASHBOARD HASIL GENERATE AI ---
            
            # A. Grafik Pai Distribusi Sentimen
            st.header("📊 Ringkasan Distribusi Sentimen")
            fig_pie = px.pie(df, names='Status_Teks', hole=0.4, color='Status_Teks', color_discrete_map={'Membeli (Positif)': '#10b981', 'Tidak Membeli (Negatif)': '#ef4444'})
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.divider() 
            
            # B. Word Cloud
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

            # D. Tombol Unduh Hasil Laporan
            st.header("📥 Pusat Unduh")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df[[kolom_ulasan, 'Status_Teks']].to_excel(writer, index=False, sheet_name='Hasil_Analisis_AI')
            st.download_button(label="📥 Unduh Hasil Analisis Lengkap (.xlsx)", data=output.getvalue(), file_name="Hasil_Dynamic_Analytics.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
            st.divider()

            # E. KOTAK UJI COBA CEPAT (Menggunakan Otak yang baru saja di-generate)
            st.header("🧠 Kotak Uji Coba Cepat")
            input_text = st.text_input("Ketik sampel kalimat baru untuk diuji langsung oleh Otak AI baru:")
            if input_text:
                vec_input = vectorizer.transform([input_text])
                pred = model.predict(vec_input)[0]
                
                if str(pred) == "1" or "membeli" in str(pred).lower() or "positif" in str(pred).lower():
                    st.markdown("### Hasil Analisis: :green[Membeli (Positif)]")
                else:
                    st.markdown("### Hasil Analisis: :red[Tidak Membeli (Negatif)]")
        else:
            st.error("⚠️ Berkas yang diunggah harus memiliki kolom teks ulasan dan kolom label/status agar AI bisa digenerate!")
else:
    # Halaman awal bersih saat pertama kali dibuka publik
    st.info("👋 Sistem siap! Silakan unggah file Excel data latihan Anda di menu sidebar untuk mulai men-generate Otak AI dan melihat visualisasi dashboard secara otomatis.")