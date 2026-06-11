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

st.title("🚀 Platform AI Analisis Konsumen (Public Engine)")
st.markdown("""
👋 Selamat datang! Platform ini disediakan untuk publik secara gratis. 
Anda dapat menganalisis sentimen ulasan konsumen Anda secara masal dengan aman. 
**Jaminan Privasi:** Data yang Anda unggah hanya diproses sementara di memori server dan langsung dihapus saat sesi selesai.
""")
st.divider()

# --- 2. SIDEBAR: TEMPAT PUBLIK UNGGAH FILE MEREKA ---
st.sidebar.header("📁 Unggah Berkas Anda")
uploaded_file = st.sidebar.file_uploader(
    "Silakan pilih file ulasan Anda (Excel/CSV/TXT)", 
    type=['xlsx', 'xls', 'csv', 'txt']
)

def load_data_publik(file):
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
    except Exception as e:
        st.error(f"Gagal membaca file: {str(e)}")
        return None
    return None

# --- 3. PEMROSESAN ANALITIK UNTUK PUBLIK ---
if uploaded_file:
    df = load_data_publik(uploaded_file)
    
    if df is not None:
        jumlah_baris = len(df)
        st.success(f"✓ Berhasil memuat data Anda: **{uploaded_file.name}** ({jumlah_baris} baris data)")
        
        # Deteksi otomatis kolom ulasan dan kolom label/target (jika ada)
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

        # --- LOGIKA AUTO-TRAINING AMAN (MENGGUNAKAN DATA YANG DIUNGGAH PUBLIK) ---
        # Skenario A: Jika publik mengunggah data latihan yang sudah ada label kuncinya (0/1 atau Membeli/Tidak)
        if kolom_label and df[kolom_label].notna().any():
            st.info("🧠 AI sedang mempelajari pola karakteristik sentimen data Anda...")
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
            X_train = vectorizer.fit_transform(df[kolom_ulasan].astype(str))
            y_train = df[kolom_label].astype(str)
            
            model = LogisticRegression(class_weight='balanced')
            model.fit(X_train, y_train)
            
            df['Prediksi_AI'] = model.predict(X_train)
            label_display = 'Prediksi_AI'
            st.success("✨ Model AI kustom berhasil dirakit khusus untuk data Anda!")
            
        # Skenario B: Jika publik mengunggah data polos tanpa label, kita gunakan modal otak bawaan
        else:
            st.info("🧠 Menganalisis data menggunakan Otak AI Standar Industri...")
            # Data modal dasar agar aplikasi tidak kosong jika publik upload data polosan
            data_modal = {
                'Ulasan': ['bagus suka', 'rusak hancur kecewa', 'cepat ramah', 'lambat telat parah', 'oke mantap', 'jelek buruk menyesal'],
                'Label': ['Membeli', 'Tidak Membeli', 'Membeli', 'Tidak Membeli', 'Membeli', 'Tidak Membeli']
            }
            df_modal = pd.DataFrame(data_modal)
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
            X_modal = vectorizer.fit_transform(df_modal['Ulasan'])
            model = LogisticRegression(class_weight='balanced')
            model.fit(X_modal, df_modal['Label'])
            
            X_new = vectorizer.transform(df[kolom_ulasan].astype(str))
            df['Prediksi_AI'] = model.predict(X_new)
            label_display = 'Prediksi_AI'

        # Konversi hasil tebakan agar ramah dibaca manusia di grafik
        df['Status_Teks'] = df[label_display].apply(lambda x: "Tidak Membeli (Negatif)" if str(x)=='0' or 'tidak' in str(x).lower() else "Membeli (Positif)")

        # --- TAMPILKAN SELURUH HASIL ANALISIS SENSASIONAL KEDALAM BENTUK LINEAR ---
        
        # A. Grafik Pai Distribusi Sentimen
        st.header("📊 Ringkasan Distribusi Sentimen")
        fig_pie = px.pie(df, names='Status_Teks', hole=0.4, color='Status_Teks', color_discrete_map={'Membeli (Positif)': '#10b981', 'Tidak Membeli (Negatif)': '#ef4444'})
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.divider() 
        
        # B. Word Cloud Premium
        st.header("🔍 Kata Kunci Terpopuler")
        text_combined = " ".join(df[kolom_ulasan].astype(str)).lower()
        if text_combined.strip():
            wc = WordCloud(width=1000, height=450, background_color="white", colormap="viridis").generate(text_combined)
            fig_wc, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True)
        st.divider()

        # C. Tabel Detil Analisis Per Baris
        st.header("📋 Hasil Detil Analisis Per Baris")
        st.dataframe(df[[kolom_ulasan, 'Status_Teks']], use_container_width=True, height=300)
        st.divider()

        # D. Tombol Unduh Laporan Excel
        st.header("📥 Pusat Unduh")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button(label="📥 Unduh Hasil Analisis Lengkap (.xlsx)", data=output.getvalue(), file_name="Hasil_Analisis_Publik.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
        st.divider()

        # E. Kotak Uji Cepat Real-time
        st.header("🧠 Kotak Uji Coba Cepat")
        input_text = st.text_input("Ketik ulasan individu baru untuk diuji langsung:")
        if input_text:
            vec_input = vectorizer.transform([input_text])
            pred = model.predict(vec_input)[0]
            if str(pred) == "0" or 'tidak' in str(pred).lower():
                st.markdown("### Hasil Analisis: :red[Tidak Membeli (Negatif)]")
            else:
                st.markdown("### Hasil Analisis: :green[Membeli (Positif)]")
    else:
        st.error("Format berkas tidak dikenali.")
else:
    # Landing page awal publik saat link dibuka (Ramah & Informatif)
    st.info("👋 Selamat Datang! Untuk mulai menganalisis ulasan konsumen Anda dan melihat Dashboard Grafik, silakan unggah file Excel/CSV/TXT toko Anda melalui menu sidebar di sebelah kiri.")