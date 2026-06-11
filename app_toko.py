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

st.title("🚀 Platform AI Analisis Konsumen (Advanced Pure Analytics)")
st.markdown("""
👋 Selamat datang! Sistem ini dikonfigurasi untuk menganalisis data ulasan konsumen yang **polosan (tanpa label)**.
Otak AI otomatis di-generate di latar belakang menggunakan basis data master.
""")
st.divider()

# --- 2. SIDEBAR: PUSAT UNGGAH DATA POLOS ---
st.sidebar.header("📁 Unggah Berkas Anda")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file Excel/CSV ulasan kosong Anda", 
    type=['xlsx', 'xls', 'csv']
)

# --- 3. INIDIALISASI OTAK AI OTOMATIS (Mencegah NameError) ---
@st.cache_resource
def buat_otak_ai_otomatis():
    # Sistem mencoba membaca file master dari folder proyek Anda
    nama_file_master = 'basis_data_pintar.xlsx'
    
    if os.path.exists(nama_file_master):
        df_master = pd.read_excel(nama_file_master)
    else:
        # Cadangan otomatis jika file excel master belum Anda upload ke GitHub
        data_cadangan = {
            'Ulasan': [
                'bagus banget suka pas mantap', 'rusak hancur kecewa jelek buruk', 
                'cepat ramah kurir baik', 'lambat telat parah kapok', 
                'oke produk original sesuai', 'menyesal salah warna robek',
                'barang rusak tapi admin bersedia mengganti untung bertanggung jawab',
                'salah warna tetapi respon cepat dan dikirim ulang makasih seller',
                'ada kendala produk tapi diganti baru pelayanan luar biasa'
            ],
            'Label': ['Membeli', 'Tidak Membeli', 'Membeli', 'Tidak Membeli', 'Membeli', 'Tidak Membeli', 'Membeli', 'Membeli', 'Membeli']
        }
        df_master = pd.DataFrame(data_cadangan)
        
    vectorizer_master = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
    X_train = vectorizer_master.fit_transform(df_master['Ulasan'].astype(str))
    
    model_master = LogisticRegression(class_weight='balanced')
    model_master.fit(X_train, df_master['Label'].astype(str))
    
    return vectorizer_master, model_master

# Mengaktifkan variabel vectorizer dan model secara global agar baris di bawahnya tidak eror
vectorizer, model = buat_otak_ai_otomatis()

# --- 4. PEMROSESAN & EKSEKUSI PREDIKSI MANDIRI ---
if uploaded_file:
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
        st.success(f"✓ Berhasil memuat: **{uploaded_file.name}** ({jumlah_baris} baris ulasan terdeteksi)")
        
        # Deteksi otomatis letak kolom teks ulasan
        kolom_ulasan = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'ulasan' in col_lower or 'text' in col_lower or 'komentar' in col_lower or 'review' in col_lower:
                kolom_ulasan = col
                break
        
        if kolom_ulasan is None and len(df.columns) > 0:
            kolom_ulasan = df.columns[0]
            df = df.rename(columns={kolom_ulasan: 'Ulasan'})
            kolom_ulasan = 'Ulasan'
            
        # PROSES INTI: Menggunakan vectorizer yang sudah dijamin terdefinisi di atas
        if kolom_ulasan:
            st.info("🧠 AI sedang menyisir teks ulasan dan melakukan prediksi sentimen mandiri...")
            
            X_new = vectorizer.transform(df[kolom_ulasan].astype(str))
            df['Hasil_Analisis_AI'] = model.predict(X_new)
            
            df['Status_Teks'] = df['Hasil_Analisis_AI'].apply(
                lambda x: "Membeli (Positif)" if str(x).lower() == 'membeli' else "Tidak Membeli (Negatif)"
            )
            
            # --- TAMPILAN DASHBOARD ---
            st.header("📊 Ringkasan Distribusi Sentimen")
            fig_pie = px.pie(df, names='Status_Teks', hole=0.4, color='Status_Teks', color_discrete_map={'Membeli (Positif)': '#10b981', 'Tidak Membeli (Negatif)': '#ef4444'})
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.divider() 
            
            st.header("🔍 Kata Kunci Terpopuler")
            text_combined = " ".join(df[kolom_ulasan].astype(str)).lower()
            if text_combined.strip():
                wc = WordCloud(width=1000, height=450, background_color="white", colormap="viridis").generate(text_combined)
                fig_wc, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc, use_container_width=True)
            st.divider()

            st.header("📋 Hasil Detil Analisis Per Baris")
            st.dataframe(df[[kolom_ulasan, 'Status_Teks']], use_container_width=True, height=300)
            st.divider()

            st.header("📥 Pusat Unduh")
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df[[kolom_ulasan, 'Status_Teks']].to_excel(writer, index=False, sheet_name='Hasil_Analisis_AI')
            st.download_button(label="📥 Unduh Hasil Analisis Lengkap (.xlsx)", data=output.getvalue(), file_name="Laporan_Analisis_Sentimen_AI.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
            st.divider()

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
    st.info("👋 Sistem Pure Analytics Siap! Silakan unggah file Excel ulasan polosan Anda di sidebar.")