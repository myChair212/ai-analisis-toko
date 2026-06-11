import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# --- 1. PENGATURAN HALAMAN & GAYA VISUAL ---
st.set_page_config(page_title="AI Analisis Toko Pro", layout="centered")

# Menggunakan CSS bawaan untuk mempercantik font dan spasi card
st.markdown("""
    <style>
    .reportview-container { background: #0f172a; }
    h1, h2, h3 { font-family: 'Helvetica Neue', Arial, sans-serif; letter-spacing: -0.5px; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #00c6ff, #0072ff); }
    </style>
""", unsafe_allowed_html=True)

st.title("🚀 Platform AI Analisis Konsumen")
st.markdown("---")

# --- 2. DATA SETTINGAN BAWAAN (Modal Otak AI) ---
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

# --- 3. SIDEBAR: PUSAT UNGGAH DATA ---
st.sidebar.header("📁 Pusat Data")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file ulasan Anda", 
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

# --- 4. PEMROSESAN DATA & TAMPILAN LINEAR ---
if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"✓ Berhasil memuat data: {uploaded_file.name}")
        
        # Deteksi otomatis kolom ulasan
        kolom_ulasan = None
        for col in df.columns:
            if 'ulasan' in str(col).lower() or 'text' in str(col).lower() or 'komentar' in str(col).lower():
                kolom_ulasan = col
                break
        
        if kolom_ulasan is None and len(df.columns) > 0:
            kolom_ulasan = df.columns[0]
            df = df.rename(columns={kolom_ulasan: 'Ulasan'})
            kolom_ulasan = 'Ulasan'

        if kolom_ulasan:
            # Jalankan prediksi masal AI
            X_new = vectorizer_base.transform(df[kolom_ulasan].astype(str))
            df['Prediksi_AI'] = model_base.predict(X_new)
            
            # ---------------------------------------------------------
            # BAGIAN A: GRAFIK PAI (DISTRIBUSI SENTIMEN)
            # ---------------------------------------------------------
            st.header("📊 Ringkasan Distribusi Sentimen")
            
            # Modifikasi warna chart agar elegan (Teal untuk Membeli, Coral-Red untuk Tidak Membeli)
            warna_custom = {'Membeli': '#10b981', 'Tidak Membeli': '#ef4444'}
            fig_pie = px.pie(
                df, 
                names='Prediksi_AI', 
                hole=0.4, 
                color='Prediksi_AI',
                color_discrete_map=warna_custom
            )
            fig_pie.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider() # Garis pembatas elegan
            
            # ---------------------------------------------------------
            # BAGIAN B: WORD CLOUD PREMIUM (TERLINDUNG DI LATAR BELAKANG GELAP)
            # ---------------------------------------------------------
            st.header("🔍 Kata Kunci yang Sering Muncul")
            
            text_combined = " ".join(df[kolom_ulasan].astype(str)).lower()
            if text_combined.strip():
                # Settingan Word Cloud Mewah
                wc = WordCloud(
                    width=1000, height=450,
                    background_color="#1e293b", # Warna gelap charcoal murni
                    colormap="GnBu",            # Gradasi warna Hijau-Teal-Biru yang sejuk
                    prefer_horizontal=0.85,     # Dominan horizontal agar mudah dibaca
                    max_words=100,              # Batasi agar tidak terlalu penuh sesak
                    contour_width=1,
                    contour_color="#334155"
                ).generate(text_combined)
                
                fig_wc, ax = plt.subplots(figsize=(10, 4.5), facecolor="#1e293b")
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                
                # Render ke layar secara penuh
                st.pyplot(fig_wc, use_container_width=True)
            else:
                st.text("Teks tidak cukup untuk membuat analisis kata kunci.")
                
            st.divider()

            # ---------------------------------------------------------
            # BAGIAN C: DATA DETIL TABEL
            # ---------------------------------------------------------
            st.header("📋 Hasil Detil Analisis Per Baris")
            st.markdown("Menampilkan ulasan asli beserta tebakan otomatis dari sistem AI:")
            st.dataframe(df[[kolom_ulasan, 'Prediksi_AI']], use_container_width=True, height=350)
            
            st.divider()

            # ---------------------------------------------------------
            # BAGIAN D: TOMBOL UNDUH LAPORAN
            # ---------------------------------------------------------
            st.header("📥 Pusat Unduh")
            st.markdown("Unduh hasil analisis di atas ke dalam format Excel untuk bahan laporan internal:")
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Analisis_AI_Toko')
            
            st.download_button(
                label="📥 Unduh Laporan Excel Lengkap (.xlsx)",
                data=output.getvalue(),
                file_name="Laporan_Analisis_Masal_AI.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True # Tombol memanjang penuh agar terlihat kokoh
            )
            
            st.divider()

            # ---------------------------------------------------------
            # BAGIAN E: UJI COBA REAL-TIME (Paling Bawah sebagai Bonus)
            # ---------------------------------------------------------
            st.header("🧠 Kotak Uji Coba Cepat")
            input_text = st.text_input("Ketik ulasan sembarang di bawah ini untuk melihat reaksi kilat AI:")
            if input_text:
                vec_input = vectorizer_base.transform([input_text])
                pred = model_base.predict(vec_input)[0]
                warna = "green" if pred == "Membeli" else "red"
                st.markdown(f"### Hasil Analisis: :{warna}[{pred}]")
                
        else:
            st.error("File tidak memiliki kolom teks yang bisa dibaca.")
else:
    # Tampilan awal saat baru membuka web (Elegan & Bersih)
    st.info("👋 Selamat Datang! Silakan unggah file Excel/CSV data toko Anda melalui menu sidebar di sebelah kiri untuk memulai analisis otomatis.")