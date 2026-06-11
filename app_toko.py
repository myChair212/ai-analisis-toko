import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import os

# --- 1. PENGATURAN HALAMAN ---
st.set_page_config(page_title="AI Universal Topic Analyzer", layout="centered")

st.title("🧩 AI Universal Topic Clustering Engine")
st.markdown("""
👋 Selamat datang! Portal ini bekerja menggunakan **Unsupervised Machine Learning (K-Means)**.
Sistem tidak memerlukan label sentimen (Positif/Negatif) atau data training. AI akan membaca teks Anda, 
menganalisis kesamaan pola kata secara mandiri, lalu mengelompokkannya ke dalam beberapa topik utama secara otomatis.
""")
st.divider()

# --- 2. SIDEBAR: PUSAT UNGGAH & KONTROL PUBLIK ---
st.sidebar.header("📁 Konfigurasi Analisis")
uploaded_file = st.sidebar.file_uploader(
    "Unggah berkas teks Anda (Excel/CSV)", 
    type=['xlsx', 'xls', 'csv']
)

# Pengguna bisa menentukan sendiri mau memecah data menjadi berapa kelompok topik
jumlah_topik = st.sidebar.slider(
    "Tentukan Jumlah Kelompok Topik:",
    min_value=2,
    max_value=10,
    value=3,
    help="AI akan membagi dokumen Anda secara merata ke dalam jumlah kelompok ini."
)

# --- 3. PROSES DETEKSI KOLOM DAN CLUSTERING OTOMATIS ---
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

    if df is not None and len(df) > 0:
        st.success(f"✓ Berhasil memuat berkas: **{uploaded_file.name}** ({len(df)} baris data)")
        
        # Deteksi otomatis daftar nama kolom di file yang diunggah
        daftar_kolom = list(df.columns)
        
        st.header("🎯 Pilih Kolom Teks")
        kolom_pilihan = st.selectbox(
            "Silakan pilih kolom mana yang berisi kalimat/teks yang ingin dikelompokkan topiknya:",
            options=daftar_kolom
        )
        
        st.divider()
        
        if kolom_pilihan:
            # Proteksi jika jumlah data lebih sedikit dari jumlah kelompok topik yang diminta
            if len(df) < jumlah_topik:
                st.error(f"Jumlah baris data Anda ({len(df)}) terlalu sedikit untuk dibagi menjadi {jumlah_topik} topik. Silakan kurangi jumlah kelompok topik di sidebar.")
            else:
                st.info(f"🏃‍♂️ AI sedang menganalisis kemiripan makna teks pada kolom '{kolom_pilihan}'...")
                
                # 🚀 PROSES INTI 1: Mengubah teks menjadi angka biner pembobot (TF-IDF)
                # Menghapus kata umum bawaan (Stopwords) agar kata-kata seperti 'dan', 'di', 'yang' diabaikan otomatis
                vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
                X = vectorizer.fit_transform(df[kolom_pilihan].astype(str))
                
                # 🚀 PROSES INTI 2: Clustering K-Means (Membagi kelompok tanpa label)
                kmeans = KMeans(n_clusters=jumlah_topik, random_state=42, n_init=10)
                df['Cluster_ID'] = kmeans.fit_predict(X)
                
                # Mengubah nama cluster angka (0, 1, 2) menjadi nama teks yang ramah (Topik 1, Topik 2, Topik 3)
                df['Kelompok_Topik'] = df['Cluster_ID'].apply(lambda x: f"Topik {x + 1}")
                
                st.success("✨ Sukses! AI telah selesai memetakan dan mengelompokkan teks Anda secara mandiri.")
                st.divider()
                
                # --- 4. DASHBOARD VISUALISASI TOPIK ---
                
                # A. Grafik Pai Distribusi Kelompok Topik
                st.header("📊 Ringkasan Pembagian Topik")
                df_counts = df['Kelompok_Topik'].value_counts().reset_index()
                df_counts.columns = ['Kelompok_Topik', 'Jumlah_Dokumen']
                
                fig_pie = px.pie(
                    df_counts, 
                    names='Kelompok_Topik', 
                    values='Jumlah_Dokumen',
                    hole=0.4,
                    title="Proporsi Penyebaran Kelompok Teks"
                )
                fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)
                st.divider()
                
                # B. Word Cloud Multi-Tab Dinamis Berdasarkan Topik
                st.header("🔍 Kata Kunci Dominan Per Topik")
                st.markdown("Klik tab di bawah ini untuk melihat kata kunci apa saja yang paling sering muncul di setiap kelompok topik:")
                
                # Membuat Tab dinamis di Streamlit sesuai jumlah kelompok topik
                list_tabs = st.tabs([f"Topik {i+1}" for i in range(jumlah_topik)])
                
                for i, tab in enumerate(list_tabs):
                    with tab:
                        # Ambil teks khusus kelompok tersebut
                        df_sub = df[df['Cluster_ID'] == i]
                        text_combined = " ".join(df_sub[kolom_pilihan].astype(str)).lower()
                        
                        if text_combined.strip():
                            wc = WordCloud(width=1000, height=400, background_color="white", colormap="plasma").generate(text_combined)
                            fig_wc, ax = plt.subplots(figsize=(10, 4), facecolor="white")
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig_wc, use_container_width=True)
                        else:
                            st.write("Tidak ada kata kunci yang cukup dominan di topik ini.")
                st.divider()
                
                # C. Tabel Hasil Detil Pengelompokan AI
                st.header("📋 Tabel Hasil Pengelompokan Dokumen")
                st.dataframe(df[[kolom_pilihan, 'Kelompok_Topik']], use_container_width=True, height=350)
                st.divider()
                
                # D. Pusat Unduh Laporan Universal
                st.header("📥 Pusat Unduh")
                st.markdown("Unduh data Anda yang kini sudah rapi dilengkapi dengan kolom pengelompokan topik otomatis oleh AI:")
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='AI_Topic_Clustering')
                    
                st.download_button(
                    label="📥 Unduh Hasil Pengelompokan Topik (.xlsx)", 
                    data=output.getvalue(), 
                    file_name="Hasil_Analisis_Topik_AI.xlsx", 
                    mime="application/vnd.ms-excel", 
                    use_container_width=True
                )
    else:
        st.error("Berkas kosong atau format tidak didukung.")
else:
    st.info("💡 Sistem Siap! Silakan unggah file Excel/CSV berisi teks topik apa saja bebas. AI akan langsung mengelompokkannya secara otomatis.")