import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. MEMBUAT TAMPILAN WEB (FRONTEND) DENGAN PYTHON
st.set_page_config(page_title="AI Analisis Toko", layout="centered")
st.title("🚀 Aplikasi AI Analisis Pelanggan")
st.write("Silakan unggah file Excel data toko Anda untuk melatih AI secara otomatis.")

# KODE UTAMA: Membuat Tombol Upload File di Halaman Web
file_terunggah = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"])

# 2. PROSES JIKA PENGGUNA SUDAH MENGUNGGAH FILE
if file_terunggah is not None:
    # Membaca file Excel yang diunggah pengguna langsung dari memori web
    df = pd.read_excel(file_terunggah, sheet_name='Data Pelanggan')
    
    # Menampilkan tabel data di halaman web
    st.success("File berhasil diunggah! Berikut adalah 5 data teratas Anda:")
    st.dataframe(df.head(5))
    
    # Menghitung baris otomatis untuk otak adaptif kita
    jumlah_baris = len(df)
    st.info(f"Sistem mendeteksi data latihan: {jumlah_baris} baris.")
    
    # --- PROSES OTAK AI ADAPTIF (Sama seperti skrip Anda sebelumnya) ---
    kata_tidak_penting = ['dan', 'di', 'saat', 'ini', 'itu', 'yang', 'saya', 'juga', 'pas', 'agak', 'sih']
    
    if jumlah_baris <= 50:
        vektor_nlp = TfidfVectorizer(max_features=10, stop_words=kata_tidak_penting)
        mode_hybrid = True
    else:
        vektor_nlp = TfidfVectorizer(max_features=200, stop_words=kata_tidak_penting)
        mode_hybrid = False
        
    # Preprocessing
    df['Gender'] = df['Gender'].map({'Pria': 0, 'Wanita': 1})
    df_tabular = pd.get_dummies(df[['Umur', 'Gender', 'Lokasi', 'Skor Buka App']], columns=['Lokasi'], dtype=int)
    X_tabular_angka = df_tabular.values 
    X_nlp_angka = vektor_nlp.fit_transform(df['Ulasan Pelanggan']).toarray()
    X_gabungan = np.hstack((X_tabular_angka, X_nlp_angka))
    y = df['Status Membeli'].values
    
    # Latih AI
    model_super = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_super.fit(X_gabungan, y)
    st.write("🧠 **Status Otak AI:** Sukses dilatih dan siap memprediksi!")
    
    # --- FORM INPUT UNTUK MEMPREDIKSI DATA BARU ---
    st.subheader("🔮 Uji Coba Prediksi Konsumen Baru")
    ulasan_input = st.text_input("Masukkan Kalimat Ulasan Konsumen Baru:", "Barangnya hancur pas sampai kecewa")
    
    if st.button("Analisis Sekarang"):
        # Dummy data untuk tabular baru (Umur 30, Wanita, Skor 8, Surabaya)
        data_tabular_baru = [30, 1, 8, 0, 0, 1]
        
        if mode_hybrid and any(kata in ulasan_input.lower() for kata in ['kecewa', 'rusak', 'hancur', 'jelek']):
            hasil = "Tidak Membeli (Negatif/Komplain) [Sistem Satpam]"
        else:
            data_nlp_baru = vektor_nlp.transform([ulasan_input]).toarray()[0]
            fitur_baru = np.hstack((data_tabular_baru, data_nlp_baru)).reshape(1, -1)
            tebakan = model_super.predict(fitur_baru)
            hasil = "Membeli" if tebakan[0] == 1 else "Masih Mikir-mikir" if tebakan[0] == 2 else "Tidak Membeli"
            
        # TAMPILAN KOTAK WARNA ADAPTIF (Sempurna untuk Mode Satpam maupun Mode AI)
        if "Negatif" in hasil or "Sistem" in hasil or "Tidak Membeli" in hasil:
            st.error(f"**Hasil Analisis:** {hasil}")
        elif "Mikir" in hasil:
            st.warning(f"**Hasil Analisis:** {hasil}")
        else:
            st.success(f"**Hasil Analisis:** {hasil}")