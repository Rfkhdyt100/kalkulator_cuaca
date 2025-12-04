import streamlit as st
import time
import joblib
import numpy as np
import pandas as pd

# =========================
# Loading Animation
# =========================
st.set_page_config(page_title="Kalkulator Cuaca", layout="centered")

with st.spinner("Memuat aplikasi..."):
    time.sleep(0.1)

success_msg = st.empty()
success_msg.markdown("""
<div class="fade-message">
    <p style="color: white; font-size:18px; padding:10px; background:#4BB543; border-radius:8px;">
        ✔ Aplikasi siap digunakan!
    </p>
</div>

<style>
@keyframes fadeOut {
    0%   { opacity: 1; }
    50%  { opacity: 0.6; }
    100% { opacity: 0; }
}

.fade-message {
    animation: fadeOut 2s ease-in-out forwards;
}
</style>
""", unsafe_allow_html=True)

st.title("🌤️ Kalkulator Prediksi Cuaca")

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

artifact = load_model()
model = artifact["model"]
le_target = artifact["le_target"]
feature_cols = artifact["feature_cols"]

# Load dataset untuk statistik
df = pd.read_csv("data/cuaca.csv")

# =========================
# Input
# =========================
suhu = st.number_input("Suhu (°C)", 0.0, 50.0, 30.0)
kelembaban = st.number_input("Kelembaban (%)", 0.0, 100.0, 70.0)
angin = st.number_input("Kecepatan Angin (km/jam)", 0.0, 50.0, 10.0)

# =========================
# Prediction
# =========================
if st.button("Prediksi Cuaca"):
    input_data = np.array([[suhu, kelembaban, angin]])
    pred = model.predict(input_data)[0]
    hasil = le_target.inverse_transform([pred])[0]

    st.success(f"🌦️ **Prediksi Cuaca: {hasil}**")

    # Detail input
    st.subheader("📊 Detail Data yang Anda Masukkan:")
    st.write(f"""
    • **Suhu:** {suhu} °C  
    • **Kelembaban:** {kelembaban} %  
    • **Kecepatan Angin:** {angin} km/jam  
    """)

    # Statistik
    with st.expander("📈 Lihat Statistik Cuaca"):
        st.subheader("Statistik Umum Dataset Cuaca")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata Suhu", f"{df['Suhu'].mean():.2f} °C")
        col2.metric("Rata-rata Kelembaban", f"{df['Kelembaban'].mean():.2f} %")
        col3.metric("Rata-rata Angin", f"{df['Angin'].mean():.2f} km/jam")

        st.write("### Distribusi Cuaca")
        st.bar_chart(df["Cuaca"].value_counts())

    # Penjelasan
    st.info("""
    **Bagaimana Kalkulator Ini Bekerja:**

    1. Masukkan suhu, kelembaban, dan kecepatan angin.  
    2. Data dibandingkan dengan banyak contoh dalam dataset.  
    3. Model memprediksi kondisi dengan pola yang paling mirip.

    **Sederhananya:** sistem memprediksi cuaca berdasarkan pola dari data sebelumnya.
    """)

