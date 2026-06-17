import json
import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download  # Tambahan untuk Hugging Face

# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="Clickbait Detector",
    layout="centered"
)

# ============================================================
# Definisi model — HANYA HYBRID
# ============================================================
class HybridIndoBERTBiLSTM(nn.Module):
    def __init__(self, model_name, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(model_name)
        self.lstm      = nn.LSTM(768, hidden_dim, num_layers=2,
                                 bidirectional=True, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        seq_out     = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(seq_out)
        attn_scores = self.attention(lstm_out)
        attn_scores = attn_scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0, -1e9
        )
        attn_w  = torch.softmax(attn_scores, dim=1)
        context = (attn_w * lstm_out).sum(dim=1)
        return self.fc(self.dropout(context))


# ============================================================
# Load model & tokenizer (Integrasi Hugging Face)
# ============================================================
@st.cache_resource(show_spinner="Mengunduh model dari Hugging Face (hanya saat pertama kali)...")
def load_assets():
    # ---------------------------------------------------------
    # UBAH BAGIAN INI DENGAN REPOSITORI HUGGING FACE ANDA
    REPO_ID = "Myopianoda/deteksi_clickbait" 
    # ---------------------------------------------------------
    
    device = torch.device("cpu")
    
    # 1. Unduh file model_config.json dari Hugging Face
    config_path = hf_hub_download(repo_id=REPO_ID, filename="model_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # 2. Load tokenizer (Jika folder 'tokenizer' tetap ditaruh di GitHub, biarkan ini)
    # Jika file tokenizer juga diunggah ke Hugging Face, ubah menjadi: AutoTokenizer.from_pretrained(REPO_ID)
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    
    # 3. Inisialisasi Arsitektur
    hybrid = HybridIndoBERTBiLSTM(cfg["model_name"], hidden_dim=128)
    
    # 4. Unduh file best_model.pth dari Hugging Face
    model_path = hf_hub_download(repo_id=REPO_ID, filename="best_model.pth")
    
    # 5. Muat bobot ke dalam model
    hybrid.load_state_dict(torch.load(model_path, map_location=device))
    hybrid.eval()

    return tokenizer, hybrid, cfg, device

tokenizer, hybrid_model, cfg, device = load_assets()

# Inisialisasi LIME Explainer
explainer = LimeTextExplainer(class_names=["Non-Clickbait", "Clickbait"])

# ============================================================
# Fungsi prediksi untuk UI utama
# ============================================================
def predict(text, model):
    enc = tokenizer(
        text,
        add_special_tokens=True,
        max_length=cfg["max_len"],
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device),
                       enc["attention_mask"].to(device))
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()
        
    label = "Clickbait" if probs[1] > probs[0] else "Non-Clickbait"
    return label, probs[0], probs[1]

# ============================================================
# Fungsi Wrapper khusus untuk LIME
# ============================================================
def predictor_for_lime(texts):
    # LIME akan mengirimkan list of strings (teks yang sudah diacak/di-mask)
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=cfg["max_len"],
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = hybrid_model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs = torch.softmax(logits, dim=1)
    
    # LIME wajib menerima output dalam bentuk numpy array 2D
    return probs.cpu().numpy()


# ============================================================
# UI Streamlit & Navigasi
# ============================================================
menu = st.sidebar.selectbox("Menu Navigasi", ["Beranda", "Tentang Sistem"])

if menu == "Beranda":
    st.title("Clickbait Headline Detector")
    st.caption("Deteksi judul berita clickbait menggunakan IndoBERT-BiLSTM & LIME XAI")
    st.divider()

    st.markdown("#### Masukkan judul berita")
    user_input = st.text_area("", placeholder="Contoh: Viral! Bocah Ini Bikin Semua Orang Tercengang...", height=100, label_visibility="collapsed")

    if st.button("Deteksi & Analisis", use_container_width=True, type="primary"):
        if not user_input.strip():
            st.warning("Masukkan judul berita terlebih dahulu.")
        else:
            with st.spinner("Menganalisis kalimat..."):
                label, prob_ncb, prob_cb = predict(user_input.strip(), hybrid_model)

            st.divider()
            st.markdown("### Hasil Prediksi")

            col1, col2 = st.columns(2)
            if label == "Clickbait":
                col1.error(f"Kategori: **{label}**")
            else:
                col1.success(f"Kategori: **{label}**")
                
            col2.metric("Tingkat Keyakinan (Probabilitas)", f"{max(prob_cb, prob_ncb)*100:.1f}%")
            
            st.progress(prob_cb, text="Kecenderungan Clickbait")
            
            # --- BAGIAN LIME XAI ---
            st.markdown("---")
            st.markdown("### Analisis Explainable AI (LIME)")
            st.caption("Visualisasi pengaruh kata terhadap hasil prediksi model.")
            
            with st.spinner("Membangun visualisasi analisis kata..."):
                exp = explainer.explain_instance(
                    user_input.strip(), 
                    predictor_for_lime, 
                    num_features=10, 
                    num_samples=250
                )
                
                # 1. Ekstrak data mentah dari LIME
                lime_data = exp.as_list()
                words = [item[0] for item in lime_data]
                weights = [item[1] for item in lime_data]
                
                # 2. Pembuatan Visualisasi Formal dengan Matplotlib
                fig, ax = plt.subplots(figsize=(8, 4.5))
                
                # Warna: Merah untuk Clickbait (>0), Hijau untuk Non-Clickbait (<0)
                colors = ['#D32F2F' if w > 0 else '#388E3C' for w in weights]
                
                # Membuat grafik batang horizontal yang lebih ramping dan modern (tidak kaku)
                bars = ax.barh(words, weights, color=colors, height=0.4, alpha=0.85)
                ax.invert_yaxis()  # Mengurutkan kata dengan pengaruh terbesar di posisi paling atas
                
                # 3. Kustomisasi Gaya Akademik (Bersih dan Dinamis)
                ax.set_xlabel("Bobot Pengaruh (LIME Weight)", fontsize=10, color="#424242")
                ax.set_title("Kontribusi Kata terhadap Prediksi Model", fontsize=12, fontweight='bold', pad=15, color="#212121")
                
                # Menambahkan gridline vertikal transparan untuk mempermudah pembacaan nilai
                ax.grid(axis='x', linestyle='-', alpha=0.15, color="#000000")
                
                # Menghapus garis tepi agar grafik terlihat lebih mengalir dan tidak berbentuk kotak kaku
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_color('#BDBDBD')
                
                # Menambahkan garis vertikal di titik 0 sebagai pemisah yang elegan
                ax.axvline(x=0, color='#616161', linewidth=1)
                
                # 4. Tampilkan di Streamlit
                st.pyplot(fig)
                
                # 5. Keterangan Grafis
                st.markdown("""
                **Keterangan Visualisasi:**
                * <span style='color:#D32F2F; font-weight:bold;'>■ Merah:</span> Kata yang mendorong model memprediksi teks sebagai **Clickbait**.
                * <span style='color:#388E3C; font-weight:bold;'>■ Hijau:</span> Kata yang menahan model memprediksi teks sebagai **Non-Clickbait**.
                * *Semakin panjang garis batang, semakin kuat pengaruh kata tersebut terhadap keputusan akhir model.*
                """, unsafe_allow_html=True)

    st.divider()
    st.caption("Model dilatih menggunakan dataset CLICK-ID · Skripsi 2026")

elif menu == "Tentang Sistem":
    st.title("Tentang Sistem")
    st.markdown("Sistem ini dikembangkan untuk mengklasifikasikan judul artikel berita ke dalam kategori **Clickbait** atau **Non-Clickbait**.")
    
    st.markdown("### 1. Arsitektur Hybrid IndoBERT-BiLSTM")
    st.write("Sistem ini menggabungkan dua algoritma utama:")
    st.write("- **IndoBERT (Bidirectional Encoder Representations from Transformers):** Model pra-latih yang bertugas mengekstrak fitur dan memahami konteks semantik bahasa Indonesia.")
    st.write("- **BiLSTM (Bidirectional Long Short-Term Memory):** Memproses fitur dari IndoBERT secara sekuensial dari dua arah untuk menangkap dependensi panjang dalam sebuah kalimat.")

    st.markdown("### 2. Explainable AI (LIME)")
    st.write("Sistem diintegrasikan dengan modul **LIME** (Local Interpretable Model-agnostic Explanations) untuk membedah *black box* pada model Deep Learning. LIME memberikan transparansi dengan menunjukkan seberapa besar kontribusi (bobot) setiap kata terhadap keputusan akhir yang diambil oleh algoritma.")
    
    st.divider()
    st.markdown("#### Informasi Pengembang")
    st.markdown("**Penulis:** Rizky Syahrul Maulid")
    st.markdown("**Program Studi:** Informatika")
    st.markdown("**Institusi:** STT Wastukancana Purwakarta")