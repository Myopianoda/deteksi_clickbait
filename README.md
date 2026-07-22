# Deteksi Clickbait Bahasa Indonesia

Aplikasi Streamlit untuk klasifikasi judul berita menggunakan:

```text
IndoBERT
→ BiLSTM dua arah
→ attention pooling
→ classifier
```

Checkpoint deployment:

```text
Varian        : target_q0150
Training seed : 2027
```

## Struktur repository

```text
deteksi_clickbait/
├── apps.py
├── model.py
├── lime_explainer.py
├── requirements.txt
├── README.md
├── DEPLOYMENT_CHECKLIST.md
├── HF_MODEL_CARD.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## File Hugging Face yang wajib tersedia

Repository:

```text
Myopianoda/deteksi_clickbait
```

File:

```text
best_model_target_q0150_seed2027.pth
run_config_target_q0150_seed2027.json
```

## Menjalankan secara lokal

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m streamlit run apps.py
```

## LIME

Aplikasi menyediakan:

```text
Prediksi cepat
Prediksi + LIME
```

Nilai bawaan LIME aplikasi adalah 500 sampel agar lebih responsif.
Analisis final skripsi menggunakan 5.000 sampel dan tidak dijalankan
secara otomatis pada setiap prediksi aplikasi.
