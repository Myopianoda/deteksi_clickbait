# Checklist Deployment

## A. Hugging Face

Repository:

```text
Myopianoda/deteksi_clickbait
```

Unggah dua file dari Google Drive:

```text
MyDrive/Skripsi_Clickbait/
└── phase_b_confirmatory_v1/
    └── seed_2027/
        └── target_q0150/
            ├── best_model.pth
            └── run_config.json
```

Ubah nama saat diunggah:

```text
best_model.pth
→ best_model_target_q0150_seed2027.pth

run_config.json
→ run_config_target_q0150_seed2027.json
```

Salin isi `HF_MODEL_CARD.md` ke `README.md` repository Hugging Face.

## B. GitHub

Ganti atau tambahkan:

```text
apps.py
model.py
lime_explainer.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml
```

Folder tokenizer lama tidak lagi digunakan dan boleh dihapus:

```text
tokenizer/
```

## C. Streamlit Community Cloud

1. Pastikan entry point tetap `apps.py`.
2. Commit semua perubahan GitHub.
3. Buka pengaturan aplikasi Streamlit.
4. Reboot aplikasi.
5. Jika model lama masih muncul, clear cache lalu reboot lagi.
6. Muat halaman pertama kali dan tunggu checkpoint selesai diunduh.
