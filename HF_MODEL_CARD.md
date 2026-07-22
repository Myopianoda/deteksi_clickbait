# Deteksi Clickbait Bahasa Indonesia

Model klasifikasi judul berita berbahasa Indonesia menggunakan arsitektur
Hybrid IndoBERT–BiLSTM dengan attention pooling.

## Identitas checkpoint

| Komponen | Nilai |
|---|---|
| Arsitektur | Hybrid IndoBERT–BiLSTM |
| Varian pelatihan | `target_q0150` |
| Training seed | 2027 |
| Model dasar | `indolem/indobert-base-uncased` |
| Dataset | CLICK-ID `all_agree` |
| Masukan | Teks judul berita |
| Keluaran | Non-clickbait atau clickbait |

`target_q0150` bukan arsitektur model ketiga. Kode tersebut menunjukkan
varian pelatihan Hybrid dengan targeted weighted counterfactual punctuation
augmentation.

## Hasil test

| Metrik | Nilai |
|---|---:|
| Accuracy | 94,11% |
| Macro-F1 | 93,74% |
| Precision clickbait | 93,76% |
| Recall clickbait | 90,74% |
| F1 clickbait | 92,23% |
| F1 non-clickbait | 95,26% |

Confusion matrix:

```text
TN = 763
FP = 30
FN = 46
TP = 451
```

## Ketahanan tanda baca

| Pengujian | Nilai |
|---|---:|
| Flip rate setelah `!` | 3,95% |
| Flip rate setelah `?` | 8,75% |
| Mean flip rate `!/?` | 6,35% |
| Mean FPR setelah `!/?` | 10,40% |
| Flip rate penghapusan `!?` | 13,71% |

## Batasan

Model hanya menganalisis teks judul. Model tidak membaca isi artikel,
tidak menentukan kebenaran berita, tidak mendeteksi hoaks, dan tidak
menilai kredibilitas media.
