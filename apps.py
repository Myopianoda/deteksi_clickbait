from __future__ import annotations

import re as _re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from lime_explainer import build_lime_figure, explain_prediction
from model import load_runtime, predict

HF_REPO_ID = "Myopianoda/deteksi_clickbait"
HF_CHECKPOINT_FILENAME = "best_model_target_q0150_seed2027.pth"
HF_CONFIG_FILENAME = "run_config_target_q0150_seed2027.json"

APP_TITLE = "Deteksi Clickbait Bahasa Indonesia"
MODEL_CAPTION = (
    "Hybrid IndoBERT–BiLSTM · target_q0150 · training seed 2027"
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📰",
    layout="centered",
)


@st.cache_resource(show_spinner="Memuat model final...")
def get_runtime():
    return load_runtime(
        repo_id=HF_REPO_ID,
        checkpoint_filename=HF_CHECKPOINT_FILENAME,
        config_filename=HF_CONFIG_FILENAME,
    )


def show_prediction(result, positive_class_id):
    predicted_clickbait = (
        result["label_id"] == positive_class_id
    )
    label = (
        "CLICKBAIT"
        if predicted_clickbait
        else "NON-CLICKBAIT"
    )

    if predicted_clickbait:
        st.error(f"{label} · {result['confidence']:.2%}")
    else:
        st.success(f"{label} · {result['confidence']:.2%}")

    left, middle, right = st.columns(3)
    left.metric("Prediksi", label)
    middle.metric(
        "Skor clickbait",
        f"{result['clickbait_score']:.2%}",
    )
    right.metric(
        "Skor non-clickbait",
        f"{result['non_clickbait_score']:.2%}",
    )

    st.progress(
        float(result["clickbait_score"]),
        text="Kecenderungan model terhadap kelas clickbait",
    )


def show_lime(lime_result):
    predicted_class_name = lime_result["predicted_class_name"]

    st.subheader("Penjelasan lokal LIME")
    st.caption(
        "Bobot positif mendukung kelas yang diprediksi. "
        "Bobot negatif menahan kelas yang diprediksi."
    )

    figure = build_lime_figure(
        features=lime_result["features"],
        predicted_class_name=predicted_class_name,
    )
    st.pyplot(
        figure,
        use_container_width=True,
    )
    plt.close(figure)

    table = pd.DataFrame(
        lime_result["features"]
    ).rename(
        columns={
            "feature": "Fitur",
            "weight": "Bobot LIME",
            "direction": "Arah kontribusi",
        }
    )

    table = table.sort_values(
        "Bobot LIME",
        key=lambda series: series.abs(),
        ascending=False,
    )

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Bobot LIME": st.column_config.NumberColumn(
                format="%.6f"
            )
        },
    )

    st.caption(
        f"Kelas yang dijelaskan: {predicted_class_name} | "
        f"Sampel variasi: {lime_result['num_samples']:,} | "
        f"Fidelitas lokal R²: "
        f"{lime_result['local_fidelity']:.4f}"
    )

    if lime_result["local_fidelity"] < 0.5:
        st.info(
            "Fidelitas lokal relatif rendah. Interpretasi fitur "
            "perlu dibaca sebagai pendekatan lokal, bukan penyebab "
            "pasti keputusan model."
        )


def _format_indonesian_list(items):
    """Menggabungkan daftar menjadi frasa bahasa Indonesia yang alami."""
    clean_items = [
        str(item).strip()
        for item in items
        if str(item).strip()
    ]

    if not clean_items:
        return ""

    if len(clean_items) == 1:
        return clean_items[0]

    if len(clean_items) == 2:
        return f"{clean_items[0]} dan {clean_items[1]}"

    return ", ".join(clean_items[:-1]) + f", dan {clean_items[-1]}"


def _find_language_indicators(text):
    """
    Mendeteksi indikasi gaya bahasa dengan aturan sederhana.

    Hasilnya hanya penjelasan tambahan, bukan kelas prediksi model.
    """
    original_text = str(text).strip()
    lowered_text = original_text.lower()

    indicator_rules = {
        "sensasional": [
            r"\bheboh\b",
            r"\bviral\b",
            r"\bgempar\b",
            r"\bmenggemparkan\b",
            r"\bmengejutkan\b",
            r"\bmencengangkan\b",
            r"\btercengang\b",
            r"\bluar biasa\b",
            r"\bspektakuler\b",
            r"\bfantastis\b",
            r"\bbikin geger\b",
            r"\bbikin heboh\b",
            r"\bbikin tercengang\b",
        ],
        "provokatif": [
            r"\bgeram\b",
            r"\bmarah\b",
            r"\bmemalukan\b",
            r"\bskandal\b",
            r"\bbongkar\b",
            r"\bmembongkar\b",
            r"\bkecam\b",
            r"\bmengecam\b",
            r"\bserang\b",
            r"\bmenyerang\b",
            r"\bboikot\b",
            r"\bancam\b",
            r"\bmengancam\b",
            r"\bpengkhianat\b",
            r"\bbikin panas\b",
            r"\bbikin emosi\b",
        ],
        "membangun rasa penasaran": [
            r"\bternyata\b",
            r"\bini alasannya\b",
            r"\binilah alasannya\b",
            r"\brahasia\b",
            r"\bsiapa sangka\b",
            r"\btak disangka\b",
            r"\btidak disangka\b",
            r"\bkamu tidak akan percaya\b",
            r"\banda tidak akan percaya\b",
            r"\bapa yang terjadi\b",
            r"\bbegini caranya\b",
            r"\bbegini faktanya\b",
            r"\bterungkap\b",
            r"\bbikin penasaran\b",
            r"\bnomor\s+\d+\b",
        ],
        "urgensi atau ajakan kuat": [
            r"\bsegera\b",
            r"\bjangan lewatkan\b",
            r"\bsebelum terlambat\b",
            r"\bwajib tahu\b",
            r"\bharus tahu\b",
            r"\bbaca sekarang\b",
            r"\bcek sekarang\b",
        ],
    }

    categories = []

    for category, patterns in indicator_rules.items():
        if any(
            _re.search(pattern, lowered_text, flags=_re.IGNORECASE)
            for pattern in patterns
        ):
            categories.append(category)

    has_repeated_punctuation = bool(
        _re.search(r"!{2,}|\?{2,}|!\?|\?!|\.{3,}", original_text)
    )
    has_exclamation = "!" in original_text
    has_uppercase_emphasis = any(
        word not in {"COVID", "PRESIDEN", "INDONESIA"}
        for word in _re.findall(r"\b[A-Z]{4,}\b", original_text)
    )

    if (
        has_repeated_punctuation
        or has_exclamation
        or has_uppercase_emphasis
    ):
        categories.append("penekanan berlebihan")

    return list(dict.fromkeys(categories))


def _get_supporting_lime_features(lime_result, limit=3):
    """Mengambil fitur LIME terkuat yang mendukung kelas prediksi."""
    if not lime_result:
        return []

    supporting_features = [
        feature
        for feature in lime_result.get("features", [])
        if float(feature.get("weight", 0.0)) > 0
        and str(feature.get("feature", "")).strip()
    ]

    supporting_features.sort(
        key=lambda feature: abs(
            float(feature.get("weight", 0.0))
        ),
        reverse=True,
    )

    return supporting_features[: int(limit)]


def show_final_explanation(
    text,
    prediction_result,
    positive_class_id,
    lime_result=None,
):
    """Menampilkan ringkasan singkat tepat setelah hasil analisis."""
    predicted_clickbait = (
        int(prediction_result["label_id"])
        == int(positive_class_id)
    )

    label = (
        "CLICKBAIT"
        if predicted_clickbait
        else "NON-CLICKBAIT"
    )

    relevant_score = (
        float(prediction_result["clickbait_score"])
        if predicted_clickbait
        else float(prediction_result["non_clickbait_score"])
    )

    categories = _find_language_indicators(text)
    supporting_features = _get_supporting_lime_features(
        lime_result=lime_result,
        limit=3,
    )

    st.subheader("Kesimpulan")

    with st.container(border=True):
        st.markdown(
            f"**{label}** · tingkat keyakinan **{relevant_score:.2%}**"
        )

        if categories:
            st.markdown(
                "**Indikasi gaya bahasa:** "
                + _format_indonesian_list(categories)
                + "."
            )
        else:
            st.markdown(
                "**Indikasi gaya bahasa:** tidak ada pola dominan "
                "yang terdeteksi oleh aturan sederhana."
            )

        if supporting_features:
            feature_names = _format_indonesian_list(
                [
                    f"“{feature['feature']}”"
                    for feature in supporting_features
                ]
            )
            st.markdown(
                f"**Fitur LIME yang paling mendukung:** {feature_names}."
            )
        elif lime_result is None:
            st.caption(
                "Gunakan Prediksi + LIME untuk melihat fitur pendukung."
            )

        st.caption(
            "Indikasi gaya bahasa merupakan penjelasan berbasis aturan, "
            "bukan kelas tambahan dari model."
        )


def _open_detector():
    """Memindahkan navigasi dari halaman pengantar ke halaman prediksi."""
    st.session_state["nav_menu"] = "Coba Sistem"


menu = st.sidebar.radio(
    "Menu navigasi",
    ["Tentang Clickbait", "Coba Sistem"],
    key="nav_menu",
)


if menu == "Tentang Clickbait":
    st.title("Mengenal Clickbait")
    st.caption(
        "Pahami pengertian dan ciri-cirinya sebelum mencoba sistem deteksi."
    )

    st.markdown(
        """
**Clickbait** adalah judul yang dibuat untuk menarik perhatian dan mendorong
orang membuka sebuah berita atau konten. Judul seperti ini biasanya memancing
rasa penasaran, emosi, atau rasa mendesak, tetapi tidak langsung menyampaikan
informasi utama secara jelas.

Clickbait **tidak selalu berarti hoaks**. Sebuah judul dapat bersifat clickbait
meskipun isi beritanya benar. Perbedaannya terletak pada cara judul tersebut
menarik perhatian pembaca.
"""
    )

    st.subheader("Ciri-ciri yang sering ditemukan")
    st.markdown(
        """
- **Menahan informasi penting**, misalnya memakai kata “ternyata” atau
  “ini alasannya” tanpa menjelaskan jawabannya.
- **Menggunakan bahasa emosional atau sensasional**, seperti “heboh”,
  “mengejutkan”, atau “bikin geger”.
- **Menciptakan rasa mendesak**, misalnya “wajib tahu”, “baca sekarang”,
  atau “sebelum terlambat”.
- **Menggunakan penekanan berlebihan**, seperti huruf kapital dan tanda baca
  berulang: `!!!`, `???`, atau `...`.
"""
    )

    st.subheader("Contoh sederhana")
    clickbait_column, informative_column = st.columns(2)

    with clickbait_column:
        with st.container(border=True):
            st.markdown("**Cenderung clickbait**")
            st.markdown(
                "“Viral! Anda Tidak Akan Percaya Apa yang Terjadi Setelah Ini...”"
            )
            st.caption(
                "Informasi utama ditahan dan judul memakai kata emosional "
                "untuk memancing rasa penasaran."
            )

    with informative_column:
        with st.container(border=True):
            st.markdown("**Cenderung non-clickbait**")
            st.markdown(
                "“BMKG Memprakirakan Hujan Lebat di Jakarta pada Jumat Sore”"
            )
            st.caption(
                "Judul menyampaikan informasi utama secara langsung dan spesifik."
            )

    st.info(
        "Satu ciri saja belum tentu membuat sebuah judul menjadi clickbait. "
        "Penilaian perlu melihat keseluruhan pola bahasa pada judul."
    )

    st.subheader("Apa yang dilakukan sistem ini?")
    st.markdown(
        """
1. Pengguna memasukkan satu judul berita berbahasa Indonesia.
2. Model menganalisis pola kata dan tanda baca pada judul tersebut.
3. Sistem menampilkan prediksi **CLICKBAIT** atau **NON-CLICKBAIT** beserta
   skor keyakinannya.
4. Fitur **LIME** dapat digunakan untuk melihat kata atau tanda baca yang
   mendukung dan menahan hasil prediksi.
"""
    )

    st.warning(
        "Sistem hanya membaca teks judul. Hasil prediksi tidak menentukan "
        "kebenaran berita, tidak mendeteksi hoaks, tidak membaca isi artikel, "
        "dan tidak menilai kredibilitas media."
    )

    with st.expander("Informasi teknis sistem"):
        st.markdown(
            """
**Arsitektur model**

1. **IndoBERT** menghasilkan representasi kontekstual setiap token.
2. **BiLSTM dua arah** memproses urutan token dari dua arah.
3. **Attention pooling** membentuk representasi judul.
4. **Classifier** menghasilkan skor untuk dua kelas.

**Performa model pengujian**

- Accuracy: **94,11%**
- Macro-F1: **93,74%**
- Mean flip rate tanda `!` dan `?`: **6,35%**

**Konfigurasi aplikasi**

- Varian pelatihan: `target_q0150`
- Training seed: `2027`
- LIME aplikasi: `500` sampel secara bawaan
- Random state LIME: `42`
"""
        )

    st.button(
        "Mulai Coba Sistem",
        type="primary",
        use_container_width=True,
        on_click=_open_detector,
    )

    st.divider()
    st.markdown(
        """
**Penulis:** Rizky Syahrul Maulid  
**Program Studi:** Teknik Informatika  
**Institusi:** STT Wastukancana Purwakarta
"""
    )


elif menu == "Coba Sistem":
    try:
        runtime = get_runtime()
    except Exception as error:
        st.error(f"Model gagal dimuat: {error}")
        st.stop()

    st.title(APP_TITLE)
    st.caption(MODEL_CAPTION)

    st.markdown(
        "Masukkan judul berita untuk mengetahui apakah pola bahasanya "
        "lebih dekat dengan **clickbait** atau **non-clickbait**."
    )

    st.warning(
        "Sistem hanya menganalisis pola pada teks judul. "
        "Hasil prediksi tidak menentukan kebenaran berita, "
        "kredibilitas media, atau kesesuaian judul dengan isi artikel."
    )

    user_input = st.text_area(
        "Judul berita",
        placeholder=(
            "Contoh: Viral! Bocah Ini Bikin Semua Orang Tercengang..."
        ),
        height=120,
    )

    with st.expander("Pengaturan LIME"):
        lime_samples = st.select_slider(
            "Jumlah sampel variasi",
            options=[
                300,
                500,
                750,
                1000,
                1500,
                2000,
            ],
            value=500,
            help=(
                "Semakin besar jumlah sampel, penjelasan cenderung "
                "lebih stabil tetapi prosesnya lebih lama."
            ),
        )

        lime_features = st.slider(
            "Jumlah fitur yang ditampilkan",
            min_value=3,
            max_value=15,
            value=10,
        )

    predict_column, lime_column = st.columns(2)

    predict_clicked = predict_column.button(
        "Prediksi cepat",
        use_container_width=True,
    )

    lime_clicked = lime_column.button(
        "Prediksi + LIME",
        type="primary",
        use_container_width=True,
    )

    clean_text = user_input.strip()

    if predict_clicked or lime_clicked:
        if not clean_text:
            st.warning("Masukkan judul berita terlebih dahulu.")
            st.stop()

        with st.spinner("Menjalankan prediksi..."):
            prediction_result = predict(
                runtime=runtime,
                text=clean_text,
            )

        lime_result = None

        if lime_clicked:
            cache_key = (
                clean_text,
                int(lime_samples),
                int(lime_features),
            )

            existing_cache = st.session_state.get(
                "last_lime_cache"
            )

            if (
                existing_cache
                and existing_cache.get("key") == cache_key
            ):
                lime_result = existing_cache["result"]
            else:
                try:
                    with st.spinner(
                        "Membuat penjelasan LIME..."
                    ):
                        lime_result = explain_prediction(
                            runtime=runtime,
                            text=clean_text,
                            predicted_label=int(
                                prediction_result["label_id"]
                            ),
                            num_samples=int(lime_samples),
                            num_features=int(lime_features),
                            batch_size=16,
                            random_state=42,
                        )
                except Exception as error:
                    st.error(
                        f"Penjelasan LIME gagal dibuat: {error}"
                    )
                    st.stop()

                st.session_state["last_lime_cache"] = {
                    "key": cache_key,
                    "result": lime_result,
                }

        st.session_state["analysis"] = {
            "text": clean_text,
            "prediction": prediction_result,
            "lime": lime_result,
        }

    analysis = st.session_state.get("analysis")

    if analysis and analysis["text"] == clean_text:
        st.divider()

        show_prediction(
            result=analysis["prediction"],
            positive_class_id=runtime.positive_class_id,
        )

        if analysis["lime"] is not None:
            show_lime(analysis["lime"])
        else:
            st.caption(
                "Gunakan tombol Prediksi + LIME untuk melihat "
                "kontribusi lokal kata dan tanda baca."
            )

        show_final_explanation(
            text=analysis["text"],
            prediction_result=analysis["prediction"],
            positive_class_id=runtime.positive_class_id,
            lime_result=analysis.get("lime"),
        )

    st.divider()
    st.caption(
        "Dataset CLICK-ID · Skripsi 2026 · "
        "Checkpoint target_q0150 seed 2027"
    )
