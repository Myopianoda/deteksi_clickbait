from __future__ import annotations

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


try:
    runtime = get_runtime()
except Exception as error:
    st.error(f"Model gagal dimuat: {error}")
    st.stop()


menu = st.sidebar.selectbox(
    "Menu navigasi",
    ["Beranda", "Tentang Sistem"],
)


if menu == "Beranda":
    st.title(APP_TITLE)
    st.caption(MODEL_CAPTION)

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

    st.divider()
    st.caption(
        "Dataset CLICK-ID · Skripsi 2026 · "
        "Checkpoint target_q0150 seed 2027"
    )


elif menu == "Tentang Sistem":
    st.title("Tentang Sistem")

    st.markdown(
        '''
Sistem mengklasifikasikan judul berita berbahasa Indonesia ke dalam
kelas **clickbait** atau **non-clickbait**. Model hanya membaca teks
judul dan tidak membaca isi artikel.
'''
    )

    st.subheader("Arsitektur")
    st.markdown(
        '''
1. **IndoBERT** menghasilkan representasi kontekstual token.
2. **BiLSTM dua arah** memproses urutan token dari dua arah.
3. **Attention pooling** membentuk representasi judul.
4. **Classifier** menghasilkan skor dua kelas.
'''
    )

    st.subheader("Varian deployment")
    st.write(
        {
            "Arsitektur": "Hybrid IndoBERT–BiLSTM",
            "Varian pelatihan": "target_q0150",
            "Training seed": 2027,
            "Fokus mitigasi": "Sensitivitas terhadap tanda ! dan ?",
            "Test accuracy": "94,11%",
            "Test Macro-F1": "93,74%",
            "Mean flip rate !/?": "6,35%",
        }
    )

    st.subheader("Explainable AI")
    st.markdown(
        '''
LIME membuat variasi kecil dari satu judul, menjalankan prediksi ulang,
lalu memperkirakan fitur yang mendukung atau menahan kelas prediksi.
Penjelasan bersifat lokal dan bukan hubungan sebab-akibat.
'''
    )

    st.subheader("Konfigurasi laporan")
    st.write(
        {
            "LIME laporan": "5.000 sampel",
            "LIME aplikasi": "500 sampel bawaan",
            "Jumlah fitur bawaan": 10,
            "Random state": 42,
        }
    )

    st.divider()
    st.markdown(
        '''
**Penulis:** Rizky Syahrul Maulid  
**Program Studi:** Teknik Informatika  
**Institusi:** STT Wastukancana Purwakarta
'''
    )


# ============================================================================
# TAMBAHAN: PENJELASAN/KESIMPULAN OTOMATIS DI BAGIAN PALING BAWAH APLIKASI
# Catatan: bagian ini tidak mengubah prediksi model. Kategori gaya bahasa
# di bawah merupakan indikasi berbasis pola teks agar hasil lebih mudah
# dipahami oleh pengguna.
# ============================================================================

import re as _re


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
    Mendeteksi indikasi gaya bahasa menggunakan aturan sederhana.

    Hasil fungsi ini bukan kelas tambahan dari model. Aturan hanya dipakai
    untuk membantu menjelaskan pola bahasa yang tampak pada judul.
    """
    original_text = str(text).strip()
    lowered_text = original_text.lower()

    indicator_rules = {
        "sensasional": {
            "patterns": [
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
            "explanation": (
                "menggunakan kata yang memberi penekanan emosional atau "
                "kesan peristiwa yang sangat luar biasa"
            ),
        },
        "provokatif": {
            "patterns": [
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
            "explanation": (
                "memakai diksi yang dapat memancing kemarahan, pertentangan, "
                "atau reaksi emosional pembaca"
            ),
        },
        "membangun rasa penasaran": {
            "patterns": [
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
            "explanation": (
                "menahan sebagian informasi utama sehingga pembaca didorong "
                "untuk membuka berita demi memperoleh jawabannya"
            ),
        },
        "urgensi atau ajakan kuat": {
            "patterns": [
                r"\bsegera\b",
                r"\bjangan lewatkan\b",
                r"\bsebelum terlambat\b",
                r"\bwajib tahu\b",
                r"\bharus tahu\b",
                r"\bbaca sekarang\b",
                r"\bcek sekarang\b",
            ],
            "explanation": (
                "menciptakan kesan mendesak atau mendorong pembaca untuk "
                "segera melakukan tindakan"
            ),
        },
    }

    indicators = []

    for category, rule in indicator_rules.items():
        matches = []

        for pattern in rule["patterns"]:
            for match in _re.finditer(
                pattern,
                lowered_text,
                flags=_re.IGNORECASE,
            ):
                matched_text = match.group(0).strip()

                if matched_text and matched_text not in matches:
                    matches.append(matched_text)

        if matches:
            indicators.append(
                {
                    "category": category,
                    "matches": matches,
                    "explanation": rule["explanation"],
                }
            )

    emphasis_matches = []

    if _re.search(r"!{2,}|\?{2,}|!\?|\?!", original_text):
        emphasis_matches.append("tanda baca berulang")

    if original_text.count("!") >= 1:
        emphasis_matches.append("tanda seru")

    if _re.search(r"\.{3,}", original_text):
        emphasis_matches.append("elipsis (...) ")

    uppercase_words = [
        word
        for word in _re.findall(r"\b[A-Z]{4,}\b", original_text)
        if word not in {"COVID", "PRESIDEN", "INDONESIA"}
    ]

    if uppercase_words:
        emphasis_matches.append(
            "huruf kapital pada "
            + _format_indonesian_list(
                [f'“{word}”' for word in uppercase_words[:3]]
            )
        )

    emphasis_matches = list(dict.fromkeys(emphasis_matches))

    if emphasis_matches:
        indicators.append(
            {
                "category": "penekanan berlebihan",
                "matches": emphasis_matches,
                "explanation": (
                    "menggunakan tanda baca atau bentuk penulisan yang "
                    "meningkatkan intensitas judul"
                ),
            }
        )

    return indicators


def _get_supporting_lime_features(lime_result, limit=3):
    """Mengambil fitur LIME terkuat yang mendukung kelas prediksi."""
    if not lime_result:
        return []

    features = lime_result.get("features", [])
    supporting_features = [
        feature
        for feature in features
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
    """Menampilkan kesimpulan akhir otomatis setelah seluruh hasil analisis."""
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

    indicators = _find_language_indicators(text)
    category_names = [
        indicator["category"]
        for indicator in indicators
    ]

    st.divider()
    st.subheader("Kesimpulan Akhir")

    with st.container(border=True):
        if predicted_clickbait:
            st.markdown(
                f"Judul ini diprediksi sebagai **{label}** dengan "
                f"skor clickbait **{relevant_score:.2%}**."
            )

            if category_names:
                st.markdown(
                    "Berdasarkan pola bahasa pada judul, sistem menemukan "
                    "indikasi **"
                    + _format_indonesian_list(category_names)
                    + "**."
                )
            else:
                st.markdown(
                    "Model menemukan pola yang lebih dekat dengan kelas "
                    "clickbait, tetapi aturan tambahan tidak menemukan kata "
                    "atau bentuk penulisan yang cukup kuat untuk diberi "
                    "kategori sensasional, provokatif, atau rasa penasaran."
                )
        else:
            st.markdown(
                f"Judul ini diprediksi sebagai **{label}** dengan "
                f"skor non-clickbait **{relevant_score:.2%}**."
            )

            if category_names:
                st.markdown(
                    "Secara umum model menilai judul lebih dekat dengan pola "
                    "informatif. Namun, aturan bahasa masih menemukan indikasi "
                    "**"
                    + _format_indonesian_list(category_names)
                    + "**. Indikasi tersebut belum cukup untuk mengubah "
                    "keputusan akhir model menjadi clickbait."
                )
            else:
                st.markdown(
                    "Judul cenderung menyampaikan informasi secara langsung. "
                    "Tidak ditemukan indikasi kuat berupa bahasa sensasional, "
                    "provokatif, rasa penasaran, urgensi, atau penekanan "
                    "berlebihan."
                )

        if indicators:
            st.markdown("**Alasan indikasi gaya bahasa:**")

            for indicator in indicators:
                readable_matches = _format_indonesian_list(
                    [
                        f'“{match.strip()}”'
                        for match in indicator["matches"][:4]
                    ]
                )

                st.markdown(
                    f"- **{indicator['category'].capitalize()}**: "
                    f"{indicator['explanation']}. "
                    f"Indikator yang ditemukan: {readable_matches}."
                )

        supporting_features = _get_supporting_lime_features(
            lime_result=lime_result,
            limit=3,
        )

        if supporting_features:
            lime_feature_names = _format_indonesian_list(
                [
                    f"“{feature['feature']}”"
                    for feature in supporting_features
                ]
            )

            predicted_class_name = lime_result.get(
                "predicted_class_name",
                label,
            )

            st.markdown(
                "Berdasarkan penjelasan lokal LIME, fitur "
                f"{lime_feature_names} termasuk fitur terkuat yang "
                f"mendukung prediksi **{predicted_class_name}** pada judul "
                "ini."
            )
        elif lime_result is None:
            st.caption(
                "Penjelasan kata dari LIME belum tersedia. Gunakan tombol "
                "Prediksi + LIME untuk menambahkan informasi fitur yang "
                "mendukung hasil prediksi."
            )

        st.info(
            "CLICKBAIT/NON-CLICKBAIT merupakan hasil prediksi model. "
            "Sensasional, provokatif, rasa penasaran, urgensi, dan penekanan "
            "berlebihan merupakan indikasi berbasis aturan bahasa, bukan "
            "kelas tambahan yang dipelajari model. Hasil ini tidak menentukan "
            "kebenaran isi berita atau maksud penulis."
        )


# Pemanggilan diletakkan paling bawah agar kesimpulan menjadi bagian terakhir
# pada halaman Beranda, sesuai dengan permintaan penambahan tanpa menghapus
# atau memindahkan kode yang sudah ada.
if menu == "Beranda":
    _final_analysis = st.session_state.get("analysis")

    if (
        _final_analysis
        and _final_analysis.get("text") == clean_text
    ):
        show_final_explanation(
            text=_final_analysis["text"],
            prediction_result=_final_analysis["prediction"],
            positive_class_id=runtime.positive_class_id,
            lime_result=_final_analysis.get("lime"),
        )
