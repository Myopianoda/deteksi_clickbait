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
