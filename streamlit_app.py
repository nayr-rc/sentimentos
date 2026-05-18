import streamlit as st
from transformers import pipeline
from typing import List, Tuple

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource
def load_sentiment_pipeline(model_name: str = MODEL_NAME):
    return pipeline("sentiment-analysis", model=model_name)


def analyze_text(text: str, classifier) -> List[dict]:
    texts = [line.strip() for line in text.splitlines() if line.strip()]
    if not texts:
        return []
    return classifier(texts)


def format_sentiment_label(label: str) -> str:
    return "Positivo" if label.lower().startswith("pos") else "Negativo" if label.lower().startswith("neg") else label


def build_result_summary(results: List[dict]) -> List[Tuple[str, str, float]]:
    return [
        (format_sentiment_label(item["label"]), item["label"], item["score"])
        for item in results
    ]


def main() -> None:
    st.set_page_config(
        page_title="Análise de Sentimento",
        page_icon="🧠",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        "<style>
        .big-title { font-size: 42px; font-weight: 800; }
        .subtitle { color: #6c757d; margin-top: -12px; margin-bottom: 24px; }
        .stButton>button { background-color: #2563eb; color: white; border-radius: 10px; }
        .stTextArea textarea { min-height: 200px; }
        </style>",
        unsafe_allow_html=True,
    )

    header_col, _ = st.columns([4, 1])
    with header_col:
        st.markdown("<div class='big-title'>Análise de Sentimento</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='subtitle'>Cole um texto ou várias frases abaixo para identificar sentimento positivo ou negativo imediatamente.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    st.sidebar.header("Sobre o app")
    st.sidebar.write(
        "Este app usa um modelo pré-treinado do Hugging Face para classificar textos em sentimento positivo ou negativo."
    )
    st.sidebar.write(f"**Modelo usado:** `{MODEL_NAME}`")

    with st.sidebar.expander("Exemplos rápidos", expanded=True):
        st.write("• Hoje foi um dia excelente e produtivo.")
        st.write("• Estou muito frustrado com o serviço.")
        st.write("• A experiência foi razoável, mas pode melhorar.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Como usar")
    st.sidebar.write(
        "1. Cole o texto no campo principal.\n"
        "2. Clique em **Analisar**.\n"
        "3. Veja o resultado com confiança e gráficos."  # noqa: E501
    )

    user_text = st.text_area(
        "Digite aqui o texto para análise",
        placeholder="Escreva uma frase ou cole várias linhas de texto...",
        height=260,
    )

    analyze_button = st.button("Analisar sentimento")

    if analyze_button:
        if not user_text.strip():
            st.warning("Por favor, insira um texto antes de clicar em Analisar.")
            return

        with st.spinner("Carregando modelo e avaliando o sentimento..."):
            classifier = load_sentiment_pipeline()
            results = analyze_text(user_text, classifier)

        if not results:
            st.warning("Nenhum texto válido foi encontrado para análise.")
            return

        summary = build_result_summary(results)

        positive_count = sum(1 for sentiment, _, _ in summary if sentiment == "Positivo")
        negative_count = sum(1 for sentiment, _, _ in summary if sentiment == "Negativo")
        average_confidence = sum(score for _, _, score in summary) / len(summary)

        st.success("Análise concluída com sucesso.")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Positivos", positive_count, delta=f"{positive_count} texto(s)")
        metric_col2.metric("Negativos", negative_count, delta=f"{negative_count} texto(s)")
        metric_col3.metric("Confiança média", f"{average_confidence:.0%}")

        table_data = [
            {
                "Texto": index,
                "Sentimento": sentiment,
                "Rótulo do modelo": raw_label,
                "Confiança": f"{score:.2%}",
            }
            for index, (sentiment, raw_label, score) in enumerate(summary, start=1)
        ]

        st.markdown("### Resultados detalhados")
        st.table(table_data)

        st.markdown("### Confiança por texto")
        st.bar_chart([score for _, _, score in summary])

        st.info(
            "O modelo usa `distilbert-base-uncased-finetuned-sst-2-english`, indicado para análise de sentimento em frases curtas."  # noqa: E501
        )

    st.markdown("---")
    st.caption("Feito para implantação rápida no Streamlit Cloud com interface moderna.")


if __name__ == "__main__":
    main()
