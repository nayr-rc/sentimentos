import streamlit as st
from transformers import pipeline
from typing import List

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource
def load_sentiment_pipeline(model_name: str = MODEL_NAME):
    return pipeline("sentiment-analysis", model=model_name)


def analyze_text(text: str, classifier) -> list[dict[str, float]]:
    texts = [line.strip() for line in text.splitlines() if line.strip()]
    if not texts:
        return []
    return classifier(texts)


def format_sentiment_label(label: str) -> str:
    return "Positivo" if label.lower().startswith("pos") else "Negativo" if label.lower().startswith("neg") else label


def build_result_summary(results: list[dict[str, float]]) -> list[tuple[str, str, float]]:
    return [
        (format_sentiment_label(item["label"]), item["label"], item["score"])
        for item in results
    ]


def main() -> None:
    st.set_page_config(
        page_title="Análise de Sentimento",
        page_icon="🧠",
        layout="centered",
    )

    st.title("Análise de Sentimento com Streamlit")
    st.markdown(
        "Use este app para avaliar se um texto é positivo ou negativo usando um modelo pré-treinado do Hugging Face."
    )

    st.sidebar.header("Configuração")
    st.sidebar.write(f"Modelo: `{MODEL_NAME}`")
    st.sidebar.write("Digite um ou mais textos na área principal e clique em \"Analisar\".")

    user_text = st.text_area(
        "Texto para análise",
        placeholder="Escreva aqui uma frase em português ou inglês...",
        height=180,
    )

    analyze_button = st.button("Analisar")

    if analyze_button:
        if not user_text.strip():
            st.warning("Por favor, insira um texto antes de clicar em Analisar.")
            return

        with st.spinner("Carregando modelo e analisando..."):
            classifier = load_sentiment_pipeline()
            results = analyze_text(user_text, classifier)

        if not results:
            st.warning("Nenhum texto válido foi encontrado para análise.")
            return

        summary = build_result_summary(results)

        st.success("Análise concluída com sucesso.")

        for index, item in enumerate(summary, start=1):
            sentiment, raw_label, score = item
            st.markdown(f"**Texto {index}**")
            st.write(f"- Resultado: **{sentiment}**")
            st.write(f"- Rótulo do modelo: `{raw_label}`")
            st.write(f"- Confiança: **{score:.2f}**")
            if index < len(summary):
                st.markdown("---")

        scores = [item[2] for item in summary]
        st.bar_chart(scores)

        st.write(
            "O modelo usa `distilbert-base-uncased-finetuned-sst-2-english`, que funciona bem para frases curtas de sentimento geral."
        )

    st.markdown("---")
    st.caption("Criado para implantação no Streamlit Cloud.")


if __name__ == "__main__":
    main()
