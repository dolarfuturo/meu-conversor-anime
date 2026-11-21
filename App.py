import streamlit as st
import requests
import time
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Meu Estúdio de Vídeo IA", page_icon="🏎️", layout="centered")

st.title("🏎️ Estúdio de Vídeo IA")
st.write("Replique cenas reais em qualquer estilo ou converta para desenho.")

api_url = st.text_input(
    "🔗 Cole o link do Cloudflare (gerado no Google Colab):",
    placeholder="https://trycloudflare.com"
).strip().rstrip("/")

estilo = st.radio(
    "🎨 Escolha o modo de criação:",
    [
        "✨ Recriar Cena com IA (Você escreve o estilo)",
        "✏️ Somente Linhas (Rápido)",
        "🎨 Desenho 2D Colorido (Rápido)"
    ]
)

prompt_en = ""
forca_ia = 0.45

if "Recriar" in estilo:
    modo_api = "ia_prompt"
    prompt_pt = st.text_area(
        "✍️ Descreva como quer recriar o vídeo (em português):",
        value="anime japonês de corrida noturna, holofotes brilhantes, faíscas saindo dos carros, traço limpo, alta qualidade",
        height=80
    )
    forca_ia = st.slider(
        "🎛️ Liberdade criativa da IA (Menor = mais fiel ao vídeo real | Maior = mais desenhado):",
        min_value=0.30, max_value=0.65, value=0.45, step=0.05
    )
    if prompt_pt:
        try:
            prompt_en = GoogleTranslator(source="pt", target="en").translate(prompt_pt)
        except:
            prompt_en = prompt_pt
elif "Linhas" in estilo:
    modo_api = "linhas"
else:
    modo_api = "colorido"

video_file = st.file_uploader("📹 Envie o vídeo guia (ex: ultrapassagem real em MP4)", type=["mp4", "mov"])

if video_file is not None:
    st.video(video_file)

    if st.button("🚀 Gerar Vídeo", type="primary", use_container_width=True):
        if not api_url:
            st.error("Por favor, cole o link do Google Colab no campo acima primeiro.")
        else:
            try:
                with st.spinner("Enviando vídeo para a GPU..."):
                    files = {"file": ("video.mp4", video_file.getvalue(), "video/mp4")}
                    params = {"modo": modo_api, "prompt": prompt_en, "forca": forca_ia}
                    res = requests.post(f"{api_url}/iniciar", files=files, params=params, timeout=60)

                if res.status_code != 200:
                    st.error(f"Erro ao enviar ({res.status_code}): {res.text}")
                else:
                    with st.spinner("A IA está recriando a cena quadro a quadro... (Pode levar 1 a 2 min no modo IA)"):
                        while True:
                            time.sleep(4)
                            st_res = requests.get(f"{api_url}/status", timeout=30).json()
                            if st_res["estado"] == "pronto":
                                break
                            elif st_res["estado"] == "erro":
                                st.error(f"Erro interno: {st_res['erro']}")
                                st.stop()

                    with st.spinner("Baixando resultado..."):
                        vid_res = requests.get(f"{api_url}/baixar", timeout=60)
                        st.success("✅ Vídeo recriado com sucesso!")
                        st.video(vid_res.content)
                        st.download_button(
                            label="⬇️ Baixar Vídeo Final",
                            data=vid_res.content,
                            file_name=f"video_{modo_api}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Falha na conexão: {e}")
