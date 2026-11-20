import streamlit as st
import requests
import time

st.set_page_config(page_title="Meu Conversor Anime", page_icon="🎌", layout="centered")

st.title("🎌 Conversor de Vídeo para Desenho")
st.write("Escolha o estilo e transforme seus vídeos sem limite de créditos.")

api_url = st.text_input(
    "🔗 Cole o link do Cloudflare (gerado no Google Colab):",
    placeholder="https://trycloudflare.com"
).strip().rstrip("/")

# Seletor de estilo direto na tela
estilo = st.radio(
    "🎨 Escolha o estilo do vídeo:",
    ["✏️ Somente Linhas (Preto e Branco)", "🎨 Desenho 2D Colorido (Estilo Vetor)"],
    horizontal=True
)

modo_api = "linhas" if "Linhas" in estilo else "colorido"

video_file = st.file_uploader("📹 Escolha o vídeo original (MP4 ou MOV)", type=["mp4", "mov"])

if video_file is not None:
    st.video(video_file)

    if st.button("✨ Converter Vídeo", type="primary", use_container_width=True):
        if not api_url:
            st.error("Por favor, cole o link do Google Colab no campo acima primeiro.")
        else:
            try:
                with st.spinner("Enviando vídeo para o servidor..."):
                    files = {"file": ("video.mp4", video_file.getvalue(), "video/mp4")}
                    res = requests.post(f"{api_url}/iniciar?modo={modo_api}", files=files, timeout=60)

                if res.status_code != 200:
                    st.error(f"Erro ao enviar ({res.status_code}): {res.text}")
                else:
                    with st.spinner("Desenhando os quadros na GPU... Aguarde."):
                        while True:
                            time.sleep(3)
                            st_res = requests.get(f"{api_url}/status", timeout=30).json()
                            if st_res["estado"] == "pronto":
                                break
                            elif st_res["estado"] == "erro":
                                st.error(f"Erro interno: {st_res['erro']}")
                                st.stop()

                    with st.spinner("Baixando vídeo pronto..."):
                        vid_res = requests.get(f"{api_url}/baixar", timeout=60)
                        st.success("✅ Vídeo convertido com sucesso!")
                        st.video(vid_res.content)
                        st.download_button(
                            label="⬇️ Baixar Vídeo Pronto",
                            data=vid_res.content,
                            file_name=f"video_{modo_api}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Falha na conexão: {e}")
