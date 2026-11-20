import streamlit as st
import requests

st.set_page_config(page_title="Meu Conversor Anime", page_icon="🎌", layout="centered")

st.title("🎌 Conversor de Vídeo para Anime")
st.write("Transforme vídeos comuns em animação sem limites de créditos.")

# Campo para colar o link gerado no Google Colab
api_url = st.text_input(
    "🔗 Cole o link do Cloudflare (gerado no Google Colab):",
    placeholder="https://trycloudflare.com"
).strip().rstrip("/")

# Upload do vídeo
video_file = st.file_uploader("📹 Escolha um vídeo (MP4 ou MOV)", type=["mp4", "mov"])

if video_file is not None:
    st.video(video_file)

    if st.button("✨ Converter para Anime", type="primary", use_container_width=True):
        if not api_url:
            st.error("Por favor, cole o link do Google Colab no campo acima primeiro.")
        else:
            with st.spinner("Processando cada quadro na GPU do Colab... Aguarde alguns segundos."):
                try:
                    files = {"file": (video_file.name, video_file.getvalue(), "video/mp4")}
                    response = requests.post(f"{api_url}/converter", files=files, timeout=300)

                    if response.status_code == 200:
                        st.success("✅ Vídeo convertido com sucesso!")
                        st.video(response.content)
                        st.download_button(
                            label="⬇️ Baixar Vídeo Anime",
                            data=response.content,
                            file_name="video_anime.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Erro no servidor ({response.status_code}). Verifique se o Colab ainda está rodando.")
                except Exception as e:
                    st.error(f"Falha na conexão com o Colab: {e}")
