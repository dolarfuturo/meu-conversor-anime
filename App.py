import streamlit as st
import requests
import time

st.set_page_config(page_title="Meu Conversor Anime", page_icon="🎌", layout="centered")

st.title("🎌 Conversor de Vídeo para Desenho")
st.write("Transforme vídeos comuns em desenho animado limpo sem limite de tempo.")

api_url = st.text_input(
    "🔗 Cole o link do Cloudflare (gerado no Google Colab):",
    placeholder="https://trycloudflare.com"
).strip().rstrip("/")

video_file = st.file_uploader("📹 Escolha o vídeo original (MP4 ou MOV)", type=["mp4", "mov"])

if video_file is not None:
    st.video(video_file)

    if st.button("✨ Converter para Desenho", type="primary", use_container_width=True):
        if not api_url:
            st.error("Por favor, cole o link do Google Colab no campo acima primeiro.")
        else:
            try:
                with st.spinner("Enviando vídeo para o servidor..."):
                    files = {"file": ("video.mp4", video_file.getvalue(), "video/mp4")}
                    res = requests.post(f"{api_url}/iniciar", files=files, timeout=60)

                if res.status_code != 200:
                    st.error(f"Erro ao enviar ({res.status_code}): {res.text}")
                else:
                    with st.spinner("Desenhando os quadros na GPU... Isso leva cerca de 15 a 30 segundos."):
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
                            label="⬇️ Baixar Vídeo Desenho",
                            data=vid_res.content,
                            file_name="video_desenho.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Falha na conexão: {e}")
