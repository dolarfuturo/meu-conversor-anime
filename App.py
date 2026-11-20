!pip install -q diffusers transformers accelerate controlnet_aux

import os, cv2, torch, subprocess, threading, uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pycloudflared import try_cloudflare
from controlnet_aux import LineartDetector
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

app = FastAPI()
status_tarefa = {"estado": "ocioso", "erro": ""}

print("⏳ Carregando Lineart + ControlNet + Modelo Anime na GPU T4...")
if "lineart" not in globals():
    lineart = LineartDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

if "pipe" not in globals():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stablediffusionapi/anything-v5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()

def gerar_camada_linhas_fortes(resized_bgr):
    lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

    pil_img = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        linhas_pil = lineart(pil_img)

    gray_linhas = cv2.cvtColor(np.array(linhas_pil), cv2.COLOR_RGB2GRAY)
    linhas_fortes = np.clip(gray_linhas.astype(np.float32) * 2.2, 0, 255).astype(np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    linhas_conectadas = cv2.morphologyEx(linhas_fortes, cv2.MORPH_CLOSE, kernel)

    papel_branco = cv2.bitwise_not(linhas_conectadas)
    papel_branco[papel_branco > 225] = 255
    return cv2.cvtColor(papel_branco, cv2.COLOR_GRAY2BGR)

def modo_linhas_ou_cor(frame_bgr, target_w, target_h, colorir=False):
    escala = 720 / max(target_h, target_w) if max(target_h, target_w) > 720 else 1.0
    new_w = max(8, (int(target_w * escala) // 8) * 8)
    new_h = max(8, (int(target_h * escala) // 8) * 8)

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    linhas_bgr = gerar_camada_linhas_fortes(resized)

    if not colorir:
        return cv2.resize(linhas_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    linhas_bgr = cv2.resize(linhas_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cores_chapadas = (resized // 28) * 28 + 14
    hsv = cv2.cvtColor(cores_chapadas.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10, 0, 255)
    cores_vivas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    colorido = (cores_vivas.astype(np.uint16) * linhas_bgr.astype(np.uint16) // 255).astype(np.uint8)
    return cv2.resize(colorido, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

frame_anterior_ia = None

def modo_recriar_com_ia(frame_bgr, target_w, target_h, prompt_en, forca):
    global frame_anterior_ia
    escala = 640 / max(target_h, target_w)
    new_w = max(8, (int(target_w * escala) // 8) * 8)
    new_h = max(8, (int(target_h * escala) // 8) * 8)

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if frame_anterior_ia is not None and frame_anterior_ia.shape == resized.shape:
        entrada = cv2.addWeighted(resized, 0.80, frame_anterior_ia, 0.20, 0)
    else:
        entrada = resized

    init_image = Image.fromarray(cv2.cvtColor(entrada, cv2.COLOR_BGR2RGB))
    control_image = lineart(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))).resize((new_w, new_h))

    generator = torch.Generator(device="cuda").manual_seed(42)
    prompt_final = f"{prompt_en}, clean lineart, vibrant colors, sharp details, masterpiece"

    with torch.no_grad():
        result = pipe(
            prompt=prompt_final,
            negative_prompt="blurry, distorted, ugly, deformed car, low quality, messy text, watermark",
            image=init_image,
            control_image=control_image,
            strength=float(forca),
            controlnet_conditioning_scale=0.90,
            num_inference_steps=10,
            guidance_scale=7.0,
            generator=generator
        ).images[0]

    out_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    frame_anterior_ia = out_bgr.copy()
    return cv2.resize(out_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

def processar_em_segundo_plano(input_path, temp_video, output_path, modo, prompt, forca):
    global status_tarefa, frame_anterior_ia
    frame_anterior_ia = None
    try:
        cap = cv2.VideoCapture(input_path)
        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 30
        w = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2) * 2
        h = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2) * 2

        pulo = max(1, round(fps_orig / 12)) if modo == "ia_prompt" else 1
        fps_saida = fps_orig / pulo

        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"MJPG"), fps_saida, (w, h))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % pulo == 0:
                if modo == "linhas":
                    out.write(modo_linhas_ou_cor(frame, w, h, colorir=False))
                elif modo == "colorido":
                    out.write(modo_linhas_ou_cor(frame, w, h, colorir=True))
                else:
                    out.write(modo_recriar_com_ia(frame, w, h, prompt, forca))
            idx += 1

        cap.release()
        out.release()

        subprocess.run([
            "ffmpeg", "-y", "-i", temp_video, "-i", input_path,
            "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?",
            output_path
        ], check=True)

        status_tarefa = {"estado": "pronto", "erro": ""}
    except Exception as e:
        print("ERRO:", e)
        status_tarefa = {"estado": "erro", "erro": str(e)}

@app.post("/iniciar")
async def iniciar_conversao(
    background_tasks: BackgroundTasks,
    modo: str = "linhas",
    prompt: str = "anime racing style",
    forca: float = 0.45,
    file: UploadFile = File(...)
):
    global status_tarefa
    input_path = "entrada.mp4"
    temp_video = "sem_audio.avi"
    output_path = "saida_final.mp4"

    for f in [input_path, temp_video, output_path]:
        if os.path.exists(f):
            os.remove(f)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    status_tarefa = {"estado": "processando", "erro": ""}
    background_tasks.add_task(processar_em_segundo_plano, input_path, temp_video, output_path, modo, prompt, forca)
    return {"status": "iniciado", "modo": modo}

@app.get("/status")
async def checar_status():
    return status_tarefa

@app.get("/baixar")
async def baixar_video():
    return FileResponse("saida_final.mp4", media_type="video/mp4", filename="video_recriado_ia.mp4")

def rodar_servidor():
    uvicorn.run(app, host="0.0.0.0", port=8025, log_level="error")

threading.Thread(target=rodar_servidor, daemon=True).start()
tunnel_url = try_cloudflare(port=8025)
print("\n🚀 ESTÚDIO DE VÍDEO IA PRONTO (PORTA 8025):")
print(tunnel_url.tunnel)
