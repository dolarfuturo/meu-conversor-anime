import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoClip

PILOTOS_DB = {
    "VER": {"nome": "Verstappen", "equipe": "Red Bull",     "cor": (30, 65, 175),  "detalhe": (235, 30, 45)},
    "NOR": {"nome": "Norris",     "equipe": "McLaren",      "cor": (255, 135, 0),  "detalhe": (25, 25, 25)},
    "LEC": {"nome": "Leclerc",    "equipe": "Ferrari",      "cor": (220, 20, 30),  "detalhe": (255, 220, 0)},
    "HAM": {"nome": "Hamilton",   "equipe": "Ferrari/Merc", "cor": (0, 210, 190),  "detalhe": (30, 30, 30)},
    "PIA": {"nome": "Piastri",    "equipe": "McLaren",      "cor": (255, 135, 0),  "detalhe": (240, 240, 240)},
    "RUS": {"nome": "Russell",    "equipe": "Mercedes",     "cor": (0, 180, 165),  "detalhe": (220, 220, 220)},
    "ALO": {"nome": "Alonso",     "equipe": "Aston Martin", "cor": (0, 111, 98),   "detalhe": (190, 215, 0)},
    "SAI": {"nome": "Sainz",      "equipe": "Williams",     "cor": (0, 90, 220),   "detalhe": (255, 255, 255)},
}

CIRCUITOS_DB = {
    "interlagos": {
        "titulo": "GP DE SAO PAULO - INTERLAGOS",
        "voltas_gp_real": 71,
        "pontos_norm": [
            (0.77, 0.84), (0.80, 0.64), (0.83, 0.44), (0.77, 0.31),
            (0.84, 0.22), (0.78, 0.14), (0.60, 0.12), (0.40, 0.12),
            (0.22, 0.14), (0.16, 0.24), (0.25, 0.34), (0.42, 0.39),
            (0.56, 0.43), (0.60, 0.52), (0.52, 0.58), (0.42, 0.56),
            (0.35, 0.65), (0.44, 0.73), (0.24, 0.76), (0.19, 0.84),
            (0.35, 0.91), (0.60, 0.91), (0.74, 0.89),
        ]
    }
}

CLIMAS_DB = {
    "seco": {
        "label": "SECO",
        "grama": (46, 115, 48),
        "asfalto": (58, 60, 66),
        "chuva": False
    },
    "chuva": {
        "label": "CHUVA",
        "grama": (34, 85, 38),
        "asfalto": (38, 42, 50),
        "chuva": True
    }
}

MAP_W, MAP_H = 8000, 7200
W, H = 1080, 1920
IDX_CHEGADA = 240
LARGURA_PISTA = 170
TEMPO_LUZES = 2.4


def _carregar_fonte(tamanho):
    caminhos = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "arial.ttf",
    ]
    for c in caminhos:
        try:
            return ImageFont.truetype(c, tamanho)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_TAG = _carregar_fonte(26)
FONT_PLACAR_TITULO = _carregar_fonte(26)
FONT_PLACAR = _carregar_fonte(25)
FONT_LEGENDA = _carregar_fonte(30)
FONT_BANNER = _carregar_fonte(28)
FONT_PISTA_LABEL = _carregar_fonte(24)


def spline_catmull_rom_fechada(ancoras, passos_por_segmento=180):
    pts = []
    n = len(ancoras)
    for i in range(n):
        p0 = ancoras[(i - 1) % n]
        p1 = ancoras[i]
        p2 = ancoras[(i + 1) % n]
        p3 = ancoras[(i + 2) % n]
        for s in range(passos_por_segmento):
            t = s / passos_por_segmento
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            pts.append((x, y))
    return pts


def reamostrar_distancia_uniforme(pts_brutos, num_pontos_uniformes=4600):
    arr = np.array(pts_brutos, dtype=np.float64)
    arr_next = np.roll(arr, -1, axis=0)
    dists = np.linalg.norm(arr_next - arr, axis=1)
    cum_dist = np.concatenate(([0.0], np.cumsum(dists)))
    perimetro_total = cum_dist[-1]

    alvos = np.linspace(0.0, perimetro_total, num_pontos_uniformes, endpoint=False)
    xs = np.interp(alvos, cum_dist, np.append(arr[:, 0], arr[0, 0]))
    ys = np.interp(alvos, cum_dist, np.append(arr[:, 1], arr[0, 1]))
    return list(zip(xs, ys))


def construir_mapa_base(circuito_key, clima_key):
    dados_circuito = CIRCUITOS_DB[circuito_key]
    dados_clima = CLIMAS_DB[clima_key]

    margem_x, margem_y = 650, 600
    ancoras_px = [
        (margem_x + nx * (MAP_W - 2 * margem_x), margem_y + ny * (MAP_H - 2 * margem_y))
        for nx, ny in dados_circuito["pontos_norm"]
    ]
    pts_brutos = spline_catmull_rom_fechada(ancoras_px, passos_por_segmento=180)
    pontos_pista = reamostrar_distancia_uniforme(pts_brutos, num_pontos_uniformes=4600)
    total_pts = len(pontos_pista)

    mapa = Image.new("RGB", (MAP_W, MAP_H), dados_clima["grama"])
    draw = ImageDraw.Draw(mapa)

    np.random.seed(42)
    cor_grama_escura = tuple(max(0, c - 10) for c in dados_clima["grama"])
    for _ in range(13000):
        gx = int(np.random.randint(0, MAP_W))
        gy = int(np.random.randint(0, MAP_H))
        draw.ellipse((gx, gy, gx + 18, gy + 18), fill=cor_grama_escura)

    for i in range(0, total_pts, 18):
        p1 = pontos_pista[i]
        p2 = pontos_pista[(i + 18) % total_pts]
        cor_zebra = (220, 35, 35) if (i // 18) % 2 == 0 else (240, 240, 240)
        r_z = (LARGURA_PISTA + 24) // 2
        draw.line([p1, p2], fill=cor_zebra, width=LARGURA_PISTA + 24)
        draw.ellipse((p1[0] - r_z, p1[1] - r_z, p1[0] + r_z, p1[1] + r_z), fill=cor_zebra)

    r_p = LARGURA_PISTA // 2
    for p in pontos_pista:
        draw.ellipse((p[0] - r_p, p[1] - r_p, p[0] + r_p, p[1] + r_p), fill=dados_clima["asfalto"])

    # Pit lane e boxes na reta principal
    pit_indices = list(range(4180, total_pts)) + list(range(0, 360))
    pit_pts, pit_borda_pts, boxes_pts = [], [], []
    for idx in pit_indices:
        p_cur = pontos_pista[idx]
        p_nxt = pontos_pista[(idx + 5) % total_pts]
        ang_r = math.atan2(p_nxt[1] - p_cur[1], p_nxt[0] - p_cur[0])
        perp_esq = ang_r - math.pi / 2

        pit_pts.append((p_cur[0] + math.cos(perp_esq) * 135, p_cur[1] + math.sin(perp_esq) * 135))
        pit_borda_pts.append((p_cur[0] + math.cos(perp_esq) * 78, p_cur[1] + math.sin(perp_esq) * 78))
        boxes_pts.append((p_cur[0] + math.cos(perp_esq) * 230, p_cur[1] + math.sin(perp_esq) * 230))

    r_pit = 82 // 2
    for pp in pit_pts:
        draw.ellipse((pp[0] - r_pit, pp[1] - r_pit, pp[0] + r_pit, pp[1] + r_pit), fill=dados_clima["asfalto"])

    for k in range(len(pit_borda_pts) - 1):
        cor_linha = (245, 200, 35) if k < len(pit_borda_pts) // 2 else (235, 235, 240)
        larg_linha = 6 if k < len(pit_borda_pts) // 2 else 4
        draw.line([pit_borda_pts[k], pit_borda_pts[k + 1]], fill=cor_linha, width=larg_linha)

    for idx_b in range(15, len(boxes_pts) - 15, 26):
        bx, by = boxes_pts[idx_b]
        p_a = boxes_pts[idx_b - 10]
        p_b = boxes_pts[idx_b + 10]
        ang_b = math.atan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
        w_box, h_box = 44, 22
        cantos = [
            (
                bx + math.cos(ang_b) * sx_c - math.sin(ang_b) * sy_c,
                by + math.sin(ang_b) * sx_c + math.cos(ang_b) * sy_c
            )
            for sx_c, sy_c in [(-w_box, -h_box), (w_box, -h_box), (w_box, h_box), (-w_box, h_box)]
        ]
        draw.polygon(cantos, fill=(70, 76, 88), outline=(110, 118, 132))

    if len(pit_pts) > 20:
        p_lbl_pit = pit_pts[len(pit_pts) // 3]
        draw.text((p_lbl_pit[0] - 55, p_lbl_pit[1] - 14), "PIT LANE", font=FONT_PISTA_LABEL, fill=(255, 225, 80))

    if len(boxes_pts) > 30:
        p_lbl_box = boxes_pts[len(boxes_pts) // 2]
        draw.text((p_lbl_box[0] - 42, p_lbl_box[1] - 14), "BOXES", font=FONT_PISTA_LABEL, fill=(220, 230, 245))

    # Linha de chegada quadriculada
    p_chegada = pontos_pista[IDX_CHEGADA]
    p_next = pontos_pista[(IDX_CHEGADA + 8) % total_pts]
    ang_chegada = math.atan2(p_next[1] - p_chegada[1], p_next[0] - p_chegada[0])
    perp_ang = ang_chegada + math.pi / 2

    for row in range(-1, 2):
        for col in range(-6, 7):
            cor_q = (255, 255, 255) if (row + col) % 2 == 0 else (20, 20, 20)
            cx = p_chegada[0] + math.cos(ang_chegada) * (row * 13) + math.cos(perp_ang) * (col * 13)
            cy = p_chegada[1] + math.sin(ang_chegada) * (row * 13) + math.sin(perp_ang) * (col * 13)
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=cor_q)

    return mapa, pontos_pista


def desenhar_carro_f1(cor_principal, cor_detalhe):
    cw, ch = 74, 40
    sprite = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(sprite)
    rodas = [(10, 2, 24, 9), (10, 31, 24, 38), (48, 3, 62, 10), (48, 30, 62, 37)]
    for r in rodas:
        d.rectangle(r, fill=(18, 18, 20))
    d.rectangle([3, 7, 10, 33], fill=cor_detalhe)
    d.rectangle([62, 6, 70, 34], fill=cor_detalhe)
    d.polygon([(10, 11), (44, 9), (65, 16), (65, 24), (44, 31), (10, 29)], fill=cor_principal)
    d.rectangle([21, 14, 40, 26], fill=cor_detalhe)
    d.ellipse((32, 15, 41, 25), fill=(255, 255, 255))
    return sprite


def gerar_video_corrida(
    caminho_saida,
    circuito_key="interlagos",
    clima_key="chuva",
    voltas_alvo=2,
    duracao=36,
    fps=24,
    grid_inicial=None,
    vencedor_alvo="VER",
    ultrapassagens=None,
):
    if not grid_inicial:
        grid_inicial = ["NOR", "VER", "LEC", "PIA", "HAM", "RUS", "ALO", "SAI"]
    if ultrapassagens is None:
        ultrapassagens = []

    dados_circuito = CIRCUITOS_DB[circuito_key]
    dados_clima = CLIMAS_DB[clima_key]
    mapa_base, pontos_pista = construir_mapa_base(circuito_key, clima_key)
    total_pts = len(pontos_pista)

    tempo_corrida_total = max(1.0, duracao - TEMPO_LUZES)
    distancia_total_corrida = total_pts * voltas_alvo
    vel_media = (distancia_total_corrida / tempo_corrida_total) * 0.82

    carros = []
    for pos_g, sigla in enumerate(grid_inicial):
        info = PILOTOS_DB[sigla]
        bonus_vencedor = 12.0 if sigla == vencedor_alvo else 0.0
        carros.append({
            "sigla": sigla,
            "cor": info["cor"],
            "detalhe": info["detalhe"],
            "grid": pos_g,
            "vel_base": vel_media + bonus_vencedor - (pos_g * 0.65),
            "faixa_base": -26 if pos_g % 2 == 0 else 26,
            "sprite": desenhar_carro_f1(info["cor"], info["detalhe"]),
            "offset_inicial": -32 - pos_g * 28,
        })

    def impulso_ultrapassagem(sigla, frac_corrida):
        ganho = 0.0
        desloc_extra = 0.0
        janela = 0.07
        for atacante, _, momento in ultrapassagens:
            if sigla == atacante:
                delta = frac_corrida - momento
                if -janela <= delta <= janela:
                    fase = (delta + janela) / (2 * janela)
                    sino = math.sin(fase * math.pi)
                    ganho += sino * (total_pts * 0.045)
                    desloc_extra += math.sin(fase * 2 * math.pi) * 32.0
        return ganho, desloc_extra

    def make_frame(t):
        t_corrida = max(0.0, t - TEMPO_LUZES)
        frac_corrida = min(1.0, t_corrida / tempo_corrida_total)
        estado_carros = []

        for c in carros:
            if t_corrida == 0:
                dist = 0.0
                desloc_extra = 0.0
            else:
                arrancada = min(1.0, t_corrida / 1.8)
                dist_base = c["vel_base"] * t_corrida * (0.45 + 0.55 * arrancada)
                ganho_ultra, desloc_extra = impulso_ultrapassagem(c["sigla"], frac_corrida)
                dist = dist_base + ganho_ultra

                if c["sigla"] == vencedor_alvo and frac_corrida > 0.80:
                    convergencia = (frac_corrida - 0.80) / 0.20
                    meta_lider = distancia_total_corrida * 0.96
                    if dist < meta_lider:
                        dist = dist + (meta_lider - dist) * (convergencia * 0.35)

            progresso_abs = c["offset_inicial"] + dist
            pos_idx = (IDX_CHEGADA + progresso_abs) % total_pts
            i0 = int(pos_idx)
            i1 = (i0 + 1) % total_pts
            frac = pos_idx - i0

            p0 = pontos_pista[i0]
            p1 = pontos_pista[i1]
            px = p0[0] + (p1[0] - p0[0]) * frac
            py = p0[1] + (p1[1] - p0[1]) * frac

            ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            perp = ang + math.pi / 2

            desloc = c["faixa_base"] + desloc_extra
            cx = px + math.cos(perp) * desloc
            cy = py + math.sin(perp) * desloc

            estado_carros.append({
                "sigla": c["sigla"],
                "cor": c["cor"],
                "x": cx,
                "y": cy,
                "ang_rad": ang,
                "ang_deg": -math.degrees(ang),
                "sprite": c["sprite"],
                "progresso_abs": progresso_abs,
            })

        ranking = sorted(estado_carros, key=lambda item: item["progresso_abs"], reverse=True)
        lider = ranking[0]
        volta_atual = min(voltas_alvo, max(1, int(max(0, lider["progresso_abs"]) // total_pts) + 1))

        cam_x = float(np.mean([r["x"] for r in ranking[:3]]))
        cam_y = float(np.mean([r["y"] for r in ranking[:3]]))

        left = int(max(0, min(MAP_W - W, cam_x - W // 2)))
        top = int(max(0, min(MAP_H - H, cam_y - H // 2)))
        frame = mapa_base.crop((left, top, left + W, top + H)).convert("RGBA")

        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)

        for ec in estado_carros:
            sx_centro = ec["x"] - left
            sy_centro = ec["y"] - top

            if dados_clima["chuva"] and t_corrida > 0.3:
                for s_step in range(1, 5):
                    bx = sx_centro - math.cos(ec["ang_rad"]) * (30 + s_step * 15)
                    by = sy_centro - math.sin(ec["ang_rad"]) * (30 + s_step * 15)
                    raio_spray = 9 + s_step * 3
                    alpha_spray = max(20, 100 - s_step * 20)
                    od.ellipse((bx - raio_spray, by - raio_spray, bx + raio_spray, by + raio_spray),
                               fill=(220, 235, 250, alpha_spray))

            rot_sprite = ec["sprite"].rotate(ec["ang_deg"], expand=True, resample=Image.BICUBIC)
            frame.paste(rot_sprite,
                        (int(sx_centro - rot_sprite.width // 2), int(sy_centro - rot_sprite.height // 2)),
                        rot_sprite)

            tag_w, tag_h = 74, 32
            tx = int(sx_centro - tag_w // 2)
            ty = int(sy_centro - 58)
            od.rounded_rectangle([tx, ty, tx + tag_w, ty + tag_h], radius=8,
                                 fill=ec["cor"] + (235,), outline=(255, 255, 255, 230), width=2)
            od.text((tx + 9, ty + 2), ec["sigla"], font=FONT_TAG, fill=(255, 255, 255, 255))

        if dados_clima["chuva"]:
            rng = np.random.default_rng(int(t * 100))
            xs = rng.integers(0, W, size=85)
            ys = rng.integers(0, H, size=85)
            for rx, ry in zip(xs, ys):
                od.line([(int(rx), int(ry)), (int(rx - 8), int(ry + 26))],
                        fill=(200, 225, 255, 105), width=2)

        od.rounded_rectangle([40, 45, 1040, 110], radius=16, fill=(15, 18, 25, 225))
        od.text((65, 62), f"{dados_circuito['titulo']} - CLIMA: {dados_clima['label']}", font=FONT_BANNER, fill=(255, 255, 255, 255))

        od.rounded_rectangle([40, 130, 320, 495], radius=16, fill=(15, 18, 25, 220))
        od.text((62, 146), f"VOLTA {volta_atual}/{voltas_alvo}", font=FONT_PLACAR_TITULO, fill=(255, 220, 50, 255))
        for pos, r in enumerate(ranking):
            py_row = 192 + pos * 36
            od.rectangle([62, py_row + 4, 78, py_row + 26], fill=r["cor"] + (255,))
            od.text((90, py_row), f"{pos + 1}. {r['sigla']}", font=FONT_PLACAR, fill=(245, 245, 245, 255))

        if t < 3.2:
            od.rounded_rectangle([350, 210, 920, 325], radius=20, fill=(15, 15, 20, 235))
            luzes = min(5, int((t / 1.9) * 5) + 1) if t < TEMPO_LUZES else 0
            for lz in range(5):
                lx = 415 + lz * 110
                cor_l = (240, 25, 25, 255) if lz < luzes else (55, 55, 60, 255)
                od.ellipse((lx - 28, 268 - 28, lx + 28, 268 + 28), fill=cor_l)

        od.rounded_rectangle([50, 1580, 1030, 1735], radius=22, fill=(10, 12, 18, 225))
        if t < TEMPO_LUZES:
            msg1, msg2 = f"{dados_circuito['titulo']} ({voltas_alvo} VOLTAS)", f"LARGADA AUTORIZADA ({dados_clima['label']})!"
        elif frac_corrida > 0.85:
            msg1, msg2 = f"RETA FINAL! {vencedor_alvo} NA PONTA!", "DISPUTA ATE A BANDEIRADA!"
        else:
            msg1, msg2 = f"VOLTA {volta_atual}/{voltas_alvo} EM {dados_clima['label']}!", "ULTRAPASSAGENS NO MIOLO DE INTERLAGOS!"

        od.text((85, 1605), msg1, font=FONT_LEGENDA, fill=(255, 225, 60, 255))
        od.text((85, 1660), msg2, font=FONT_LEGENDA, fill=(255, 255, 255, 255))

        final_img = Image.alpha_composite(frame, overlay).convert("RGB")
        return np.array(final_img)

    clip = VideoClip(make_frame, duration=duracao)
    clip.write_videofile(caminho_saida, fps=fps, codec="libx264", audio=False, logger=None)
    clip.close()
    return caminho_saida
