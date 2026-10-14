"""Rendert das 3D-Hellprinting-Werbevideo neu (16:9 fuer YouTube, 9:16 fuer Instagram Reels).

Aufruf: python3 render_hellprinting.py <plates_dir> <audio_src.mp4> <format y|v> <out.mp4>

plates_dir enthaelt textfreie Hintergruende: {y,v}_{A,B,C}_clean.png und {y,v}_E_min.png
(Szene "Ich bin Erick", wird per Helligkeits-Fit aus A abgeleitet).
"""
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PLATES, AUDIO, FMT, OUT = sys.argv[1:5]
FONT_DIR = "/usr/share/fonts/opentype/inter/"
FPS, DUR, SCENE = 30, 31.5, 4.5
XFADE = 0.4
RED = (228, 51, 42)
CREAM = (255, 238, 222)
WHITE = (255, 255, 255)

if FMT == "y":
    W, H = 1920, 1080
    CX, MAXW = W // 2, 1700
    HEAD, SUB, LABEL, TAG, BTN, KICK = 112, 58, 36, 32, 78, 40
    HEAD_Y = 140
    LABEL_POS, TAG_POS = (110, 56), (110, 1000)
    SCRIM_H = 0.62
else:
    W, H = 1080, 1920
    CX, MAXW = 510, 880  # optisch in die Safe Zone (rechte Reels-/Shorts-Buttons)
    HEAD, SUB, LABEL, TAG, BTN, KICK = 112, 62, 34, 30, 66, 40
    HEAD_Y = 300
    LABEL_POS, TAG_POS = (70, 225), (70, 1420)
    SCRIM_H = 0.52


def font(name, size):
    path = {"Black": "DejaVuSans-Bold.ttf", "Bold": "DejaVuSans-Bold.ttf", "SemiBold": "DejaVuSans-Bold.ttf",
            "Medium": "DejaVuSans.ttf"}[name]
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{path}", size)


def fit(lines, name, cap, maxw=None, floor=30):
    """Groesste Schrift <= cap, bei der alle Zeilen in maxw passen."""
    maxw = maxw or MAXW
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    size = cap
    while size > floor and max(d.textlength(t, font=font(name, size)) for t in lines) > maxw:
        size -= 2
    return font(name, size)


# ---------------------------------------------------------------- Hintergruende
def load(name):
    return cv2.imread(f"{PLATES}/{FMT}_{name}")


def fit_gain(src, dst, region):
    x0, y0, x1, y1 = region
    a = src[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    b = dst[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    out = src.astype(np.float32)
    for c in range(3):
        g, o = np.polyfit(a[:, c], b[:, c], 1)
        out[..., c] = out[..., c] * g + o
    return np.clip(out, 0, 255).astype(np.uint8)


plateA = load("A_clean.png")
plates = {
    "A": plateA,
    "B": load("B_clean.png"),
    "C": load("C_clean.png"),
    "E": fit_gain(plateA, load("E_min.png"),
                  (0, 500, 340, 1000) if FMT == "y" else (0, 1500, W, 1700)),
}


# ---------------------------------------------------------------- Text-Elemente
def text_layer(lines, anchor_y, align="center", ax=None, tracking=0, shadow=True):
    """lines: [(text, font, color)] -> (x, y, rgba) mit weichem Schatten."""
    ax = CX if ax is None else ax
    pad = 60
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    y = anchor_y
    for text, f, color in lines:
        if tracking:
            wtot = sum(d.textlength(ch, font=f) + tracking for ch in text) - tracking
        else:
            wtot = d.textlength(text, font=f)
        x = ax - wtot / 2 if align == "center" else ax
        if tracking:
            for ch in text:
                d.text((x, y), ch, font=f, fill=color)
                x += d.textlength(ch, font=f) + tracking
        else:
            d.text((x, y), text, font=f, fill=color)
        y += int(f.size * 1.2)
    arr = np.array(tmp)
    ys, xs = np.where(arr[..., 3] > 0)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
    arr = arr[y0:y1, x0:x1].astype(np.float32)
    if shadow:
        a = cv2.GaussianBlur(arr[..., 3], (0, 0), 10) * 0.85
        sh = np.zeros_like(arr)
        sh[..., 3] = np.clip(a, 0, 255)
        out = sh.copy()
        al = arr[..., 3:4] / 255
        out[..., :3] = arr[..., :3] * al + sh[..., :3] * (1 - al)
        out[..., 3] = np.clip(arr[..., 3] + sh[..., 3] * (1 - arr[..., 3] / 255), 0, 255)
        arr = out
    return x0, y0, arr


def rounded_button(text, f, cy):
    pw, ph = 70, 34
    tmp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    tw = d.textlength(text, font=f)
    bw, bh = int(tw + 2 * pw), int(f.size + 2 * ph)
    x0, y0 = int(CX - bw / 2), int(cy)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle((x0, y0, x0 + bw, y0 + bh), 28, fill=RED + (255,))
    glow_arr = cv2.GaussianBlur(np.array(glow), (0, 0), 26)
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), 28, fill=(190, 32, 26, 255),
                        outline=(255, 120, 100, 255), width=3)
    d.text((CX - tw / 2, y0 + ph - f.size * 0.02), text, font=f, fill=WHITE)
    arr = np.array(tmp).astype(np.float32)
    g = glow_arr.astype(np.float32)
    g[..., 3] *= 0.55
    al = arr[..., 3:4] / 255
    out = arr.copy()
    out[..., :3] = arr[..., :3] * al + g[..., :3] * (1 - al)
    out[..., 3] = np.clip(arr[..., 3] + g[..., 3] * (1 - al[..., 0]), 0, 255)
    pad = 80
    xa, ya = max(x0 - pad, 0), max(y0 - pad, 0)
    xb, yb = min(x0 + bw + pad, W), min(y0 + bh + pad, H)
    return xa, ya, out[ya:yb, xa:xb], (x0 + bw / 2 - xa, y0 + bh / 2 - ya)


def stack(blocks, y0, gap=26):
    """blocks: [(lines, fontname, cap, color)] -> Layer-Liste, vertikal gestapelt."""
    out, y = [], y0
    for lines, name, cap, color in blocks:
        f = fit(lines, name, cap)
        out.append(text_layer([(t, f, color) for t in lines], y))
        y += int(f.size * 1.2) * len(lines) + gap
    return out


V = FMT == "v"
S = []
S.append(("A", stack([(["DEINE IDEE.", "IN FORM", "GEBRACHT."] if V else ["DEINE IDEE.", "IN FORM GEBRACHT."], "Black", HEAD, WHITE),
                      (["3D-DRUCKSERVICE", "EINZELSTÜCKE", "KLEINSERIEN"] if V else
                       ["3D-DRUCKSERVICE · EINZELSTÜCKE · KLEINSERIEN"], "Medium", SUB, CREAM)], HEAD_Y), "top"))
S.append(("B", stack([(["FUNKTIONS-", "TEILE"] if V else ["FUNKTIONSTEILE"], "Black", HEAD, WHITE),
                      (["HALTER · CLIPS", "ERSATZTEILE", "PROTOTYPEN &", "KLEINSERIEN"] if V else
                       ["HALTER · CLIPS · ERSATZTEILE", "PROTOTYPEN & KLEINSERIEN"], "Medium", SUB, CREAM)], HEAD_Y), "top"))
S.append(("A", stack([(["3D-DRUCK"], "Black", HEAD, WHITE),
                      (["Einzelstücke und", "Kleinserien passend", "zu deinem Vorhaben"] if V else
                       ["Einzelstücke und Kleinserien", "passend zu deinem Vorhaben"], "Medium", SUB, CREAM)], HEAD_Y), "top"))
S.append(("C", stack([(["LASERGRAVUR"], "Black", HEAD, WHITE),
                      (["Individuelle Gravuren", "mit Charakter"] if V else
                       ["Individuelle Gravuren mit Charakter"], "Medium", SUB, CREAM)], HEAD_Y), "top"))
S.append(("B", stack([(["DAS PASSENDE", "MATERIAL"] if V else ["DAS PASSENDE MATERIAL"], "Black", HEAD, WHITE),
                      (["PLA · PETG", "PETG-CF/GF · PA-CF/GF", "ABS · ASA · TPU"] if V else
                       ["PLA · PETG · PETG-CF/GF", "PA-CF/GF · ABS · ASA · TPU"], "Medium", SUB, CREAM)], HEAD_Y), "top"))
S.append(("B", stack([(["DU HAST SCHON", "EINE DATEI?"] if V else ["DU HAST SCHON EINE DATEI?"], "Black", HEAD, WHITE),
                      (["STL · STEP · 3MF", "OBJ · DXF · PDF · BILD"] if V else
                       ["STL · STEP · 3MF · OBJ · DXF · PDF · BILD"], "Medium", SUB, CREAM),
                      (["Dateiprüfung: 10 €", "bei Auftrag angerechnet"] if V else
                       ["Technische Dateiprüfung: 10 €", "Bei Druckauftrag vollständig angerechnet"],
                       "Bold", SUB - (0 if V else 8), WHITE)], HEAD_Y, gap=20), "top"))
btn_y = 800 if FMT == "y" else 1275
kick = text_layer([("JETZT ONLINE ANFRAGEN", font("Bold", KICK), CREAM)], btn_y - 80, tracking=4)
_bf = fit(["3d-hellprinting.de"], "Black", BTN, maxw=MAXW - 140)
button = rounded_button("3d-hellprinting.de", _bf, btn_y)
S.append(("E", stack([(["ICH BIN", "ERICK."] if V else ["ICH BIN ERICK."], "Black", HEAD, WHITE),
                      (["Du sprichst direkt mit mir.", "Von der Anfrage bis", "zum fertigen Teil."] if V else
                       ["Du sprichst direkt mit mir.", "Von der Anfrage bis zum fertigen Teil."], "Medium", SUB, CREAM)],
                     HEAD_Y) + [kick, button], "both"))

label = text_layer([("3D HELLPRINTING AND MORE", font("Bold", LABEL), WHITE)], LABEL_POS[1],
                   align="left", ax=LABEL_POS[0], tracking=3)
lw = ImageDraw.Draw(Image.new("L", (1, 1))).textlength("3D HELLPRINTING AND MORE", font=font("Bold", LABEL)) + 3 * 23
_tag_f = font("Bold", TAG)
if FMT == "y":
    tag = text_layer([("DÜSTER. PRÄZISE. INDIVIDUELL.", _tag_f, (240, 120, 70))], TAG_POS[1],
                     align="left", ax=TAG_POS[0], tracking=3)
else:  # Hochkant: Claim nur im Schluss-Frame, sonst liegt er ueber den Produkten
    tag = text_layer([("DÜSTER. PRÄZISE. INDIVIDUELL.", _tag_f, (240, 120, 70))], 1455, tracking=3)

# Scrims (Lesbarkeit): oben abdunkeln, im CTA-Teil auch unten
yy = np.linspace(0, 1, H)[:, None].astype(np.float32)
top = np.clip(1 - yy / SCRIM_H, 0, 1) ** 1.3 * 0.80
bottom = np.clip((yy - 0.62) / 0.38, 0, 1) ** 1.2 * 0.72
SCRIMS = {"top": top, "both": np.maximum(top, bottom)}


# ---------------------------------------------------------------- Rendering
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def blit(frame, layer, alpha, dy=0, scale=1.0, center=None):
    x, y, arr = layer[:3]
    if scale != 1.0 and center is not None:
        h, w = arr.shape[:2]
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        x = int(x + center[0] - center[0] * scale)
        y = int(y + center[1] - center[1] * scale)
    h, w = arr.shape[:2]
    y = int(y + dy)
    fx0, fy0 = max(x, 0), max(y, 0)
    fx1, fy1 = min(x + w, W), min(y + h, H)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub_ = arr[fy0 - y:fy1 - y, fx0 - x:fx1 - x]
    a = (sub_[..., 3:4] / 255.0) * alpha
    roi = frame[fy0:fy1, fx0:fx1]
    roi[:] = roi * (1 - a) + sub_[..., [2, 1, 0]] * a  # PIL RGB -> BGR


def render_scene(si, lt):
    key, elems, scrim = S[si]
    zoom = 1.0 + 0.07 * (max(lt, 0) + 0.4) / (SCENE + 0.8)
    plate = plates[key]
    ph, pw = plate.shape[:2]
    cw, ch = pw / zoom, ph / zoom
    x0, y0 = (pw - cw) / 2, (ph - ch) / 2
    M = np.float32([[W / cw, 0, -x0 * W / cw], [0, H / ch, -y0 * H / ch]])
    frame = cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
    frame *= (1 - SCRIMS[scrim])[..., None]
    # Marke + Tagline
    blit(frame, label, min(max((lt + 0.4) / 0.5, 0), 1))
    ul_w = int(lw * min(max((lt + 0.4) / 0.7, 0), 1))
    cv2.line(frame, (LABEL_POS[0], LABEL_POS[1] + LABEL + 14), (LABEL_POS[0] + ul_w, LABEL_POS[1] + LABEL + 14),
             (RED[2], RED[1], RED[0]), 3)
    if (key != "E") == (FMT == "y"):
        blit(frame, tag, 0.9)
    for i, el in enumerate(elems):
        t0 = 0.15 + 0.16 * i
        p = ease((lt - t0) / 0.55)
        if el is button:
            pulse = 1 + 0.035 * np.sin(max(lt - 1.0, 0) * 2 * np.pi / 1.4) * min((lt - 1.0) / 0.5, 1) if lt > 1.0 else 1
            blit(frame, el, p, dy=(1 - p) * 30, scale=pulse, center=el[3])
        else:
            blit(frame, el, p, dy=(1 - p) * 34)
    return frame


def render(t):
    si = min(int(t // SCENE), 6)
    lt = t - si * SCENE
    f = render_scene(si, lt)
    if lt > SCENE - XFADE and si < 6:
        k = (lt - (SCENE - XFADE)) / XFADE
        f = f * (1 - k) + render_scene(si + 1, lt - SCENE) * k
    return np.clip(f, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    n = int(DUR * FPS)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
           "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags",
           "+faststart", "-c:a", "copy", "-t", str(DUR), OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(render(i / FPS).tobytes())
    p.stdin.close()
    p.wait()
    print("done", OUT)
