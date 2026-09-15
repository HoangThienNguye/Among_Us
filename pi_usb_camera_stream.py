"""
Draai dit bestand OP de Raspberry Pi.

Leest de USB-camera uit, herkent rood/blauw Among Us,
en streamt het beeld met percentage en vakje.

Nodig in dezelfde map: amongus_rood.jpeg en amongus_blauw.jpeg
Stoppen: Ctrl+C
"""

import os
import sys
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import cv2
import numpy as np

if sys.platform.startswith("win"):
    print("Dit bestand is voor de Raspberry Pi.")
    print("Op je laptop: start opdracht_amongus_raspberry.py")
    raise SystemExit(1)

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")


MIN_OVEREENKOMST_PROCENT = 60
CAMERA = 0
POORT = 8080
BREEDTE = 640
HOOGTE = 480
JPEG_KWALITEIT = 70
ZOEK_W, ZOEK_H = 320, 240

box_kleuren = {
    "rood": (0, 0, 255),
    "blauw": (255, 0, 0),
}

laatste_jpg = None
slot = Lock()
positie_geschiedenis = deque(maxlen=5)


def get_amongus_position(x, y, w, h):
    """Middelpunt van het vakje, een beetje gladgestreken. Later te versturen."""
    cx = x + w / 2.0
    cy = y + h / 2.0
    positie_geschiedenis.append((cx, cy))
    avg_x = sum(p[0] for p in positie_geschiedenis) / len(positie_geschiedenis)
    avg_y = sum(p[1] for p in positie_geschiedenis) / len(positie_geschiedenis)
    return avg_x, avg_y


def knip_amongus(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return gray[y:y + h, x:x + w]


def roteer(img, hoek):
    if hoek == 0:
        return img

    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), hoek, 1)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=255)


def beste_match(gray, templates):
    best_value = -1.0
    best_location = None
    best_w = 0
    best_h = 0

    gh, gw = gray.shape
    for template in templates:
        h, w = template.shape
        if w < 12 or h < 12 or w > gw - 6 or h > gh - 6:
            continue

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        if result.size == 0:
            continue

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if not np.isfinite(max_val):
            continue
        max_val = float(max_val)
        if max_val > best_value:
            best_value = max_val
            best_location = max_loc
            best_w = w
            best_h = h

    return best_value, best_location, best_w, best_h


def kleur_maskers(hsv, soepel=True):
    s_min = 15 if soepel else 40
    v_min = 20 if soepel else 45
    # Rood: niet tot 160, want paarsblauw zit daar tegenaan
    rood = cv2.inRange(hsv, (0, s_min, v_min), (10, 255, 255))
    rood |= cv2.inRange(hsv, (172, s_min, v_min), (180, 255, 255))
    # Blauw + paarsblauw (het blauwe poppetje is paarsachtig)
    blauw = cv2.inRange(hsv, (90, max(10, s_min - 8), v_min), (170, 255, 255))
    return rood, blauw


def kleur_van_stuk(stuk):
    if stuk.size < 20:
        return 0.0, 0.0

    hsv = cv2.cvtColor(stuk, cv2.COLOR_BGR2HSV)
    n = float(stuk.shape[0] * stuk.shape[1])
    rood, blauw = kleur_maskers(hsv, soepel=True)
    return cv2.countNonZero(rood) / n, cv2.countNonZero(blauw) / n


def herken_kleur(roi):
    if roi.size == 0:
        return None

    h, w = roi.shape[:2]
    if h < 14 or w < 12:
        return None

    # Lijf zit links/midden; vizier rechts niet meenemen
    lichaam = roi[int(h * 0.20):int(h * 0.84), int(w * 0.06):int(w * 0.55)]
    if lichaam.size < 20:
        lichaam = roi

    rood_deel, blauw_deel = kleur_van_stuk(lichaam)
    b = float(np.mean(lichaam[:, :, 0]))
    g = float(np.mean(lichaam[:, :, 1]))
    r = float(np.mean(lichaam[:, :, 2]))

    # Paarsblauw: B hoog, G laag, R mag meedoen. Eerst blauw checken.
    if blauw_deel > 0.08 and blauw_deel >= rood_deel:
        return "blauw"
    if b > g + 6 and b >= r - 8:
        return "blauw"

    if rood_deel > 0.10 and rood_deel > blauw_deel + 0.04:
        return "rood"
    if r > g + 12 and r > b + 18:
        return "rood"

    return None


print("Templates laden...")
templates = []
templates_zoek = []

for bestand in ("amongus_rood.jpeg", "amongus_blauw.jpeg"):
    image = cv2.imread(bestand)
    if image is None:
        print("Template niet gevonden:", bestand)
        raise SystemExit(1)

    vorm = knip_amongus(image)
    vorm = cv2.resize(vorm, (int(vorm.shape[1] * 80 / vorm.shape[0]), 80))
    vorm = cv2.equalizeHist(vorm)

    for scale in np.geomspace(0.14, 3.10, 12):
        w = int(vorm.shape[1] * scale)
        h = int(vorm.shape[0] * scale)
        if w < 12 or h < 12:
            continue
        basis = cv2.resize(vorm, (w, h))
        templates_zoek.append(basis)
        for hoek in (-18, 0, 18):
            templates.append(roteer(basis, hoek))

templates_klein = [
    cv2.resize(t, (max(10, t.shape[1] // 2), max(10, t.shape[0] // 2)))
    for t in templates_zoek
]
drempel = MIN_OVEREENKOMST_PROCENT / 100.0
print("Templates klaar.")


def verfinen(gray, x, y, bw, bh):
    pad = int(0.40 * max(bw, bh))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + bw + pad)
    y2 = min(gray.shape[0], y + bh + pad)
    lokale = [t for t in templates if 0.50 * bw < t.shape[1] < 1.70 * bw]
    if not lokale:
        lokale = templates
    waarde, loc, w, h = beste_match(gray[y1:y2, x1:x2], lokale)
    if loc is None:
        return -1.0, None, 0, 0
    return waarde, (loc[0] + x1, loc[1] + y1), w, h


def zoek_overal(gray):
    klein = cv2.resize(gray, (ZOEK_W, ZOEK_H))
    waarde, loc, w, h = beste_match(klein, templates_klein)
    if loc is None:
        return waarde, None, 0, 0

    x, y = loc[0] * 2, loc[1] * 2
    bw, bh = max(12, w * 2), max(12, h * 2)
    fwaarde, floc, fw, fh = verfinen(gray, x, y, bw, bh)
    if floc is not None and fwaarde >= waarde * 0.85:
        return fwaarde, floc, fw, fh
    return waarde, (x, y), bw, bh


def verwerk_frame(frame, vorige):
    frame = cv2.resize(frame, (BREEDTE, HOOGTE)).copy()
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    loc = None
    best_value = -1.0
    best_w = 0
    best_h = 0

    if vorige is not None:
        x, y, bw, bh = vorige
        best_value, loc, best_w, best_h = verfinen(gray, x, y, bw, bh)

    if loc is None or best_value < 0.48:
        gwaarde, gloc, gw, gh = zoek_overal(gray)
        if gloc is not None and gwaarde >= best_value:
            best_value, loc, best_w, best_h = gwaarde, gloc, gw, gh

    kleur = None
    if loc is not None:
        x, y = max(0, loc[0]), max(0, loc[1])
        loc = (x, y)
        x2 = min(frame.shape[1], x + best_w)
        y2 = min(frame.shape[0], y + best_h)
        roi = frame[y:y2, x:x2]
        kleur = herken_kleur(roi)

    if not np.isfinite(best_value) or best_value < 0:
        procent = 0
    else:
        procent = int(np.clip(best_value, 0.0, 1.0) * 100)

    geaccepteerd = kleur is not None and best_value >= drempel

    cv2.putText(
        frame,
        f"match: {procent}%  nodig: {MIN_OVEREENKOMST_PROCENT}%",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if geaccepteerd else (0, 0, 255),
        2
    )

    if geaccepteerd:
        x, y = loc
        vorige = (x, y, best_w, best_h)
        mx, my = get_amongus_position(x, y, best_w, best_h)
        print(f"amongus {kleur}  x={mx:.1f}  y={my:.1f}", flush=True)

        kleur_box = box_kleuren[kleur]
        cv2.rectangle(frame, (x, y), (x + best_w, y + best_h), kleur_box, 3)
        cv2.drawMarker(frame, (int(mx), int(my)), kleur_box, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            frame,
            f"{kleur}  {procent}%  ({int(mx)},{int(my)})",
            (x, max(y - 10, 50)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            kleur_box,
            2
        )
    else:
        vorige = None
        positie_geschiedenis.clear()

    return frame, vorige


def open_camera():
    print("Beschikbare video-apparaten:")
    for i in range(10):
        pad = f"/dev/video{i}"
        if os.path.exists(pad):
            print(" ", pad)

    for index in range(8):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, BREEDTE)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HOOGTE)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"USB-camera gevonden op index {index}")
                return cap
            cap.release()
        else:
            cap.release()

    raise SystemExit("USB-camera niet gevonden. Steek de camera in een USB-poort en probeer opnieuw.")


def camera_loop(cap):
    global laatste_jpg
    vorige = None
    print("USB-camera gestart, herkenning aan.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame, vorige = verwerk_frame(frame, vorige)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_KWALITEIT])
        if not ok:
            continue
        with slot:
            laatste_jpg = jpg.tobytes()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, formaat, *args):
        return

    def do_GET(self):
        if self.path not in ("/", "/stream", "/stream.mjpg"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                with slot:
                    jpg = laatste_jpg
                if jpg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            return


if __name__ == "__main__":
    cap = open_camera()
    Thread(target=camera_loop, args=(cap,), daemon=True).start()
    os.system(f"fuser -k {POORT}/tcp >/dev/null 2>&1")
    time.sleep(0.8)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", POORT), StreamHandler)
    print(f"Camera-stream: http://thien.local:{POORT}/stream.mjpg")
    print("Coordinaten (x, y) verschijnen HIER in deze terminal als er een Among Us is.")
    print("Laat dit venster open. Stoppen: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStream gestopt.")
