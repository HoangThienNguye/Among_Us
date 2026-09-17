"""
Leest de USB-camera uit, herkent Among Us,
en streamt het beeld met percentage en vakje.

Hoe het werkt:
1. Bij het opstarten worden de voorbeeldfoto's omgezet naar "templates".
2. Per camerabeeld schuift OpenCV die over het beeld (template matching).
3. Past de vorm en is de kleur rood of blauw, dan komt er een vakje om.
4. Het beeld gaat als JPEG naar de browser (MJPEG-stream).

main.py importeert dit bestand en gebruikt verwerk_frame() en open_camera().

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

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")


MIN_OVEREENKOMST_PROCENT = 60   # hoe goed het op een Among Us moet lijken voor we het accepteren
CAMERA = 0                      # camera-index; open_camera() zoekt er zelf ook naar
POORT = 8080                    # poort waarop de stream te bekijken is
BREEDTE = 640                   # elk frame wordt naar dit formaat geschaald, dus alle
HOOGTE = 480                    # coordinaten in dit bestand liggen in 640x480
JPEG_KWALITEIT = 70             # lager = kleiner plaatje, minder netwerk, minder scherp
ZOEK_W, ZOEK_H = 320, 240       # halve grootte; hierop zoeken we grofweg, dat is sneller

# Kleur van het vakje in het beeld.
box_kleuren = {
    "rood": (0, 0, 255),
    "blauw": (255, 0, 0),
}

laatste_jpg = None
slot = Lock()
positie_geschiedenis = deque(maxlen=5)  # laatste 5 middelpunten, om te gladstrijken


def get_amongus_position(x, y, w, h):
    """Middelpunt van het vakje in pixels, gemiddeld over de laatste 5 frames.

    (x, y) is de linkerbovenhoek. Het gemiddelde haalt het gewiebel eruit,
    maar loopt daardoor iets achter.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0
    positie_geschiedenis.append((cx, cy))
    avg_x = sum(p[0] for p in positie_geschiedenis) / len(positie_geschiedenis)
    avg_y = sum(p[1] for p in positie_geschiedenis) / len(positie_geschiedenis)
    return avg_x, avg_y


def knip_amongus(image):
    """Snijdt het poppetje uit een voorbeeldfoto, in grijswaarden.

    THRESH_OTSU kiest zelf de grens tussen licht en donker. Van de
    gevonden vlekken is de grootste het poppetje.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return gray[y:y + h, x:x + w]


def roteer(img, hoek):
    """Draait een template, voor als het poppetje schuin hangt.

    Het canvas groeit mee (nw, nh), anders vallen de hoeken eraf. Lege
    plekken worden wit, net als de achtergrond van de foto.
    """
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
    """Probeert alle templates en geeft (score, (x, y), breedte, hoogte) terug.

    De score loopt van -1 tot 1, waarbij 1 perfect is.
    """
    best_value = -1.0
    best_location = None
    best_w = 0
    best_h = 0

    gh, gw = gray.shape
    for template in templates:
        h, w = template.shape
        # Te klein, of groter dan het beeld: daar kan matchTemplate niks mee.
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
    """Maakt twee maskers: waar zit rood, en waar zit blauw.

    HSV in plaats van BGR, want de kleurtoon blijft gelijk bij ander licht.
    s_min en v_min houden grijs en bijna-zwart buiten de deur.
    """
    s_min = 15 if soepel else 40
    v_min = 20 if soepel else 45
    # Rood ligt aan beide uiteinden van de HSV-schaal (rond 0 en rond 180),
    # daarom twee bereiken. Niet tot 160, want paarsblauw zit daar tegenaan.
    rood = cv2.inRange(hsv, (0, s_min, v_min), (10, 255, 255))
    rood |= cv2.inRange(hsv, (172, s_min, v_min), (180, 255, 255))
    # Blauw + paarsblauw (het blauwe poppetje is paarsachtig)
    blauw = cv2.inRange(hsv, (90, max(10, s_min - 8), v_min), (170, 255, 255))
    return rood, blauw


def kleur_van_stuk(stuk):
    """Welk deel van dit stukje beeld is rood, en welk deel blauw.

    Twee getallen tussen 0 en 1; 0.35 betekent 35% van de pixels.
    """
    if stuk.size < 20:
        return 0.0, 0.0

    hsv = cv2.cvtColor(stuk, cv2.COLOR_BGR2HSV)
    n = float(stuk.shape[0] * stuk.shape[1])
    rood, blauw = kleur_maskers(hsv, soepel=True)
    return cv2.countNonZero(rood) / n, cv2.countNonZero(blauw) / n


def herken_kleur(roi):
    """Geeft "rood", "blauw" of None voor het gevonden poppetje.

    roi is het uitgesneden stukje beeld. Eerst wordt geteld hoeveel pixels
    in het kleurbereik vallen; lukt dat niet, dan worden de gemiddelden van
    de losse kanalen vergeleken (voor als het licht de kleur laat verwateren).
    """
    if roi.size == 0:
        return None

    h, w = roi.shape[:2]
    if h < 14 or w < 12:
        return None

    # Alleen het lijf (links/midden). Het vizier is bij beide poppetjes
    # lichtblauw-grijs en zou de meting vertroebelen.
    lichaam = roi[int(h * 0.20):int(h * 0.84), int(w * 0.06):int(w * 0.55)]
    if lichaam.size < 20:
        lichaam = roi

    rood_deel, blauw_deel = kleur_van_stuk(lichaam)
    # In BGR is kanaal 0 blauw, 1 groen en 2 rood.
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


# Dit blok loopt één keer, zodra dit bestand gestart of geïmporteerd
# wordt. Uit de twee voorbeeldfoto's maken we hier alle templates die
# later bij elk camerabeeld gebruikt worden. Dat kost een paar seconden,
# vandaar de meldingen in de terminal.
#
# templates       = alle formaten én draaihoeken, voor het nauwkeurig zoeken
# templates_zoek  = alleen de rechte versies
# templates_klein = die rechte versies op halve grootte, voor het snelle
#                   grove zoeken over het hele beeld
print("Templates laden...")
templates = []
templates_zoek = []

for bestand in ("amongus_rood.jpeg", "amongus_blauw.jpeg"):
    image = cv2.imread(bestand)
    if image is None:
        print("Template niet gevonden:", bestand)
        raise SystemExit(1)

    vorm = knip_amongus(image)
    # Eerst op een vaste hoogte van 80 pixels zetten, zodat beide foto's
    # even groot beginnen (de breedte schaalt mee). equalizeHist trekt
    # licht en donker gelijk, waardoor de match minder afhangt van de
    # lamp in de kamer. Datzelfde doen we straks met het camerabeeld.
    vorm = cv2.resize(vorm, (int(vorm.shape[1] * 80 / vorm.shape[0]), 80))
    vorm = cv2.equalizeHist(vorm)

    # Het poppetje kan dichtbij of ver weg zijn, dus maken we 12 formaten
    # van klein (0.14x) tot groot (3.10x). geomspace verdeelt die stappen
    # steeds met dezelfde factor in plaats van met een vast verschil, wat
    # beter past bij afstand: van 0.14 naar 0.20 is net zo'n grote stap
    # als van 2.0 naar 3.0.
    for scale in np.geomspace(0.14, 3.10, 12):
        w = int(vorm.shape[1] * scale)
        h = int(vorm.shape[0] * scale)
        if w < 12 or h < 12:
            continue
        basis = cv2.resize(vorm, (w, h))
        templates_zoek.append(basis)
        # Per formaat ook een versie naar links en naar rechts gekanteld,
        # zodat een schuin hangend poppetje ook past.
        for hoek in (-18, 0, 18):
            templates.append(roteer(basis, hoek))

templates_klein = [
    cv2.resize(t, (max(10, t.shape[1] // 2), max(10, t.shape[0] // 2)))
    for t in templates_zoek
]
drempel = MIN_OVEREENKOMST_PROCENT / 100.0  # 60 procent wordt 0.60, want scores lopen van 0 tot 1
print("Templates klaar.")


def verfinen(gray, x, y, bw, bh):
    """Zoekt nauwkeurig, maar alleen in een klein gebied rond een eerdere vondst.

    Dit is veel sneller dan het hele beeld afzoeken, en dat kan ook: tussen
    twee camerabeelden beweegt het poppetje maar een klein stukje. We nemen
    het oude vakje plus 40% marge eromheen (pad) en zoeken alleen daarin.

    Ook gebruiken we alleen templates die qua breedte in de buurt liggen van
    wat we vorige keer vonden (tussen de helft en 1,7 keer zo breed), want de
    afstand tot de camera verandert niet ineens.

    Aan het eind worden x en y opgeteld bij x1 en y1, om de plek terug te
    rekenen naar coordinaten in het hele beeld. beste_match() rekent namelijk
    binnen het uitgesneden stukje.
    """
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
    """Zoekt in het hele beeld, in twee stappen.

    Eerst op halve grootte (320x240) met de kleine templates. Dat is vier
    keer zo weinig pixels en dus een stuk sneller. De gevonden plek en maten
    worden daarna met 2 vermenigvuldigd om terug te komen in 640x480.

    Daarna nog een keer verfinen() op de volle resolutie voor een preciezer
    vakje. Levert dat een duidelijk slechtere score op (minder dan 85% van de
    grove score), dan houden we toch de grove uitkomst.
    """
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
    """Verwerkt één camerabeeld: zoeken, kleur bepalen en tekenen.

    Dit is de kern van het bestand, en de functie die main.py gebruikt.

    Meegeven:
      frame   - het ruwe beeld van de camera
      vorige  - waar we het vorige frame iets vonden, of None als we
                niets hadden. Daarmee kan er in de buurt gezocht worden
                in plaats van overal.

    Geeft vier dingen terug:
      frame   - hetzelfde beeld, nu met tekst en eventueel een vakje erop
      vorige  - (x, y, breedte, hoogte) van deze vondst, of None. Die geef
                je bij het volgende frame weer mee.
      positie - (x, y) middelpunt van het poppetje, of None
      kleur   - "rood", "blauw" of None
    """
    frame = cv2.resize(frame, (BREEDTE, HOOGTE)).copy()
    # equalizeHist ook hier, net als bij de templates. Zo vergelijken we
    # appels met appels, ongeacht hoe licht of donker het beeld is.
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    loc = None
    best_value = -1.0
    best_w = 0
    best_h = 0

    # Stap 1: hadden we vorige keer iets gevonden? Kijk dan eerst daar in
    # de buurt, want dat is snel.
    if vorige is not None:
        x, y, bw, bh = vorige
        best_value, loc, best_w, best_h = verfinen(gray, x, y, bw, bh)

    # Stap 2: niets gevonden, of de score is te laag om te vertrouwen.
    # Dan toch het hele beeld afzoeken. Dat kost meer tijd, dus doen we
    # het alleen als het nodig is.
    if loc is None or best_value < 0.48:
        gwaarde, gloc, gw, gh = zoek_overal(gray)
        if gloc is not None and gwaarde >= best_value:
            best_value, loc, best_w, best_h = gwaarde, gloc, gw, gh

    # Iets gevonden dat qua vorm klopt? Kijk dan of de kleur ook past.
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

    # Alleen goedkeuren als de vorm genoeg lijkt EN de kleur duidelijk is.
    # Een van de twee is niet genoeg; anders gaat hij op alles reageren.
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
        return frame, vorige, (mx, my), kleur

    # Niet goedgekeurd: vorige op None zetten, zodat we het volgende frame
    # weer overal zoeken. De geschiedenis leegmaken zorgt ervoor dat oude
    # posities niet meetellen als het poppetje straks ergens anders opduikt.
    vorige = None
    positie_geschiedenis.clear()
    return frame, vorige, None, None


def open_camera():
    """Zoekt de USB-camera en geeft een geopende VideoCapture terug.

    BUFFERSIZE op 1 zodat we steeds het nieuwste beeld krijgen en niet een
    paar frames achterlopen. Dat is belangrijk als de servo's erop reageren.

    """
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
    """Blijft camerabeelden lezen, verwerken en als JPEG klaarzetten.

    Deze functie wordt alleen gebruikt als je dit bestand zelf start.
    main.py heeft zijn eigen lus, omdat daar ook de laser en de servo's
    bij komen.
    """
    global laatste_jpg
    vorige = None
    print("USB-camera gestart, herkenning aan.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame, vorige, _positie, _kleur = verwerk_frame(frame, vorige)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_KWALITEIT])
        if not ok:
            continue
        with slot:
            laatste_jpg = jpg.tobytes()


class StreamHandler(BaseHTTPRequestHandler):
    """Kleine webserver die het laatste beeld als MJPEG-stream doorgeeft.

    MJPEG is niets anders dan losse JPEG's achter elkaar binnen één
    verbinding. De browser houdt die verbinding open en vervangt steeds het
    plaatje, waardoor het video lijkt. Daar is dus geen videobibliotheek
    voor nodig.
    """

    def log_message(self, formaat, *args):
        # Standaard logt deze webserver elke opvraging. Dat zou de
        # coordinaten in de terminal onleesbaar maken, dus we doen niets.
        return

    def do_GET(self):
        """Handelt een verzoek van de browser af. Alles buiten deze paden is 404."""
        if self.path not in ("/", "/stream", "/stream.mjpg"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        # multipart/x-mixed-replace vertelt de browser: er komen steeds
        # nieuwe plaatjes die het vorige vervangen. "boundary=frame" is het
        # woord waarmee we hieronder elk nieuw plaatje aankondigen.
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
                time.sleep(0.03)  # ongeveer 30 beelden per seconde
        except (BrokenPipeError, ConnectionResetError):
            # Browser is weg (tabblad gesloten). Geen fout, gewoon stoppen.
            return


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        print("Dit bestand is voor de Raspberry Pi.")
        print("Op je laptop: start opdracht_amongus_raspberry.py")
        raise SystemExit(1)

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
