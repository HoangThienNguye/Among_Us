"""
Hoofdprogramma voor de Raspberry Pi.

1. Among Us herkennen
2. Laser herkennen (zelfde camerabeeld)
3. Servo's bijsturen tot de laser op de Among Us staat

Aan/uit-knop: GPIO 26 naar GND. Uit = servo's terug naar beginpositie.

Stoppen: Ctrl+C
Stream bekijken: http://thien.local:8080/stream.mjpg
"""

import os
import sys
import time
from http.server import ThreadingHTTPServer
from threading import Lock, Thread

import cv2

import pi_usb_camera_stream as amongus
import servo_controller as servo
from laser_detection import LaserDetector

# X: grotere hoek = laser naar links in beeld (min-teken nodig).
# Y: grotere hoek = laser naar beneden in beeld (geen omkering).
TEKEN_X = -1
TEKEN_Y = 1

START_X = 90.0
START_Y = 90.0
HOEK_MIN = 5.0
HOEK_MAX = 175.0

# Pixels: kleiner dan dit is "raak".
DREMPEL = 12

# Graden per pixel-fout, per frame. Klein houden.
K = 0.02
MAX_STAP = 2.0

# Laser mag niet tot tegen de beeldrand.
RAND = 24

# Alleen sturen na stabiele herkenning. Anders terug naar start.
MIN_HITS_AMONGUS = 5
MIN_HITS_LASER = 4
MAX_LASER_SPRONG = 60
KWIJT_NAAR_START = 10

KNOP_PIN = 26

staat = {
    "aan": False,
    "naar_begin": False,
}
staat_slot = Lock()


def begrens(waarde, laag, hoog):
    return max(laag, min(hoog, waarde))


def sturing_aan():
    with staat_slot:
        return staat["aan"]


def haal_naar_begin():
    with staat_slot:
        if not staat["naar_begin"]:
            return False
        staat["naar_begin"] = False
        return True


def zet_naar_begin(angle_x, angle_y):
    servo.set_angle(servo.servoX, START_X)
    servo.set_angle(servo.servoY, START_Y)
    return START_X, START_Y


def schakel():
    with staat_slot:
        staat["aan"] = not staat["aan"]
        if staat["aan"]:
            print("AAN: servo's volgen de Among Us", flush=True)
        else:
            staat["naar_begin"] = True
            print("UIT: servo's naar beginpositie", flush=True)


def teken_laser(frame, laser_pos):
    if laser_pos is None:
        cv2.putText(
            frame, "laser: --", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )
        return
    lx, ly = laser_pos
    cv2.circle(frame, (int(lx), int(ly)), 8, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"laser: ({int(lx)},{int(ly)})",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def stuur_servos(amongus_pos, laser_pos, angle_x, angle_y):
    ax, ay = amongus_pos
    lx, ly = laser_pos

    x_min, x_max = RAND, amongus.BREEDTE - RAND
    y_min, y_max = RAND, amongus.HOOGTE - RAND

    doel_x = begrens(ax, x_min, x_max)
    doel_y = begrens(ay, y_min, y_max)
    fout_x = doel_x - lx
    fout_y = doel_y - ly

    stap_x = begrens(TEKEN_X * K * fout_x, -MAX_STAP, MAX_STAP) if abs(fout_x) > DREMPEL else 0.0
    stap_y = begrens(TEKEN_Y * K * fout_y, -MAX_STAP, MAX_STAP) if abs(fout_y) > DREMPEL else 0.0

    # TEKEN * stap = voorspelde laserbeweging in beeld (rechts/omlaag positief).
    beeld_dx = TEKEN_X * stap_x
    beeld_dy = TEKEN_Y * stap_y
    if beeld_dx < 0 and lx <= x_min:
        stap_x = 0.0
    if beeld_dx > 0 and lx >= x_max:
        stap_x = 0.0
    if beeld_dy < 0 and ly <= y_min:
        stap_y = 0.0
    if beeld_dy > 0 and ly >= y_max:
        stap_y = 0.0

    if stap_x != 0.0:
        angle_x = begrens(angle_x + stap_x, HOEK_MIN, HOEK_MAX)
        servo.set_angle(servo.servoX, angle_x)
    if stap_y != 0.0:
        angle_y = begrens(angle_y + stap_y, HOEK_MIN, HOEK_MAX)
        servo.set_angle(servo.servoY, angle_y)

    return angle_x, angle_y, fout_x, fout_y


def camera_lus(cap, laser, start_x, start_y):
    vorige = None
    angle_x = start_x
    angle_y = start_y
    laatste_melding = None
    hits_amongus = 0
    hits_laser = 0
    frames_kwijt = 0
    vorige_laser = None
    print("Camera gestart. Volgen alleen bij een echte Among Us + laserstip.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        if haal_naar_begin():
            angle_x, angle_y = zet_naar_begin(angle_x, angle_y)
            hits_amongus = 0
            hits_laser = 0
            frames_kwijt = 0
            vorige_laser = None

        schoon = cv2.resize(frame, (amongus.BREEDTE, amongus.HOOGTE))

        beeld, vorige, amongus_pos, kleur = amongus.verwerk_frame(schoon.copy(), vorige)
        laser_pos = laser.detect_on_frame(schoon)

        if amongus_pos is None:
            hits_amongus = 0
            frames_kwijt += 1
        else:
            hits_amongus += 1
            frames_kwijt = 0

        if laser_pos is None:
            hits_laser = 0
            vorige_laser = None
        else:
            if vorige_laser is not None:
                sprong = (
                    (laser_pos[0] - vorige_laser[0]) ** 2
                    + (laser_pos[1] - vorige_laser[1]) ** 2
                ) ** 0.5
                if sprong > MAX_LASER_SPRONG:
                    laser_pos = None
                    hits_laser = 0
                    vorige_laser = None
                else:
                    hits_laser += 1
                    vorige_laser = laser_pos
            else:
                hits_laser += 1
                vorige_laser = laser_pos

        teken_laser(beeld, laser_pos)
        cv2.rectangle(
            beeld,
            (RAND, RAND),
            (amongus.BREEDTE - RAND, amongus.HOOGTE - RAND),
            (80, 80, 80),
            1,
        )

        aan = sturing_aan()
        cv2.putText(
            beeld,
            "AAN" if aan else "UIT",
            (beeld.shape[1] - 90, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if aan else (0, 0, 255),
            2,
        )

        mag_sturen = (
            aan
            and amongus_pos is not None
            and laser_pos is not None
            and hits_amongus >= MIN_HITS_AMONGUS
            and hits_laser >= MIN_HITS_LASER
        )

        if mag_sturen:
            angle_x, angle_y, dx, dy = stuur_servos(
                amongus_pos, laser_pos, angle_x, angle_y
            )
            print(
                f"{kleur}  amongus=({amongus_pos[0]:.0f},{amongus_pos[1]:.0f})  "
                f"laser=({laser_pos[0]:.0f},{laser_pos[1]:.0f})  "
                f"dx={dx:.0f} dy={dy:.0f}  "
                f"servo=({angle_x:.1f},{angle_y:.1f})",
                flush=True,
            )
            laatste_melding = "sturen"
        elif aan and amongus_pos is None and frames_kwijt == KWIJT_NAAR_START:
            angle_x, angle_y = zet_naar_begin(angle_x, angle_y)
            print("Among Us weg, terug naar startpositie", flush=True)
            laatste_melding = "start"
        elif amongus_pos is None:
            if laatste_melding != "geen amongus":
                print("geen Among Us — servo blijft/gaat naar start", flush=True)
                laatste_melding = "geen amongus"
        else:
            if laatste_melding != "geen laser":
                print("Among Us gezien, geen geldige laserstip", flush=True)
                laatste_melding = "geen laser"

        cv2.putText(
            beeld, f"servo X={angle_x:.0f} Y={angle_y:.0f}",
            (10, beeld.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        ok, jpg = cv2.imencode(
            ".jpg", beeld, [int(cv2.IMWRITE_JPEG_QUALITY), amongus.JPEG_KWALITEIT]
        )
        if ok:
            with amongus.slot:
                amongus.laatste_jpg = jpg.tobytes()


def main():
    if sys.platform.startswith("win"):
        print("Dit programma is voor de Raspberry Pi.")
        print("Kopieer de map naar de Pi en start daar: python3 main.py")
        raise SystemExit(1)

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

    cap = None
    try:
        laser = LaserDetector(
            open_camera=False,
            min_peak_brightness=180,
            peak_margin=10,
            min_area=2,
            max_area=80,
            min_circularity=0.65,
            smoothing_window=3,
        )
        cap = amongus.open_camera()
        LaserDetector.configure_capture(cap)

        servo.connect()
        servo.set_angle(servo.servoX, START_X)
        servo.set_angle(servo.servoY, START_Y)
        time.sleep(0.3)

        from gpiozero import Button

        knop = Button(
            KNOP_PIN,
            pull_up=True,
            bounce_time=0.2,
            pin_factory=servo.pin_factory(),
        )
        knop.when_pressed = schakel
        print(f"Aan/uit-knop: GPIO {KNOP_PIN} naar GND. Start in UIT.")

        Thread(
            target=camera_lus,
            args=(cap, laser, START_X, START_Y),
            daemon=True,
        ).start()

        os.system(f"fuser -k {amongus.POORT}/tcp >/dev/null 2>&1")
        time.sleep(0.8)
        ThreadingHTTPServer.allow_reuse_address = True
        server = ThreadingHTTPServer(("0.0.0.0", amongus.POORT), amongus.StreamHandler)
        print(f"Camera-stream: http://thien.local:{amongus.POORT}/stream.mjpg")
        print("Among Us eerst, dan laser, dan servo's. Stoppen: Ctrl+C")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGestopt.")
    finally:
        servo.stop()
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
