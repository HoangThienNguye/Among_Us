"""
Laserstip-detectie met OpenCV voor de Among Us laser-wijs-opdracht.

Idee: i.p.v. op kleur te filteren (onbetrouwbaar, want het Among Us
plaatje bevat zelf ook rode/groene/felle kleuren) of op een vaste
helderheids-drempel (onbetrouwbaar, want die klopt niet meer zodra de
belichting/hardware-instellingen veranderen), kijken we per frame naar
het felste punt en pakken daar een marge onder. Zolang de laser
duidelijk feller is dan zijn omgeving, werkt dit ongeacht belichting.

Gebruik:
    python laser_detect.py --camera 0 --debug

Als module:
    from laser_detect import LaserDetector
    det = LaserDetector(camera_index=0)
    x, y = det.get_laser_position()   # None als niet gevonden
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np


class LaserDetector:
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        exposure: int = 0,           # laat op 0; adaptieve threshold regelt de rest
        min_peak_brightness: int = 40,   # frame moet minstens dit peak-niveau halen
        peak_margin: int = 15,           # hoeveel onder de piek nog meetelt als "laser"
        min_area: int = 2,
        max_area: int = 400,
        min_circularity: float = 0.5,
        smoothing_window: int = 5,
    ):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kan camera {camera_index} niet openen")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self._configure_exposure(exposure)

        self.min_peak_brightness = min_peak_brightness
        self.peak_margin = peak_margin
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

        self._history: deque[tuple[float, float]] = deque(maxlen=smoothing_window)

    def _configure_exposure(self, hw_brightness: int) -> None:
        """Optionele hardware-hint; de adaptieve threshold doet het zware werk.

        We laten dit standaard op 0 (ongemoeid) staan: uit metingen bleek
        dat de 'brightness'-control op deze camera een simpele optelling/
        aftrekking op elke pixel is (geen echte exposure/gain) — die trekt
        de laser even hard omlaag als de achtergrond, en lost dus niets op.
        De adaptieve peak-margin threshold (zie _brightness_mask) is
        robuuster: die kijkt per frame naar het felste punt, ongeacht hoe
        de hardware verder staat ingesteld.
        """
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual (V4L2), indien aanwezig
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, hw_brightness)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)

        print(
            f"[camera] brightness gevraagd={hw_brightness} "
            f"werkelijk={self.cap.get(cv2.CAP_PROP_BRIGHTNESS)}, "
            f"auto_wb werkelijk={self.cap.get(cv2.CAP_PROP_AUTO_WB)}"
        )

    def _read_frame(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    @staticmethod
    def _brightness_map(frame: np.ndarray) -> np.ndarray:
        """Per-pixel helderheid, robuust voor gekleurde (bv. rode) lasers.

        Standaard cv2.COLOR_BGR2GRAY gebruikt luminance-weging
        (~0.30*R + 0.59*G + 0.11*B). Een verzadigd rood puntje
        (R=255,G=0,B=0) komt daarmee op grijswaarde ~76 uit, ver onder
        een threshold als 250 — terwijl het punt met het blote oog
        (of in de kleurenframe) juist heel fel is. Door per pixel het
        MAXIMUM van de drie kanalen te pakken, telt een verzadigd
        kanaal in elke kleur even zwaar mee.
        """
        channel_max = np.max(frame, axis=2)
        return cv2.GaussianBlur(channel_max, (5, 5), 0)

    def _brightness_mask(self, gray: np.ndarray) -> tuple[np.ndarray, int]:
        """Adaptieve threshold i.p.v. een vaste waarde.

        Een vaste drempel (bv. 250) faalt zodra de belichting/hardware-
        brightness verandert: soms haalt zelfs de laser die waarde niet
        (te donker ingesteld), soms haalt de hele omgeving 'm juist wel
        (te licht, of camera compenseert bij beweging). Door te kijken
        naar het felste punt ván dit specifieke frame, en daar een marge
        onder te pakken, werkt dit ongeacht de exacte belichting — zolang
        de laser duidelijk feller is dan de rest van het beeld.

        Geeft (mask, peak) terug; peak wordt ook gebruikt om frames zonder
        zichtbare laser te negeren (peak < min_peak_brightness).
        """
        peak = int(gray.max())
        if peak < self.min_peak_brightness:
            return np.zeros_like(gray, dtype=np.uint8), peak

        threshold = max(peak - self.peak_margin, 0)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return mask, peak

    def _find_candidates(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        """Geeft lijst van (x, y, circularity) kandidaten terug."""
        gray = self._brightness_map(frame)
        mask, _ = self._brightness_mask(gray)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            candidates.append((cx, cy, circularity))

        return candidates

    def get_laser_position(self) -> tuple[float, float] | None:
        """Geeft de (gesmoothde) pixelpositie van de laser terug, of None."""
        frame = self._read_frame()
        if frame is None:
            return None

        candidates = self._find_candidates(frame)
        if not candidates:
            return None

        # Pak de meest cirkelvormige kandidaat (grootste kans op laser i.p.v. ruis)
        cx, cy, _ = max(candidates, key=lambda c: c[2])
        self._history.append((cx, cy))

        avg_x = sum(p[0] for p in self._history) / len(self._history)
        avg_y = sum(p[1] for p in self._history) / len(self._history)
        return avg_x, avg_y

    def get_last_frame(self) -> np.ndarray | None:
        """Voor debug-doeleinden: laatst gelezen frame opnieuw ophalen."""
        return self._read_frame()

    def release(self) -> None:
        self.cap.release()


def run_debug(camera_index: int) -> None:
    """Live debug-view: toont de camera, de threshold-mask, en de gedetecteerde positie."""
    det = LaserDetector(camera_index=camera_index)
    prev_time = time.time()

    try:
        while True:
            frame = det._read_frame()
            if frame is None:
                print("Geen frame ontvangen, stoppen.")
                break

            candidates = det._find_candidates(frame)
            pos = None
            if candidates:
                cx, cy, _ = max(candidates, key=lambda c: c[2])
                det._history.append((cx, cy))
                avg_x = sum(p[0] for p in det._history) / len(det._history)
                avg_y = sum(p[1] for p in det._history) / len(det._history)
                pos = (avg_x, avg_y)

            display = frame.copy()
            if pos:
                cv2.circle(display, (int(pos[0]), int(pos[1])), 8, (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"laser: ({pos[0]:.0f}, {pos[1]:.0f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display, "geen laser gevonden", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            now = time.time()
            fps = 1 / (now - prev_time) if now != prev_time else 0
            prev_time = now
            cv2.putText(
                display, f"{fps:.1f} fps", (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

            cv2.imshow("camera", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        det.release()
        cv2.destroyAllWindows()


def run_snapshot(camera_index: int, count: int, outdir: str) -> None:
    """Schrijft `count` frames + hun threshold-mask weg als PNG's.

    Gebruik dit als je geen scherm/X11 hebt: scp de outdir terug naar je
    eigen pc en bekijk de plaatjes. Zo zie je precies wat de camera opneemt
    en of de threshold de laser wel/niet raakt, zonder cv2.imshow nodig te
    hebben op de Pi zelf.
    """
    import os

    os.makedirs(outdir, exist_ok=True)
    det = LaserDetector(camera_index=camera_index)

    try:
        for i in range(count):
            frame = det._read_frame()
            if frame is None:
                print(f"frame {i}: geen frame ontvangen")
                continue

            gray = det._brightness_map(frame)
            mask, peak = det._brightness_mask(gray)

            cv2.imwrite(f"{outdir}/frame_{i:02d}.png", frame)
            cv2.imwrite(f"{outdir}/mask_{i:02d}.png", mask)
            print(f"frame {i}: piek-helderheid={peak}, opgeslagen")
            time.sleep(0.3)
    finally:
        det.release()

    print(f"Klaar. Bekijk de PNG's in {outdir}/ (scp terug naar je pc).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laserstip-detectie debug tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--debug", action="store_true", help="Open live debug-venster")
    parser.add_argument(
        "--snapshot", type=int, default=0,
        help="Aantal frames+masks wegschrijven naar ./snapshots i.p.v. live tonen",
    )
    args = parser.parse_args()

    if args.snapshot:
        run_snapshot(args.camera, args.snapshot, "snapshots")
    elif args.debug:
        run_debug(args.camera)
    else:
        detector = LaserDetector(camera_index=args.camera)
        try:
            while True:
                pos = detector.get_laser_position()
                print(pos)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            detector.release()