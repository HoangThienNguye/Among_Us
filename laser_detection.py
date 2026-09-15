"""
Laser dot detection with OpenCV for the Among Us laser-pointing assignment.

Idea: instead of filtering on color (unreliable, since the Among Us
picture itself also has red/green/bright colors) or a fixed brightness
threshold (unreliable, since that stops being correct as soon as
exposure/hardware settings change), we look at the brightest point in
each frame and take a margin below it. As long as the laser is clearly
brighter than its surroundings, this works regardless of exposure.

Usage:
    python laser_detect.py --camera 0 --debug

As a module:
    from laser_detect import LaserDetector
    det = LaserDetector(camera_index=0)
    x, y = det.get_laser_position()   # None if not found
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
        exposure: int = 0,           # leave at 0; the adaptive threshold does the rest
        min_peak_brightness: int = 40,   # frame must reach at least this peak level
        peak_margin: int = 15,           # how far below the peak still counts as "laser"
        min_area: int = 2,
        max_area: int = 400,
        min_circularity: float = 0.5,
        smoothing_window: int = 5,
    ):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

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
        """Optional hardware hint; the adaptive threshold does the heavy lifting.

        We leave this at 0 (untouched) by default: measurements showed
        that the 'brightness' control on this camera is a simple
        add/subtract on every pixel (not a real exposure/gain control) —
        it pulls the laser down just as much as the background, so it
        doesn't actually solve anything. The adaptive peak-margin
        threshold (see _brightness_mask) is more robust: it looks at the
        brightest point per frame, regardless of how the hardware itself
        is configured.
        """
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual (V4L2), if supported
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, hw_brightness)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)

        print(
            f"[camera] brightness requested={hw_brightness} "
            f"actual={self.cap.get(cv2.CAP_PROP_BRIGHTNESS)}, "
            f"auto_wb actual={self.cap.get(cv2.CAP_PROP_AUTO_WB)}"
        )

    def _read_frame(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    @staticmethod
    def _brightness_map(frame: np.ndarray) -> np.ndarray:
        """Per-pixel brightness, robust for colored (e.g. red) lasers.

        The standard cv2.COLOR_BGR2GRAY conversion uses luminance
        weighting (~0.30*R + 0.59*G + 0.11*B). A saturated red dot
        (R=255,G=0,B=0) ends up at a gray value of only ~76 with that
        formula, far below a threshold like 250 — even though the point
        looks very bright to the eye (or in the color frame). By taking
        the MAXIMUM of the three channels per pixel instead, a saturated
        channel counts equally regardless of which color it is.
        """
        channel_max = np.max(frame, axis=2)
        return cv2.GaussianBlur(channel_max, (5, 5), 0)

    def _brightness_mask(self, gray: np.ndarray) -> tuple[np.ndarray, int]:
        """Adaptive threshold instead of a fixed value.

        A fixed threshold (e.g. 250) breaks as soon as exposure/hardware
        brightness changes: sometimes even the laser doesn't reach it
        (set too dark), sometimes the whole scene reaches it anyway (too
        bright, or the camera compensating during movement). By looking
        at the brightest point of this specific frame and taking a
        margin below it, this works regardless of the exact exposure —
        as long as the laser is clearly brighter than the rest of the
        image.

        Returns (mask, peak); peak is also used to ignore frames with no
        visible laser (peak < min_peak_brightness).
        """
        peak = int(gray.max())
        if peak < self.min_peak_brightness:
            return np.zeros_like(gray, dtype=np.uint8), peak

        threshold = max(peak - self.peak_margin, 0)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return mask, peak

    def _find_candidates(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        """Returns a list of (x, y, circularity) candidates."""
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
        """Returns the (smoothed) pixel position of the laser, or None."""
        frame = self._read_frame()
        if frame is None:
            return None

        candidates = self._find_candidates(frame)
        if not candidates:
            return None

        # Pick the most circular candidate (most likely to be the laser, not noise)
        cx, cy, _ = max(candidates, key=lambda c: c[2])
        self._history.append((cx, cy))

        avg_x = sum(p[0] for p in self._history) / len(self._history)
        avg_y = sum(p[1] for p in self._history) / len(self._history)
        return avg_x, avg_y

    def get_last_frame(self) -> np.ndarray | None:
        """For debug purposes: read and return the latest frame again."""
        return self._read_frame()

    def release(self) -> None:
        self.cap.release()


def run_debug(camera_index: int) -> None:
    """Live debug view: shows the camera feed, the threshold mask, and the detected position."""
    det = LaserDetector(camera_index=camera_index)
    prev_time = time.time()

    try:
        while True:
            frame = det._read_frame()
            if frame is None:
                print("No frame received, stopping.")
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
                    display, "no laser found", (10, 30),
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
    """Writes `count` frames + their threshold mask out as PNGs.

    Use this if you don't have a display/X11: scp the outdir back to your
    own machine and inspect the images. This lets you see exactly what
    the camera is capturing and whether the threshold does/doesn't hit
    the laser, without needing cv2.imshow on the Pi itself.
    """
    import os

    os.makedirs(outdir, exist_ok=True)
    det = LaserDetector(camera_index=camera_index)

    try:
        for i in range(count):
            frame = det._read_frame()
            if frame is None:
                print(f"frame {i}: no frame received")
                continue

            gray = det._brightness_map(frame)
            mask, peak = det._brightness_mask(gray)

            cv2.imwrite(f"{outdir}/frame_{i:02d}.png", frame)
            cv2.imwrite(f"{outdir}/mask_{i:02d}.png", mask)
            print(f"frame {i}: peak brightness={peak}, saved")
            time.sleep(0.3)
    finally:
        det.release()

    print(f"Done. Check the PNGs in {outdir}/ (scp them back to your machine).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laser dot detection debug tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--debug", action="store_true", help="Open a live debug window")
    parser.add_argument(
        "--snapshot", type=int, default=0,
        help="Write N frames+masks to ./snapshots instead of showing them live",
    )
    parser.add_argument(
        "--button", type=int, default=None,
        help="GPIO pin number of a start button. If set, waits for a press "
             "before starting detection (wire the other leg to GND — "
             "gpiozero uses the internal pull-up, no resistor needed).",
    )
    args = parser.parse_args()

    if args.button is not None:
        from gpiozero import Button  # imported lazily: only needed when --button is used

        print(f"Waiting for button press on GPIO{args.button}...")
        Button(args.button).wait_for_press()
        print("Button pressed, starting.")

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