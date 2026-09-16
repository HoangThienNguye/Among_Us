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
    from laser_detection import LaserDetector
    det = LaserDetector(camera_index=0)
    x, y = det.get_laser_position()   # None if not found

    # Same frame as another detector (no extra camera):
    det = LaserDetector(open_camera=False)
    x, y = det.detect_on_frame(frame)
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
        exposure: int = 0,           # leave at 0; local-contrast detection does the rest
        background_blur: int = 41,       # kernel size for estimating local background
        min_peak_brightness: int = 25,   # min local contrast (not raw brightness) to consider
        peak_margin: int = 10,           # how far below the strongest local contrast still counts
        min_area: int = 2,
        max_area: int = 400,
        min_circularity: float = 0.6,
        smoothing_window: int = 5,
        miss_tolerance: int = 3,     # keep last known position for this many consecutive misses
        roi: tuple[int, int, int, int] | None = None,  # (x1, y1, x2, y2): only look inside this box
        open_camera: bool = True,
    ):
        self.min_peak_brightness = min_peak_brightness
        self.peak_margin = peak_margin
        self.background_blur = background_blur | 1  # must be odd for cv2.blur/GaussianBlur-style kernels
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.miss_tolerance = miss_tolerance
        self.roi = roi
        self._miss_count = 0
        self._last_valid: tuple[float, float] | None = None
        self._history: deque[tuple[float, float]] = deque(maxlen=smoothing_window)

        self.cap = None
        if not open_camera:
            return

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self._configure_exposure(exposure)

    @staticmethod
    def configure_capture(cap, hw_brightness: int = 0) -> None:
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
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual (V4L2), if supported
        cap.set(cv2.CAP_PROP_BRIGHTNESS, hw_brightness)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_BACKLIGHT, 0)

        print(
            f"[camera] brightness requested={hw_brightness} "
            f"actual={cap.get(cv2.CAP_PROP_BRIGHTNESS)}, "
            f"auto_wb actual={cap.get(cv2.CAP_PROP_AUTO_WB)}"
        )

    def _configure_exposure(self, hw_brightness: int) -> None:
        if self.cap is not None:
            self.configure_capture(self.cap, hw_brightness)

    def _read_frame(self) -> np.ndarray | None:
        if self.cap is None:
            return None
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
        """Local-contrast threshold instead of a global-peak-relative one.

        Comparing every pixel to the frame's single brightest pixel breaks
        down on an evenly, brightly lit background (e.g. a well-lit white
        wall): the whole wall sits close to the peak, so any margin wide
        enough to also catch a somewhat-fainter laser ends up including
        the entire wall too — there's nothing to tell "uniformly bright"
        apart from "one bright spot" if you only ever look at the global
        max.

        Instead, estimate a local background level per pixel (a heavy
        blur — the average of a fairly large neighborhood) and look at
        how much each pixel exceeds ITS OWN neighborhood. A laser dot
        stands out from its immediate surroundings regardless of whether
        the overall scene is bright or dark; a uniformly lit wall does
        not stand out from its own neighborhood at all, so it disappears
        from this mask even though it may be close to the global peak
        brightness.

        Returns (mask, peak_contrast); peak_contrast is also used to
        ignore frames where nothing stands out locally at all.
        """
        background = cv2.blur(gray, (self.background_blur, self.background_blur))
        contrast = cv2.subtract(gray, background)  # clips to 0, so only "brighter than local bg" survives

        peak = int(contrast.max())
        if peak < self.min_peak_brightness:
            return np.zeros_like(gray, dtype=np.uint8), peak

        threshold = max(peak - self.peak_margin, 0)
        _, mask = cv2.threshold(contrast, threshold, 255, cv2.THRESH_BINARY)
        return mask, peak

    def _find_candidates(self, frame: np.ndarray) -> list[tuple[float, float, float]]:
        """Returns a list of (x, y, circularity) candidates, in full-frame coordinates.

        No morphological opening here: the laser dot is often only a
        handful of pixels after local-contrast subtraction, and an
        opening step (erosion+dilation) with a kernel that size erases
        genuine tiny detections entirely rather than cleaning up noise.
        Closing (to fill small holes) is harmless and kept.

        min_circularity matters more than it might seem at this scale:
        printed edges/outlines (e.g. a picture taped to the wall) can
        produce small curved-line fragments that pass the brightness
        mask too. A short curved line segment has much lower circularity
        (empirically ~0.1-0.6 in testing) than an actual filled laser dot
        (~0.8-0.9), so a cutoff around 0.6 cleanly rejects those edge
        fragments while keeping genuine dot detections. min_area/max_area
        additionally reject large irregular bright regions (e.g. a lit
        window).

        If `roi` is set, only that sub-region is analyzed at all — useful
        for excluding a known competing bright feature (e.g. a reflective
        clock face) that would otherwise win over the actual laser purely
        because it has more local contrast. Coordinates are converted back
        to full-frame terms before being returned.
        """
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            sub_frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            sub_frame = frame
            offset_x, offset_y = 0, 0

        gray = self._brightness_map(sub_frame)
        mask, _ = self._brightness_mask(gray)
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
            cx = M["m10"] / M["m00"] + offset_x
            cy = M["m01"] / M["m00"] + offset_y
            candidates.append((cx, cy, circularity))

        return candidates

    def detect_on_frame(self, frame: np.ndarray) -> tuple[float, float] | None:
        """Laser position in an existing frame, without opening an extra camera.

        Tolerates a few consecutive missed frames (miss_tolerance) before
        actually reporting None: a real laser that's genuinely detectable
        can still momentarily fall just under the threshold on a single
        frame due to sensor noise/exposure jitter, which otherwise shows
        up as random-looking flicker even though nothing really changed.
        """
        if frame is None or frame.size == 0:
            return None

        candidates = self._find_candidates(frame)
        if not candidates:
            self._miss_count += 1
            if self._last_valid is not None and self._miss_count <= self.miss_tolerance:
                return self._last_valid
            self._history.clear()
            self._last_valid = None
            return None

        # Pick the most circular candidate (most likely to be the laser, not noise)
        cx, cy, _ = max(candidates, key=lambda c: c[2])
        self._miss_count = 0
        self._history.append((cx, cy))

        avg_x = sum(p[0] for p in self._history) / len(self._history)
        avg_y = sum(p[1] for p in self._history) / len(self._history)
        self._last_valid = (avg_x, avg_y)
        return self._last_valid

    def get_laser_position(self) -> tuple[float, float] | None:
        """Returns the (smoothed) pixel position of the laser, or None."""
        frame = self._read_frame()
        if frame is None:
            return None
        return self.detect_on_frame(frame)

    def get_last_frame(self) -> np.ndarray | None:
        """For debug purposes: read and return the latest frame again."""
        return self._read_frame()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def run_debug(camera_index: int, **detector_kwargs) -> None:
    """Live debug view: shows the camera feed, the threshold mask, and the detected position."""
    det = LaserDetector(camera_index=camera_index, **detector_kwargs)
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


def run_snapshot(camera_index: int, count: int, outdir: str, **detector_kwargs) -> None:
    """Writes `count` frames + their threshold mask + an annotated overlay as PNGs.

    Use this if you don't have a display/X11: scp the outdir back to your
    own machine and inspect the images. `frame_XX.png` is the raw camera
    frame, `mask_XX.png` is the pre-filter brightness mask (can include
    blobs that later get rejected — e.g. an overexposed window/light), and
    `annotated_XX.png` shows the actual final decision: a green circle at
    the (x, y) get_laser_position() would return, or a red "no candidate"
    label if nothing passed the area/circularity filters. That last file
    is the one that tells you whether the code picked the right spot.
    """
    import os

    os.makedirs(outdir, exist_ok=True)
    det = LaserDetector(camera_index=camera_index, **detector_kwargs)

    try:
        for i in range(count):
            frame = det._read_frame()
            if frame is None:
                print(f"frame {i}: no frame received")
                continue

            gray = det._brightness_map(frame)
            mask, peak = det._brightness_mask(gray)
            candidates = det._find_candidates(frame)

            annotated = frame.copy()
            if candidates:
                cx, cy, circ = max(candidates, key=lambda c: c[2])
                cv2.circle(annotated, (int(cx), int(cy)), 10, (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"({cx:.0f},{cy:.0f}) circ={circ:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    annotated, "no candidate passed filters",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            cv2.imwrite(f"{outdir}/frame_{i:02d}.png", frame)
            cv2.imwrite(f"{outdir}/mask_{i:02d}.png", mask)
            cv2.imwrite(f"{outdir}/annotated_{i:02d}.png", annotated)
            print(f"frame {i}: peak brightness={peak}, candidates={len(candidates)}, saved")
            time.sleep(0.3)
    finally:
        det.release()

    print(f"Done. Check the PNGs in {outdir}/ (scp them back to your machine).")


def run_with_toggle_button(camera_index: int, button_pin: int, **detector_kwargs) -> None:
    """Press the button to start detection, press again to stop — repeatable.

    Uses gpiozero's when_pressed callback instead of a blocking wait, since
    we need to react to a second press while the detection loop is already
    running. `running` is flipped on each press; the loop only calls
    get_laser_position() while `running` is True, otherwise it idles.
    """
    from gpiozero import Button  # imported lazily: only needed in this mode

    state = {"running": False}

    def toggle() -> None:
        state["running"] = not state["running"]
        print("Started detection" if state["running"] else "Stopped detection")

    button = Button(button_pin, bounce_time=0.2)
    button.when_pressed = toggle

    print(f"Ready. Press the button on GPIO{button_pin} to start/stop.")
    detector = LaserDetector(camera_index=camera_index, **detector_kwargs)
    try:
        while True:
            if state["running"]:
                print(detector.get_laser_position())
                time.sleep(0.05)
            else:
                time.sleep(0.1)  # idle, just waiting for a press
    except KeyboardInterrupt:
        pass
    finally:
        detector.release()


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
        help="GPIO pin number of a start/stop button (toggle). Wire the "
             "other leg to GND — gpiozero uses the internal pull-up, no "
             "resistor needed. Only applies to the normal run mode, not "
             "--debug or --snapshot.",
    )
    parser.add_argument(
        "--peak-margin", type=int, default=10,
        help="How far below the frame's brightest point still counts as "
             "'laser'. Raise this (try 40-60) if the laser gets missed "
             "when something else in view (e.g. a ceiling light) is "
             "brighter than the laser itself.",
    )
    parser.add_argument(
        "--min-peak", type=int, default=25,
        help="Minimum local contrast (not raw brightness) before we "
             "even look for a laser. Lower this if the laser's contrast "
             "against its background is just generally weak.",
    )
    parser.add_argument(
        "--background-blur", type=int, default=41,
        help="Size of the neighborhood (in pixels) used to estimate local "
             "background brightness. Larger tolerates broader smooth "
             "lighting gradients; smaller reacts more locally but can be "
             "noisier. Must end up odd (even values get +1 automatically).",
    )
    parser.add_argument(
        "--miss-tolerance", type=int, default=3,
        help="Number of consecutive frames the last known laser position "
             "is kept before reporting 'not found'. Raise this if "
             "detection flickers on/off frame-to-frame even though the "
             "laser hasn't actually moved or disappeared.",
    )
    parser.add_argument(
        "--roi", type=int, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
        help="Only look for the laser inside this box (x1 y1 x2 y2, pixel "
             "coordinates). Use this to exclude a known competing bright "
             "feature in the shot (e.g. a reflective clock face) that "
             "would otherwise out-compete the actual laser for local "
             "contrast. Check annotated_XX.png from --snapshot to read "
             "off good coordinates.",
    )
    parser.add_argument(
        "--min-circularity", type=float, default=0.6,
        help="Minimum roundness (0-1) a bright blob must have to count as "
             "the laser. Printed edges/outlines (e.g. taped-up pictures) "
             "can create small curved fragments that are bright enough "
             "but not round; a real dot measured ~0.8-0.9 in testing "
             "versus ~0.1-0.6 for edge fragments. Lower this only if the "
             "real dot itself is being rejected (check circ= in "
             "annotated_XX.png from --snapshot).",
    )
    args = parser.parse_args()

    detector_kwargs = {
        "peak_margin": args.peak_margin,
        "min_peak_brightness": args.min_peak,
        "background_blur": args.background_blur,
        "miss_tolerance": args.miss_tolerance,
        "roi": tuple(args.roi) if args.roi else None,
        "min_circularity": args.min_circularity,
    }

    if args.snapshot:
        run_snapshot(args.camera, args.snapshot, "snapshots", **detector_kwargs)
    elif args.debug:
        run_debug(args.camera, **detector_kwargs)
    elif args.button is not None:
        run_with_toggle_button(args.camera, args.button, **detector_kwargs)
    else:
        detector = LaserDetector(camera_index=args.camera, **detector_kwargs)
        try:
            while True:
                pos = detector.get_laser_position()
                print(pos)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            detector.release()