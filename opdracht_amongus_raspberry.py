"""
Toont de USB-camera van de Raspberry Pi, inclusief Among Us-vakjes.

De herkenning gebeurt op de Pi. Dit script laat alleen het beeld zien.
Stoppen: q
"""

import time
import urllib.request

import cv2
import numpy as np

PI_HOST = "thien.local"
PI_POORT = 8080
STREAM_URL = f"http://{PI_HOST}:{PI_POORT}/stream.mjpg"


def frames_van_pi(url):
    print("Verbinden met Raspberry Pi:", url)
    print("Kijk in het venster 'Among Us (Raspberry Pi)'.")
    while True:
        try:
            stream = urllib.request.urlopen(url, timeout=8)
            print("Verbonden. Je moet linksboven 'match: ...%' zien.")
            data = b""
            while True:
                stuk = stream.read(4096)
                if not stuk:
                    raise ConnectionError("Stream gestopt.")
                data += stuk
                start = data.find(b"\xff\xd8")
                einde = data.find(b"\xff\xd9")
                if start == -1 or einde == -1 or einde < start:
                    if len(data) > 2_000_000:
                        data = b""
                    continue
                jpg = data[start:einde + 2]
                data = data[einde + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
        except Exception as fout:
            print("Geen beeld van de Pi:", fout)
            print("De streamer op de Raspberry draait waarschijnlijk niet.")
            time.sleep(2)


if __name__ == "__main__":
    cv2.namedWindow("Among Us (Raspberry Pi)", cv2.WINDOW_NORMAL)
    for frame in frames_van_pi(STREAM_URL):
        cv2.imshow("Among Us (Raspberry Pi)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
