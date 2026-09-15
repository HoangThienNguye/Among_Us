"""
Among Us handmatig aan- en uitzetten. Start niet vanzelf bij opstarten.

Op de Pi, in de console:

    python3 bestuur.py start
    python3 bestuur.py stop
    python3 bestuur.py status

Of interactief (daarna start / stop / status / q typen):

    python3 bestuur.py
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MAP = Path(__file__).resolve().parent
MAIN = MAP / "main.py"
PID_BESTAND = MAP / "amongus.pid"
LOG_BESTAND = MAP / "amongus.log"


def _pid():
    if not PID_BESTAND.exists():
        return None
    try:
        pid = int(PID_BESTAND.read_text().strip())
    except ValueError:
        PID_BESTAND.unlink(missing_ok=True)
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        PID_BESTAND.unlink(missing_ok=True)
        return None
    return pid


def status():
    pid = _pid()
    if pid is None:
        print("Among Us staat UIT.")
        return False
    print(f"Among Us staat AAN (pid {pid}).")
    return True


def _pigpiod_aan():
    """Zorg dat de pigpio-daemon draait voordat main.py de servo's gebruikt."""
    try:
        import pigpio
        pi = pigpio.pi()
        ok = bool(pi.connected)
        pi.stop()
        if ok:
            return
    except Exception:
        pass

    subprocess.run(["sudo", "-n", "pigpiod"], check=False)
    time.sleep(0.4)


def start():
    if _pid() is not None:
        print("Among Us draait al.")
        return

    if not MAIN.exists():
        print("main.py niet gevonden in", MAP)
        raise SystemExit(1)

    _pigpiod_aan()

    with open(LOG_BESTAND, "a", encoding="utf-8") as log:
        log.write("\n--- start ---\n")
        log.flush()
        proc = subprocess.Popen(
            [sys.executable, "-u", str(MAIN)],
            cwd=str(MAP),
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    PID_BESTAND.write_text(str(proc.pid), encoding="utf-8")

    ok = False
    for _ in range(25):
        time.sleep(0.4)
        if proc.poll() is not None:
            PID_BESTAND.unlink(missing_ok=True)
            print("Starten mislukt. Laatste log:")
            try:
                print(LOG_BESTAND.read_text(encoding="utf-8", errors="replace")[-1500:])
            except OSError:
                pass
            raise SystemExit(1)
        try:
            staart = LOG_BESTAND.read_text(encoding="utf-8", errors="replace")[-2500:]
        except OSError:
            continue
        if "Camera-stream:" in staart or "Aan/uit-knop" in staart:
            ok = True
            break
    if not ok and proc.poll() is None:
        ok = True
    if not ok:
        print("Starten duurde te lang. Probeer: python3 bestuur.py status")
        return
    print("Among Us AAN.")
    print("Stream: http://thien.local:8080/stream.mjpg")
    print("Uitzetten: stop")


def stop():
    pid = _pid()
    if pid is None:
        subprocess.run(["pkill", "-f", "/home/thien/main.py"], check=False)
        PID_BESTAND.unlink(missing_ok=True)
        print("Among Us staat al UIT.")
        return

    try:
        os.kill(pid, signal.SIGINT)
    except OSError:
        pass

    for _ in range(20):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.2)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    subprocess.run(["pkill", "-f", "/home/thien/main.py"], check=False)
    PID_BESTAND.unlink(missing_ok=True)
    print("Among Us UIT.")


def helptekst():
    print("Typ: start  of  stop  of  status  of  q")


def interactief():
    helptekst()
    while True:
        try:
            cmd = input("amongus> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if cmd in ("start", "aan", "a"):
            start()
        elif cmd in ("stop", "uit", "u"):
            stop()
        elif cmd in ("status", "s"):
            status()
        elif cmd in ("q", "quit", "exit"):
            return
        elif cmd:
            helptekst()


def main():
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        interactief()
        return
    if args[0] in ("start", "aan"):
        start()
    elif args[0] in ("stop", "uit"):
        stop()
    elif args[0] in ("status",):
        status()
    else:
        print("Gebruik: python3 bestuur.py [start|stop|status]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
