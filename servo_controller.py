import time

try:
    import pigpio
except ImportError:
    pigpio = None

servoX = 22  # Servo X op GPIO 22
servoY = 23  # Servo Y op GPIO 23

pwm = None
_pigpio_factory = None


def connect():
    """Verbind met pigpiod. Alleen aanroepen op de Raspberry Pi."""
    global pwm
    if pigpio is None:
        raise RuntimeError("pigpio is niet geinstalleerd. Installeer python3-pigpio en start pigpiod.")
    if pwm is not None and pwm.connected:
        return pwm

    pwm = pigpio.pi()
    if not pwm.connected:
        raise RuntimeError(
            "Kan geen verbinding maken met pigpio. "
            "Start de daemon eerst: sudo pigpiod"
        )

    pwm.set_mode(servoX, pigpio.OUTPUT)
    pwm.set_mode(servoY, pigpio.OUTPUT)
    return pwm


def pin_factory():
    """gpiozero-knop over dezelfde pigpio-daemon als de servo's."""
    global _pigpio_factory
    connect()
    if _pigpio_factory is None:
        from gpiozero.pins.pigpio import PiGPIOFactory
        _pigpio_factory = PiGPIOFactory()
    return _pigpio_factory


def _require_pwm():
    if pwm is None or not pwm.connected:
        connect()
    return pwm


def servo_off(servo):
    """Servo-pulsen stoppen."""
    p = _require_pwm()
    p.set_servo_pulsewidth(servo, 0)
    p.set_PWM_dutycycle(servo, 0)
    p.set_PWM_frequency(servo, 0)


def servo_move(servo, pulsewidth):
    """Servo aansturen met pulsewidth in microseconden."""
    p = _require_pwm()
    p.set_PWM_frequency(servo, 50)
    p.set_servo_pulsewidth(servo, pulsewidth)


def angle_to_pulsewidth(angle):
    """
    Zet hoek om naar pulsewidth.

    0 graden   = 500 us
    90 graden  = 1500 us
    180 graden = 2500 us
    """
    pulsewidth = 500 + (angle / 180) * 2000
    return int(pulsewidth)


def set_angle(servo, angle):
    """Zet een servo op een hoek (0-180) en laat hem daar staan."""
    angle = max(0.0, min(180.0, float(angle)))
    servo_move(servo, angle_to_pulsewidth(angle))
    return angle


def stop():
    """Beide servo's uitzetten en pigpio loslaten."""
    global pwm, _pigpio_factory
    if pwm is not None:
        try:
            servo_off(servoX)
            servo_off(servoY)
            pwm.stop()
        except Exception:
            pass
    pwm = None
    _pigpio_factory = None


if __name__ == "__main__":
    connect()

    print("Servo X begint...")
    angle = 1
    while angle <= 90:
        set_angle(servoX, angle)
        print(f"Servo X: {angle} graden")
        time.sleep(0.1)
        angle += 1
    print("Servo X klaar.")

    print("Servo Y begint...")
    angle = 1
    while angle <= 90:
        set_angle(servoY, angle)
        print(f"Servo Y: {angle} graden")
        time.sleep(0.1)
        angle += 1
    print("Servo Y klaar.")

    print("Reset naar 90 graden...")
    set_angle(servoX, 90)
    time.sleep(1)
    set_angle(servoY, 90)
    time.sleep(1)
    print("Beide servo's staan op 90 graden.")
    stop()
