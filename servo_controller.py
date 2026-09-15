import pigpio
import time

servoX = 22  # Servo X op GPIO 22
servoY = 23  # Servo Y op GPIO 23

pwm = pigpio.pi()

if not pwm.connected:
    print("Kan geen verbinding maken met pigpio!")
    exit()


pwm.set_mode(servoX, pigpio.OUTPUT)
pwm.set_mode(servoY, pigpio.OUTPUT)


def servo_off(servo):
    """
    Servo volledig stoppen.
    """
    pwm.set_servo_pulsewidth(servo, 0)
    pwm.set_PWM_dutycycle(servo, 0)
    pwm.set_PWM_frequency(servo, 0)


def servo_move(servo, pulsewidth):
    """
    Servo aansturen met een bepaalde pulsewidth.
    """
    pwm.set_PWM_frequency(servo, 50)
    pwm.set_servo_pulsewidth(servo, pulsewidth)


def angle_to_pulsewidth(angle):
    """
    Zet hoek om naar pulsewidth.

    0 graden   = 500 us
    90 graden  = 1500 us
    180 graden = 2500 us
    """

    pulsewidth = 500 + (angle / 180) * 2000

    return int(pulsewidth)

# Zorg dat beide servo's uit staan voordat we beginnen.
servo_off(servoX)
servo_off(servoY)


# ==========================
# SERVO X: 1 -> 90 GRADEN
# ==========================

print("Servo X begint...")

# Zorg dat Servo Y uit staat.
servo_off(servoY)

angle = 1

while angle <= 90:

    pulsewidth = angle_to_pulsewidth(angle)

    servo_move(servoX, pulsewidth)

    print(f"Servo X: {angle} graden")

    time.sleep(0.1)

    angle += 1


# Servo X uitzetten
servo_off(servoX)

print("Servo X klaar.")


# ==========================
# SERVO Y: 1 -> 90 GRADEN
# ==========================

print("Servo Y begint...")

# Zorg dat Servo X uit staat.
servo_off(servoX)

angle = 1

while angle <= 90:

    pulsewidth = angle_to_pulsewidth(angle)

    servo_move(servoY, pulsewidth)

    print(f"Servo Y: {angle} graden")

    time.sleep(0.1)

    angle += 1


# Servo Y uitzetten
servo_off(servoY)

print("Servo Y klaar.")


# ==========================
# RESET NAAR 90 GRADEN
# ==========================

print("Reset naar 90 graden...")

# Servo X naar 90 graden
servo_off(servoY)

servo_move(servoX, angle_to_pulsewidth(90))
time.sleep(1)

servo_off(servoX)


# Servo Y naar 90 graden
servo_off(servoX)

servo_move(servoY, angle_to_pulsewidth(90))
time.sleep(1)

servo_off(servoY)

print("Beide servo's staan op 90 graden.")

pwm.stop()