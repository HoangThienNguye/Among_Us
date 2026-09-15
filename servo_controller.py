#!/usr/bin/python3
import pigpio
import time

servoX = 22  # GPIO 22
servoY = 23  # GPIO 23

pwm = pigpio.pi()

if not pwm.connected:
    print("Kan geen verbinding maken met pigpio!")
    exit()

pwm.set_mode(servoX, pigpio.OUTPUT)
pwm.set_mode(servoY, pigpio.OUTPUT)


def servo_off(servo):
    pwm.set_servo_pulsewidth(servo, 0)
    pwm.set_PWM_dutycycle(servo, 0)
    pwm.set_PWM_frequency(servo, 0)


def servo_on(servo, pulsewidth):
    # first we turn off both servo's
    servo_off(servoX)
    servo_off(servoY)

    # turn on requested servo
    pwm.set_PWM_frequency(servo, 50)
    pwm.set_servo_pulsewidth(servo, pulsewidth)


# ==========================
# SERVO Y
# ==========================

print("Servo Y - 0 graden")
servo_on(servoY, 500)
time.sleep(3)

print("Servo Y - 90 graden")
servo_on(servoY, 1500)
time.sleep(3)

print("Servo Y - 180 graden")
servo_on(servoY, 2500)
time.sleep(3)

servo_off(servoY)


# ==========================
# SERVO X
# ==========================

print("Servo X - 0 graden")
servo_on(servoX, 500)
time.sleep(3)

print("Servo X - 90 graden")
servo_on(servoX, 1500)
time.sleep(3)

print("Servo X - 180 graden")
servo_on(servoX, 2500)
time.sleep(3)

servo_off(servoX)


# ==========================
# RESET
# ==========================

print("Reset: Servo X naar 90 graden")
servo_on(servoX, 1500)
time.sleep(1)
servo_off(servoX)

print("Reset: Servo Y naar 90 graden")
servo_on(servoY, 1500)
time.sleep(1)
servo_off(servoY)

print("Beide servo's staan op 90 graden.")

# pigpio afsluiten
pwm.stop()