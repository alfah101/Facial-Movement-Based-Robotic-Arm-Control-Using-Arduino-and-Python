from pyfirmata import Arduino, SERVO, util

port = "COM9"
board = Arduino(port)

board.digital[9].mode = SERVO
board.digital[3].mode = SERVO
board.digital[4].mode = SERVO
board.digital[5].mode = SERVO
board.digital[6].mode = SERVO
board.digital[7].mode = SERVO

board.digital[9].write(90)
board.digital[3].write(70)
board.digital[4].write(180)
board.digital[5].write(90)
board.digital[6].write(120)
board.digital[7].write(30)


def rotateservo(pin, angle):
    board.digital[pin].write(angle)
