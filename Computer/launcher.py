import serial
import subprocess
import time
import signal
import os

PORT = "/dev/ttyACM1"
BAUDRATE = 9600

EMOTION_CMD = ["python3", "emotion_detection2_pneu.py"]
YOLO_CMD    = ["python3", "objet.py"]

current_process = None
current_mode = None

def stop_current():
    global current_process
    if current_process is not None:
        print("Arrêt du script courant")
        current_process.send_signal(signal.SIGINT)
        current_process.wait()
        current_process = None

def start_mode(mode):
    global current_process, current_mode

    if mode == current_mode:
        return

    stop_current()

    if mode == 1:
        print("Lancement Emotion Detection")
        current_process = subprocess.Popen(EMOTION_CMD)
    elif mode == 2:
        print("Lancement Obstacle Detection")
        current_process = subprocess.Popen(YOLO_CMD)
    elif mode == 3:
        print("Mode NEUTRAL")

    current_mode = mode

def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print("Mode launcher prêt")

    while True:
        line = ser.readline().decode().strip()
        if line.startswith("MODE:"):
            mode = int(line.split(":")[1])
            start_mode(mode)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_current()
        print("\nArrêt launcher")


