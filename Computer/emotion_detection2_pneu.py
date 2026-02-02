import cv2
import time
import serial
from fer.fer import FER

import socket
import struct

import numpy as np

EMOTION_MAP = {
    "happy": "happy",
    "surprise": "happy",
    "sad": "sad",
    "angry": "neutral",
    "fear": "sad",
    "disgust": "sad",
    "neutral": "neutral",
}

CONFIDENCE_THRESHOLD = 0.35
FRAME_SKIP = 3
RESIZE_FX = 0.6
RESIZE_FY = 0.6

def reduce_to_three(emotions: dict) -> tuple[str, float]:
    reduced = {"happy": 0.0, "sad": 0.0, "neutral": 0.0}
    for emo, val in emotions.items():
        reduced[EMOTION_MAP[emo]] += float(val)

    top = max(reduced, key=reduced.get)
    score = reduced[top]

    if score < CONFIDENCE_THRESHOLD:
        return "neutral", score
    return top, score

ARDUINO_PORT = "/dev/ttyACM1"
ARDUINO_BAUDRATE = 9600
arduino = None

def init_arduino():
    global arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE, timeout=1)
        time.sleep(2)
        print("Arduino connected")
        return True
    except Exception as e:
        print(f"Arduino connection error: {e}")
        return False

# def send_command(command):
#     if arduino and arduino.is_open:
#         arduino.write(f"{command}\n".encode())
#         print(f"{command}")

def send_command(command):
    if arduino and arduino.is_open:
        arduino.write(f"{command}\n".encode())

PNEUMATIC_MAP = {
    "happy": {
        "left": ("arreter", 0),
        "right": ("cycle", 100),
    },
    "sad": {
        "left": ("cycle", 100),
        "right": ("arreter", 0),
    },
    "neutral": {
        "left": ("arreter", 0),
        "right": ("arreter", 0),
    },
}

def process_emotion_pneumatic(emotion, confidence):
    print(f"\nEmotion: {emotion} ({confidence:.2f})")

    config = PNEUMATIC_MAP[emotion]

    # LEFT side
    left_action, intensity = config["left"]
    if left_action == "cycle":
        send_command(f"GG,{intensity}")
        time.sleep(0.3)
        send_command(f"AG,{intensity}")
    else:
        send_command("STOP_G")

    # RIGHT side
    right_action, intensity = config["right"]
    if right_action == "cycle":
        send_command(f"GD,{intensity}")
        time.sleep(0.3)
        send_command(f"AD,{intensity}")
    else:
        send_command("STOP_D")

    send_command("DONE")
    raise SystemExit

def main():

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 5000))
    detector = FER(mtcnn=False)
    # cap = cv2.VideoCapture(0)

    # if not cap.isOpened():
    #     raise RuntimeError("Could not open webcam.")

    frame_count = 0
    last_boxes_and_labels = []

    while True:
        packet, addr= sock.recvfrom(65536)
        #print("Reçu de :", addr)        
        size = struct.unpack("H", packet[:2])[0]
        data = packet[2:2+size]

        # Décodage JPEG
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("Emotion Detection + Pneumatique", frame)
        # ret, frame = cap.read()
        # if not ret:
        #     break

        frame = cv2.resize(frame, None, fx=RESIZE_FX, fy=RESIZE_FY)
        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_emotions(rgb)
            last_boxes_and_labels = []

            if results:
                h, w, _ = frame.shape
                cx, cy = w / 2, h / 2

                best = min(
                    results,
                    key=lambda r: (
                        (r["box"][0] + r["box"][2] / 2 - cx) ** 2 +
                        (r["box"][1] + r["box"][3] / 2 - cy) ** 2
                    )
                )

                x, y, bw, bh = best["box"]
                top_emotion, score = reduce_to_three(best["emotions"])

                process_emotion_pneumatic(top_emotion, score)

                last_boxes_and_labels.append((x, y, bw, bh, top_emotion, score))
            # else:
                # send_command("STOP")

        for (x, y, bw, bh, emo, score) in last_boxes_and_labels:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emo} ({score:.2f})",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Emotion Detection + Pneumatique", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino and arduino.is_open:
        arduino.close()

if __name__ == "__main__":
    if init_arduino():
        main()
    else:
        print("Arduino not available")
