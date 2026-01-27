from picamera2 import Picamera2
import cv2
import socket
import pickle
import struct

###### code pour envoyer le flux vidéo via UDP ######
# ici c'est le code coté RPI

PC_IP = "10.42.0.2" # IP de l'ordinateur receveur
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()

    # Compression JPEG
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    data = buf.tobytes()
    size = len(data)

    # UDP fragmentation simple
    if size > 60000:
        continue  # Ignore frames trop gros (rare)
    sock.sendto(struct.pack("H", size) + data, (PC_IP, PORT))
