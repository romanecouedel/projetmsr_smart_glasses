import socket
import struct
import cv2
import numpy as np

UDP_IP = "0.0.0.0"
UDP_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("En écoute sur le port", UDP_PORT)

while True:
    packet, addr = sock.recvfrom(65536)  # On peut recevoir jusqu'à 65536 bytes

    if len(packet) < 2:
        continue  # paquet trop petit, on ignore

    # 1) Lire la taille depuis les 2 premiers octets
    size = struct.unpack("H", packet[:2])[0]

    # 2) Le reste est la data JPEG
    jpeg_data = packet[2:2+size]

    # 3) Décoder en image OpenCV
    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if frame is not None:
        cv2.imshow("Frame UDP", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC pour quitter
            break

cv2.destroyAllWindows()
