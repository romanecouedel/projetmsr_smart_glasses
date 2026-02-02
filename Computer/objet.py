from ultralytics import YOLO
import cv2
import torch
import numpy as np
import serial
import time
from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2


import socket
import struct


ARDUINO_PORT = "/dev/ttyACM1"
ARDUINO_BAUDRATE = 9600
arduino = None

OBJECT_CLASSES = [0, 13, 28, 56, 59, 26, 57]

# Noms des classes COCO
COCO_NAMES = {
    0: 'person', 13: 'bench', 28:'suitcase', 57:'couch',
    56: 'chair', 59: 'bed', 26: 'handbag'
}

class ObjectTracker:
    """Suit les objets détectés au fil du temps pour éviter les changements brusques"""
    
    def __init__(self, stability_duration=1.0, intensity_step=15):
        """
        stability_duration: durée minimale (secondes) qu'un objet doit être détecté
        intensity_step: changement max d'intensité par frame (pour lisser)
        """
        self.tracked_objects = {}  # {side: {'class': int, 'first_seen': time, 'last_seen': time, 'intensity': int}}
        self.stability_duration = stability_duration
        self.intensity_step = intensity_step
        self.last_intensities = {'left': 0, 'right': 0}
    
    def update(self, side, obj_class, intensity, current_time):
        """
        Met à jour le tracking d'un objet
        Retourne l'intensité lissée si l'objet est stable, None sinon
        """
        key = side
        
        # Nouvel objet ou objet différent détecté
        if key not in self.tracked_objects or self.tracked_objects[key]['class'] != obj_class:
            self.tracked_objects[key] = {
                'class': obj_class,
                'first_seen': current_time,
                'last_seen': current_time,
                'intensity': intensity
            }
            return None  # Pas encore stable
        
        # Même objet détecté, mettre à jour
        self.tracked_objects[key]['last_seen'] = current_time
        
        # Vérifier si l'objet est stable (détecté depuis assez longtemps)
        time_stable = current_time - self.tracked_objects[key]['first_seen']
        if time_stable < self.stability_duration:
            return None  # Pas encore stable
        
        # Objet stable, lisser l'intensité
        last_intensity = self.last_intensities[side]
        smoothed_intensity = self._smooth_intensity(last_intensity, intensity)
        self.last_intensities[side] = smoothed_intensity
        
        return smoothed_intensity
    
    def _smooth_intensity(self, current, target):
        """Lisse le changement d'intensité avec un step maximum"""
        diff = target - current
        
        if abs(diff) <= self.intensity_step:
            return target
        elif diff > 0:
            return current + self.intensity_step
        else:
            return current - self.intensity_step
    
    def clear_side(self, side, current_time):
        """Nettoie un côté si aucun objet n'est détecté"""
        if side in self.tracked_objects:
            # Garder pendant un court moment pour éviter les flickerings
            if current_time - self.tracked_objects[side]['last_seen'] > 0.5:
                del self.tracked_objects[side]
                self.last_intensities[side] = 0
    
    def reset(self):
        """Réinitialise complètement le tracker"""
        self.tracked_objects = {}
        self.last_intensities = {'left': 0, 'right': 0}

def init_arduino():
    global arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE, timeout=1)
        time.sleep(2)
        print("Connexion Arduino établie")
        return True
    except Exception as e:
        print(f"Erreur connexion Arduino: {e}")
        print("Vérifiez le port et les droits d'accès")
        return False

def send_command(command):
    """Envoie commande à l'Arduino"""
    if arduino and arduino.is_open:
        try:
            arduino.write(f"{command}\n".encode())
            print(f"Arduino: {command}")
        except Exception as e:
            print(f"Erreur envoi: {e}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb'

print(f"Chargement du modèle Depth Anything V2 ({encoder}) sur {DEVICE}...")
depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()
print("Modèle Depth chargé\n")

print("Chargement du modèle YOLO...")
yolo_model = YOLO("yolo11x-seg.pt")
print("Modèle YOLO chargé\n")

def visualize_depth(depth_map):
    """Convertit la carte de profondeur en image colorée"""
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_normalized = 1 - depth_normalized
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return depth_colored

def calculate_mask_depth(mask, depth_map):
    """
    Calcule la profondeur moyenne d'un objet à partir de son masque
    mask: masque binaire de l'objet (numpy array)
    depth_map: carte de profondeur (numpy array)
    Returns: profondeur moyenne normalisée (0-1, 0=proche, 1=loin)
    """
    # Redimensionner le masque à la taille de la depth map si nécessaire
    if mask.shape[:2] != depth_map.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]))
    
    # Extraire les valeurs de profondeur là où le masque est actif
    mask_bool = mask > 0.5
    if not mask_bool.any():
        return None
    
    depth_values = depth_map[mask_bool]
    avg_depth = np.mean(depth_values)
    
    # Normaliser par rapport à min/max de toute la carte
    normalized_depth = (avg_depth - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return normalized_depth

def depth_to_intensity(normalized_depth, closest_depth):
    """
    Convertit une profondeur normalisée en intensité (0-100)
    L'objet le plus proche est toujours à 100%
    Les autres sont calculés proportionnellement
    
    IMPORTANT: Dans depth map, valeurs ÉLEVÉES = PROCHE, valeurs BASSES = LOIN
    normalized_depth: profondeur de l'objet actuel (1=proche, 0=loin)
    closest_depth: profondeur de l'objet le plus proche globalement (1=proche, 0=loin)
    """
    # Vérifier si c'est l'objet le plus proche (ou très proche)
    if normalized_depth >= closest_depth * 0.99:  # Tolérance de 1%
        return 100
    
    # Calcul proportionnel: plus loin = moins intense
    # Si closest = 0.8 et current = 0.4, alors ratio = 0.4/0.8 = 0.5 -> 50%
    if closest_depth > 0:
        intensity = int((normalized_depth / closest_depth) * 100)
    else:
        intensity = 10
    
    return max(10, min(100, intensity))  # Limiter entre 10 et 100

def is_in_roi(box_center_x, box_center_y, frame_width, frame_height):
    """Vérifie si le centre de la bounding box est dans la zone d'intérêt"""
    roi_top = 0
    roi_bottom = frame_height
    
    if box_center_y < roi_top:
        return False
    
    progress = (box_center_y - roi_top) / (roi_bottom - roi_top)
    roi_width_ratio = 0.2 + (0.6 * progress)
    
    roi_left = frame_width * (1 - roi_width_ratio) / 2
    roi_right = frame_width * (1 + roi_width_ratio) / 2
    
    return roi_left <= box_center_x <= roi_right

def draw_roi(frame, frame_width, frame_height):
    """Dessine la zone d'intérêt en forme de trapèze"""
    roi_top = 0
    roi_bottom = frame_height
    
    top_width = int(frame_width * 0.2)
    bottom_width = int(frame_width * 0.8)
    pts = np.array([
        [frame_width // 2 - top_width // 2, roi_top],
        [frame_width // 2 + top_width // 2, roi_top],
        [frame_width // 2 + bottom_width // 2, roi_bottom],
        [frame_width // 2 - bottom_width // 2, roi_bottom]
    ], np.int32)
    
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0, 30))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    
    return frame

def process_detection_with_depth(objects_data, frame_width, frame_height, tracker, current_time):
    """
    Analyse les objets avec leur profondeur et contrôle les ballons
    objects_data: liste de dict avec 'box', 'mask', 'depth', 'class', 'center'
    tracker: ObjectTracker pour le filtrage temporel
    current_time: timestamp actuel
    
    CONTRAINTES:
    - L'objet le plus proche globalement est toujours à intensité 100%
    - Les autres objets ont une intensité proportionnelle à leur distance
    - On ne garde qu'UN SEUL objet par côté (le plus proche)
    - Filtrage: objet doit être détecté pendant 1s minimum
    - Lissage: changement d'intensité progressif
    """
    center_x = frame_width / 2
    left_objects = []
    right_objects = []
    
    # Filtrer les objets dans la ROI
    roi_objects = []
    for obj in objects_data:
        box_center_x, box_center_y = obj['center']
        if is_in_roi(box_center_x, box_center_y, frame_width, frame_height):
            roi_objects.append(obj)
            if box_center_x < center_x:
                left_objects.append(obj)
            else:
                right_objects.append(obj)
    
    if len(roi_objects) == 0:
        # Nettoyer les côtés et arrêter progressivement
        tracker.clear_side('left', current_time)
        tracker.clear_side('right', current_time)
        
        # Envoyer STOP seulement si les deux intensités sont à 0
        if tracker.last_intensities['left'] == 0 and tracker.last_intensities['right'] == 0:
            send_command("STOP")
            print("\nAucun obstacle dans la ROI")
        return
    
    # Trouver l'objet le plus proche globalement (profondeur MAXIMALE car proche = valeur élevée)
    closest_global = max(roi_objects, key=lambda x: x['depth'])
    closest_depth = closest_global['depth']
    
    # Recalculer les intensités en fonction de l'objet le plus proche
    for obj in roi_objects:
        obj['intensity'] = depth_to_intensity(obj['depth'], closest_depth)
    
    print("\n" + "="*60)
    print("OBJETS DÉTECTÉS DANS LA ROI:")
    print("="*60)
    
    # CONTRAINTE: Garder UN SEUL objet par côté (le plus proche = depth max)
    selected_left = None
    selected_right = None
    final_left_intensity = None
    final_right_intensity = None
    
    if len(left_objects) > 0:
        selected_left = max(left_objects, key=lambda x: x['depth'])
        
        # Appliquer le filtrage temporel
        smoothed_intensity = tracker.update('left', selected_left['class'], 
                                           selected_left['intensity'], current_time)
        
        if smoothed_intensity is not None:
            final_left_intensity = smoothed_intensity
            side = "GAUCHE"
            class_name = COCO_NAMES.get(selected_left['class'], f"class_{selected_left['class']}")
            depth_percent = selected_left['depth'] * 100
            status = "STABLE" if smoothed_intensity == selected_left['intensity'] else "LISSAGE"
            print(f"  [{side:6}] {class_name:10} | D:{depth_percent:5.1f}% | I:{smoothed_intensity:3}% | {status}")
        else:
            # Objet pas encore stable
            class_name = COCO_NAMES.get(selected_left['class'], f"class_{selected_left['class']}")
            print(f"  [GAUCHE] {class_name:10} |  EN ATTENTE (stabilisation...)")
    else:
        tracker.clear_side('left', current_time)
    
    if len(right_objects) > 0:
        selected_right = max(right_objects, key=lambda x: x['depth'])
        
        # Appliquer le filtrage temporel
        smoothed_intensity = tracker.update('right', selected_right['class'], 
                                           selected_right['intensity'], current_time)
        
        if smoothed_intensity is not None:
            final_right_intensity = smoothed_intensity
            side = "DROITE"
            class_name = COCO_NAMES.get(selected_right['class'], f"class_{selected_right['class']}")
            depth_percent = selected_right['depth'] * 100
            status = "STABLE" if smoothed_intensity == selected_right['intensity'] else "LISSAGE"
            print(f"  [{side:6}] {class_name:10} | D:{depth_percent:5.1f}% | I:{smoothed_intensity:3}% | {status}")
        else:
            # Objet pas encore stable
            class_name = COCO_NAMES.get(selected_right['class'], f"class_{selected_right['class']}")
            print(f"  [DROITE] {class_name:10} |  EN ATTENTE (stabilisation...)")
    else:
        tracker.clear_side('right', current_time)
    
    # Contrôler les ballons SEULEMENT avec les objets stables
    if final_left_intensity is not None and final_right_intensity is not None:
        send_command(f"GG,{final_left_intensity+30}")
        time.sleep(0.3)
        send_command(f"GD,{final_right_intensity}")
        time.sleep(0.3)
        print(f"\nALERTE: Obstacles des DEUX côtés")
        
    elif final_left_intensity is not None:
        send_command(f"GG,{final_left_intensity}")
        time.sleep(0.3)
        # Réduire progressivement le côté droit si nécessaire
        if tracker.last_intensities['right'] > 0:
            new_right = max(0, tracker.last_intensities['right'] - tracker.intensity_step)
            if new_right > 0:
                send_command(f"GD,{new_right}")
                time.sleep(0.3)
                tracker.last_intensities['right'] = new_right
        print(f"\nALERTE: Obstacle à GAUCHE")
        
    elif final_right_intensity is not None:
        send_command(f"GD,{final_right_intensity}")
        time.sleep(0.3)
        # Réduire progressivement le côté gauche si nécessaire
        if tracker.last_intensities['left'] > 0:
            new_left = max(0, tracker.last_intensities['left'] - tracker.intensity_step)
            if new_left > 0:
                
                send_command(f"GG,{new_left}")
                time.sleep(0.3)
                tracker.last_intensities['left'] = new_left
        print(f"\nALERTE: Obstacle à DROITE")
    
    print("="*60)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Caméra non trouvée")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Caméra activée ({frame_width}x{frame_height}px)")
    print("Appuyez sur 'q' pour quitter\n")

    tracker = ObjectTracker(stability_duration=1.0, intensity_step=15)
    
    # Créer les 4 fenêtres
    cv2.namedWindow("1. Camera RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. Depth Estimation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("3. Object Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("4. Objects + Depth Info", cv2.WINDOW_NORMAL)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 5000))
    
    try:
        while True:

            print("-------------------------------")

            packet, addr= sock.recvfrom(65536)
            #print("Reçu de :", addr)
            
            size = struct.unpack("H", packet[:2])[0]
            data = packet[2:2+size]

            # Décodage JPEG
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
           
            current_time = time.time() 


            # ret, frame = cap.read()
            # if not ret:
            #     break
            
            # FENÊTRE 1 : Caméra RGB Brute
            frame_rgb = frame.copy()
            cv2.imshow("1. Camera RGB", frame_rgb)
            
            # FENÊTRE 2 : Depth Estimation
            with torch.no_grad():
                depth = depth_model.infer_image(frame)
            depth_colored = visualize_depth(depth)
            if depth_colored.shape[:2] != frame.shape[:2]:
                depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            cv2.imshow("2. Depth Estimation", depth_colored)
            
            # FENÊTRE 3 : Object Detection avec Masques
            results = yolo_model(frame, conf=0.5, iou=0.45, verbose=False,classes=OBJECT_CLASSES)
            
            # Collecter les données des objets
            objects_data = []
            
            for result in results:
                boxes = result.boxes
                masks = result.masks
                
                if masks is None:
                    continue
                
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    if cls not in OBJECT_CLASSES:
                        continue
                    
                    # Récupérer le masque
                    mask = masks.data[i].cpu().numpy()
                    
                    # Calculer la profondeur moyenne du masque
                    avg_depth = calculate_mask_depth(mask, depth)
                    
                    if avg_depth is None:
                        continue
                    
                    # Centre de la bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    objects_data.append({
                        'box': (x1, y1, x2, y2),
                        'mask': mask,
                        'depth': avg_depth,
                        'intensity': 0,  # Sera calculé plus tard
                        'class': cls,
                        'center': (center_x, center_y)
                    })
            
            # Traiter les détections avec profondeur (calcule les intensités)
            process_detection_with_depth(objects_data, frame_width, frame_height,tracker,current_time)
            
            # Créer frame annotée
            annotated_frame = results[0].plot()
            annotated_frame = draw_roi(annotated_frame, frame_width, frame_height)
            
            # Ligne de séparation
            cv2.line(annotated_frame, 
                    (frame_width // 2, 0), 
                    (frame_width // 2, frame_height), 
                    (255, 0, 0), 2)
            
            cv2.imshow("3. Object Detection", annotated_frame)
            
            # FENÊTRE 4 : Info Objets + Profondeur
            info_frame = frame.copy()
            
            for obj in objects_data:
                x1, y1, x2, y2 = obj['box']
                class_name = COCO_NAMES.get(obj['class'], f"class_{obj['class']}")
                depth_percent = obj['depth'] * 100
                
                # Dessiner bounding box
                color = (0, 255, 0) if obj['center'][0] < frame_width / 2 else (255, 0, 0)
                cv2.rectangle(info_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Afficher infos
                label = f"{class_name}"
                depth_label = f"D:{depth_percent:.1f}% I:{obj['intensity']}%"
                
                cv2.putText(info_frame, label, (int(x1), int(y1)-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(info_frame, depth_label, (int(x1), int(y1)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Ligne de séparation
            cv2.line(info_frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 255, 255), 2)
            
            cv2.imshow("4. Objects + Depth Info", info_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nArrêt du programme...")
    
    finally:
        send_command("STOP")
        cap.release()
        cv2.destroyAllWindows()
        if arduino and arduino.is_open:
            arduino.close()
        print("✓ Nettoyage terminé")

if __name__ == "__main__":
    if init_arduino():
        main()
    else:
        print("Impossible de démarrer sans Arduino")
