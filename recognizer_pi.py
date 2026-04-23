import cv2
import os
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2
import random

# --- Configuration ---
# https://github.com/deepinsight/insightface/releases/tag/v0.7
MODEL_PATH = "./models/buffalo_m/w600k_r50.onnx"  # great
# MODEL_PATH = "./models/buffalo_s/w600k_mbf.onnx" # nice
# MODEL_PATH = "./models/antelopev2/glintr100.onnx" # too slow
# MODEL_PATH = "./models/buffalo_sc/w600k_mbf.onnx"  # optimal
# MODEL_PATH = "./models/buffalo_l/w600k_r50.onnx" # mem photo, slow
DATASET_DIR = "./Dataset/"
FRAME_SKIP = 1          # Process every frame (no skip needed with static image)
CONFIDENCE_THRESHOLD = 30.0  # % — minimal confidence to accept match
FADE_TIMEOUT_SEC = 0.5  
FADE_SPEED = 2          # Fade-out speed

# Try to load ONNX model
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print("✅ ONNX model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def preprocess_face(img):
    if img is None or img.size == 0:
        raise ValueError("Invalid image")
    img_resized = cv2.resize(img, (112, 112))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = (img_rgb / 127.5) - 1.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    return np.expand_dims(img_transposed, axis=0).astype(np.float32)

def get_embedding_from_frame(face_image):
    try:
        preprocessed = preprocess_face(face_image)
        embedding = session.run(None, {input_name: preprocessed})[0]
        emb_norm = embedding.flatten()
        norm = np.linalg.norm(emb_norm)
        return emb_norm / norm if norm != 0 else emb_norm
    except Exception as e:
        print(f"⚠ Error extracting embedding: {e}")
        return None

def load_known_faces():
    known_faces = {}
    if not os.path.exists(DATASET_DIR):
        print(f"⚠ Dataset directory '{DATASET_DIR}' not found.")
        return known_faces
    
    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_folder):
            embeddings_list = []
            for file in os.listdir(person_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_folder, file)
                    try:
                        img = cv2.imread(img_path)
                        emb = get_embedding_from_frame(img)
                        if emb is not None:
                            embeddings_list.append(emb)
                    except Exception as e:
                        print(f"⚠ Error loading {img_path}: {e}")
            if embeddings_list:
                known_faces[person_name] = embeddings_list
    return known_faces

def convert_distance_to_confidence(distance: float) -> float:
    max_dist = 2.5
    if distance <= 0:
        return 100.0
    if distance >= max_dist:
        return 0.0
    confidence = (max_dist - distance) / max_dist * 100.0
    return max(0.0, min(100.0, confidence))

def iou(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def generate_random_color():
    """Generate a random RGB color."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (b, g, r)  # OpenCV uses BGR

def get_text_color_for_background(rgb):
    """Determine black/white text based on background brightness."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    luminance = 0.299*r + 0.587*g + 0.114*b
    if luminance > 128:
        return (0, 0, 0)  # Black text for light background
    else:
        return (255, 255, 255)  # White text for dark background

def recognize_face_in_frame(face_image, known_faces_dict):
    face_emb = get_embedding_from_frame(face_image)
    if face_emb is None:
        return "Error", -1

    best_match = "Unknown"
    best_confidence = 0.0
    
    for person, embeddings in known_faces_dict.items():
        for emb in embeddings:
            dist = np.linalg.norm(emb - face_emb)
            conf = convert_distance_to_confidence(dist)

            if conf > best_confidence:
                best_confidence = conf
                best_match = person

    return best_match, best_confidence

def main():
    face_cache = {}
    # Load known faces first
    known_faces = load_known_faces()
    if not known_faces:
        print("⚠ No faces found in Dataset. Cannot proceed.")
        return
    
    print(f"✅ Loaded {len(known_faces)} persons with embeddings.")

    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure for still capture (better quality, no preview stream)
    config = picam2.create_still_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    
    print("📸 Starting camera...")
    picam2.start()
    time.sleep(2)  # Allow auto-exposure and AWB to settle

    # Initialize Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Color cache per person (to keep colors consistent)
    person_colors = {}
    
    print("🔄 Starting recognition loop...")
    try:
        while True:
            # Capture a frame from Picamera2
            frame = picam2.capture_array()  # Returns RGB array
            
            if frame is None or frame.size == 0:
                print("⚠ Failed to capture image")
                continue

            # Convert RGB (Picamera) to BGR (OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            current_detections = []
            
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30))

                for (x, y, w, h) in faces:
                    # Extract ROI
                    face_roi = frame_bgr[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        continue

                    name, conf = recognize_face_in_frame(face_roi, known_faces)

                    # Filter by confidence threshold
                    if conf >= CONFIDENCE_THRESHOLD:
                        current_detections.append({
                            'rect': (int(x), int(y), int(w), int(h)),
                            'name': name,
                            'confidence': float(conf),
                            'timestamp': time.time()
                        })

                # Maintain face cache with fading effect
                current_time = time.time()
                new_cache = {}
                
                for det in current_detections:
                    best_key_to_replace = None
                    for key, entry in list(face_cache.items()):
                        rect_entry = entry.get('rect')
                        if rect_entry is None:
                            continue
                        
                        if iou(det['rect'], rect_entry) > 0.3 and \
                           det['confidence'] > entry.get('confidence', 0):
                            best_key_to_replace = key
                            break

                    new_entry = {
                        'rect': (int(det['rect'][0]), int(det['rect'][1]),
                                int(det['rect'][2]), int(det['rect'][3])),
                        'name': det['name'],
                        'confidence': float(det['confidence']),
                        'timestamp': current_time
                    }

                    # Assign random color if new person
                    if new_entry['name'] not in person_colors:
                        person_colors[new_entry['name']] = generate_random_color()

                    if best_key_to_replace is not None:
                        new_cache[best_key_to_replace] = new_entry.copy()
                        new_cache[best_key_to_replace]['color'] = face_cache.get(best_key_to_replace, {}).get('color', 
                            person_colors[new_entry['name']])
                    else:
                        idx = len(new_cache) + 1000
                        while f"face_{idx}" in face_cache and f"face_{idx}" in new_cache:
                            idx += 1
                        key = f"face_{idx}"
                        new_entry['color'] = person_colors[new_entry['name']]
                        new_cache[key] = new_entry

                # Preserve fading entries from old cache
                for key, entry in face_cache.items():
                    if key not in new_cache:
                        timestamp = entry.get('timestamp', 0)
                        if current_time - timestamp <= FADE_TIMEOUT_SEC + FADE_SPEED * 0.5:
                            new_cache[key] = entry.copy()

                face_cache = new_cache.copy()

            except Exception as e:
                print(f"⚠ Error processing frame: {e}")

            # --- Drawing logic ---
            for key, cached_data in list(face_cache.items()):
                rect = cached_data.get('rect')
                if rect is None:
                    continue
                
                x = int(rect[0])
                y = int(rect[1])
                w = int(rect[2])
                h = int(rect[3])
                
                # Get person color
                name_value = cached_data.get('name', "Unknown")
                
                if name_value in person_colors:
                    base_color = list(person_colors[name_value])
                else:
                    base_color = [36, 255, 12]  # Default greenish

                time_since_detection = current_time - cached_data.get('timestamp', current_time)
                
                if time_since_detection <= FADE_TIMEOUT_SEC:
                    alpha = 1.0
                else:
                    elapsed_after_fade_start = time_since_detection - FADE_TIMEOUT_SEC
                    alpha = max(0.05, 1.0 - (elapsed_after_fade_start / FADE_SPEED))

                color_bgr = tuple(int(c * alpha) for c in base_color)

                cv2.rectangle(frame_bgr,
                              (x, y),
                              (x + w, y + h),
                              color_bgr,
                              int(2 * alpha))

                # Determine text color based on background brightness
                bg_rgb = [base_color[2], base_color[1], base_color[0]]
                text_color_bgr = get_text_color_for_background(bg_rgb)
                
                conf = float(cached_data.get('confidence', 0))
                label = f"{name_value} ({conf:.1f}%)" if name_value != "Unknown" else "Unknown"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, int(2 * alpha))
                text_w, text_h = text_size

                fade_total_time = FADE_TIMEOUT_SEC + FADE_SPEED * 0.5
                if time_since_detection <= fade_total_time:
                    bg_rect_color = tuple(int(c * alpha) for c in base_color)
                    cv2.rectangle(frame_bgr,
                                  (x, y - 35),
                                  (x + text_w + 10, y),
                                  bg_rect_color,
                                  -1)

                # Put text
                cv2.putText(frame_bgr,
                           label,
                           (x, y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6,
                           tuple(int(c * alpha) for c in text_color_bgr),
                           int(2 * alpha),
                           cv2.LINE_AA)

            # Show frame
            cv2.imshow('Face Recognition [Raspberry Pi Picamera]', frame_bgr)

            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

# Global cache for face tracking
face_cache = {}

if __name__ == "__main__":
    main()