"""
Hilfsfunktionen für Datenmanagement und Bildverarbeitung.
"""
import os
import io
import json
import numpy as np
import face_recognition
from PIL import Image
from typing import List, Dict, Optional, Tuple

from config import (
    EMBEDDINGS_FILE, FACES_META_FILE, PERSONS_FILE, 
    HEIF_SUPPORT, FACES_FOLDER
)


def load_data() -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """Lade Embeddings, Gesichts-Metadaten und Personen."""
    # Embeddings laden
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        embeddings = np.empty((0, 128))
    
    # Gesichts-Metadaten laden
    if os.path.exists(FACES_META_FILE):
        with open(FACES_META_FILE) as f:
            faces_meta = json.load(f)
    else:
        faces_meta = []
    
    # Personen laden
    if os.path.exists(PERSONS_FILE):
        with open(PERSONS_FILE) as f:
            persons = json.load(f)
    else:
        persons = []
    
    return embeddings, faces_meta, persons


def save_data(embeddings: np.ndarray, faces_meta: List[Dict], persons: List[Dict]) -> None:
    """Speichere Embeddings, Gesichts-Metadaten und Personen."""
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(FACES_META_FILE, 'w') as f:
        json.dump(faces_meta, f, indent=2)
    with open(PERSONS_FILE, 'w') as f:
        json.dump(persons, f, indent=2)


def process_image_for_faces(file_bytes: bytes, filename: str, upscale: bool = True) -> Tuple[Optional[np.ndarray], str]:
    """Verarbeite ein Bild und bereite es für die Gesichtserkennung vor."""
    try:
        fname = filename.lower()
        
        # HEIC/HEIF zuerst konvertieren
        if fname.endswith((".heic", ".heif")):
            if HEIF_SUPPORT:
                pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                buf.seek(0)
                image = face_recognition.load_image_file(buf)
            else:
                return None, f"HEIC/HEIF kann nicht verarbeitet werden (pillow-heif nicht installiert)"
        else:
            image = face_recognition.load_image_file(io.BytesIO(file_bytes))
        
        # Bild ggf. vergrößern für bessere Erkennung
        if upscale and max(image.shape) < 800:
            try:
                import cv2
                scale = 800 / max(image.shape)
                image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
            except ImportError:
                pass  # Kein cv2 verfügbar, verwende Originalgröße
        
        return image, "OK"
    except Exception as e:
        return None, f"Fehler beim Verarbeiten des Bildes: {str(e)}"


def detect_faces_robust(image: np.ndarray, use_cnn: bool = False) -> Tuple[List, List]:
    """Robuste Gesichtserkennung mit Fallback."""
    model = 'cnn' if use_cnn else 'hog'
    
    try:
        face_locations = face_recognition.face_locations(image, model=model)
        if not face_locations:
            return [], []
        encs = face_recognition.face_encodings(image, known_face_locations=face_locations)
    except Exception:
        # Fallback auf HOG-Modell
        try:
            face_locations = face_recognition.face_locations(image, model='hog')
            if not face_locations:
                return [], []
            encs = face_recognition.face_encodings(image, known_face_locations=face_locations)
        except Exception:
            return [], []
    
    # Validiere Encodings
    valid_encs = []
    valid_locations = []
    for i, (loc, enc) in enumerate(zip(face_locations, encs)):
        if enc is not None and len(enc) > 0:
            valid_encs.append(enc)
            valid_locations.append(loc)
    
    return valid_locations, valid_encs


def get_person_encodings(person_id: str, faces_meta: List[Dict]) -> List[np.ndarray]:
    """Hole alle Gesichts-Encodings für eine bestimmte Person."""
    import face_recognition
    
    person_encodings = []
    person_faces = [meta for meta in faces_meta if meta.get('person_id') == person_id]
    
    for face_meta in person_faces:
        face_path = os.path.join(FACES_FOLDER, face_meta['filename'])
        if os.path.exists(face_path):
            try:
                face_img = face_recognition.load_image_file(face_path)
                face_encs = face_recognition.face_encodings(face_img)
                if face_encs and len(face_encs) > 0:
                    person_encodings.append(face_encs[0])
            except:
                continue
    
    return person_encodings


def get_average_encoding_for_person(person_id: str, faces_meta: List[Dict]) -> Optional[np.ndarray]:
    """Berechne das Durchschnitts-Encoding für eine Person basierend auf allen trainierten Gesichtern."""
    encodings = get_person_encodings(person_id, faces_meta)
    
    if not encodings:
        return None
    
    # Berechne Durchschnitt aller Encodings dieser Person
    avg_encoding = np.mean(encodings, axis=0)
    return avg_encoding


def improved_face_distance(query_encoding: np.ndarray, person_id: Optional[str], faces_meta: List[Dict]) -> float:
    """Verbesserte Distanzberechnung unter Verwendung trainierter Gesichter."""
    if not person_id:
        # Keine trainierte Person - verwende Standard-Distanz
        return float('inf')
    
    # Hole Durchschnitts-Encoding für diese Person
    avg_encoding = get_average_encoding_for_person(person_id, faces_meta)
    
    if avg_encoding is None:
        return float('inf')
    
    # Berechne Distanz zum Durchschnitts-Encoding
    distance = np.linalg.norm(query_encoding - avg_encoding)
    return distance


def get_person_name(person_id: Optional[str], persons: List[Dict]) -> Tuple[str, str]:
    """Hole Personendaten basierend auf person_id."""
    if not person_id:
        return "(unbekannt)", ""
    
    person = next((p for p in persons if p['person_id'] == person_id), None)
    if person:
        name_str = f"{person.get('vorname','')} {person.get('nachname','')}".strip()
        label = person.get('name','')
        return name_str, label
    return "(unbekannt)", ""
