"""
Moderne Face Search Engine mit aktuellen Computer Vision Bibliotheken.
Ersetzt die komplexe Vector DB und face_recognition mit einfacheren, effektiveren Lösungen.
"""
import os
import time
import numpy as np
import streamlit as st
from PIL import Image
from typing import List, Dict, Optional, Tuple
import json
import hashlib

from config import IMAGE_FOLDER

# Lazy imports für bessere Performance
def get_cv2():
    """Lazy import für OpenCV"""
    try:
        import cv2
        return cv2
    except ImportError:
        st.error("❌ OpenCV nicht installiert! Installiere mit: pip install opencv-python")
        return None

def get_mtcnn():
    """Lazy import für MTCNN (bessere Face Detection)"""
    try:
        from mtcnn import MTCNN
        return MTCNN()
    except ImportError:
        st.warning("⚠️ MTCNN nicht installiert. Installiere mit: pip install mtcnn")
        return None

def get_dlib_models():
    """Lazy import für dlib (falls verfügbar)"""
    try:
        import dlib
        # Versuche shape predictor zu laden
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)
            face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
            return detector, predictor, face_rec_model
    except ImportError:
        pass
    return None, None, None

def get_deepface():
    """Lazy import für DeepFace (modernste Option)"""
    try:
        from deepface import DeepFace
        return DeepFace
    except ImportError:
        st.warning("⚠️ DeepFace nicht installiert. Installiere mit: pip install deepface")
        return None


class ModernFaceDetector:
    """
    Moderne Face Detection mit mehreren Backends.
    """
    
    def __init__(self):
        self.cv2 = get_cv2()
        self.mtcnn = get_mtcnn()
        self.deepface = get_deepface()
        self.dlib_detector, self.dlib_predictor, self.dlib_face_rec = get_dlib_models()
        
        # Lade OpenCV Face Cascade als Fallback
        if self.cv2:
            cascade_path = self.cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = self.cv2.CascadeClassifier(cascade_path)
            else:
                self.face_cascade = None
    
    def detect_faces(self, image_path: str, method="auto") -> List[Dict]:
        """
        Erkenne Gesichter mit dem besten verfügbaren Verfahren.
        """
        try:
            # Lade Bild
            if self.cv2:
                img = self.cv2.imread(image_path)
                if img is None:
                    return []
                rgb_img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
            else:
                pil_img = Image.open(image_path)
                rgb_img = np.array(pil_img.convert('RGB'))
            
            faces = []
            
            # Method 1: MTCNN (beste moderne Face Detection)
            if method in ["auto", "mtcnn"] and self.mtcnn:
                try:
                    result = self.mtcnn.detect_faces(rgb_img)
                    for i, face_data in enumerate(result):
                        if face_data['confidence'] > 0.9:  # Nur sehr sichere Detections
                            box = face_data['box']
                            x, y, w, h = box
                            faces.append({
                                'method': 'MTCNN',
                                'bbox': (x, y, x+w, y+h),
                                'confidence': face_data['confidence'],
                                'landmarks': face_data.get('keypoints', {}),
                                'quality_score': face_data['confidence']
                            })
                except Exception as e:
                    st.warning(f"MTCNN Fehler: {e}")
            
            # Method 2: DeepFace (wenn MTCNN nicht verfügbar/erfolgreich)
            if not faces and method in ["auto", "deepface"] and self.deepface:
                try:
                    # DeepFace hat eingebaute Face Detection
                    temp_results = self.deepface.extract_faces(
                        img_path=image_path,
                        target_size=(224, 224),
                        detector_backend="opencv",  # Oder "mtcnn", "retinaface"
                        enforce_detection=False
                    )
                    
                    for i, face_array in enumerate(temp_results):
                        if face_array is not None and face_array.size > 0:
                            # DeepFace liefert normalisierte Faces, wir brauchen Bounding Boxes
                            # Vereinfachte Annahme: Gesicht ist zentral im extractierten Bereich
                            h, w = rgb_img.shape[:2]
                            faces.append({
                                'method': 'DeepFace',
                                'bbox': (w//4, h//4, 3*w//4, 3*h//4),  # Grobe Schätzung
                                'confidence': 0.8,  # DeepFace liefert keine direkte Confidence
                                'quality_score': 0.8,
                                'face_data': face_array
                            })
                except Exception as e:
                    st.warning(f"DeepFace Fehler: {e}")
            
            # Method 3: dlib (präzise, aber langsamer)
            if not faces and method in ["auto", "dlib"] and self.dlib_detector:
                try:
                    gray = self.cv2.cvtColor(rgb_img, self.cv2.COLOR_RGB2GRAY)
                    face_rects = self.dlib_detector(gray)
                    
                    for rect in face_rects:
                        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                        
                        # Qualitätsbewertung basierend auf Größe
                        area = (x2 - x1) * (y2 - y1)
                        total_area = rgb_img.shape[0] * rgb_img.shape[1]
                        quality = min(1.0, area / (total_area * 0.01))  # Mindestens 1% des Bildes
                        
                        faces.append({
                            'method': 'dlib',
                            'bbox': (x1, y1, x2, y2),
                            'confidence': 0.9,  # dlib ist sehr präzise
                            'quality_score': quality
                        })
                except Exception as e:
                    st.warning(f"dlib Fehler: {e}")
            
            # Method 4: OpenCV Haar Cascade (Fallback)
            if not faces and method in ["auto", "opencv"] and self.face_cascade:
                try:
                    gray = self.cv2.cvtColor(rgb_img, self.cv2.COLOR_RGB2GRAY)
                    face_rects = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(30, 30)
                    )
                    
                    for (x, y, w, h) in face_rects:
                        # Qualitätsbewertung
                        area = w * h
                        total_area = rgb_img.shape[0] * rgb_img.shape[1]
                        quality = min(1.0, area / (total_area * 0.005))
                        
                        faces.append({
                            'method': 'OpenCV',
                            'bbox': (x, y, x+w, y+h),
                            'confidence': 0.7,  # OpenCV ist weniger zuverlässig
                            'quality_score': quality
                        })
                except Exception as e:
                    st.warning(f"OpenCV Fehler: {e}")
            
            # Sortiere nach Qualität
            faces.sort(key=lambda x: x['quality_score'], reverse=True)
            
            return faces
            
        except Exception as e:
            st.error(f"Fehler bei Face Detection in {image_path}: {e}")
            return []


class ModernFaceEncoder:
    """
    Moderne Face Encoding mit mehreren Backends.
    """
    
    def __init__(self):
        self.deepface = get_deepface()
        self.cv2 = get_cv2()
        
        # Verfügbare DeepFace Modelle (vom besten zum schlechtesten)
        self.model_priority = [
            "ArcFace",      # Beste Genauigkeit
            "Facenet512",   # Sehr gut und schnell  
            "Facenet",      # Gut und schnell
            "DeepFace",     # Basis-Modell
            "VGG-Face",     # Fallback
        ]
        
        self.current_model = None
        self._initialize_best_model()
    
    def _initialize_best_model(self):
        """Initialisiere das beste verfügbare Modell."""
        if not self.deepface:
            return
        
        for model_name in self.model_priority:
            try:
                # Teste das Modell mit einem kleinen Dummy-Array
                test_img = np.zeros((224, 224, 3), dtype=np.uint8)
                self.deepface.represent(test_img, model_name=model_name, enforce_detection=False)
                self.current_model = model_name
                st.success(f"✅ Face Encoding Modell: {model_name}")
                break
            except Exception as e:
                st.warning(f"⚠️ Modell {model_name} nicht verfügbar: {e}")
                continue
        
        if not self.current_model:
            st.error("❌ Kein DeepFace Modell verfügbar!")
    
    def encode_face(self, image_path: str, face_bbox: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """
        Erstelle Face Encoding für ein Bild oder Gesichtsbereich.
        """
        if not self.deepface or not self.current_model:
            return None
        
        try:
            # Verwende DeepFace für Face Encoding
            if face_bbox:
                # Crop das Gesicht aus dem Bild
                pil_img = Image.open(image_path)
                x1, y1, x2, y2 = face_bbox
                cropped_face = pil_img.crop((x1, y1, x2, y2))
                
                # Speichere temporär (DeepFace braucht Dateipfad)
                temp_path = "/tmp/temp_face.jpg"
                cropped_face.save(temp_path)
                image_to_encode = temp_path
            else:
                image_to_encode = image_path
            
            # Erstelle Encoding
            embedding = self.deepface.represent(
                img_path=image_to_encode,
                model_name=self.current_model,
                enforce_detection=False,  # Wichtig: Gesicht ist schon detektiert
                detector_backend="skip"   # Skip Detection, da wir schon das Gesicht haben
            )
            
            # Cleanup
            if face_bbox and os.path.exists("/tmp/temp_face.jpg"):
                os.remove("/tmp/temp_face.jpg")
            
            # Konvertiere zu numpy array
            if isinstance(embedding, list) and len(embedding) > 0:
                encoding = np.array(embedding[0]['embedding'])
                return encoding
            elif isinstance(embedding, dict) and 'embedding' in embedding:
                encoding = np.array(embedding['embedding'])
                return encoding
            else:
                return None
                
        except Exception as e:
            st.warning(f"Face Encoding Fehler: {e}")
            return None
    
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Vergleiche zwei Face Encodings und gib Distanz zurück.
        """
        try:
            # Cosine Distance (0 = identisch, 2 = komplett unterschiedlich)
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                return 2.0  # Maximale Distanz
            
            cosine_sim = dot_product / (norm1 * norm2)
            cosine_distance = 1 - cosine_sim
            
            return float(cosine_distance)
            
        except Exception as e:
            st.warning(f"Face Comparison Fehler: {e}")
            return 2.0  # Maximale Distanz bei Fehler


class SimpleFaceDatabase:
    """
    Einfache, dateibasierte Face Database ohne komplexe Vector Indizes.
    """
    
    def __init__(self):
        self.db_file = os.path.join(os.path.dirname(__file__), 'simple_face_db.json')
        self.detector = ModernFaceDetector()
        self.encoder = ModernFaceEncoder()
        self.database = self._load_database()
    
    def _load_database(self) -> Dict:
        """Lade Database aus Datei."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Database Load Fehler: {e}")
        
        return {'faces': [], 'version': '1.0', 'created': time.time()}
    
    def _save_database(self):
        """Speichere Database in Datei."""
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.database, f, indent=2)
        except Exception as e:
            st.error(f"Database Save Fehler: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Erstelle Hash für Datei."""
        try:
            stat = os.stat(file_path)
            return hashlib.md5(f"{file_path}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()
        except:
            return ""
    
    def build_database(self, progress_callback=None):
        """Baue Database neu auf."""
        st.info("🔄 Baue moderne Face Database auf...")
        
        # Sammle alle Bilder
        all_images = []
        for root, dirs, files in os.walk(IMAGE_FOLDER):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.webp')):
                    full_path = os.path.join(root, file)
                    if os.path.getsize(full_path) > 1024:  # Mindestens 1KB
                        all_images.append(full_path)
        
        if not all_images:
            st.warning("Keine Bilder gefunden!")
            return
        
        st.info(f"📊 Verarbeite {len(all_images)} Bilder...")
        
        # Reset Database
        self.database = {'faces': [], 'version': '1.0', 'created': time.time()}
        
        processed = 0
        faces_found = 0
        
        for image_path in all_images:
            try:
                if progress_callback:
                    progress_callback(processed / len(all_images))
                
                # Face Detection
                faces = self.detector.detect_faces(image_path)
                
                if faces:
                    rel_path = os.path.relpath(image_path, IMAGE_FOLDER)
                    file_hash = self._get_file_hash(image_path)
                    
                    for i, face in enumerate(faces):
                        # Face Encoding
                        encoding = self.encoder.encode_face(image_path, face['bbox'])
                        
                        if encoding is not None:
                            self.database['faces'].append({
                                'image_path': rel_path,
                                'full_path': image_path,
                                'face_index': i,
                                'bbox': face['bbox'],
                                'confidence': face['confidence'],
                                'method': face['method'],
                                'quality_score': face['quality_score'],
                                'encoding': encoding.tolist(),  # JSON serializable
                                'file_hash': file_hash,
                                'timestamp': time.time()
                            })
                            faces_found += 1
                
                processed += 1
                
                # Speichere regelmäßig
                if processed % 50 == 0:
                    self._save_database()
                    st.info(f"💾 Zwischenspeicherung: {faces_found} Gesichter in {processed} Bildern")
                
            except Exception as e:
                st.warning(f"Fehler bei {os.path.basename(image_path)}: {e}")
                processed += 1
                continue
        
        # Finale Speicherung
        self._save_database()
        
        if progress_callback:
            progress_callback(1.0)
        
        st.success(f"✅ Database aufgebaut: {faces_found} Gesichter in {processed} Bildern!")
    
    def search_similar_faces(self, query_encoding: np.ndarray, max_results=100, threshold=0.6) -> List[Dict]:
        """
        Suche ähnliche Gesichter in der Database.
        """
        if not self.database['faces']:
            return []
        
        results = []
        
        for face_data in self.database['faces']:
            try:
                # Lade Encoding
                stored_encoding = np.array(face_data['encoding'])
                
                # Berechne Distanz
                distance = self.encoder.compare_faces(query_encoding, stored_encoding)
                
                # Nur wenn unter Threshold
                if distance <= threshold:
                    result = {
                        'filename': face_data['image_path'],
                        'full_path': face_data['full_path'],
                        'dist': distance,
                        'confidence': face_data['confidence'],
                        'method': face_data['method'],
                        'quality_score': face_data['quality_score'],
                        'bbox': face_data['bbox'],
                        'source': 'modern_db'
                    }
                    results.append(result)
                    
            except Exception as e:
                continue
        
        # Sortiere nach Distanz
        results.sort(key=lambda x: x['dist'])
        
        return results[:max_results]
    
    def get_stats(self) -> Dict:
        """Hole Database Statistiken."""
        total_faces = len(self.database['faces'])
        unique_images = len(set(face['image_path'] for face in self.database['faces']))
        
        db_size_mb = 0
        if os.path.exists(self.db_file):
            db_size_mb = os.path.getsize(self.db_file) / 1024 / 1024
        
        return {
            'total_faces': total_faces,
            'unique_images': unique_images,
            'database_size_mb': db_size_mb,
            'model': self.encoder.current_model
        }


class ModernFaceSearchUI:
    """
    Moderne UI für Face Search.
    """
    
    def __init__(self):
        self.db = SimpleFaceDatabase()
        self.detector = ModernFaceDetector()
        self.encoder = ModernFaceEncoder()
    
    def render_modern_search_page(self):
        """Rendere die moderne Suchseite."""
        st.title("🚀 Moderne Gesichtssuche")
        st.markdown("---")
        
        # Database Status
        stats = self.db.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Gesichter", stats['total_faces'])
        with col2:
            st.metric("🖼️ Bilder", stats['unique_images'])
        with col3:
            st.metric("💾 DB Größe", f"{stats['database_size_mb']:.1f} MB")
        with col4:
            st.metric("🧠 Modell", stats.get('model', 'Nicht geladen'))
        
        # Database Management
        st.markdown("---")
        st.subheader("🔧 Database Management")
        
        col_rebuild, col_stats = st.columns(2)
        
        with col_rebuild:
            if st.button("🔄 Database neu aufbauen", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Fortschritt: {progress:.0%}")
                
                self.db.build_database(progress_callback)
                
                progress_bar.empty()
                status_text.empty()
                st.rerun()
        
        with col_stats:
            if st.button("📊 Detaillierte Statistiken"):
                st.json(stats)
        
        # Gesichtssuche
        if stats['total_faces'] > 0:
            st.markdown("---")
            st.subheader("🔍 Gesichtssuche")
            
            uploaded_file = st.file_uploader(
                "Lade ein Bild zum Suchen hoch",
                type=['png', 'jpg', 'jpeg', 'heic', 'heif', 'bmp', 'webp']
            )
            
            if uploaded_file is not None:
                # Zeige Upload
                col_upload, col_settings = st.columns([2, 1])
                
                with col_upload:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Suchbild", use_container_width=True)
                
                with col_settings:
                    st.subheader("⚙️ Einstellungen")
                    
                    search_threshold = st.slider(
                        "Suchgenauigkeit",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.6,
                        step=0.05,
                        help="Niedriger = strenger, Höher = mehr Ergebnisse"
                    )
                    
                    max_results = st.number_input(
                        "Max. Ergebnisse",
                        min_value=10,
                        max_value=500,
                        value=50,
                        step=10
                    )
                
                # Such-Button
                if st.button("🔍 Suche starten", type="primary", use_container_width=True):
                    self._perform_search(uploaded_file, search_threshold, max_results)
        else:
            st.warning("📝 Baue zuerst die Database auf, um die Suche zu nutzen.")
    
    def _perform_search(self, uploaded_file, threshold, max_results):
        """Führe Gesichtssuche durch."""
        with st.spinner("🔍 Analysiere Suchbild..."):
            # Speichere temporär
            temp_path = "/tmp/search_image.jpg"
            image = Image.open(uploaded_file)
            image.save(temp_path)
            
            # Face Detection im Upload
            faces = self.detector.detect_faces(temp_path)
            
            if not faces:
                st.error("❌ Keine Gesichter im Suchbild gefunden!")
                return
            
            st.success(f"✅ {len(faces)} Gesicht(er) gefunden!")
            
            # Wähle bestes Gesicht (höchste Qualität)
            best_face = max(faces, key=lambda x: x['quality_score'])
            
            # Face Encoding
            query_encoding = self.encoder.encode_face(temp_path, best_face['bbox'])
            
            if query_encoding is None:
                st.error("❌ Konnte kein Face Encoding erstellen!")
                return
        
        with st.spinner("🚀 Durchsuche Database..."):
            # Suche in Database
            results = self.db.search_similar_faces(
                query_encoding, 
                max_results=max_results, 
                threshold=threshold
            )
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Zeige Ergebnisse
        if results:
            st.success(f"🎉 {len(results)} ähnliche Gesichter gefunden!")
            
            # Ergebnis-Statistiken
            distances = [r['dist'] for r in results]
            st.info(f"📏 Distanz-Bereich: {min(distances):.3f} - {max(distances):.3f}")
            
            # Zeige Ergebnisse in Grid
            self._render_results_grid(results)
        else:
            st.warning(f"😔 Keine ähnlichen Gesichter unter Threshold {threshold:.2f} gefunden.")
            st.info("💡 Tipp: Erhöhe die Suchgenauigkeit oder verwende ein anderes Bild.")
    
    def _render_results_grid(self, results):
        """Rendere Ergebnis-Grid."""
        cols_per_row = 4
        
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, result in enumerate(results[i:i + cols_per_row]):
                with cols[j]:
                    try:
                        # Lade Bild
                        image_path = os.path.join(IMAGE_FOLDER, result['filename'])
                        
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            
                            # Caption mit Infos
                            caption = f"""
                            **{os.path.basename(result['filename'])}**
                            📏 Distanz: {result['dist']:.3f}
                            🎯 Qualität: {result['quality_score']:.2f}
                            🔧 Methode: {result['method']}
                            """
                            
                            st.image(image, caption=caption, use_container_width=True)
                        else:
                            st.error(f"❌ Bild nicht gefunden: {result['filename']}")
                    except Exception as e:
                        st.error(f"❌ Fehler: {e}")


def render_modern_search_page():
    """Hauptfunktion für moderne Gesichtssuche."""
    ui = ModernFaceSearchUI()
    ui.render_modern_search_page()


if __name__ == "__main__":
    render_modern_search_page()
