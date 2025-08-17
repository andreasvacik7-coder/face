"""
Vereinfachte, moderne Face Search Engine.
Basiert nur auf OpenCV und einfachen Algorithmen - keine komplexen Dependencies.
"""
import os
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from typing import List, Dict, Optional, Tuple
import json
import hashlib
import math

from config import IMAGE_FOLDER

def get_cv2():
    """Lazy import für OpenCV"""
    try:
        import cv2
        return cv2
    except ImportError:
        st.error("❌ OpenCV nicht installiert! Installiere mit: pip install opencv-python")
        return None

def get_sklearn():
    """Lazy import für scikit-learn (für feature comparison)"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity
    except ImportError:
        st.warning("⚠️ scikit-learn nicht verfügbar - verwende einfache Cosine Similarity")
        return None


class SimpleFaceDetector:
    """
    Einfache, robuste Face Detection nur mit OpenCV.
    """
    
    def __init__(self):
        self.cv2 = get_cv2()
        self.face_cascade = None
        self.eye_cascade = None
        self.profile_cascade = None
        
        if self.cv2:
            # Lade verschiedene Haar Cascades
            try:
                cascade_path = self.cv2.data.haarcascades
                self.face_cascade = self.cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = self.cv2.CascadeClassifier(cascade_path + 'haarcascade_eye.xml')
                self.profile_cascade = self.cv2.CascadeClassifier(cascade_path + 'haarcascade_profileface.xml')
                
                # Teste ob Cascades geladen wurden
                if self.face_cascade.empty():
                    self.face_cascade = None
                if self.eye_cascade.empty():
                    self.eye_cascade = None  
                if self.profile_cascade.empty():
                    self.profile_cascade = None
                    
            except Exception as e:
                st.warning(f"Haar Cascade Fehler: {e}")
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Erkenne Gesichter mit OpenCV Haar Cascades.
        """
        if not self.cv2 or not self.face_cascade:
            return []
        
        try:
            # Lade Bild
            img = self.cv2.imread(image_path)
            if img is None:
                return []
            
            gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            faces = []
            
            # Methode 1: Standard Face Detection
            face_rects = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=self.cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in face_rects:
                # Qualitätsbewertung
                area = w * h
                total_area = height * width
                area_ratio = area / total_area
                
                # Verhältnis prüfen (Gesichter sind meist 3:4 oder 1:1)
                aspect_ratio = w / h
                aspect_score = 1.0 if 0.7 <= aspect_ratio <= 1.4 else 0.5
                
                # Position-Score (zentrale Gesichter sind oft wichtiger)
                center_x = x + w // 2
                center_y = y + h // 2
                dist_from_center = math.sqrt((center_x - width//2)**2 + (center_y - height//2)**2)
                max_dist = math.sqrt((width//2)**2 + (height//2)**2)
                position_score = 1.0 - (dist_from_center / max_dist)
                
                # Augen-Verifikation (wenn Eye Cascade verfügbar)
                eye_score = 0.5  # Default
                if self.eye_cascade:
                    face_roi = gray[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3)
                    if len(eyes) >= 2:
                        eye_score = 1.0
                    elif len(eyes) == 1:
                        eye_score = 0.7
                
                # Gesamt-Qualitätsscore
                quality_score = (
                    min(1.0, area_ratio * 20) * 0.4 +  # Größe (max bei 5% des Bildes)
                    aspect_score * 0.2 +                # Form
                    position_score * 0.2 +              # Position  
                    eye_score * 0.2                     # Augen-Verifikation
                )
                
                faces.append({
                    'method': 'OpenCV-Haar',
                    'bbox': (x, y, x + w, y + h),
                    'confidence': min(0.9, quality_score + 0.1),
                    'quality_score': quality_score,
                    'area_ratio': area_ratio,
                    'aspect_ratio': aspect_ratio,
                    'eyes_detected': len(eyes) if self.eye_cascade else 0
                })
            
            # Methode 2: Profile Face Detection (falls verfügbar)
            if self.profile_cascade and len(faces) < 2:
                profile_rects = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in profile_rects:
                    # Prüfe Überlappung mit bereits gefundenen Gesichtern
                    overlap = False
                    for existing_face in faces:
                        ex1, ey1, ex2, ey2 = existing_face['bbox']
                        if not (x > ex2 or x + w < ex1 or y > ey2 or y + h < ey1):
                            overlap = True
                            break
                    
                    if not overlap:
                        area = w * h
                        total_area = height * width
                        quality_score = min(0.8, (area / total_area) * 15)  # Profile etwas schlechter bewertet
                        
                        faces.append({
                            'method': 'OpenCV-Profile',
                            'bbox': (x, y, x + w, y + h),
                            'confidence': min(0.8, quality_score + 0.1),
                            'quality_score': quality_score,
                            'area_ratio': area / total_area,
                            'aspect_ratio': w / h,
                            'eyes_detected': 0
                        })
            
            # Sortiere nach Qualität
            faces.sort(key=lambda x: x['quality_score'], reverse=True)
            
            return faces
            
        except Exception as e:
            st.error(f"Face Detection Fehler in {image_path}: {e}")
            return []


class SimpleFeatureExtractor:
    """
    Einfache Feature-Extraktion für Gesichtsvergleich.
    Verwendet OpenCV ohne komplexe Deep Learning Modelle.
    """
    
    def __init__(self):
        self.cv2 = get_cv2()
        self.cosine_similarity = get_sklearn()
        
        # Initialisiere ORB Feature Detector
        if self.cv2:
            try:
                self.orb = self.cv2.ORB_create(nfeatures=500)
                self.sift = None
                
                # Versuche SIFT (bessere Qualität, falls verfügbar)
                try:
                    self.sift = self.cv2.SIFT_create(nfeatures=100)
                except AttributeError:
                    # SIFT nicht verfügbar in OpenCV-Python
                    pass
                    
            except Exception as e:
                st.warning(f"Feature Extractor Initialisierung: {e}")
                self.orb = None
                self.sift = None
    
    def extract_features(self, image_path: str, face_bbox: Tuple[int, int, int, int] = None) -> Optional[np.ndarray]:
        """
        Extrahiere einfache Features aus einem Gesicht.
        """
        if not self.cv2 or not self.orb:
            return None
        
        try:
            # Lade Bild
            img = self.cv2.imread(image_path)
            if img is None:
                return None
            
            gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
            
            # Crop Gesicht falls bbox gegeben
            if face_bbox:
                x1, y1, x2, y2 = face_bbox
                face_gray = gray[y1:y2, x1:x2]
            else:
                face_gray = gray
            
            # Resize auf Standard-Größe für Konsistenz
            face_resized = self.cv2.resize(face_gray, (128, 128))
            
            # Feature-Extraktion mit mehreren Methoden
            features = []
            
            # 1. Histogramm Features (einfach aber effektiv)
            hist_features = self._extract_histogram_features(face_resized)
            features.extend(hist_features)
            
            # 2. Texture Features (LBP-ähnlich)
            texture_features = self._extract_texture_features(face_resized)
            features.extend(texture_features)
            
            # 3. ORB Keypoint Features (falls verfügbar)
            if self.orb:
                orb_features = self._extract_orb_features(face_resized)
                if orb_features is not None:
                    features.extend(orb_features)
            
            # 4. Gradient Features
            gradient_features = self._extract_gradient_features(face_resized)
            features.extend(gradient_features)
            
            # Normalisiere Features
            features_array = np.array(features, dtype=np.float32)
            if len(features_array) > 0:
                # L2 Normalisierung
                norm = np.linalg.norm(features_array)
                if norm > 0:
                    features_array = features_array / norm
            
            return features_array
            
        except Exception as e:
            st.warning(f"Feature Extraction Fehler: {e}")
            return None
    
    def _extract_histogram_features(self, face_gray: np.ndarray) -> List[float]:
        """Extrahiere Histogramm-Features."""
        try:
            # Gesamt-Histogramm
            hist = self.cv2.calcHist([face_gray], [0], None, [64], [0, 256])
            hist_features = hist.flatten().tolist()
            
            # Regional-Histogramme (4 Quadranten)
            h, w = face_gray.shape
            quadrants = [
                face_gray[0:h//2, 0:w//2],      # Top-left
                face_gray[0:h//2, w//2:w],      # Top-right
                face_gray[h//2:h, 0:w//2],      # Bottom-left
                face_gray[h//2:h, w//2:w]       # Bottom-right
            ]
            
            for quad in quadrants:
                if quad.size > 0:
                    quad_hist = self.cv2.calcHist([quad], [0], None, [16], [0, 256])
                    hist_features.extend(quad_hist.flatten().tolist())
            
            return hist_features
            
        except Exception:
            return [0.0] * 128  # Fallback
    
    def _extract_texture_features(self, face_gray: np.ndarray) -> List[float]:
        """Extrahiere einfache Texture-Features."""
        try:
            features = []
            
            # Sobel Gradients
            grad_x = self.cv2.Sobel(face_gray, self.cv2.CV_64F, 1, 0, ksize=3)
            grad_y = self.cv2.Sobel(face_gray, self.cv2.CV_64F, 0, 1, ksize=3)
            
            features.extend([
                np.mean(np.abs(grad_x)),
                np.std(grad_x),
                np.mean(np.abs(grad_y)),
                np.std(grad_y)
            ])
            
            # Laplacian
            laplacian = self.cv2.Laplacian(face_gray, self.cv2.CV_64F)
            features.extend([
                np.mean(np.abs(laplacian)),
                np.std(laplacian)
            ])
            
            # Lokale Standardabweichung (vereinfachtes LBP)
            kernel = np.ones((3, 3), np.float32) / 9
            blurred = self.cv2.filter2D(face_gray, -1, kernel)
            local_std = np.sqrt(self.cv2.filter2D((face_gray - blurred)**2, -1, kernel))
            features.extend([
                np.mean(local_std),
                np.std(local_std)
            ])
            
            return features
            
        except Exception:
            return [0.0] * 8  # Fallback
    
    def _extract_orb_features(self, face_gray: np.ndarray) -> Optional[List[float]]:
        """Extrahiere ORB Keypoint Features."""
        try:
            if not self.orb:
                return None
            
            # Finde Keypoints und Descriptors
            keypoints, descriptors = self.orb.detectAndCompute(face_gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # Aggregiere Descriptors zu einem Feature Vector
                # Verwende statistische Maße
                features = [
                    np.mean(descriptors, axis=0),
                    np.std(descriptors, axis=0),
                    np.median(descriptors, axis=0)
                ]
                
                # Flattene alle Features
                orb_features = []
                for feat_array in features:
                    orb_features.extend(feat_array.tolist())
                
                # Begrenze auf 100 Features für Performance
                return orb_features[:100]
            
            return None
            
        except Exception:
            return None
    
    def _extract_gradient_features(self, face_gray: np.ndarray) -> List[float]:
        """Extrahiere Gradient-basierte Features."""
        try:
            features = []
            
            # Unterschiedliche Kernel-Größen
            for ksize in [3, 5]:
                grad_x = self.cv2.Sobel(face_gray, self.cv2.CV_64F, 1, 0, ksize=ksize)
                grad_y = self.cv2.Sobel(face_gray, self.cv2.CV_64F, 0, 1, ksize=ksize)
                
                # Magnitude und Richtung
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                direction = np.arctan2(grad_y, grad_x)
                
                features.extend([
                    np.mean(magnitude),
                    np.std(magnitude),
                    np.mean(direction),
                    np.std(direction)
                ])
            
            return features
            
        except Exception:
            return [0.0] * 8  # Fallback
    
    def compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Vergleiche zwei Feature-Vektoren und gib Distanz zurück.
        """
        try:
            if len(features1) != len(features2):
                return 2.0  # Maximale Distanz
            
            # Verwende Cosine Distance wenn verfügbar
            if self.cosine_similarity:
                similarity = self.cosine_similarity([features1], [features2])[0][0]
                distance = 1.0 - similarity
            else:
                # Einfache Cosine Distance Implementation
                dot_product = np.dot(features1, features2)
                norm1 = np.linalg.norm(features1)
                norm2 = np.linalg.norm(features2)
                
                if norm1 == 0 or norm2 == 0:
                    return 2.0
                
                cosine_sim = dot_product / (norm1 * norm2)
                distance = 1.0 - cosine_sim
            
            # Stelle sicher dass Distanz zwischen 0 und 2 liegt
            return max(0.0, min(2.0, distance))
            
        except Exception as e:
            st.warning(f"Feature Comparison Fehler: {e}")
            return 2.0  # Maximale Distanz bei Fehler


class SimpleFaceDatabase:
    """
    Einfache, JSON-basierte Face Database.
    """
    
    def __init__(self):
        self.db_file = os.path.join(os.path.dirname(__file__), 'simple_face_db.json')
        self.detector = SimpleFaceDetector()
        self.extractor = SimpleFeatureExtractor()
        self.database = self._load_database()
    
    def _load_database(self) -> Dict:
        """Lade Database aus JSON Datei mit Fehlerbehandlung."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validiere geladene Daten
                if not isinstance(data, dict):
                    raise ValueError("Database ist kein Dictionary")
                
                if 'faces' not in data:
                    raise ValueError("Database hat kein 'faces' Feld")
                
                if not isinstance(data['faces'], list):
                    raise ValueError("'faces' ist keine Liste")
                
                print(f"✅ Database geladen: {len(data['faces'])} Gesichter")
                return data
                
            except json.JSONDecodeError as e:
                st.error(f"JSON Parse Fehler: {e}")
                # Versuche Backup zu laden
                backup_file = self.db_file + '.backup'
                if os.path.exists(backup_file):
                    try:
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        st.warning("Backup-Database geladen")
                        return data
                    except Exception:
                        st.error("Auch Backup ist korrupt")
                
            except Exception as e:
                st.error(f"Database Load Fehler: {e}")
        
        # Erstelle neue Database
        print("📝 Erstelle neue leere Database")
        return {
            'faces': [], 
            'version': '2.0', 
            'created': time.time(),
            'feature_type': 'simple_cv'
        }
    
    def _save_database(self):
        """Speichere Database als JSON mit Fehlerbehandlung."""
        try:
            # Erstelle Backup der existierenden Datei
            if os.path.exists(self.db_file):
                backup_file = self.db_file + '.backup'
                import shutil
                shutil.copy2(self.db_file, backup_file)
            
            # Validiere Database vor dem Speichern
            if not isinstance(self.database, dict):
                raise ValueError("Database ist kein Dictionary")
            
            if 'faces' not in self.database:
                raise ValueError("Database hat kein 'faces' Feld")
            
            # Speichere in temporäre Datei zuerst
            temp_file = self.db_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.database, f, indent=2, ensure_ascii=False)
            
            # Teste ob die temporäre Datei gültig ist
            with open(temp_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                if 'faces' not in test_data:
                    raise ValueError("Gespeicherte Datei ist ungültig")
            
            # Wenn alles ok, ersetze die echte Datei
            import shutil
            shutil.move(temp_file, self.db_file)
            
            print(f"✅ Database gespeichert: {len(self.database['faces'])} Gesichter")
                
        except Exception as e:
            st.error(f"Database Save Fehler: {e}")
            # Cleanup der temporären Datei
            temp_file = self.db_file + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def _get_file_hash(self, file_path: str) -> str:
        """Erstelle Hash für Datei."""
        try:
            stat = os.stat(file_path)
            return hashlib.md5(f"{file_path}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()
        except:
            return ""
    
    def build_database(self, progress_callback=None):
        """Baue Database neu auf."""
        st.info("🔄 Baue einfache Face Database auf...")
        
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
        
        st.info(f"📊 Verarbeite {len(all_images)} Bilder mit OpenCV...")
        
        # Reset Database
        self.database = {
            'faces': [], 
            'version': '2.0', 
            'created': time.time(),
            'feature_type': 'simple_cv'
        }
        
        processed = 0
        faces_found = 0
        errors = 0
        
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
                        # Feature Extraction
                        features = self.extractor.extract_features(image_path, face['bbox'])
                        
                        if features is not None and len(features) > 0:
                            self.database['faces'].append({
                                'image_path': rel_path,
                                'full_path': image_path,
                                'face_index': i,
                                'bbox': face['bbox'],
                                'confidence': face['confidence'],
                                'method': face['method'],
                                'quality_score': face['quality_score'],
                                'features': features.tolist(),  # JSON serializable
                                'feature_length': len(features),
                                'file_hash': file_hash,
                                'timestamp': time.time(),
                                'area_ratio': face.get('area_ratio', 0),
                                'eyes_detected': face.get('eyes_detected', 0)
                            })
                            faces_found += 1
                
                processed += 1
                
                # Speichere regelmäßig
                if processed % 100 == 0:
                    self._save_database()
                    st.info(f"💾 Zwischenspeicherung: {faces_found} Gesichter in {processed} Bildern")
                
            except Exception as e:
                errors += 1
                if errors < 10:  # Zeige nur erste 10 Fehler
                    st.warning(f"Fehler bei {os.path.basename(image_path)}: {e}")
                processed += 1
                continue
        
        # Finale Speicherung
        self._save_database()
        
        if progress_callback:
            progress_callback(1.0)
        
        st.success(f"✅ Einfache Database aufgebaut!")
        st.info(f"""
        📊 **Statistiken:**
        - **Gesichter gefunden:** {faces_found}
        - **Bilder verarbeitet:** {processed}
        - **Fehler:** {errors}
        - **Feature-Typ:** OpenCV + Histogramme + ORB
        - **Durchschnitt:** {faces_found/max(1,processed-errors):.1f} Gesichter pro erfolgreichem Bild
        """)
    
    def search_similar_faces(self, query_features: np.ndarray, max_results=100, threshold=0.8) -> List[Dict]:
        """
        Suche ähnliche Gesichter in der Database.
        """
        if not self.database['faces'] or query_features is None:
            return []
        
        results = []
        
        for face_data in self.database['faces']:
            try:
                # Lade gespeicherte Features
                stored_features = np.array(face_data['features'])
                
                # Berechne Distanz
                distance = self.extractor.compare_features(query_features, stored_features)
                
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
                        'area_ratio': face_data.get('area_ratio', 0),
                        'eyes_detected': face_data.get('eyes_detected', 0),
                        'source': 'simple_cv_db'
                    }
                    results.append(result)
                    
            except Exception as e:
                continue
        
        # Sortiere nach Distanz (beste zuerst)
        results.sort(key=lambda x: x['dist'])
        
        return results[:max_results]
    
    def get_stats(self) -> Dict:
        """Hole Database Statistiken."""
        total_faces = len(self.database['faces'])
        unique_images = len(set(face['image_path'] for face in self.database['faces']))
        
        db_size_mb = 0
        if os.path.exists(self.db_file):
            db_size_mb = os.path.getsize(self.db_file) / 1024 / 1024
        
        # Feature-Statistiken
        if total_faces > 0:
            feature_lengths = [len(face['features']) for face in self.database['faces']]
            avg_feature_length = sum(feature_lengths) / len(feature_lengths)
            
            # Qualitäts-Statistiken
            quality_scores = [face['quality_score'] for face in self.database['faces']]
            avg_quality = sum(quality_scores) / len(quality_scores)
        else:
            avg_feature_length = 0
            avg_quality = 0
        
        return {
            'total_faces': total_faces,
            'unique_images': unique_images,
            'database_size_mb': db_size_mb,
            'feature_type': self.database.get('feature_type', 'unknown'),
            'avg_feature_length': avg_feature_length,
            'avg_quality_score': avg_quality,
            'version': self.database.get('version', '1.0')
        }


class SimpleFaceSearchUI:
    """
    Einfache UI für die vereinfachte Face Search.
    """
    
    def __init__(self):
        self.db = SimpleFaceDatabase()
        self.detector = SimpleFaceDetector()
        self.extractor = SimpleFeatureExtractor()
    
    def render_simple_search_page(self):
        """Rendere die vereinfachte Suchseite."""
        st.title("🎯 Vereinfachte Gesichtssuche")
        st.info("💡 **Einfache OpenCV-basierte Lösung** - keine komplexen AI-Modelle, schnell und zuverlässig!")
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
            st.metric("⭐ Ø Qualität", f"{stats.get('avg_quality_score', 0):.2f}")
        
        # Zusätzliche Infos
        if stats['total_faces'] > 0:
            st.info(f"""
            🔧 **Technische Details:**
            - Feature-Typ: {stats.get('feature_type', 'unbekannt')}
            - Ø Feature-Länge: {stats.get('avg_feature_length', 0):.0f}
            - Database Version: {stats.get('version', '1.0')}
            """)
        
        # Database Management
        st.markdown("---")
        st.subheader("🔧 Database Management")
        
        col_rebuild, col_test = st.columns(2)
        
        with col_rebuild:
            if st.button("🔄 Database neu aufbauen", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Fortschritt: {progress:.0%}")
                
                start_time = time.time()
                self.db.build_database(progress_callback)
                duration = time.time() - start_time
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Database aufgebaut in {duration:.1f}s")
                st.rerun()
        
        with col_test:
            if st.button("🧪 OpenCV Test"):
                self._test_opencv_functionality()
        
        # Gesichtssuche
        if stats['total_faces'] > 0:
            st.markdown("---")
            st.subheader("🔍 Gesichtssuche")
            
            uploaded_file = st.file_uploader(
                "Lade ein Bild zum Suchen hoch",
                type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
                help="HEIC/HEIF wird noch nicht unterstützt in der einfachen Version"
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
                        "Ähnlichkeits-Schwelle",
                        min_value=0.1,
                        max_value=1.5,
                        value=0.8,
                        step=0.05,
                        help="Niedriger = strenger (mehr ähnlich), Höher = lockerer"
                    )
                    
                    max_results = st.number_input(
                        "Max. Ergebnisse",
                        min_value=10,
                        max_value=200,
                        value=50,
                        step=10
                    )
                    
                    show_details = st.checkbox(
                        "Detaillierte Infos zeigen",
                        value=False,
                        help="Zeige technische Details zu jedem Ergebnis"
                    )
                
                # Such-Button
                if st.button("🔍 Suche starten", type="primary", use_container_width=True):
                    self._perform_simple_search(uploaded_file, search_threshold, max_results, show_details)
        else:
            st.warning("📝 Baue zuerst die Database auf, um die Suche zu nutzen.")
            
        # Hilfe-Bereich
        st.markdown("---")
        with st.expander("ℹ️ Wie funktioniert die vereinfachte Suche?"):
            st.markdown("""
            **🎯 Einfache, robuste Technologie:**
            - **Face Detection:** OpenCV Haar Cascades (frontal + profile)
            - **Features:** Histogramme + Texturen + ORB Keypoints + Gradienten
            - **Vergleich:** Cosine Distance zwischen Feature-Vektoren
            - **Speicherung:** Einfache JSON-Datei (keine komplexe Vector DB)
            
            **✅ Vorteile:**
            - Keine komplexen AI-Abhängigkeiten
            - Schnelle Installation (nur OpenCV)
            - Robust und stabil
            - Einfach zu verstehen und debuggen
            
            **⚠️ Einschränkungen:**
            - Weniger präzise als moderne Deep Learning Modelle
            - Haar Cascades finden nicht alle Gesichter
            - Funktioniert am besten mit frontalen, gut beleuchteten Gesichtern
            
            **💡 Tipps für beste Ergebnisse:**
            - Verwende klare, gut beleuchtete Fotos
            - Frontale Gesichter funktionieren am besten
            - Experimentiere mit der Ähnlichkeits-Schwelle
            - Werte um 0.6-0.8 sind meist optimal
            """)
    
    def _test_opencv_functionality(self):
        """Teste OpenCV Funktionalität."""
        st.info("🧪 Teste OpenCV Komponenten...")
        
        cv2 = get_cv2()
        if not cv2:
            st.error("❌ OpenCV nicht verfügbar!")
            return
        
        st.success("✅ OpenCV verfügbar")
        
        # Teste Haar Cascades
        try:
            cascade_path = cv2.data.haarcascades
            face_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_eye.xml')
            profile_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_profileface.xml')
            
            st.success("✅ Haar Cascades verfügbar (Face, Eye, Profile)")
            
            # Teste ORB
            orb = cv2.ORB_create(nfeatures=100)
            st.success("✅ ORB Feature Detector verfügbar")
            
            # Teste SIFT (optional)
            try:
                sift = cv2.SIFT_create(nfeatures=50)
                st.success("✅ SIFT Feature Detector verfügbar (Premium)")
            except:
                st.info("ℹ️ SIFT nicht verfügbar (benötigt opencv-contrib-python)")
            
        except Exception as e:
            st.error(f"❌ OpenCV Komponenten-Fehler: {e}")
        
        # Teste scikit-learn
        sklearn_cosine = get_sklearn()
        if sklearn_cosine:
            st.success("✅ scikit-learn Cosine Similarity verfügbar")
        else:
            st.info("ℹ️ scikit-learn nicht verfügbar - verwende NumPy Fallback")
    
    def _perform_simple_search(self, uploaded_file, threshold, max_results, show_details):
        """Führe vereinfachte Gesichtssuche durch."""
        with st.spinner("🔍 Analysiere Suchbild..."):
            # Speichere temporär
            temp_path = "/tmp/search_image.jpg"
            image = Image.open(uploaded_file)
            image.save(temp_path)
            
            # Face Detection im Upload
            faces = self.detector.detect_faces(temp_path)
            
            if not faces:
                st.error("❌ Keine Gesichter im Suchbild gefunden!")
                st.info("💡 Tipp: Verwende ein Bild mit klaren, frontalen Gesichtern")
                return
            
            st.success(f"✅ {len(faces)} Gesicht(er) gefunden!")
            
            # Zeige gefundene Gesichter
            if show_details:
                for i, face in enumerate(faces):
                    st.info(f"""
                    **Gesicht {i+1}:**
                    - Methode: {face['method']}
                    - Qualität: {face['quality_score']:.2f}
                    - Confidence: {face['confidence']:.2f}
                    - Augen erkannt: {face.get('eyes_detected', 0)}
                    """)
            
            # Wähle bestes Gesicht
            best_face = max(faces, key=lambda x: x['quality_score'])
            st.info(f"🎯 Verwende Gesicht mit Qualität {best_face['quality_score']:.2f}")
            
            # Feature Extraction
            query_features = self.extractor.extract_features(temp_path, best_face['bbox'])
            
            if query_features is None:
                st.error("❌ Konnte keine Features aus dem Gesicht extrahieren!")
                return
            
            if show_details:
                st.info(f"📊 Feature-Vektor: {len(query_features)} Dimensionen")
        
        with st.spinner("🚀 Durchsuche Database..."):
            start_time = time.time()
            
            # Suche in Database
            results = self.db.search_similar_faces(
                query_features, 
                max_results=max_results, 
                threshold=threshold
            )
            
            search_time = time.time() - start_time
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Zeige Ergebnisse
        if results:
            st.success(f"🎉 {len(results)} ähnliche Gesichter gefunden in {search_time:.2f}s!")
            
            # Ergebnis-Statistiken
            distances = [r['dist'] for r in results]
            st.info(f"📏 Distanz-Bereich: {min(distances):.3f} - {max(distances):.3f}")
            
            # Zeige Ergebnisse in Grid
            self._render_simple_results_grid(results, show_details)
        else:
            st.warning(f"😔 Keine ähnlichen Gesichter unter Schwelle {threshold:.2f} gefunden.")
            st.info("💡 Tipps: Erhöhe die Ähnlichkeits-Schwelle oder verwende ein anderes Bild.")
    
    def _render_simple_results_grid(self, results, show_details):
        """Rendere Ergebnis-Grid für vereinfachte Suche."""
        cols_per_row = 3 if show_details else 4
        
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, result in enumerate(results[i:i + cols_per_row]):
                with cols[j]:
                    try:
                        # Lade Bild
                        image_path = os.path.join(IMAGE_FOLDER, result['filename'])
                        
                        if os.path.exists(image_path):
                            # Erstelle Bild mit Gesichts-Markierung
                            marked_image = self._create_simple_marked_image(image_path, result['bbox'])
                            
                            # Caption mit Infos
                            caption = f"""
                            **{os.path.basename(result['filename'])}**
                            📏 Distanz: {result['dist']:.3f}
                            🎯 Qualität: {result['quality_score']:.2f}
                            """
                            
                            if show_details:
                                caption += f"""
                                🔧 Methode: {result['method']}
                                👁️ Augen: {result.get('eyes_detected', 0)}
                                📐 Fläche: {result.get('area_ratio', 0):.1%}
                                """
                            
                            st.image(marked_image, caption=caption, use_container_width=True)
                        else:
                            st.error(f"❌ Bild nicht gefunden: {result['filename']}")
                    except Exception as e:
                        st.error(f"❌ Fehler: {e}")
    
    def _create_simple_marked_image(self, image_path: str, face_bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Erstelle Bild mit einfacher Gesichts-Markierung.
        """
        try:
            image = Image.open(image_path)
            
            # Erstelle Kopie für Markierungen
            marked_image = image.copy()
            draw = ImageDraw.Draw(marked_image)
            
            # Entpacke Gesichts-Position
            x1, y1, x2, y2 = face_bbox
            
            # Zeichne grünen Rahmen
            line_width = max(2, min(6, (x2 - x1) // 50))
            
            draw.rectangle(
                [(x1 - line_width, y1 - line_width), (x2 + line_width, y2 + line_width)], 
                outline="lime", 
                width=line_width
            )
            
            # Füge Checkmark hinzu
            try:
                draw.text(
                    (x1, y1 - 20), 
                    "✓", 
                    fill="lime"
                )
            except:
                pass
            
            return marked_image
            
        except Exception:
            # Bei Fehler: Original-Bild zurückgeben
            return Image.open(image_path)


def render_simple_search_page():
    """Hauptfunktion für vereinfachte Gesichtssuche."""
    ui = SimpleFaceSearchUI()
    ui.render_simple_search_page()


if __name__ == "__main__":
    render_simple_search_page()
