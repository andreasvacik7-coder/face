"""
Überarbeitete Suchfunktionen für effektive Gesichtserkennung.
Fokus auf Präzision und nur relevante Ergebnisse.
"""
import os
import gc
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional

from config import FACES_FOLDER, IMAGE_FOLDER, DIST_THRESHOLD, MAX_SEARCH_IMAGES, ENABLE_VECTOR_DB
from utils import load_data, get_person_name

# Lazy imports für bessere Performance
def get_face_recognition():
    """Lazy import für face_recognition"""
    try:
        import face_recognition
        return face_recognition
    except ImportError:
        return None

def get_vector_db():
    """Lazy import für Vector DB"""
    try:
        from vector_db import search_faces_with_vector_db, get_vector_db
        return search_faces_with_vector_db, get_vector_db
    except ImportError:
        return None, None


class PrecisionFaceSearch:
    """
    Präzise Gesichtssuche mit intelligenter Filterung.
    Zeigt nur wirklich passende Gesichter.
    """
    
    def __init__(self):
        self.face_recognition = get_face_recognition()
        self.vector_search, self.vector_db_func = get_vector_db()
        
        # VERSCHÄRFTE Qualitäts-Schwellwerte basierend auf User-Feedback
        self.quality_thresholds = {
            'perfect': 0.020,     # Praktisch identisch (sehr streng)
            'excellent': 0.030,   # Exzellent (streng)
            'very_good': 0.045,   # Sehr gut 
            'good': 0.055,        # Gut (0.057 war User-Grenze für korrekte Ergebnisse)
            'acceptable': 0.065   # Grenzwertig (0.068 war schon zu ungenau)
        }
    
    def search_faces(self, query_encoding: np.ndarray, progress_callback=None) -> List[Dict]:
        """
        Hauptsuchfunktion mit intelligenter Strategie-Auswahl.
        """
        if not self.face_recognition:
            return []
        
        # Prüfe Vector Database Verfügbarkeit
        vector_db_ready = self._check_vector_db_status()
        
        if vector_db_ready and ENABLE_VECTOR_DB:
            return self._search_with_vector_db(query_encoding, progress_callback)
        else:
            return self._search_traditional_precise(query_encoding, progress_callback)
    
    def _check_vector_db_status(self) -> bool:
        """Prüfe ob Vector Database bereit ist."""
        if not self.vector_db_func or not self.vector_search:
            return False
        
        try:
            vector_db = self.vector_db_func()
            return vector_db and vector_db.index and vector_db.index.ntotal > 0
        except:
            return False
    
    def _search_with_vector_db(self, query_encoding: np.ndarray, progress_callback=None) -> List[Dict]:
        """
        Suche mit Vector Database + intelligente Nachverifikation.
        """
        try:
            if progress_callback:
                progress_callback(0.1, "Verwende Vector Database für Ultra-Speed...")
            
            # Vector DB Suche
            vector_results = self.vector_search(query_encoding, progress_callback)
            
            if not vector_results:
                return []
            
            # Filtere nach Qualität
            filtered_results = []
            for result in vector_results:
                if result['dist'] <= self.quality_thresholds['acceptable']:
                    # Bestimme Qualitätslevel
                    quality = self._determine_quality(result['dist'])
                    result['quality'] = quality
                    result['quality_score'] = self._get_quality_score(quality)
                    filtered_results.append(result)
            
            # Sortiere nach Qualität
            filtered_results.sort(key=lambda x: x['dist'])
            
            # Begrenzt Ergebnisse auf Top-Matches ABER NUR wenn zu viele
            # ENTFERNE künstliche Begrenzung - zeige alle passenden Gesichter!
            if len(filtered_results) > 200:  # Nur bei extrem vielen Ergebnissen begrenzen
                print(f"⚠️ Sehr viele Ergebnisse ({len(filtered_results)}). Zeige beste 200.")
                filtered_results = filtered_results[:200]
            
            if progress_callback:
                progress_callback(1.0, f"Vector DB: {len(filtered_results)} präzise Matches gefunden")
            
            return filtered_results
            
        except Exception as e:
            print(f"Vector DB Fehler: {e}")
            return self._search_traditional_precise(query_encoding, progress_callback)
    
    def _search_traditional_precise(self, query_encoding: np.ndarray, progress_callback=None) -> List[Dict]:
        """
        Traditionelle Suche mit Fokus auf Präzision statt Vollständigkeit.
        """
        if not self.face_recognition:
            return []
        
        results = []
        
        try:
            if progress_callback:
                progress_callback(0.1, "Sammle Bilddateien für präzise Suche...")
            
            # Sammle Bilder mit Smart-Sampling für Performance
            all_images = self._collect_images_smart()
            
            if not all_images:
                return []
            
            total_images = len(all_images)
            processed = 0
            
            if progress_callback:
                progress_callback(0.2, f"Analysiere {total_images} Bilder mit hoher Präzision...")
            
            # Batch-Verarbeitung für bessere Performance
            batch_size = 15  # Moderate Batch-Größe für Balance zwischen Speed und RAM
            
            for i in range(0, len(all_images), batch_size):
                batch = all_images[i:i + batch_size]
                
                batch_results = self._process_image_batch(batch, query_encoding)
                results.extend(batch_results)
                
                processed += len(batch)
                
                # Progress Update
                if progress_callback:
                    progress = 0.2 + (processed / total_images) * 0.7
                    progress_callback(progress, f"Verarbeitet: {processed}/{total_images}")
                
                # Memory Management
                gc.collect()
            
            # Finale Filterung und Sortierung
            results = self._finalize_results(results)
            
            if progress_callback:
                progress_callback(1.0, f"Traditionelle Suche: {len(results)} präzise Matches")
            
            return results
            
        except Exception as e:
            print(f"Traditionelle Suche Fehler: {e}")
            return []
    
    def _collect_images_smart(self) -> List[Tuple[str, str]]:
        """
        Sammle Bilder mit intelligenter Strategie für beste Performance.
        """
        all_images = []
        
        # Sammle rekursiv alle Bilder
        for root, dirs, files in os.walk(IMAGE_FOLDER):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.webp')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, IMAGE_FOLDER)
                    all_images.append((rel_path, full_path))
        
        # Intelligente Begrenzung für Performance
        if len(all_images) > MAX_SEARCH_IMAGES:
            # Priorisiere Bilder aus Unterordnern (oft besser organisiert)
            subfolder_images = [img for img in all_images if os.sep in img[0]]
            main_folder_images = [img for img in all_images if os.sep not in img[0]]
            
            # Nimm alle Unterordner-Bilder + eine Auswahl aus dem Hauptordner
            remaining_slots = MAX_SEARCH_IMAGES - len(subfolder_images)
            if remaining_slots > 0:
                all_images = subfolder_images + main_folder_images[:remaining_slots]
            else:
                all_images = subfolder_images[:MAX_SEARCH_IMAGES]
        
        return all_images
    
    def _process_image_batch(self, batch: List[Tuple[str, str]], query_encoding: np.ndarray) -> List[Dict]:
        """
        Verarbeite einen Batch von Bildern für Face-Matching.
        """
        batch_results = []
        
        for rel_path, full_path in batch:
            try:
                # Lade Bild
                img_array = self.face_recognition.load_image_file(full_path)
                
                # Finde Gesichter (HOG für Performance, CNN für wichtige Bilder)
                face_locations = self.face_recognition.face_locations(img_array, model="hog")
                
                if not face_locations:
                    continue
                
                # Extrahiere Encodings
                face_encodings = self.face_recognition.face_encodings(img_array, known_face_locations=face_locations)
                
                # Prüfe jedes Gesicht im Bild
                for j, encoding in enumerate(face_encodings):
                    if encoding is None:
                        continue
                    
                    # Berechne Distanz
                    distance = self.face_recognition.face_distance([query_encoding], encoding)[0]
                    
                    # Nur sehr passende Gesichter aufnehmen
                    if distance <= self.quality_thresholds['acceptable']:
                        quality = self._determine_quality(distance)
                        
                        result = {
                            'filename': rel_path,
                            'full_path': full_path,
                            'dist': float(distance),
                            'quality': quality,
                            'quality_score': self._get_quality_score(quality),
                            'face_index': j,
                            'source': 'traditional_precise'
                        }
                        batch_results.append(result)
                
                # Memory cleanup
                del img_array, face_encodings
                
            except Exception as e:
                # Ignoriere fehlerhafter Bilder ohne die Suche zu stoppen
                continue
        
        return batch_results
    
    def _finalize_results(self, results: List[Dict]) -> List[Dict]:
        """
        Finale Verarbeitung der Suchergebnisse.
        """
        if not results:
            return []
        
        # Sortiere nach Qualität (beste zuerst)
        results.sort(key=lambda x: (x['dist'], -x['quality_score']))
        
        # Entferne Duplikate (gleiche Datei, verschiedene Gesichter)
        seen_files = set()
        unique_results = []
        
        for result in results:
            file_key = result['filename']
            if file_key not in seen_files:
                seen_files.add(file_key)
                unique_results.append(result)
        
        # Begrenzt auf vernünftige Anzahl NUR bei extrem vielen Ergebnissen
        if len(unique_results) > 150:  # Erhöht von 50 auf 150
            print(f"ℹ️ Viele gute Ergebnisse gefunden ({len(unique_results)}). Zeige beste 150.")
            unique_results = unique_results[:150]
        
        return unique_results
    
    def _determine_quality(self, distance: float) -> str:
        """Bestimme Qualitätslevel basierend auf Distanz."""
        if distance <= self.quality_thresholds['perfect']:
            return 'perfect'
        elif distance <= self.quality_thresholds['excellent']:
            return 'excellent'
        elif distance <= self.quality_thresholds['very_good']:
            return 'very_good'
        elif distance <= self.quality_thresholds['good']:
            return 'good'
        else:
            return 'acceptable'
    
    def _get_quality_score(self, quality: str) -> int:
        """Hole numerischen Score für Qualitätslevel."""
        scores = {
            'perfect': 5,
            'excellent': 4,
            'very_good': 3,
            'good': 2,
            'acceptable': 1
        }
        return scores.get(quality, 0)


# Globale Instanz für Backward-Kompatibilität
_search_engine = None

def get_search_engine():
    """Hole oder erstelle globale Suchmaschine."""
    global _search_engine
    if _search_engine is None:
        _search_engine = PrecisionFaceSearch()
    return _search_engine


def search_similar_faces_live(
    query_encoding: np.ndarray, 
    progress_callback: Optional[Callable] = None, 
    results_callback: Optional[Callable] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Hauptsuchfunktion für Backward-Kompatibilität.
    Nutzt die neue präzise Suchmaschine.
    """
    search_engine = get_search_engine()
    
    # Führe präzise Suche durch
    all_results = search_engine.search_faces(query_encoding, progress_callback)
    
    # Trenne Ergebnisse in faces und images (für Kompatibilität)
    faces_results = []
    images_results = []
    
    for result in all_results:
        # Bestimme ob es ein Face oder Image ist
        if result['filename'].startswith('face_'):
            faces_results.append(result)
        else:
            images_results.append(result)
    
    return faces_results, images_results


# Neue direkte API für moderne Nutzung
def search_faces_precise(query_encoding: np.ndarray, progress_callback=None) -> List[Dict]:
    """
    Moderne API für präzise Gesichtssuche.
    Gibt alle Ergebnisse in einer Liste zurück.
    """
    search_engine = get_search_engine()
    return search_engine.search_faces(query_encoding, progress_callback)
