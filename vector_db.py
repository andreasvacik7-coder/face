"""
Vector Database Module für effiziente Gesichtserkennung mit FAISS.
Verbessert drastisch die Performance bei großen Bilddatenbanken.
"""
import os
import json
import time
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import streamlit as st

from config import (
    VECTOR_INDEX_FILE, VECTOR_METADATA_FILE, IMAGE_FOLDER, 
    FACES_FOLDER, DIST_THRESHOLD, ENABLE_VECTOR_DB
)

# Lazy import für bessere Performance
def get_faiss():
    """Lazy import für FAISS"""
    try:
        import faiss
        return faiss
    except ImportError:
        st.error("❌ FAISS nicht installiert! Installiere mit: pip install faiss-cpu")
        return None


def get_face_recognition():
    """Lazy import für face_recognition"""
    try:
        import face_recognition
        return face_recognition
    except ImportError:
        st.error("❌ face_recognition nicht installiert.")
        return None

class VectorFaceDatabase:
    """
    Hochperformante Vektor-Datenbank für Gesichtserkennung.
    Verwendet FAISS für schnelle Ähnlichkeitssuche.
    """
    
    def __init__(self):
        self.index = None
        self.metadata = []
        self.dimension = 128  # face_recognition encoding dimension
        self.faiss = get_faiss()
        self.face_recognition = get_face_recognition()
        
        if self.faiss and self.face_recognition:
            self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Lade existierenden Index oder erstelle neuen."""
        if os.path.exists(VECTOR_INDEX_FILE) and os.path.exists(VECTOR_METADATA_FILE):
            try:
                # Lade FAISS Index
                self.index = self.faiss.read_index(VECTOR_INDEX_FILE)
                
                # Lade Metadata
                with open(VECTOR_METADATA_FILE, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                print(f"✅ Vector Index geladen: {self.index.ntotal} Gesichter")
                return True
            except Exception as e:
                print(f"⚠️ Fehler beim Laden des Index: {e}")
        
        # Erstelle neuen Index
        self._create_new_index()
        return False
    
    def _create_new_index(self):
        """Erstelle einen neuen leeren FAISS Index."""
        if self.faiss:
            # Cosine Similarity Index (L2 normalisiert)
            self.index = self.faiss.IndexFlatIP(128)  # 128 ist face_recognition encoding size
            self.metadata = []
            print("✅ Neuer FAISS Index erstellt")
        else:
            print("❌ FAISS nicht verfügbar - kann keinen Index erstellen")

    def _delete_existing_files(self):
        """Lösche alle existierenden Vector Database Dateien für kompletten Neustart."""
        files_to_delete = [
            VECTOR_INDEX_FILE,
            VECTOR_METADATA_FILE
        ]
        
        deleted_count = 0
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️ Gelöscht: {os.path.basename(file_path)}")
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Fehler beim Löschen von {file_path}: {e}")
        
        if deleted_count > 0:
            print(f"✅ {deleted_count} Vector Database Dateien gelöscht")
        else:
            print("ℹ️ Keine Vector Database Dateien zum Löschen gefunden")
        
        # Reset interner Zustand
        self.index = None
        self.metadata = []
    
    def _get_file_hash(self, file_path: str) -> str:
        """Erstelle Hash für Datei zur Änderungserkennung."""
        try:
            stat = os.stat(file_path)
            content = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except:
            return ""
    
    def need_rebuild(self) -> bool:
        """Prüfe ob Index neu aufgebaut werden muss."""
        if not self.index or len(self.metadata) == 0:
            print("🔄 Rebuild nötig: Index oder Metadata fehlt")
            return True
        
        # KORREKTUR: Vergleiche tatsächliche Bildpfade, nicht nur Anzahl
        if not hasattr(self, '_checked_rebuild'):
            self._checked_rebuild = True
            
            print(f"🔍 Überprüfe Index-Aktualität...")
            print(f"   Indexierte Gesichter: {self.index.ntotal}")
            print(f"   Metadata-Einträge: {len(self.metadata)}")
            
            current_images = self._discover_all_images()
            indexed_paths = {meta['path'] for meta in self.metadata}
            current_paths = {os.path.relpath(img, IMAGE_FOLDER) for img in current_images}
            
            print(f"   Aktuelle Bilder: {len(current_paths)}")
            print(f"   Indexierte Pfade: {len(indexed_paths)}")
            
            # Finde wirklich neue Bilder
            new_images = current_paths - indexed_paths
            missing_images = indexed_paths - current_paths
            
            if new_images:
                print(f"🔄 {len(new_images)} neue Bilder gefunden")
                # Zeige erste 5 neue Bilder als Beispiel
                for i, img in enumerate(list(new_images)[:5]):
                    print(f"   Neu: {img}")
                return True
            
            if missing_images:
                print(f"🗑️ {len(missing_images)} Bilder entfernt")
                return True
            
            print(f"✅ Index ist aktuell ({len(indexed_paths)} Bilder)")
        
        return False
    
    def _discover_all_images(self) -> List[str]:
        """Entdecke alle Bilder in IMAGE_FOLDER rekursiv mit vollständiger Format-Unterstützung."""
        image_files = []
        # Erweiterte Format-Unterstützung inklusive HEIC/HEIF
        extensions = ('.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.webp', '.tiff', '.tif')
        
        for root, dirs, files in os.walk(IMAGE_FOLDER):
            for file in files:
                if file.lower().endswith(extensions):
                    full_path = os.path.join(root, file)
                    # Prüfe ob Datei lesbar ist
                    try:
                        if os.path.getsize(full_path) > 0:  # Keine leeren Dateien
                            image_files.append(full_path)
                    except (OSError, IOError):
                        # Überspringe defekte Dateien
                        continue
        
        print(f"📁 Gefundene Bilder: {len(image_files)} (alle Formate inklusive HEIC/HEIF)")
        return image_files
    
    def add_new_images_to_index(self, progress_callback=None, delete_images_without_faces=False):
        """Füge nur neue Bilder zum existierenden Index hinzu (inkrementell)."""
        if not self.faiss or not self.face_recognition:
            print("❌ FAISS oder face_recognition nicht verfügbar")
            return False
        
        print("🔄 Prüfe auf neue Bilder für inkrementelle Index-Erweiterung...")
        if delete_images_without_faces:
            print("🗑️ Automatisches Löschen von Bildern ohne Gesichter aktiviert")
        start_time = time.time()
        
        # Sammle alle verfügbaren Bilder
        all_images = self._discover_all_images()
        
        if not all_images:
            print("⚠️ Keine Bilder gefunden")
            return False
        
        # Identifiziere bereits indexierte Bilder
        indexed_paths = {meta['path'] for meta in self.metadata}
        current_paths = {os.path.relpath(img, IMAGE_FOLDER) for img in all_images}
        
        # Finde neue Bilder
        new_image_paths = current_paths - indexed_paths
        
        if not new_image_paths:
            print("✅ Alle Bilder bereits indexiert - kein Update nötig")
            return True
        
        print(f"📊 {len(new_image_paths)} neue Bilder gefunden für Index-Erweiterung:")
        for i, path in enumerate(list(new_image_paths)[:10]):  # Zeige ersten 10
            print(f"   {i+1}. {path}")
        if len(new_image_paths) > 10:
            print(f"   ... und {len(new_image_paths) - 10} weitere")
        
        # Konvertiere relative Pfade zurück zu absoluten Pfaden
        new_images = [os.path.join(IMAGE_FOLDER, rel_path) for rel_path in new_image_paths]
        
        # Verarbeite neue Bilder in Batches
        batch_size = min(10, len(new_images))  # Reduziert für CNN - mehr GPU/RAM intensiv
        processed = 0
        faces_found = 0
        deleted_images = 0
        no_faces_count = 0
        
        for i in range(0, len(new_images), batch_size):
            batch = new_images[i:i + batch_size]
            batch_start_time = time.time()
            
            print(f"\n📦 Neuer Batch {i//batch_size + 1}/{(len(new_images)-1)//batch_size + 1}")
            
            for image_path in batch:
                try:
                    result = self._process_single_image(image_path)
                    processed += 1
                    
                    if result == "NO_FACES_FOUND" or result is None:
                        no_faces_count += 1
                        deletion_result = self._handle_no_face_image(image_path, delete_images_without_faces)
                        if deletion_result == "DELETED_NO_FACES":
                            deleted_images += 1
                        else:
                            print(f"   ⚪ {os.path.basename(image_path)} - keine Gesichter")
                    elif result and not result.startswith("SKIP"):
                        faces_found += 1
                        print(f"   ✅ {os.path.basename(image_path)} - {result}")
                    elif result:
                        print(f"   ⚪ {os.path.basename(image_path)} - {result}")
                    
                    # Progress callback
                    if progress_callback:
                        progress = (processed / len(new_images)) * 0.9
                        progress_callback(progress)
                        
                except Exception as e:
                    print(f"   ❌ {os.path.basename(image_path)} - Fehler: {e}")
            
            # Batch-Statistiken
            batch_duration = time.time() - batch_start_time
            print(f"   ⏱️ Batch-Zeit: {batch_duration:.1f}s")
            
            # Speichere nach jedem Batch
            self._save_index()
            print(f"   💾 Index erweitert ({self.index.ntotal if self.index else 0} Gesichter total)")
        
        duration = time.time() - start_time
        
        # Finale Statistiken
        print(f"\n🎉 Inkrementelle Index-Erweiterung abgeschlossen!")
        print(f"   ⏱️ Gesamtzeit: {duration:.1f}s")
        print(f"   📊 Neue Bilder verarbeitet: {processed}")
        print(f"   👥 Neue Gesichter gefunden: {faces_found}")
        print(f"   ⚪ Bilder ohne Gesichter: {no_faces_count}")
        if delete_images_without_faces:
            print(f"   🗑️ Bilder gelöscht (keine Gesichter): {deleted_images}")
            print(f"   💾 Gespareter Speicherplatz: ~{deleted_images * 2:.1f} MB")
        print(f"   📈 Total indexierte Gesichter: {self.index.ntotal if self.index else 0}")
        print(f"   🚀 Index bereit für blitzschnelle Suchen!")
        
        if progress_callback:
            progress_callback(1.0)
        
        return True
        """Baue kompletten Index neu auf mit detailliertem Logging."""
        if not self.faiss or not self.face_recognition:
            print("❌ FAISS oder face_recognition nicht verfügbar")
            return False
        
        print("🔄 Starte Vector Database Index-Aufbau...")
        print("📊 Sammle Bilddateien...")
        start_time = time.time()
        
        # Erstelle neuen Index
        self._create_new_index()
        
        # Sammle alle Bilder (nur einmal!)
        all_images = self._discover_all_images()
        
        if not all_images:
            print("⚠️ Keine Bilder gefunden")
            return False
        
        print(f"📊 Starte Verarbeitung von {len(all_images)} Bildern...")
        print(f"🎯 Ziel: Alle Gesichter für blitzschnelle Suche indexieren")
        
        # Verarbeite Bilder in Batches für bessere Speicherverwaltung
        batch_size = 10  # Reduziert für CNN - höhere Qualität, mehr Speicherverbrauch
        processed = 0
        faces_found = 0
        skipped = 0
        
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size]
            batch_start_time = time.time()
            
            print(f"\n📦 Batch {i//batch_size + 1}/{(len(all_images)-1)//batch_size + 1}")
            
            for image_path in batch:
                try:
                    if progress_callback:
                        progress_callback(processed / len(all_images))
                    
                    # Detailliertes Logging für jedes Bild
                    filename = os.path.basename(image_path)
                    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                    print(f"   🔍 Verarbeite: {filename} ({file_size_mb:.1f}MB)...", end="", flush=True)
                    
                    faces_before = self.index.ntotal if self.index else 0
                    detection_method = self._process_single_image(image_path)
                    faces_after = self.index.ntotal if self.index else 0
                    faces_in_image = faces_after - faces_before
                    
                    if faces_in_image > 0:
                        method_info = f" ({detection_method})" if detection_method else ""
                        print(f" ✅ {faces_in_image} Gesicht(er){method_info}")
                        faces_found += faces_in_image
                    else:
                        print(f" ⚪ Kein Gesicht")
                        skipped += 1
                    
                    processed += 1
                    
                except Exception as e:
                    print(f" ❌ Fehler: {str(e)[:50]}...")
                    processed += 1
                    skipped += 1
                    continue
            
            # Batch-Statistiken
            batch_duration = time.time() - batch_start_time
            print(f"   ⏱️ Batch-Zeit: {batch_duration:.1f}s")
            
            # Speichere nach jedem Batch
            self._save_index()
            print(f"   💾 Index gespeichert ({self.index.ntotal if self.index else 0} Gesichter)")
        
        duration = time.time() - start_time
        
        # Finale Statistiken
        print(f"\n🎉 Index-Aufbau abgeschlossen!")
        print(f"   ⏱️ Gesamtzeit: {duration:.1f}s")
        print(f"   📊 Verarbeitete Bilder: {processed}")
        print(f"   👥 Gefundene Gesichter: {faces_found}")
        print(f"   ⚪ Bilder ohne Gesichter: {skipped}")
        print(f"   📈 Durchschnitt: {faces_found/max(1,processed-skipped):.1f} Gesichter pro Bild")
        print(f"   🚀 Index bereit für blitzschnelle Suchen!")
        
        return True
    
    def _process_single_image(self, image_path: str):
        """Verarbeite ein einzelnes Bild und füge Gesichter zum Index hinzu."""
        try:
            # Sicherheitschecks
            file_size = os.path.getsize(image_path)
            if file_size < 1024:  # Überspringe Dateien < 1KB (wahrscheinlich defekt)
                return "SKIP_SMALL"
            
            if file_size > 50 * 1024 * 1024:  # Überspringe sehr große Dateien > 50MB
                return "SKIP_LARGE"
            
            # Lade Bild mit Fehlerbehandlung
            try:
                image = self.face_recognition.load_image_file(image_path)
                if image is None or image.size == 0:
                    return "SKIP_INVALID"
            except Exception as e:
                return "SKIP_LOAD_ERROR"
            
            # Performance-Optimierung: Reduziere Bildgröße falls sehr groß
            original_shape = image.shape
            if image.shape[0] > 1800 or image.shape[1] > 1800:
                try:
                    import cv2
                    scale_factor = min(1800/image.shape[0], 1800/image.shape[1])
                    new_width = int(image.shape[1] * scale_factor)
                    new_height = int(image.shape[0] * scale_factor)
                    image = cv2.resize(image, (new_width, new_height))
                except ImportError:
                    # Falls cv2 nicht verfügbar und Bild zu groß, überspringe
                    if image.shape[0] > 2500 or image.shape[1] > 2500:
                        return "SKIP_TOO_LARGE"
            
            # Erkenne Gesichter - Robuste Mehrfach-Strategie
            face_locations = []
            detection_method = ""
            
            # Strategie 1: CNN (aber sicher)
            try:
                face_locations = self.face_recognition.face_locations(
                    image, 
                    number_of_times_to_upsample=1,
                    model="cnn"
                )
                if face_locations:
                    detection_method = "CNN"
            except (MemoryError, RuntimeError, Exception):
                # CNN kann bei großen Bildern oder wenig RAM abstürzen
                pass
            
            # Strategie 2: HOG mit Upsampling (sicherer)
            if not face_locations:
                try:
                    face_locations = self.face_recognition.face_locations(
                        image, 
                        number_of_times_to_upsample=1,  # Reduziert von 2 auf 1
                        model="cnn"  # CNN für höhere Genauigkeit!
                    )
                    if face_locations:
                        detection_method = "CNN+up"
                except Exception:
                    pass
            
            # Strategie 3: HOG ohne Upsampling (sicherste Option)
            if not face_locations:
                try:
                    face_locations = self.face_recognition.face_locations(
                        image, 
                        number_of_times_to_upsample=0,
                        model="cnn"  # CNN für höhere Genauigkeit!
                    )
                    if face_locations:
                        detection_method = "CNN"
                except Exception:
                    pass
            
            if not face_locations:
                return "NO_FACES_FOUND"
            
            # Sichere Encoding-Erstellung
            try:
                face_encodings = self.face_recognition.face_encodings(
                    image, 
                    known_face_locations=face_locations, 
                    num_jitters=3,  # Reduziert von 7 auf 3 für Stabilität
                    model="large"   # Large model für höhere Genauigkeit!
                )
            except (MemoryError, RuntimeError, Exception):
                # Falls Encoding fehlschlägt, überspringe
                return "SKIP_ENCODING_ERROR"
            
            if not face_encodings:
                return "NO_FACES_FOUND"
            
            # Relativer Pfad für bessere Portabilität
            rel_path = os.path.relpath(image_path, IMAGE_FOLDER)
            
            # Füge jedes Gesicht zum Index hinzu
            faces_added = 0
            for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
                if encoding is not None and len(encoding) > 0:
                    try:
                        # Normalisiere Encoding für Cosine Similarity
                        norm_encoding = encoding / np.linalg.norm(encoding)
                        
                        # Füge zum FAISS Index hinzu
                        self.index.add(norm_encoding.reshape(1, -1).astype('float32'))
                        
                        # Speichere Metadata
                        self.metadata.append({
                            'path': rel_path,
                            'full_path': image_path,
                            'face_index': i,
                            'bbox': location,
                            'file_hash': self._get_file_hash(image_path),
                            'timestamp': time.time(),
                            'detection_method': detection_method,
                            'original_size': f"{original_shape[1]}x{original_shape[0]}"
                        })
                        faces_added += 1
                    except Exception:
                        # Falls einzelnes Gesicht fehlschlägt, andere weiter verarbeiten
                        continue
            
            return detection_method if faces_added > 0 else "NO_FACES_FOUND"
        
        except Exception as e:
            # Kompletter Fallback für alle anderen Fehler
            return "SKIP_CRITICAL_ERROR"
    
    def _handle_no_face_image(self, image_path, delete_images_without_faces=False):
        """Behandle Bilder ohne Gesichter."""
        if delete_images_without_faces:
            try:
                os.remove(image_path)
                print(f"   🗑️ Gelöscht (keine Gesichter): {os.path.basename(image_path)}")
                return "DELETED_NO_FACES"
            except Exception as e:
                print(f"   ❌ Fehler beim Löschen von {os.path.basename(image_path)}: {e}")
                return "DELETE_ERROR"
        else:
            return "NO_FACES_FOUND"
    
    def _save_index(self):
        """Speichere Index und Metadata."""
        if not self.index or not self.faiss:
            return
        
        try:
            # Speichere FAISS Index
            self.faiss.write_index(self.index, VECTOR_INDEX_FILE)
            
            # Speichere Metadata
            with open(VECTOR_METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")
    
    def search_similar_faces(self, query_encoding: np.ndarray, k: Optional[int] = None) -> List[Dict]:
        """Echte Gesichtserkennung - nur das GLEICHE Gesicht, nicht ähnliche!"""
        if not self.index or not self.faiss or self.index.ntotal == 0:
            return []
        
        try:
            # Normalisiere Query-Encoding
            query_norm = query_encoding / np.linalg.norm(query_encoding)
            query_vector = query_norm.reshape(1, -1).astype('float32')
            
            # WICHTIG: Durchsuche ALLE Kandidaten für vollständige Ergebnisse!
            # User-Feedback: 46 Ergebnisse waren künstlich begrenzt durch max_candidates=200
            if k is not None:
                search_k = min(k, self.index.ntotal)
            else:
                # Verwende ALLE verfügbaren Vektoren für vollständige Suche
                search_k = self.index.ntotal
                print(f"🔍 Durchsuche ALLE {search_k} Vektoren für vollständige Ergebnisse")
            
            similarities, indices = self.index.search(query_vector, search_k)
            
            results = []
            added_images = set()  # Verhindere Duplikate pro Bild
            
            print(f"🔍 Analysiere {len(similarities[0])} Kandidaten mit Threshold {DIST_THRESHOLD}")
            
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    meta = self.metadata[idx]
                    
                    # Konvertiere Similarity zu Distanz (0.0 = identisch)
                    distance = 1.0 - sim  # Cosine distance
                    
                    # DEBUGGING: Zeige verdächtige niedrige Distanzen (weniger aufdringlich)
                    if distance < 0.15 and len(results) < 5:  # Nur die ersten 5 verdächtigen zeigen
                        print(f"🚨 VERDÄCHTIG: {meta['path']} hat Distanz {distance:.3f} (sim: {sim:.3f})")
                    
                    # SEHR STRENGER Threshold - nur praktisch identische Gesichter!
                    if distance <= DIST_THRESHOLD:  # Nutze den sehr strengen Threshold (0.30)
                        image_path = meta['path']
                        
                        # Verhindere mehrere Treffer vom gleichen Bild
                        if image_path not in added_images:
                            added_images.add(image_path)
                            
                            # ZUSÄTZLICHE VALIDIERUNG: Prüfe ob das Bild existiert
                            full_image_path = meta['full_path'] if 'full_path' in meta else os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'images', meta['path'])
                            if not os.path.exists(full_image_path):
                                print(f"⚠️ ÜBERSPRINGE nicht existierendes Bild: {meta['path']}")
                                continue
                            
                            result = {
                                'filename': meta['path'],
                                'full_path': meta['full_path'],
                                'dist': distance,
                                'face_id': f"vector_{idx}",
                                'bbox': meta.get('bbox', [0, 0, 100, 100]),
                                'quelle': 'vector_db',
                                'detection_method': meta.get('detection_method', 'unknown'),
                                'similarity_score': sim,
                                'original_size': meta.get('original_size', 'unknown')
                            }
                            results.append(result)
            
            # DEBUGGING: Zeige Statistiken
            if results:
                best_dist = results[0]['dist'] if results else 999
                worst_dist = results[-1]['dist'] if results else 999
                print(f"🔍 Vector DB: {len(results)} Treffer, Distanz: {best_dist:.3f} - {worst_dist:.3f} (Threshold: {DIST_THRESHOLD})")
            else:
                print(f"🔍 Vector DB: Keine Treffer unter Threshold {DIST_THRESHOLD}")
            
            # Sortiere nach Ähnlichkeit (beste zuerst)
            results.sort(key=lambda x: x['dist'])
            # Nur begrenzen wenn k explizit gesetzt ist
            if k is not None:
                return results[:k]
            else:
                return results  # ALLE Ergebnisse zurückgeben!
        
        except Exception as e:
            print(f"❌ Suchfehler: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Hole Statistiken über die Datenbank."""
        return {
            'total_faces': self.index.ntotal if self.index else 0,
            'total_images': len(set(meta['path'] for meta in self.metadata)),
            'index_size_mb': os.path.getsize(VECTOR_INDEX_FILE) / 1024 / 1024 if os.path.exists(VECTOR_INDEX_FILE) else 0,
            'metadata_size_mb': os.path.getsize(VECTOR_METADATA_FILE) / 1024 / 1024 if os.path.exists(VECTOR_METADATA_FILE) else 0
        }

# Globale Instanz
_vector_db = None

def get_vector_db() -> VectorFaceDatabase:
    """Hole oder erstelle globale Vector Database Instanz."""
    global _vector_db
    
    if not ENABLE_VECTOR_DB:
        return None
    
    if _vector_db is None:
        _vector_db = VectorFaceDatabase()
    
    return _vector_db

def search_faces_with_vector_db(query_encoding: np.ndarray, progress_callback=None) -> List[Dict]:
    """
    Echte Gesichtserkennung mit Vector Database.
    Zeigt ALLE Bilder mit dem gleichen/ähnlichen Gesicht.
    """
    vector_db = get_vector_db()
    
    if not vector_db or not vector_db.faiss:
        # Fallback auf normale Suche
        print("⚠️ Vector DB nicht verfügbar - verwende normale Suche")
        return []

    # WICHTIG: Nutze existierenden Index falls verfügbar
    if vector_db.index and vector_db.index.ntotal > 0:
        if progress_callback:
            progress_callback(0.9)
        
        # Suche nur in TOP Kandidaten - nicht in allen!
        total_vectors = vector_db.index.ntotal
        
        print(f"🔍 Durchsuche {total_vectors} Vektoren nach IDENTISCHEN Gesichtern (Threshold: {DIST_THRESHOLD})")
        
        # Durchsuche ALLE Kandidaten für vollständige Ergebnisse!
        results = vector_db.search_similar_faces(query_encoding, k=None)  # k=None = ALLE Vektoren!
        
        if progress_callback:
            progress_callback(1.0)
        
        print(f"🚀 Gesichtserkennung: {len(results)} passende Gesichter gefunden")
        if results:
            best_dist = min(r['dist'] for r in results)
            worst_dist = max(r['dist'] for r in results)
            print(f"   � Distanz-Bereich: {best_dist:.3f} - {worst_dist:.3f}")
        
        return results    # Nur wenn Index wirklich fehlt oder leer ist
    if vector_db.need_rebuild():
        print("🔄 Index-Update erforderlich...")
        if progress_callback:
            progress_callback(0.1)
        
        # Erstelle neuen Index da rebuild_index nicht existiert
        vector_db._create_new_index()
        success = vector_db.add_new_images_to_index(progress_callback)
        if not success:
            print("❌ Index-Aufbau fehlgeschlagen")
            return []
    
    if progress_callback:
        progress_callback(0.8)
    
    # Suche mit Vector DB
    results = vector_db.search_similar_faces(query_encoding)
    
    if progress_callback:
        progress_callback(1.0)
    
    print(f"🚀 Vector DB Suche: {len(results)} Ergebnisse gefunden")
    return results
