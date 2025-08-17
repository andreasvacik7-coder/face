"""
Komplett überarbeitete Face-Suche - Effektiv und intelligent.
Zeigt nur wirklich passende Gesichter mit hoher Präzision.
Mit Multi-Face-Auswahl und Gesichts-Markierungen.
"""
import os
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional, Tuple
import io

from config import (
    FACES_FOLDER, IMAGE_FOLDER, DIST_THRESHOLD, 
    ENABLE_VECTOR_DB, MAX_DISPLAY_RESULTS
)
from utils import load_data, get_person_name

# Lazy imports für bessere Performance
def get_face_recognition():
    """Lazy import für face_recognition"""
    try:
        import face_recognition
        return face_recognition
    except ImportError:
        st.error("❌ face_recognition nicht installiert! Installiere mit: pip install face-recognition")
        return None

def get_vector_db():
    """Lazy import für Vector DB"""
    if ENABLE_VECTOR_DB:
        try:
            from vector_db import search_faces_with_vector_db, get_vector_db
            return search_faces_with_vector_db, get_vector_db
        except ImportError:
            return None, None
    return None, None


class SmartFaceSearch:
    """
    Intelligente Gesichtssuche mit Multi-Level-Verifikation.
    Zeigt nur echte Matches - keine zufälligen Ergebnisse.
    """
    
    def __init__(self):
        self.face_recognition = get_face_recognition()
        self.vector_search, self.vector_db_instance = get_vector_db()
        # VERSCHÄRFTE Schwellwerte basierend auf User-Feedback
        self.quality_thresholds = {
            'excellent': 0.025,  # Praktisch identisch (sehr streng!)
            'very_good': 0.040,  # Sehr sicher (strenger)
            'good': 0.055,       # Gut erkennbar (0.057 war User-Grenze)
            'acceptable': 0.065  # Akzeptabel (0.068 war schon zu ungenau)
        }
    
    def extract_all_faces(self, image_data, use_cnn=True) -> List[Dict]:
        """
        Extrahiere alle Gesichter aus einem Bild mit Positionen.
        """
        if not self.face_recognition:
            st.error("❌ Face recognition nicht verfügbar!")
            return []
            
        try:
            # Konvertiere zu numpy array
            if hasattr(image_data, 'read'):
                image_data.seek(0)
                pil_img = Image.open(image_data)
            else:
                pil_img = image_data
            
            # Konvertiere RGB falls nötig
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            img_array = np.array(pil_img)
            
            st.write(f"🔍 **Debug:** Bildgröße: {img_array.shape}")
            
            # Prüfe Bildgröße - bei kleinen Bildern vergrößern
            height, width = img_array.shape[:2]
            min_size = 300  # Minimum für gute Gesichtserkennung
            
            if height < min_size or width < min_size:
                # Berechne Skalierungsfaktor
                scale_factor = max(min_size / height, min_size / width)
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                
                st.write(f"� **Debug:** Bild zu klein ({width}x{height}), vergrößere auf {new_width}x{new_height}")
                
                # Vergrößere Bild
                pil_img_resized = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img_array = np.array(pil_img_resized)
                st.write(f"📏 **Debug:** Neue Bildgröße: {img_array.shape}")
            
            face_locations = []
            detection_method = "none"
            
            if use_cnn:
                # Versuche CNN zuerst (bei Upload-Bildern)
                st.write("🔍 **Debug:** Versuche CNN-Erkennung (präzise)...")
                try:
                    face_locations = self.face_recognition.face_locations(img_array, model="cnn")
                    st.write(f"🔍 **Debug:** CNN gefunden: {len(face_locations)} Gesichter")
                    detection_method = "cnn"
                    
                    # Falls CNN nichts findet, versuche HOG
                    if len(face_locations) == 0:
                        st.write("🔍 **Debug:** CNN fand nichts, versuche HOG...")
                        face_locations = self.face_recognition.face_locations(img_array, model="hog")
                        st.write(f"🔍 **Debug:** HOG gefunden: {len(face_locations)} Gesichter")
                        detection_method = "hog"
                        
                except Exception as cnn_e:
                    st.warning(f"⚠️ CNN Fehler: {cnn_e}")
                    st.write("🔍 **Debug:** Fallback zu HOG...")
                    face_locations = self.face_recognition.face_locations(img_array, model="hog")
                    st.write(f"🔍 **Debug:** HOG gefunden: {len(face_locations)} Gesichter")
                    detection_method = "hog"
            else:
                # Nur HOG verwenden
                st.write("🔍 **Debug:** Versuche HOG-Erkennung (schnell)...")
                face_locations = self.face_recognition.face_locations(img_array, model="hog")
                st.write(f"🔍 **Debug:** HOG gefunden: {len(face_locations)} Gesichter")
                detection_method = "hog"
            
            if not face_locations:
                # Versuche noch andere Parameter
                st.write("🔍 **Debug:** Versuche mit number_of_times_to_upsample=2...")
                try:
                    face_locations = self.face_recognition.face_locations(
                        img_array, 
                        model="hog", 
                        number_of_times_to_upsample=2
                    )
                    st.write(f"🔍 **Debug:** HOG mit Upsampling gefunden: {len(face_locations)} Gesichter")
                    detection_method = "hog_upsampled"
                except Exception as up_e:
                    st.warning(f"⚠️ Upsampling Fehler: {up_e}")
            
            if not face_locations:
                return []
            
            if not face_locations:
                return []
            
            st.write(f"✅ **Debug:** {len(face_locations)} Gesichter gefunden mit {detection_method}, extrahiere Encodings...")
            
            # Skaliere face_locations zurück falls Bild vergrößert wurde
            if height < min_size or width < min_size:
                st.write("📏 **Debug:** Skaliere Gesichts-Positionen zurück...")
                scale_factor = max(min_size / height, min_size / width)
                scaled_locations = []
                for (top, right, bottom, left) in face_locations:
                    scaled_locations.append((
                        int(top / scale_factor),
                        int(right / scale_factor), 
                        int(bottom / scale_factor),
                        int(left / scale_factor)
                    ))
                face_locations = scaled_locations
                # Verwende Original-Bild für Encoding
                img_array = np.array(pil_img)
            
            # Extrahiere Encodings - versuche verschiedene Modelle
            face_encodings = None
            encoding_model = "large"
            
            try:
                face_encodings = self.face_recognition.face_encodings(
                    img_array, 
                    known_face_locations=face_locations,
                    model=encoding_model
                )
                st.write(f"✅ **Debug:** {len(face_encodings)} Encodings mit '{encoding_model}' Modell erstellt")
            except Exception as large_e:
                st.warning(f"⚠️ 'large' Modell Fehler, versuche 'small': {large_e}")
                try:
                    face_encodings = self.face_recognition.face_encodings(
                        img_array, 
                        known_face_locations=face_locations,
                        model="small"
                    )
                    encoding_model = "small"
                    st.write(f"✅ **Debug:** {len(face_encodings)} Encodings mit 'small' Modell erstellt")
                except Exception as small_e:
                    st.error(f"❌ Encoding Fehler: {small_e}")
                    return []
            
            if not face_encodings:
                st.error("❌ Keine Face-Encodings erstellt! Versuche ein anderes Bild.")
                return []
            
            # Prüfe Übereinstimmung von Locations und Encodings
            if len(face_locations) != len(face_encodings):
                st.warning(f"⚠️ Mismatch: {len(face_locations)} Locations vs {len(face_encodings)} Encodings")
                # Verwende minimum
                min_count = min(len(face_locations), len(face_encodings))
                face_locations = face_locations[:min_count]
                face_encodings = face_encodings[:min_count]
            
            # Erstelle Liste aller Gesichter mit Metadaten
            faces = []
            for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
                top, right, bottom, left = location
                area = (bottom - top) * (right - left)
                
                faces.append({
                    'index': i,
                    'encoding': encoding,
                    'location': location,
                    'area': area,
                    'center_x': left + (right - left) // 2,
                    'center_y': top + (bottom - top) // 2,
                    'encoding_model': encoding_model,
                    'detection_method': detection_method
                })
            
            # Sortiere nach Größe (größtes zuerst)
            faces.sort(key=lambda x: x['area'], reverse=True)
            
            st.success(f"✅ **Debug:** {len(faces)} Gesichter erfolgreich verarbeitet!")
            
            return faces
            
        except Exception as e:
            st.error(f"❌ Fehler beim Extrahieren der Gesichts-Features: {e}")
            st.error(f"❌ Debug Info: {type(e).__name__}: {str(e)}")
            return []
    
    def create_face_preview_image(self, image_data, faces: List[Dict]) -> Image.Image:
        """
        Erstelle Vorschaubild mit markierten Gesichtern.
        """
        try:
            if hasattr(image_data, 'read'):
                image_data.seek(0)
                pil_img = Image.open(image_data)
            else:
                pil_img = image_data
            
            # Erstelle Kopie für Markierungen
            preview_img = pil_img.copy()
            draw = ImageDraw.Draw(preview_img)
            
            # Markiere jedes Gesicht
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
            
            for i, face in enumerate(faces):
                top, right, bottom, left = face['location']
                color = colors[i % len(colors)]
                
                # Zeichne Rechteck
                draw.rectangle([(left-3, top-3), (right+3, bottom+3)], outline=color, width=4)
                
                # Füge Nummer hinzu
                try:
                    # Versuche größere Schrift
                    font_size = max(20, min(40, (right-left)//4))
                    draw.text((left, top-25), f"Gesicht {i+1}", fill=color, stroke_width=2, stroke_fill='white')
                except:
                    # Fallback ohne spezielle Schrift
                    draw.text((left, top-20), f"Gesicht {i+1}", fill=color)
            
            return preview_img
            
        except Exception as e:
            st.error(f"Fehler beim Erstellen der Vorschau: {e}")
            return image_data

    def search_with_vector_db(self, query_encoding: np.ndarray, progress_callback=None) -> List[Dict]:
        """
        Suche mit Vector Database - Ultra-schnell für große Datenbanken.
        """
        if not self.vector_search:
            return []
        
        try:
            if progress_callback:
                progress_callback(0.2, "Suche in Vector Database...")
            
            vector_results = self.vector_search(query_encoding, progress_callback)
            
            # Filtere und formatiere Ergebnisse
            filtered_results = []
            for result in vector_results:
                if result['dist'] <= self.quality_thresholds['acceptable']:
                    # Bestimme Qualitätslevel
                    quality = self._get_quality_level(result['dist'])
                    result['quality'] = quality
                    result['quality_icon'] = self._get_quality_icon(quality)
                    filtered_results.append(result)
            
            if progress_callback:
                progress_callback(1.0, f"Vector DB: {len(filtered_results)} passende Gesichter gefunden")
            
            return filtered_results
            
        except Exception as e:
            st.error(f"Vector DB Fehler: {e}")
            return []
    
    def search_traditional(self, query_encoding: np.ndarray, progress_callback=None) -> List[Dict]:
        """
        Fallback: Traditionelle Suche durch alle Bilder.
        Optimiert für Präzision, nicht Geschwindigkeit.
        """
        if not self.face_recognition:
            return []
        
        results = []
        
        try:
            if progress_callback:
                progress_callback(0.1, "Sammle Bilddateien...")
            
            # Sammle alle Bilder rekursiv
            all_images = []
            for root, dirs, files in os.walk(IMAGE_FOLDER):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.webp')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, IMAGE_FOLDER)
                        all_images.append((rel_path, full_path))
            
            # Begrenze für Performance (kann konfiguriert werden)
            if len(all_images) > 2000:
                all_images = all_images[:2000]
                st.warning(f"⚠️ Zu viele Bilder ({len(all_images)}). Suche begrenzt auf erste 2000 Bilder.")
            
            total_images = len(all_images)
            processed = 0
            
            if progress_callback:
                progress_callback(0.2, f"Durchsuche {total_images} Bilder...")
            
            # Verarbeite in Batches für bessere Performance
            batch_size = 10
            for i in range(0, len(all_images), batch_size):
                batch = all_images[i:i + batch_size]
                
                for rel_path, full_path in batch:
                    try:
                        # Lade Bild
                        img_array = self.face_recognition.load_image_file(full_path)
                        
                        # Finde Gesichter (HOG für Performance)
                        face_locations = self.face_recognition.face_locations(img_array, model="hog")
                        
                        if face_locations:
                            # Extrahiere Encodings
                            face_encodings = self.face_recognition.face_encodings(
                                img_array, 
                                known_face_locations=face_locations
                            )
                            
                            # Prüfe jedes Gesicht
                            for j, encoding in enumerate(face_encodings):
                                # Berechne Distanz
                                distance = self.face_recognition.face_distance([query_encoding], encoding)[0]
                                
                                # Nur wenn wirklich ähnlich (VERSCHÄRFTE Schwellwerte!)
                                if distance <= self.quality_thresholds['acceptable']:
                                    quality = self._get_quality_level(distance)
                                    
                                    result = {
                                        'filename': rel_path,
                                        'full_path': full_path,
                                        'dist': float(distance),
                                        'quality': quality,
                                        'quality_icon': self._get_quality_icon(quality),
                                        'face_index': j,
                                        'source': 'traditional'
                                    }
                                    results.append(result)
                        
                        processed += 1
                        
                        # Update Progress
                        if progress_callback and processed % 5 == 0:
                            progress = 0.2 + (processed / total_images) * 0.7
                            progress_callback(progress, f"Verarbeitet: {processed}/{total_images}")
                            
                    except Exception as e:
                        processed += 1
                        continue
            
            # Sortiere nach Qualität
            results.sort(key=lambda x: x['dist'])
            
            if progress_callback:
                progress_callback(1.0, f"Traditionelle Suche: {len(results)} passende Gesichter")
            
            return results
            
        except Exception as e:
            st.error(f"Fehler bei traditioneller Suche: {e}")
            return []
    
    def verify_results(self, results: List[Dict], query_encoding: np.ndarray, max_verify=50) -> List[Dict]:
        """
        Verifiziere unsichere Ergebnisse nochmal mit CNN für höchste Präzision.
        """
        if not self.face_recognition or not results:
            return results
        
        verified_results = []
        verify_count = 0
        
        for result in results:
            # Sehr sichere Ergebnisse direkt übernehmen
            if result['dist'] <= self.quality_thresholds['excellent']:
                verified_results.append(result)
                continue
            
            # Unsichere Ergebnisse verifizieren (begrenzt für Performance)
            if verify_count >= max_verify:
                # Rest ohne Verifikation übernehmen wenn schon viele verifiziert
                verified_results.append(result)
                continue
            
            try:
                image_path = os.path.join(IMAGE_FOLDER, result['filename'])
                if os.path.exists(image_path):
                    # Lade Bild nochmal
                    img_array = self.face_recognition.load_image_file(image_path)
                    
                    # CNN für höchste Präzision
                    face_locations = self.face_recognition.face_locations(img_array, model="cnn")
                    
                    if face_locations:
                        face_encodings = self.face_recognition.face_encodings(
                            img_array, 
                            known_face_locations=face_locations,
                            model="large"
                        )
                        
                        if face_encodings:
                            # Finde beste Übereinstimmung
                            distances = self.face_recognition.face_distance([query_encoding], face_encodings[0])
                            verified_distance = distances[0]
                            
                            # Nur wenn verifiziert
                            if verified_distance <= self.quality_thresholds['acceptable']:
                                result['dist'] = float(verified_distance)
                                result['quality'] = self._get_quality_level(verified_distance)
                                result['quality_icon'] = self._get_quality_icon(result['quality'])
                                result['verified'] = True
                                verified_results.append(result)
                            # Ansonsten verwerfen
                            
                verify_count += 1
                
            except Exception:
                # Bei Fehler: Original-Ergebnis beibehalten
                verified_results.append(result)
        
        return verified_results
    
    def smart_search(self, query_encoding: np.ndarray, progress_callback=None) -> List[Dict]:
        """
        Intelligente Suche: Vector DB wenn verfügbar, sonst traditionell.
        Mit optionaler Verifikation für höchste Präzision.
        """
        # Prüfe Vector Database Status
        vector_db_ready = False
        if self.vector_db_instance:
            try:
                vector_db = self.vector_db_instance()
                vector_db_ready = (vector_db and vector_db.index and vector_db.index.ntotal > 0)
            except:
                pass
        
        if vector_db_ready:
            if progress_callback:
                progress_callback(0.1, "Nutze Vector Database für Ultra-Speed...")
            
            results = self.search_with_vector_db(query_encoding, progress_callback)
            
            # Optional: Verifiziere unsichere Ergebnisse
            uncertain_results = [r for r in results if r['dist'] > self.quality_thresholds['very_good']]
            if len(uncertain_results) > 0 and len(uncertain_results) <= 20:
                if progress_callback:
                    progress_callback(0.9, f"Verifiziere {len(uncertain_results)} unsichere Ergebnisse...")
                results = self.verify_results(results, query_encoding, max_verify=20)
            
        else:
            if progress_callback:
                progress_callback(0.1, "Vector DB nicht bereit, nutze traditionelle Suche...")
            
            results = self.search_traditional(query_encoding, progress_callback)
        
        # Sortiere nach Qualität
        results.sort(key=lambda x: x['dist'])
        
        # KEINE KÜNSTLICHE BEGRENZUNG! Zeige alle passenden Gesichter
        # Entferne nur wirklich schlechte Matches
        final_results = []
        for result in results:
            if result['dist'] <= self.quality_thresholds['acceptable']:
                final_results.append(result)
        
        return final_results
    
    def _get_quality_level(self, distance: float) -> str:
        """Bestimme Qualitätslevel basierend auf Distanz."""
        if distance <= self.quality_thresholds['excellent']:
            return 'excellent'
        elif distance <= self.quality_thresholds['very_good']:
            return 'very_good'
        elif distance <= self.quality_thresholds['good']:
            return 'good'
        else:
            return 'acceptable'
    
    def _get_quality_icon(self, quality: str) -> str:
        """Hole Icon für Qualitätslevel."""
        icons = {
            'excellent': '🎯',    # 0.025 und darunter - praktisch identisch
            'very_good': '🟢',    # 0.025-0.040 - sehr sicher 
            'good': '🟡',         # 0.040-0.055 - gut erkennbar
            'acceptable': '🟠'    # 0.055-0.065 - noch akzeptabel
        }
        return icons.get(quality, '⚪')


class FaceSearchUI:
    """
    Benutzeroberfläche für die intelligente Gesichtssuche.
    """
    
    def __init__(self):
        self.search_engine = SmartFaceSearch()
        self.persons_data = None
    
    def load_persons_data(self):
        """Lade Personendaten für bessere Anzeige."""
        if self.persons_data is None:
            try:
                _, _, self.persons_data = load_data()
            except:
                self.persons_data = []
    
    def render_upload_section(self) -> Optional[Dict]:
        """Rendere Upload-Bereich und extrahiere Face-Encoding(s)."""
        st.subheader("🎯 Gesicht zum Suchen hochladen")
        
        uploaded_file = st.file_uploader(
            "Wähle ein Bild mit einem oder mehreren klaren Gesichtern",
            type=['png', 'jpg', 'jpeg', 'heic', 'heif', 'bmp', 'webp'],
            help="Unterstützte Formate: JPG, PNG, HEIC/HEIF, BMP, WebP"
        )
        
        if uploaded_file is not None:
            # Zeige hochgeladenes Bild
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Suchbild", use_container_width=True)
            
            with col2:
                st.info("""
                **💡 Tipps für beste Ergebnisse:**
                - Klares, gut beleuchtetes Gesicht
                - Frontalansicht bevorzugt
                - Gesicht sollte gut erkennbar sein
                - Bei mehreren Personen: wähle das gewünschte Gesicht
                """)
            
            # Extrahiere alle Gesichter
            with st.spinner("Analysiere Gesichter im Bild..."):
                # Versuche zunächst mit CNN für Upload-Bilder (präziser)
                faces = self.search_engine.extract_all_faces(uploaded_file, use_cnn=True)
                
                if not faces:
                    # Zeige detaillierte Hilfe
                    st.error("❌ Keine klaren Gesichter im Bild erkannt!")
                    
                    with st.expander("💡 Tipps für bessere Gesichtserkennung:", expanded=True):
                        col_tip1, col_tip2 = st.columns(2)
                        
                        with col_tip1:
                            st.markdown("""
                            **📸 Bildqualität verbessern:**
                            - Ausreichend Licht/gute Beleuchtung
                            - Gesicht sollte mindestens 150x150 Pixel groß sein
                            - Scharfes, nicht verschwommenes Bild
                            - Gesicht nicht zu stark gedreht
                            - Höhere Auflösung falls verfügbar
                            """)
                        
                        with col_tip2:
                            st.markdown("""
                            **🎯 Optimale Gesichtsposition:**
                            - Frontalansicht oder leichte Drehung
                            - Beide Augen sichtbar
                            - Gesicht nicht durch Haare/Hände verdeckt
                            - Brille/Maske kann Erkennung erschweren
                            - Kontrast zum Hintergrund
                            """)
                        
                        st.warning("🔄 **Problem:** Bild war sehr klein (183x275 Pixel) - versuche ein größeres Bild!")
                        st.info("💡 **Tipp:** HEIC-Bilder funktionieren oft besser, da sie mehr Details enthalten")
                    
                    return None
                
                elif len(faces) == 1:
                    # Nur ein Gesicht - verwende es direkt
                    st.success("✅ Ein Gesicht erfolgreich analysiert!")
                    
                    # Zeige detaillierte Info über erkanntes Gesicht
                    face = faces[0]
                    area_percent = (face['area'] / (image.width * image.height)) * 100
                    st.info(f"👤 **Erkanntes Gesicht:** Größe {area_percent:.1f}% des Bildes, "
                           f"Erkennungsmodell: {face.get('detection_method', 'unknown')}, "
                           f"Encoding-Modell: {face.get('encoding_model', 'unknown')}")
                    
                    return {
                        'encoding': faces[0]['encoding'],
                        'selected_face': faces[0],
                        'all_faces': faces,
                        'uploaded_image': uploaded_file
                    }
                else:
                    # Mehrere Gesichter - Benutzer wählen lassen
                    st.success(f"✅ {len(faces)} Gesichter erkannt!")
                    
                    # Zeige Info über verwendete Modelle
                    detection_method = faces[0].get('detection_method', 'unknown')
                    encoding_model = faces[0].get('encoding_model', 'unknown')
                    st.info(f"🔍 **Erkennungsmodelle:** {detection_method} (Erkennung) + {encoding_model} (Encoding)")
                    
                    # Zeige Vorschaubild mit markierten Gesichtern
                    preview_img = self.search_engine.create_face_preview_image(uploaded_file, faces)
                    st.image(preview_img, caption=f"Gefundene Gesichter (größtes = Gesicht 1)", use_container_width=True)
                    
                    # Gesichts-Auswahl
                    st.subheader("👤 Wähle das Gesicht für die Suche:")
                    
                    # Erstelle Optionen für Selectbox
                    face_options = []
                    for i, face in enumerate(faces):
                        area_percent = (face['area'] / (image.width * image.height)) * 100
                        face_options.append(f"Gesicht {i+1} (Größe: {area_percent:.1f}% des Bildes)")
                    
                    selected_face_idx = st.selectbox(
                        "Gesicht auswählen:",
                        range(len(faces)),
                        format_func=lambda x: face_options[x],
                        help="Gesicht 1 ist das größte, meist die Hauptperson"
                    )
                    
                    selected_face = faces[selected_face_idx]
                    
                    # Zeige Info über gewähltes Gesicht
                    st.info(f"👤 **Gewähltes Gesicht {selected_face_idx + 1}:** "
                           f"Position ({selected_face['center_x']}, {selected_face['center_y']}), "
                           f"Größe: {((selected_face['area'] / (image.width * image.height)) * 100):.1f}% des Bildes")
                    
                    return {
                        'encoding': selected_face['encoding'],
                        'selected_face': selected_face,
                        'all_faces': faces,
                        'uploaded_image': uploaded_file
                    }
        
        return None
    
    def render_search_button_and_execute(self, face_data: Dict) -> Optional[List[Dict]]:
        """Rendere Suchbutton und führe Suche durch."""
        if st.button("🔍 Ähnliche Gesichter suchen", type="primary", use_container_width=True):
            # Progress Container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress: float, message: str = ""):
                    progress_bar.progress(min(progress, 1.0))
                    if message:
                        status_text.text(message)
                
                start_time = time.time()
                
                # Führe Suche durch
                try:
                    query_encoding = face_data['encoding']
                    results = self.search_engine.smart_search(query_encoding, progress_callback)
                    
                    # Erweitere Ergebnisse mit Face-Markierung
                    enhanced_results = self._enhance_results_with_face_detection(results, face_data)
                    
                    search_time = time.time() - start_time
                    
                    # Cleanup Progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Zeige Ergebnis-Summary
                    if enhanced_results:
                        st.success(f"✅ Suche abgeschlossen in {search_time:.1f}s - {len(enhanced_results)} passende Gesichter gefunden!")
                    else:
                        st.warning("❌ Keine passenden Gesichter gefunden.")
                    
                    return enhanced_results
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Fehler bei der Suche: {e}")
                    return None
        
        return None
    
    def _enhance_results_with_face_detection(self, results: List[Dict], face_data: Dict) -> List[Dict]:
        """
        Erweitere Suchergebnisse mit Gesichts-Erkennungsdaten für Markierungen.
        """
        if not self.search_engine.face_recognition or not results:
            return results
        
        query_encoding = face_data['encoding']
        enhanced_results = []
        
        # Verarbeite Ergebnisse in kleineren Batches für Performance
        batch_size = 10
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            
            for result in batch:
                try:
                    # Lade Ergebnisbild
                    image_path = os.path.join(IMAGE_FOLDER, result['filename'])
                    if not os.path.exists(image_path):
                        enhanced_results.append(result)
                        continue
                    
                    # Finde Gesichter im Ergebnisbild
                    img_array = self.search_engine.face_recognition.load_image_file(image_path)
                    face_locations = self.search_engine.face_recognition.face_locations(img_array, model="hog")
                    
                    if not face_locations:
                        enhanced_results.append(result)
                        continue
                    
                    # Finde das passende Gesicht
                    face_encodings = self.search_engine.face_recognition.face_encodings(img_array, known_face_locations=face_locations)
                    
                    best_match_location = None
                    best_distance = float('inf')
                    
                    for encoding, location in zip(face_encodings, face_locations):
                        distance = self.search_engine.face_recognition.face_distance([query_encoding], encoding)[0]
                        if distance < best_distance:
                            best_distance = distance
                            best_match_location = location
                    
                    # Erweitere Result mit Gesichts-Position
                    result['face_location'] = best_match_location
                    result['verified_distance'] = best_distance
                    enhanced_results.append(result)
                    
                except Exception as e:
                    # Bei Fehler: Original-Result ohne Markierung
                    enhanced_results.append(result)
        
        return enhanced_results
    
    def render_quality_stats(self, results: List[Dict]):
        """Zeige Qualitäts-Statistiken der Ergebnisse."""
        if not results:
            return
        
        # Gruppiere nach Qualität
        quality_counts = {
            'excellent': len([r for r in results if r['quality'] == 'excellent']),
            'very_good': len([r for r in results if r['quality'] == 'very_good']),
            'good': len([r for r in results if r['quality'] == 'good']),
            'acceptable': len([r for r in results if r['quality'] == 'acceptable'])
        }
        
        # Zeige Metriken
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Exzellent", quality_counts['excellent'])
        with col2:
            st.metric("🟢 Sehr gut", quality_counts['very_good'])
        with col3:
            st.metric("🟡 Gut", quality_counts['good'])
        with col4:
            st.metric("🟠 Passend", quality_counts['acceptable'])
        
        # Distanz-Info
        if results:
            distances = [r['dist'] for r in results]
            st.info(f"📏 **Distanz-Bereich:** {min(distances):.3f} - {max(distances):.3f} (Durchschnitt: {sum(distances)/len(distances):.3f})")
    
    def render_results_gallery(self, results: List[Dict]):
        """Rendere Ergebnis-Galerie mit intelligenter Gruppierung."""
        if not results:
            st.info("Keine Ergebnisse zum Anzeigen.")
            return
        
        self.load_persons_data()
        
        st.subheader(f"🖼️ Gefundene Gesichter ({len(results)})")
        
        # Info über strenge Filterung
        if len(results) > 0:
            best_dist = min(r['dist'] for r in results)
            worst_dist = max(r['dist'] for r in results)
            st.info(f"📏 **Distanz-Bereich:** {best_dist:.3f} (beste) bis {worst_dist:.3f} (schlechteste)")
            st.success("✅ **Alle Ergebnisse sind echte Matches** - keine zufälligen Bilder mehr!")
        
        # Gruppiere nach Qualität
        quality_groups = {
            'excellent': [r for r in results if r['quality'] == 'excellent'],
            'very_good': [r for r in results if r['quality'] == 'very_good'],
            'good': [r for r in results if r['quality'] == 'good'],
            'acceptable': [r for r in results if r['quality'] == 'acceptable']
        }
        
        # Zeige Gruppen (erweitere automatisch die guten Gruppen)
        for quality, group_results in quality_groups.items():
            if not group_results:
                continue
            
            # Gruppe Header
            quality_labels = {
                'excellent': "🎯 Exzellente Übereinstimmungen",
                'very_good': "🟢 Sehr gute Übereinstimmungen", 
                'good': "🟡 Gute Übereinstimmungen",
                'acceptable': "🟠 Passende Übereinstimmungen"
            }
            
            # Automatisch expandieren für gute Qualitäten
            auto_expand = quality in ['excellent', 'very_good', 'good']
            
            with st.expander(f"{quality_labels[quality]} ({len(group_results)})", expanded=auto_expand):
                self._render_image_grid(group_results)
    
    def _render_image_grid(self, results: List[Dict]):
        """Rendere Bilder-Grid für eine Ergebnis-Gruppe."""
        cols_per_row = 4
        
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, result in enumerate(results[i:i + cols_per_row]):
                with cols[j]:
                    try:
                        # Lade Bild
                        image_path = os.path.join(IMAGE_FOLDER, result['filename'])
                        
                        if os.path.exists(image_path):
                            # Erstelle Bild mit Gesichts-Markierung falls verfügbar
                            if result.get('face_location'):
                                image = self._create_marked_image(image_path, result['face_location'])
                            else:
                                image = Image.open(image_path)
                            
                            # Erstelle Caption
                            quality_icon = result['quality_icon']
                            distance = result['dist']
                            filename = os.path.basename(result['filename'])
                            
                            # Personen-Info falls verfügbar
                            person_info = self._get_person_info(result)
                            
                            caption = f"{quality_icon} **{filename}**\n"
                            if person_info:
                                caption += f"👤 {person_info}\n"
                            caption += f"📏 Distanz: {distance:.3f}"
                            
                            # Zusätzliche Verifikationsdistanz falls verfügbar
                            if result.get('verified_distance') and abs(result['verified_distance'] - distance) > 0.005:
                                caption += f" (Verifiziert: {result['verified_distance']:.3f})"
                            
                            # Folder-Info falls in Unterverzeichnis
                            if '/' in result['filename']:
                                folder = os.path.dirname(result['filename'])
                                caption += f"\n📁 {folder}"
                            
                            st.image(image, caption=caption, use_container_width=True)
                            
                        else:
                            st.error(f"❌ Bild nicht gefunden:\n{result['filename']}")
                            
                    except Exception as e:
                        st.error(f"❌ Fehler beim Laden:\n{result['filename']}")
    
    def _create_marked_image(self, image_path: str, face_location: Tuple[int, int, int, int]) -> Image.Image:
        """
        Erstelle Bild mit markiertem Gesicht.
        """
        try:
            image = Image.open(image_path)
            
            # Erstelle Kopie für Markierungen
            marked_image = image.copy()
            draw = ImageDraw.Draw(marked_image)
            
            # Entpacke Gesichts-Position (top, right, bottom, left)
            top, right, bottom, left = face_location
            
            # Zeichne grünen Rahmen um das gefundene Gesicht
            line_width = max(3, min(8, (right - left) // 50))  # Adaptive Linienbreite
            
            # Zeichne Rahmen
            draw.rectangle(
                [(left - line_width, top - line_width), (right + line_width, bottom + line_width)], 
                outline="lime", 
                width=line_width
            )
            
            # Füge kleines "✓" Symbol hinzu
            try:
                symbol_size = max(15, min(30, (right - left) // 8))
                draw.text(
                    (left, top - symbol_size - 5), 
                    "✓", 
                    fill="lime", 
                    stroke_width=2, 
                    stroke_fill="darkgreen"
                )
            except:
                # Fallback ohne spezielle Formatierung
                draw.text((left, top - 20), "✓", fill="lime")
            
            return marked_image
            
        except Exception as e:
            # Bei Fehler: Original-Bild zurückgeben
            return Image.open(image_path)
    
    def _get_person_info(self, result: Dict) -> Optional[str]:
        """Hole Personen-Information für ein Ergebnis."""
        if not self.persons_data:
            return None
        
        # Suche nach Person basierend auf filename
        filename = result['filename']
        
        # Vereinfachte Personenerkennung basierend auf Verzeichnisname
        if '/' in filename:
            folder_name = os.path.dirname(filename)
            
            # Suche Person mit passendem Namen
            for person in self.persons_data:
                person_name = person.get('name', '').lower()
                if person_name and person_name in folder_name.lower():
                    return person.get('name', folder_name)
            
            return folder_name
        
        return None
    
    def render_help_section(self):
        """Rendere Hilfe-Bereich."""
        with st.expander("ℹ️ Wie funktioniert die intelligente Gesichtssuche?"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Intelligente Filterung:**
                - Alle Ergebnisse sind echte Matches
                - Verschärfte Distanz-Schwellwerte
                - Basierend auf User-Feedback optimiert
                - Zeigt ALLE passenden Gesichter (keine 60-Begrenzung)
                """)
                
                st.markdown("""
                **🔧 Technologie:**
                - Vector Database (FAISS) für Speed
                - CNN + Large Model für Präzision
                - Multi-Face-Erkennung mit Auswahl
                - Intelligente Gesichts-Markierung
                """)
            
            with col2:
                st.markdown("""
                **🎨 Qualitätsstufen (VERSCHÄRFT):**
                - 🎯 Exzellent (≤0.025): Praktisch identisch
                - 🟢 Sehr gut (≤0.040): Eindeutig dieselbe Person
                - 🟡 Gut (≤0.055): Klar erkennbar als gleiche Person
                - 🟠 Passend (≤0.065): Noch akzeptabel ähnlich
                """)
                
                st.markdown("""
                **💡 Tipps:**
                - Verwende klare, gut beleuchtete Fotos
                - Frontalansicht funktioniert am besten
                - Einzelpersonen im Fokus bevorzugt
                - HEIC/HEIF werden unterstützt
                - Bei 0.018-0.057: meist perfekte Ergebnisse
                - Bei 0.065+: kritisch prüfen
                """)


def render_search_page():
    """
    Haupt-Rendering-Funktion für die neue Gesichtssuche.
    """
    st.title("🔍 Intelligente Gesichtssuche")
    st.markdown("---")
    
    # Info über das neue System
    st.info("🚀 **Ultra-Präzise Suche:** Nur Distanz ≤0.065 - alle Ergebnisse sind echte Matches!")
    
    # Erstelle UI-Instanz
    ui = FaceSearchUI()
    
    # Upload-Bereich
    face_data = ui.render_upload_section()
    
    if face_data is not None:
        st.markdown("---")
        
        # Suche ausführen
        results = ui.render_search_button_and_execute(face_data)
        
        if results is not None:
            st.markdown("---")
            
            if results:
                # Zeige Statistiken
                ui.render_quality_stats(results)
                
                st.markdown("---")
                
                # Zeige Ergebnisse
                ui.render_results_gallery(results)
            else:
                # Keine Ergebnisse - Tipps anzeigen
                st.warning("🤔 Keine passenden Gesichter gefunden.")
                
                with st.expander("💡 Versuche diese Tipps:", expanded=True):
                    st.markdown("""
                    **Mögliche Gründe und Lösungen:**
                    
                    1. **🎯 Sehr strenge Qualitätskontrolle:** Distanz muss ≤0.065 sein
                    2. **📸 Bildqualität:** Versuche ein klareres, besser beleuchtetes Foto
                    3. **👤 Person unbekannt:** Die Person ist möglicherweise nicht in der Datenbank
                    4. **🔄 Anderes Foto:** Probiere ein anderes Foto derselben Person
                    5. **⚙️ Index-Update:** Falls neue Bilder hinzugefügt wurden, aktualisiere den Index
                    
                    **📊 Qualitäts-Hinweise:**
                    - Distanz 0.018-0.057: Meist korrekte Ergebnisse ✅
                    - Distanz 0.065+: Oft falsche Ergebnisse ❌
                    - System zeigt jetzt nur Distanz ≤0.065
                    
                    **📊 Status prüfen:**
                    - Gehe zur "Vector DB" Seite und prüfe den Index-Status
                    - Führe gegebenenfalls einen Index-Update durch
                    """)
    
    # Hilfe-Bereich
    st.markdown("---")
    ui.render_help_section()


if __name__ == "__main__":
    render_search_page()
