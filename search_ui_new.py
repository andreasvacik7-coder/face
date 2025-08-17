"""
Such-Modul für die Gesichtserkennungsanwendung.
"""
import os
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io

from config import FACES_FOLDER, IMAGE_FOLDER, DIST_THRESHOLD
from utils import load_data, process_image_for_faces, detect_faces_robust, get_person_name
from search import search_similar_faces_live

try:
    from deepface_analyzer import (
        analyze_uploaded_images,
        extract_features_for_person,
        ATTRIBUTE_ANALYSIS_CATEGORIES,
        update_person_features_if_needed,
        update_all_person_features,
        get_common_attributes,
        analyze_upload_with_attributes,
        render_attribute_analysis,
        create_attribute_summary,
        DEEPFACE_AVAILABLE
    )
except ImportError:
    DEEPFACE_AVAILABLE = False


def _create_image_with_face_boxes(image_path, query_encoding, distance_threshold=DIST_THRESHOLD):
    """Erstelle ein Bild mit markierten Gesichtern, die dem Query ähnlich sind."""
    try:
        import face_recognition
        
        # Lade das Bild
        pil_img = Image.open(image_path)
        img_array = np.array(pil_img)
        
        # Finde alle Gesichter im Bild
        face_locations = face_recognition.face_locations(img_array, model="hog")
        face_encodings = face_recognition.face_encodings(img_array, known_face_locations=face_locations)
        
        # Erstelle eine Kopie für das Zeichnen
        draw_img = pil_img.copy()
        draw = ImageDraw.Draw(draw_img)
        
        # Vergleiche jedes Gesicht mit dem Query
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Berechne Distanz
            distances = face_recognition.face_distance([query_encoding], encoding)
            if len(distances) > 0 and distances[0] <= distance_threshold:
                # Zeichne grünes Rechteck für ähnliche Gesichter
                draw.rectangle([(left, top), (right, bottom)], outline="green", width=3)
                # Füge Distanz als Text hinzu
                draw.text((left, top-20), f"{distances[0]:.3f}", fill="green")
        
        return draw_img
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Bildes mit Gesichtsmarkierungen: {e}")
        return Image.open(image_path)


def display_search_results(results_list, show_images=True, max_display=None, persons=None):
    """Zeige Suchergebnisse in der Streamlit-UI."""
    if not results_list:
        st.warning("Keine Ergebnisse gefunden.")
        return
    
    # Begrenze Anzeige falls gewünscht
    if max_display and len(results_list) > max_display:
        results_list = results_list[:max_display]
        st.info(f"Zeige die ersten {max_display} von {len(results_list)} Ergebnissen")
    
    # Lade Personendaten falls vorhanden
    if persons is None:
        persons = load_data()
    
    if show_images:
        # Zeige Bilder in Spalten
        cols_per_row = 4
        for i in range(0, len(results_list), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, res in enumerate(results_list[i:i+cols_per_row]):
                with cols[j]:
                    try:
                        # Lade und zeige Bild
                        image_path = os.path.join(IMAGE_FOLDER, res["filename"])
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            
                            # Personendaten abrufen
                            name_str, label = get_person_name(res.get('person_id'), persons)
                            
                            # Farbkodierung für passende Gesichter (alle sind relevant da schon gefiltert)
                            if res['dist'] < 0.35:
                                distance_color = "🟢"  # Sehr ähnlich
                            elif res['dist'] < 0.45:
                                distance_color = "🟡"  # Ähnlich  
                            else:
                                distance_color = "🟠"  # Noch passend
                            
                            caption = f"**{name_str}**\n{label}\n{distance_color} Distanz: {res['dist']:.3f}"
                            
                            # Zeige Bild mit Caption
                            st.image(image, caption=caption, use_column_width=True)
                            
                            # Zusätzliche Infos
                            if res.get('quelle'):
                                st.caption(f"Quelle: {res['quelle']}")
                                
                        else:
                            st.error(f"Bild nicht gefunden: {image_path}")
                    except Exception as e:
                        st.error(f"Fehler beim Laden des Bildes: {e}")
    else:
        # Zeige nur Liste ohne Bilder
        st.subheader("Gefundene Bilder:")
        
        try:
            import pandas as pd
            
            # Erstelle DataFrame für bessere Darstellung
            display_data = []
            for i, res in enumerate(results_list):
                # Farbkodierung für passende Gesichter
                if res['dist'] < 0.35:
                    distance_color = "🟢"  # Sehr ähnlich
                elif res['dist'] < 0.45:
                    distance_color = "🟡"  # Ähnlich
                else:
                    distance_color = "🟠"  # Noch passend
                    
                filename_display = res["filename"]
                
                display_data.append({
                    "Nr.": i+1,
                    "Datei": filename_display,
                    "Distanz": f"{distance_color} {res['dist']:.3f}",
                    "Quelle": res.get('quelle', 'unbekannt')
                })
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True)
            
        except ImportError:
            # Fallback ohne pandas
            for i, res in enumerate(results_list):
                if res['dist'] < 0.35:
                    distance_color = "🟢"
                elif res['dist'] < 0.45:
                    distance_color = "🟡"
                else:
                    distance_color = "🟠"
                    
                st.write(f"{i+1}. **{res['filename']}** - {distance_color} Distanz: {res['dist']:.3f}")


def show_image_gallery(results_list, title="Suchergebnisse"):
    """Zeige eine Bildergalerie der Suchergebnisse."""
    if not results_list:
        st.info("Keine Bilder zu zeigen.")
        return
        
    st.subheader(title)
    
    if len(results_list) > 20:
        # Große Mengen: Zeige Warning und begrenze zunächst
        st.warning("⏳ Viele Bilder gefunden. Das Laden kann etwas dauern.")
        
        # Zeige Statistik
        if results_list:
            best_dist = min(r['dist'] for r in results_list)
            worst_dist = max(r['dist'] for r in results_list)
            avg_dist = sum(r['dist'] for r in results_list) / len(results_list)
            
            st.info(f"📊 **{len(results_list)} passende Gesichter gefunden**")
            st.info(f"🎯 Distanz-Bereich: {best_dist:.3f} (beste) bis {worst_dist:.3f} (schlechteste), Ø {avg_dist:.3f}")
        
        # Bilder in Spalten anzeigen
        cols_per_row = 4
        
        with st.spinner("Lade Bilder..."):
            for i in range(0, len(results_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, res in enumerate(results_list[i:i+cols_per_row]):
                    with cols[j]:
                        # Farbkodierung für passende Gesichter
                        if res['dist'] < 0.35:
                            distance_color = "🟢"  # Sehr ähnlich
                        elif res['dist'] < 0.45:
                            distance_color = "🟡"  # Ähnlich
                        else:
                            distance_color = "🟠"  # Noch passend
                        
                        # Bestimme Pfad und Caption
                        filename_display = res["filename"]
                        
                        if "/" in filename_display:
                            folder_name = os.path.dirname(filename_display)
                            caption = f"**{folder_name}**\n{os.path.basename(filename_display)}\n{distance_color} Distanz: {res['dist']:.3f}"
                        else:
                            caption = f"**{filename_display}**\n{distance_color} Distanz: {res['dist']:.3f}"
                        
                        # Lade und zeige Bild
                        image_path = os.path.join(IMAGE_FOLDER, res["filename"])
                        if os.path.exists(image_path):
                            try:
                                image = Image.open(image_path)
                                st.image(image, caption=caption, use_column_width=True)
                            except Exception as e:
                                st.error(f"Fehler beim Laden: {res['filename']}")
                        else:
                            st.error(f"❌ Datei nicht gefunden: {res['filename']}")
    else:
        # Wenige Bilder: Normale Anzeige
        display_search_results(results_list, show_images=True)


def show_search_stats(results_list):
    """Zeige Statistiken über die Suchergebnisse."""
    if not results_list:
        return
        
    # Grundstatistiken
    total_results = len(results_list)
    
    # Gruppiere nach Qualität
    excellent = len([r for r in results_list if r['dist'] < 0.35])
    good = len([r for r in results_list if 0.35 <= r['dist'] < 0.45])
    okay = len([r for r in results_list if r['dist'] >= 0.45])
    
    # Zeige Statistiken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gesamt", total_results)
    with col2:
        st.metric("🟢 Sehr ähnlich", excellent)
    with col3:
        st.metric("🟡 Ähnlich", good)
    with col4:
        st.metric("🟠 Noch passend", okay)
    
    # Zeige Distanz-Informationen
    if results_list:
        distances = [r['dist'] for r in results_list]
        st.info(f"🎯 **Distanz-Bereich:** {min(distances):.3f} - {max(distances):.3f} (Durchschnitt: {sum(distances)/len(distances):.3f})")
    
    # Zeige Quellenverteilung
    sources = {}
    for r in results_list:
        source = r.get('quelle', 'unbekannt')
        sources[source] = sources.get(source, 0) + 1
    
    if len(sources) > 1:
        st.info(f"📂 **Quellen:** {', '.join([f'{k}: {v}' for k, v in sources.items()])}")
    
    # Zeige Verzeichnisverteilung
    folders = {}
    for r in results_list:
        filename = r['filename']
        if '/' in filename:
            folder = os.path.dirname(filename)
            folders[folder] = folders.get(folder, 0) + 1
        else:
            folders['Hauptverzeichnis'] = folders.get('Hauptverzeichnis', 0) + 1
    
    if len(folders) > 1:
        folder_info = ', '.join([f'{k}: {v}' for k, v in sorted(folders.items())])
        st.info(f"📁 **Verzeichnisse:** {folder_info}")


def render_search_page():
    """Hauptseite für die Gesichtssuche."""
    st.title("🔍 Gesichtserkennung")
    
    # Info über das neue System
    st.info("🎯 **Neue intelligente Suche:** Zeigt nur passende/ähnliche Gesichter an - keine zufälligen Ergebnisse mehr!")
    
    # Upload-Bereich
    uploaded_file = st.file_uploader(
        "Bild mit Gesicht zum Suchen hochladen",
        type=['png', 'jpg', 'jpeg', 'heic', 'HEIC']
    )
    
    if uploaded_file is not None:
        # Zeige hochgeladenes Bild
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladenes Suchbild", width=300)
        
        # Suche starten
        if st.button("🔍 Ähnliche Gesichter suchen", type="primary"):
            with st.spinner("Suche nach ähnlichen Gesichtern..."):
                
                # Fortschrittsbalken
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(progress):
                    progress_bar.progress(progress)
                    if progress < 0.1:
                        status_text.text("Bereite Suche vor...")
                    elif progress < 0.5:
                        status_text.text("Analysiere Gesicht...")
                    elif progress < 0.9:
                        status_text.text("Durchsuche Datenbank...")
                    else:
                        status_text.text("Finalisiere Ergebnisse...")
                
                try:
                    # Führe Suche durch
                    start_time = time.time()
                    results = search_similar_faces_live(
                        uploaded_file,
                        progress_callback=progress_callback,
                        use_vector_db=True  # Nutze die neue Vector DB
                    )
                    search_time = time.time() - start_time
                    
                    # Verstecke Fortschrittsbalken
                    progress_bar.empty()
                    status_text.empty()
                    
                    if results:
                        st.success(f"✅ Suche abgeschlossen in {search_time:.1f}s")
                        
                        # Zeige Statistiken
                        show_search_stats(results)
                        
                        # Zeige Ergebnisse
                        show_image_gallery(results, title="Gefundene passende Gesichter")
                        
                    else:
                        st.warning("❌ Keine passenden Gesichter gefunden.")
                        st.info("💡 **Tipps:**\n- Stelle sicher, dass ein klares Gesicht im Bild erkennbar ist\n- Probiere ein anderes Foto derselben Person\n- Die Person könnte noch nicht in der Datenbank sein")
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Fehler bei der Suche: {e}")
    
    # Zeige Informationen über das System
    st.markdown("---")
    st.subheader("ℹ️ Wie funktioniert die Gesichtserkennung?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🎯 Intelligente Filterung:**
        - Nur echte Gesichtsübereinstimmungen
        - Keine zufälligen 50-Bilder-Listen mehr
        - Dynamische Anzahl je nach Ähnlichkeit
        """)
    
    with col2:
        st.markdown("""
        **🎨 Qualitäts-Anzeige:**
        - 🟢 Sehr ähnlich (< 0.35)
        - 🟡 Ähnlich (0.35-0.45)  
        - 🟠 Noch passend (0.45-0.50)
        """)
    
    st.info("🚀 **Performance:** Nutzt Vector Database (FAISS) für schnelle Suche in großen Bilddatenbanken")


if __name__ == "__main__":
    render_search_page()
