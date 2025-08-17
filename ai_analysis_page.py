"""
Optimierte KI-Analyse Seite mit verbesserter Performance und UX.
"""
import os
import streamlit as st
from PIL import Image
import numpy as np

from deepface_analyzer import (
    analyze_uploaded_image_with_attributes,
    render_attribute_analysis,
    DEEPFACE_AVAILABLE
)
from utils import process_image_for_faces, detect_faces_robust
from config import FACES_FOLDER


def render_ai_analysis_page():
    """Rendere die optimierte KI-Analyse Seite."""
    st.header("🧠 KI-Gesichtsanalyse")
    st.markdown("Erweiterte Gesichtsattribut-Analyse mit modernster KI")
    
    if not DEEPFACE_AVAILABLE:
        st.error("❌ **DeepFace nicht verfügbar**")
        st.markdown("Bitte installieren Sie DeepFace für KI-Funktionen:")
        st.code("pip install deepface tensorflow", language="bash")
        st.info("💡 DeepFace ermöglicht Alter-, Geschlechts- und Emotionsanalyse")
        return
    
    # Tabs für verschiedene Analysemodi
    tab1, tab2, tab3 = st.tabs([
        "📤 Bild analysieren", 
        "🗂️ Datenbank-Analyse", 
        "ℹ️ Über KI-Analyse"
    ])
    
    with tab1:
        _render_upload_analysis()
    
    with tab2:
        _render_database_analysis()
    
    with tab3:
        _render_info_page()


def _render_upload_analysis():
    """Optimierte Upload-Analyse mit allen DeepFace-Attributen."""
    st.subheader("📤 Vollständige Gesichtsanalyse")
    st.markdown("**Analysiert automatisch:** Alter, Geschlecht, Emotionen und Ethnizität")
    
    # Upload-Bereich
    file = st.file_uploader(
        "Bild für vollständige KI-Analyse hochladen", 
        type=["jpg", "jpeg", "png", "heic", "heif"],
        help="Unterstützte Formate: JPG, PNG, HEIC, HEIF"
    )
    
    if file:
        # Analyseoptionen in einer kompakteren Form
        col1, col2 = st.columns(2)
        with col1:
            show_details = st.checkbox("📊 Erweiterte Details anzeigen", value=False,
                                     help="Zeigt detaillierte Analysen für alle Attribute")
        with col2:
            use_improved = st.checkbox("🚀 Robuste Multi-Backend Analyse", value=False, 
                                     help="Nutzt mehrere Fallback-Strategien (langsamer)")
        
        # Sofortige Analyse ohne zusätzliche Buttons
        _analyze_uploaded_file(file, show_details, use_improved)


def _analyze_uploaded_file(file, show_details, use_improved=False):
    """Optimierte Dateianalyse mit besserer Fehlerbehandlung."""
    file_bytes = file.read()
    
    with st.spinner("🧠 KI-Analyse läuft..."):
        # Kombinierte Verarbeitung
        image, analysis_results, error_msg = analyze_uploaded_image_with_attributes(
            file_bytes, file.name, 
            use_improved_analysis=use_improved,
            show_optimization_messages=False
        )
        
        if image is None:
            st.error(f"❌ {error_msg}")
            return
        
        # Gesichter erkennen für Markierungen
        face_locations, _ = detect_faces_robust(image, use_cnn=False)
        
        # Bild mit Markierungen anzeigen
        if face_locations:
            marked_image = _draw_face_boxes(image, face_locations)
            st.image(
                marked_image, 
                caption=f"📷 {len(face_locations)} Gesicht(er) erkannt",
                use_container_width=True
            )
        else:
            st.image(image, caption="📷 Kein Gesicht erkannt", use_container_width=True)
        
        # Analyseergebnisse
        if analysis_results:
            st.markdown("---")
            render_attribute_analysis(analysis_results, show_details=show_details)
            
            # Kompakte Export-Option
            _render_analysis_export(analysis_results)
        else:
            st.warning("⚠️ Keine KI-Analyseergebnisse verfügbar")
            st.info("💡 Versuchen Sie ein anderes Bild oder aktivieren Sie die erweiterte Analyse")


def _render_database_analysis():
    """Rendere Datenbank-Analyse Tab."""
    st.subheader("🗂️ Gespeicherte Gesichter analysieren")
    st.markdown("Analysiere bereits in der Datenbank gespeicherte Gesichter")
    
    from utils import load_data
    
    # Lade Daten
    _, faces_meta, persons = load_data()
    
    if not faces_meta:
        st.info("ℹ️ Keine gespeicherten Gesichter in der Datenbank gefunden")
        st.markdown("💡 Besuchen Sie die **Training**-Seite, um Gesichter hinzuzufügen")
        return
    
    # Person auswählen
    person_options = {}
    for person in persons:
        name = f"{person.get('vorname', '')} {person.get('nachname', '')}".strip()
        label = person.get('name', '')
        display_name = f"{name} ({label})" if label else name
        person_options[display_name] = person['person_id']
    
    person_options["🔍 Alle Personen"] = "all"
    
    selected_person = st.selectbox(
        "👤 Person auswählen",
        list(person_options.keys()),
        help="Wählen Sie eine Person für die Analyse ihrer Gesichter"
    )
    
    person_id = person_options[selected_person]
    
    # Relevante Gesichter filtern
    if person_id == "all":
        relevant_faces = faces_meta
        st.info(f"📊 Analysiere alle **{len(relevant_faces)}** Gesichter in der Datenbank")
    else:
        relevant_faces = [f for f in faces_meta if f.get("person_id") == person_id]
        st.info(f"📊 Analysiere **{len(relevant_faces)}** Gesichter für {selected_person}")
    
    if relevant_faces and st.button("🧠 Vollständige KI-Analyse starten", type="primary"):
        st.info("💡 Analysiert alle Attribute: Alter, Geschlecht, Emotionen, Ethnizität")
        _analyze_database_faces(relevant_faces, persons)


def _analyze_database_faces(faces_meta, persons):
    """Analysiere Gesichter aus der Datenbank mit allen verfügbaren Attributen."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_analyses = []
    valid_faces = []
    
    for i, face_meta in enumerate(faces_meta):
        status_text.text(f"Analysiere Gesicht {i+1}/{len(faces_meta)} - Alle Attribute")
        progress_bar.progress((i+1) / len(faces_meta))
        
        face_path = os.path.join(FACES_FOLDER, face_meta['filename'])
        
        if not os.path.exists(face_path):
            continue
        
        try:
            # Einzelnes Gesichtsbild analysieren - ALLE Attribute
            from deepface_analyzer import analyze_facial_attributes
            analysis_results = analyze_facial_attributes(
                face_path,
                actions=['age', 'gender', 'race', 'emotion']  # Alle verfügbaren Attribute
            )
            
            if analysis_results:
                all_analyses.extend(analysis_results)
                valid_faces.append(face_meta)
                
        except Exception as e:
            st.warning(f"⚠️ Fehler bei {face_meta['filename']}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if all_analyses:
        st.success(f"✅ **{len(all_analyses)}** Gesichter erfolgreich analysiert!")
        
        # Statistische Zusammenfassung
        _render_database_statistics(all_analyses, valid_faces, persons)
        
        # Detaillierte Ergebnisse
        st.markdown("### 📊 Detaillierte Analyseergebnisse")
        render_attribute_analysis(all_analyses, show_details=False)
        
    else:
        st.error("❌ Keine erfolgreichen Analysen")


def _render_database_statistics(analyses, faces_meta, persons):
    """Rendere Statistiken für Datenbankanalysen."""
    st.markdown("### 📈 Analyse-Statistiken")
    
    # Grundlegende Metriken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🧠 Analysierte Gesichter", len(analyses))
    
    with col2:
        ages = [a['age'] for a in analyses if 'age' in a]
        avg_age = sum(ages) / len(ages) if ages else 0
        st.metric("🎂 Durchschnittsalter", f"{avg_age:.1f} Jahre")
    
    with col3:
        genders = [a['gender']['dominant'] for a in analyses if 'gender' in a]
        male_count = sum(1 for g in genders if g.lower() == 'man')
        st.metric("👨 Männlich", f"{male_count}/{len(genders)}")
    
    with col4:
        emotions = [a['emotion']['dominant'] for a in analyses if 'emotion' in a]
        happy_count = sum(1 for e in emotions if e.lower() == 'happy')
        st.metric("😊 Fröhlich", f"{happy_count}/{len(emotions)}")
    
    # Detaillierte Verteilungen
    if len(analyses) > 1:
        _render_distribution_charts(analyses)


def _render_distribution_charts(analyses):
    """Rendere Verteilungsdiagramme."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎭 Emotionsverteilung:**")
        emotions = [a['emotion']['dominant'] for a in analyses if 'emotion' in a]
        if emotions:
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(emotions)) * 100
                st.progress(percentage / 100, text=f"{emotion.title()}: {count} ({percentage:.1f}%)")
    
    with col2:
        st.markdown("**👥 Geschlechterverteilung:**")
        genders = [a['gender']['dominant'] for a in analyses if 'gender' in a]
        if genders:
            gender_counts = {}
            for gender in genders:
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
            
            for gender, count in sorted(gender_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(genders)) * 100
                st.progress(percentage / 100, text=f"{gender.title()}: {count} ({percentage:.1f}%)")


def _render_info_page():
    """Rendere Info-Seite über vollständige KI-Analyse."""
    st.subheader("ℹ️ Vollständige DeepFace Gesichtsanalyse")
    
    st.markdown("### 🧠 DeepFace Technologie")
    st.markdown("Diese Anwendung nutzt **DeepFace** für state-of-the-art Gesichtsanalyse mit allen verfügbaren Attributen.")
    
    st.markdown("### 🎯 Alle analysierten Attribute")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎂 Alter (Age)**")
        st.markdown("- Geschätztes Alter in Jahren")
        st.markdown("- Altersgruppen-Kategorisierung")
        st.markdown("- Genauigkeit: ±3-5 Jahre")
        
        st.markdown("**⚧️ Geschlecht (Gender)**")
        st.markdown("- Mann / Frau Klassifikation")
        st.markdown("- Detaillierte Wahrscheinlichkeits-Scores")
        st.markdown("- Hohe Genauigkeit bei klaren Bildern")
    
    with col2:
        st.markdown("**🎭 Emotion (Emotion)**")
        st.markdown("Alle 7 Grundemotionen:")
        st.markdown("😊 Happy, 😢 Sad, 😡 Angry, 😲 Surprise")
        st.markdown("😨 Fear, 🤢 Disgust, 😐 Neutral")
        
        st.markdown("**🌍 Ethnizität (Race)**")
        st.markdown("Alle 6 Kategorien:")
        st.markdown("Asian, White, Middle Eastern")
        st.markdown("Indian, Latino, Black")
        st.caption("⚠️ Nur als statistisches Merkmal!")
    
    st.markdown("### 📊 Vollständige Datenausgabe")
    st.markdown("- **Detaillierte Scores:** Alle Wahrscheinlichkeiten")
    st.markdown("- **Konfidenz-Bewertung:** Qualitätsbewertung")
    st.markdown("- **Rohdaten-Export:** JSON-Ausgabe verfügbar")
    st.markdown("- **Technische Details:** Alle DeepFace-Parameter")
    
    st.markdown("### ⚠️ Wichtige Hinweise")
    st.warning("KI-Vorhersagen sind Schätzungen, nicht 100% akkurat")
    st.info("Ethnizität ist besonders unsicher - nur grober Hinweis")
    st.success("Alle Analysen werden lokal durchgeführt")
    
    # Beispiel-Code
    with st.expander("💻 DeepFace Code-Beispiel"):
        st.code("""
# Vollständige DeepFace Analyse - wie in dieser App
from deepface import DeepFace

objs = DeepFace.analyze(
    img_path = "image.jpg", 
    actions = ['age', 'gender', 'race', 'emotion'],
    enforce_detection = False
)

# Ergebnis enthält alle Attribute für jedes erkannte Gesicht:
# - age: Geschätztes Alter
# - gender: {'Woman': 67.3, 'Man': 32.7}
# - race: {'asian': 45.2, 'white': 30.1, 'black': 12.3, ...}
# - emotion: {'happy': 78.5, 'neutral': 15.2, 'sad': 4.1, ...}
        """, language="python")


def _draw_face_boxes(image, face_locations):
    """Zeichne Bounding Boxes um erkannte Gesichter."""
    from PIL import Image, ImageDraw
    
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Bounding Box
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        
        # Gesichtsnummer
        draw.text((left, top-20), f"Gesicht {i+1}", fill="red")
    
    return np.array(pil_img)


def _render_analysis_export(analysis_results):
    """Rendere Export-Optionen für Analyseergebnisse."""
    with st.expander("📁 Analyseergebnisse exportieren"):
        import json
        
        # JSON Export
        json_data = json.dumps(analysis_results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 Als JSON herunterladen",
            data=json_data,
            file_name="ki_analyse_ergebnisse.json",
            mime="application/json"
        )
        
        # CSV Export für tabellarische Daten
        csv_data = _convert_to_csv(analysis_results)
        if csv_data:
            st.download_button(
                label="📊 Als CSV herunterladen",
                data=csv_data,
                file_name="ki_analyse_ergebnisse.csv",
                mime="text/csv"
            )


def _convert_to_csv(analysis_results):
    """Konvertiere Analyseergebnisse zu CSV."""
    import io
    import csv
    
    if not analysis_results:
        return None
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    headers = ['Gesicht_Nr', 'Alter', 'Geschlecht', 'Geschlecht_Konfidenz', 
               'Emotion', 'Emotion_Konfidenz', 'Ethnizität', 'Ethnizität_Konfidenz']
    writer.writerow(headers)
    
    # Daten
    for i, result in enumerate(analysis_results):
        row = [i+1]
        
        # Alter
        row.append(result.get('age', ''))
        
        # Geschlecht
        gender_info = result.get('gender', {})
        row.append(gender_info.get('dominant', ''))
        row.append(gender_info.get('confidence', ''))
        
        # Emotion
        emotion_info = result.get('emotion', {})
        row.append(emotion_info.get('dominant', ''))
        row.append(emotion_info.get('confidence', ''))
        
        # Ethnizität
        race_info = result.get('race', {})
        row.append(race_info.get('dominant', ''))
        row.append(race_info.get('confidence', ''))
        
        writer.writerow(row)
    
    return output.getvalue()
