"""
Training-Modul für die Gesichtserkennungsanwendung.
"""
import os
import time
import uuid
import streamlit as st
import numpy as np
from PIL import Image

from config import FACES_FOLDER
from utils import load_data, save_data, process_image_for_faces, detect_faces_robust
from deepface_analyzer import (
    analyze_uploaded_image_with_attributes, 
    render_attribute_analysis,
    create_attribute_summary,
    DEEPFACE_AVAILABLE
)


def render_training_page():
    """Rendere die Training-Seite."""
    st.header("🎯 Gesichter trainieren")
    st.markdown("Laden Sie Bilder hoch, um neue Gesichter zu trainieren und der Datenbank hinzuzufügen.")
    
    # Lade Daten
    embeddings, faces_meta, persons = load_data()

    # Personen-Management
    st.subheader("👤 Personen-Verwaltung")
    
    # Personen-Auswahl oder neue Person anlegen
    person_labels = [f"{p.get('vorname','')} {p.get('nachname','')} ({p['person_id'][:8]})".strip() for p in persons]
    person_labels.append("➕ Neue Person anlegen")
    
    selected = st.selectbox(
        "Person auswählen oder neue anlegen", 
        person_labels,
        help="Wählen Sie eine bestehende Person oder legen Sie eine neue an"
    )

    if 'new_person' not in st.session_state:
        st.session_state['new_person'] = None

    if selected == "➕ Neue Person anlegen":
        person_id = _handle_new_person_creation(persons, embeddings, faces_meta)
    else:
        st.session_state['new_person'] = None
        idx = person_labels.index(selected)
        person = persons[idx]
        person_id = person["person_id"]

    if person_id:
        _handle_person_training(person_id, persons, embeddings, faces_meta)


def _handle_new_person_creation(persons, embeddings, faces_meta):
    """Behandle die Erstellung einer neuen Person."""
    st.markdown("### ➕ Neue Person hinzufügen")
    col1, col2 = st.columns(2)
    
    with col1:
        vorname = st.text_input("Vorname", key="vorname_new", placeholder="z.B. Max")
    with col2:
        nachname = st.text_input("Nachname", key="nachname_new", placeholder="z.B. Mustermann")
    
    name = st.text_input("Optional: Label/Spitzname", key="label_new", placeholder="z.B. Maxi")
    
    if st.button("👤 Person anlegen", type="primary"):
        if vorname.strip() or nachname.strip():
            person_id = str(uuid.uuid4())
            persons.append({
                "person_id": person_id, 
                "vorname": vorname.strip(), 
                "nachname": nachname.strip(), 
                "name": name.strip()
            })
            save_data(embeddings, faces_meta, persons)
            st.session_state['new_person'] = person_id
            st.success(f"✅ Person '{vorname} {nachname}' wurde angelegt!")
            st.rerun()
        else:
            st.error("❌ Bitte mindestens Vor- oder Nachname eingeben!")

    # Wenn gerade angelegt, sofort als ausgewählt behandeln
    if st.session_state['new_person']:
        return st.session_state['new_person']
    else:
        return None


def _handle_person_training(person_id, persons, embeddings, faces_meta):
    """Behandle das Training für eine spezifische Person."""
    person_data = next((p for p in persons if p['person_id'] == person_id), None)
    if not person_data:
        st.error("❌ Person nicht gefunden!")
        return
    
    vorname = person_data.get('vorname', '')
    nachname = person_data.get('nachname', '')
    person_display_name = f"{vorname} {nachname}".strip()
    
    st.markdown("---")
    st.subheader(f"📷 Trainingsbilder für {person_display_name}")
    
    # Zeige Einstellungen
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Tipp:** Für beste Erkennung laden Sie Bilder aus verschiedenen Blickwinkeln hoch (frontal, seitlich, verschiedene Lichtverhältnisse etc.)")
    with col2:
        analyze_attributes = st.checkbox(
            "🧠 KI-Analyse aktivieren", 
            value=True,
            help="Analysiere Alter, Geschlecht, Emotionen und Ethnizität der Gesichter"
        )
    
    # File Upload
    files = st.file_uploader(
        "Bilder hochladen (mehrere möglich)", 
        type=["jpg", "jpeg", "png", "heic", "heif"], 
        accept_multiple_files=True, 
        key=f"upload_{person_id}",
        help="Unterstützte Formate: JPG, PNG, HEIC, HEIF"
    )
    
    if files:
        _process_uploaded_files(files, person_id, person_display_name, embeddings, faces_meta, persons, analyze_attributes)
    
    # Zeige alle gespeicherten Gesichter dieser Person
    _show_existing_faces(person_id, person_display_name, faces_meta, embeddings, persons)


def _process_uploaded_files(files, person_id, person_display_name, embeddings, faces_meta, persons, analyze_attributes=True):
    """Verarbeite hochgeladene Dateien."""
    from config import HEIF_SUPPORT
    
    if any(f.name.lower().endswith((".heic", ".heif")) for f in files) and not HEIF_SUPPORT:
        st.warning("⚠️ Für HEIC/HEIF-Unterstützung bitte das Paket 'pillow-heif' installieren: `pip install pillow-heif`")
    
    # Fortschrittsbalken für Verarbeitung
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Alle Crops aller Bilder sammeln
    all_crops = []
    all_analysis_results = []
    
    for file_idx, file in enumerate(files):
        status_text.text(f"Verarbeite Bild {file_idx + 1}/{len(files)}: {file.name}")
        progress_bar.progress((file_idx) / len(files))
        
        # Bild verarbeiten mit optionaler KI-Analyse
        file_bytes = file.read()
        
        if analyze_attributes:
            image, analysis_results, error_msg = analyze_uploaded_image_with_attributes(
                file_bytes, file.name, use_improved_analysis=False  # Deaktiviert für bessere Performance
            )
        else:
            image, error_msg = process_image_for_faces(file_bytes, file.name)
            analysis_results = None
        
        if image is None:
            st.warning(f"❌ {file.name}: {error_msg}")
            continue
        
        # Gesichter erkennen
        face_locations, encs = detect_faces_robust(image, use_cnn=False)
        
        if not face_locations:
            st.warning(f"❌ Kein Gesicht erkannt in {file.name}")
            continue
        
        # Crops erstellen
        for i, (loc, enc) in enumerate(zip(face_locations, encs)):
            top, right, bottom, left = loc
            crop = image[top:bottom, left:right]
            
            # KI-Analyseergebnis für dieses spezifische Gesicht (falls verfügbar)
            face_analysis = None
            if analysis_results and i < len(analysis_results):
                face_analysis = analysis_results[i]
            
            all_crops.append({
                'enc': enc,
                'crop': crop,
                'file_name': file.name,
                'idx': i,
                'location': loc,
                'analysis': face_analysis
            })
            
            if analysis_results:
                all_analysis_results.extend(analysis_results)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Verarbeitung abgeschlossen!")
    
    if not all_crops:
        st.error("❌ Keine Gesichter in den hochgeladenen Bildern erkannt.")
    else:
        # Zeige Gesamtanalyse falls vorhanden
        if analyze_attributes and all_analysis_results:
            st.markdown("### 🧠 KI-Analyseergebnisse aller erkannten Gesichter")
            render_attribute_analysis(all_analysis_results, show_details=False)
        
        _handle_face_selection(all_crops, person_id, person_display_name, embeddings, faces_meta, persons)


def _handle_face_selection(all_crops, person_id, person_display_name, embeddings, faces_meta, persons):
    """Behandle die Auswahl der zu speichernden Gesichter."""
    st.markdown("### 🎯 Gesichter-Auswahl")
    st.markdown("**Wählen Sie die Gesichter aus, die für diese Person gespeichert werden sollen:**")
    
    # Multiselect mit Default: alle
    crop_labels = [f"{crop['file_name']} - Gesicht {crop['idx']+1}" for crop in all_crops]
    default_indices = list(range(len(all_crops)))
    
    selected_indices = st.multiselect(
        "Gesichter auswählen:", 
        options=list(range(len(crop_labels))),
        default=default_indices,
        format_func=lambda x: crop_labels[x],
        help="Standardmäßig sind alle erkannten Gesichter ausgewählt"
    )
    
    selected_crops = [all_crops[i] for i in selected_indices]
    
    # Zeige die ausgewählten Gesichter
    if selected_crops:
        st.markdown("### 👁️ Vorschau der ausgewählten Gesichter")
        
        # Grid Layout für Vorschau
        cols_per_row = 4
        for i in range(0, len(selected_crops), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, crop_data in enumerate(selected_crops[i:i+cols_per_row]):
                with cols[j]:
                    # Caption mit KI-Analyseergebnis
                    caption = f"Gesicht {i+j+1}\n{crop_data['file_name']}"
                    if crop_data.get('analysis'):
                        ai_summary = create_attribute_summary([crop_data['analysis']])
                        if ai_summary != "Keine Attributanalyse verfügbar":
                            caption += f"\n🧠 {ai_summary}"
                    
                    st.image(
                        crop_data['crop'], 
                        caption=caption, 
                        use_container_width=True
                    )
        
        # Speichern Button
        if st.button("💾 Ausgewählte Gesichter speichern", type="primary"):
            _save_selected_faces(selected_crops, person_id, person_display_name, embeddings, faces_meta, persons)


def _save_selected_faces(selected_crops, person_id, person_display_name, embeddings, faces_meta, persons):
    """Speichere die ausgewählten Gesichter."""
    with st.spinner("💾 Speichere Gesichter..."):
        new_embs = []
        new_meta = []
        
        for crop_data in selected_crops:
            # Gesichtsbild speichern
            face_pil = Image.fromarray(crop_data['crop'])
            face_id = str(uuid.uuid4())
            crop_fname = f"face_{face_id}.jpg"
            crop_path = os.path.join(FACES_FOLDER, crop_fname)
            face_pil.save(crop_path)
            
            # Metadaten erstellen
            meta = {
                "face_id": face_id,
                "filename": crop_fname,
                "person_id": person_id,
                "created_at": time.time()
            }
            
            new_embs.append(crop_data['enc'])
            new_meta.append(meta)
        
        # Embeddings und Metadaten speichern
        if new_embs:
            if embeddings.shape[0] == 0:
                all_embs = np.array(new_embs)
            else:
                all_embs = np.vstack([embeddings, np.array(new_embs)])
            
            faces_meta.extend(new_meta)
            save_data(all_embs, faces_meta, persons)
            
            st.success(f"✅ {len(new_embs)} neue Gesichter für {person_display_name} gespeichert!")
            st.balloons()
        else:
            st.info("ℹ️ Keine neuen Gesichter ausgewählt.")


def _show_existing_faces(person_id, person_display_name, faces_meta, embeddings, persons):
    """Zeige alle gespeicherten Gesichter einer Person."""
    st.markdown("---")
    st.subheader(f"🖼️ Alle gespeicherten Gesichter für {person_display_name}")
    
    person_faces = [m for m in faces_meta if m.get("person_id") == person_id]
    
    if person_faces:
        st.info(f"📊 **{len(person_faces)} Gesichter** für diese Person gespeichert")
        
        # Grid Layout für gespeicherte Gesichter
        cols_per_row = 4
        for i in range(0, len(person_faces), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, meta in enumerate(person_faces[i:i+cols_per_row]):
                with cols[j]:
                    image_path = os.path.join(FACES_FOLDER, meta["filename"])
                    if os.path.exists(image_path):
                        st.image(
                            image_path, 
                            caption=f"{person_display_name}\n{meta['face_id'][:8]}", 
                            use_container_width=True
                        )
                        
                        if st.button(
                            "🗑️ Löschen", 
                            key=f"del_{meta['face_id']}", 
                            help="Dieses Gesicht löschen"
                        ):
                            _delete_face(meta, faces_meta, embeddings, persons, image_path)
                    else:
                        st.error(f"❌ Bild nicht gefunden: {meta['filename']}")
    else:
        st.info("ℹ️ Noch keine Gesichter für diese Person gespeichert. Laden Sie Bilder hoch, um zu beginnen!")


def _delete_face(meta, faces_meta, embeddings, persons, image_path):
    """Lösche ein Gesicht aus der Datenbank."""
    try:
        # Datei löschen
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Aus Metadaten entfernen
        idx = faces_meta.index(meta)
        faces_meta.pop(idx)
        
        # Aus Embeddings entfernen
        if embeddings.shape[0] > 0:
            embs = np.delete(embeddings, idx, axis=0)
            save_data(embs, faces_meta, persons)
        else:
            save_data(embeddings, faces_meta, persons)
        
        st.success("✅ Gesicht gelöscht!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Fehler beim Löschen: {e}")
