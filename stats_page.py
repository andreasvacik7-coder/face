"""
Statistiken-Modul für die Gesichtserkennungsanwendung.
"""
import os
import json
import time
import streamlit as st
import numpy as np
import face_recognition

from config import (
    EMBEDDINGS_FILE, FACES_META_FILE, PERSONS_FILE, FACES_FOLDER, IMAGE_FOLDER,
    DIST_THRESHOLD, MAX_DISPLAY_RESULTS, HEIF_SUPPORT
)
from utils import load_data, save_data


def render_statistics_page():
    """Rendere die Statistiken-Seite."""
    st.header("📊 Datenbank-Statistiken & Analytics")
    st.markdown("Detaillierte Einblicke in Ihre Gesichtsdatenbank")
    
    # Lade aktuelle Daten
    embeddings, faces_meta, persons = load_data()
    
    # Grundlegende Metriken
    _show_basic_metrics(embeddings, faces_meta, persons)
    
    if len(persons) > 0:
        st.markdown("---")
        _show_person_statistics(persons, faces_meta)
    
    # Systemstatistiken
    st.markdown("---")
    _show_system_information()
    
    # Datenbankqualität
    st.markdown("---")
    _show_database_quality(faces_meta, persons)
    
    # Datenbank-Aktionen
    st.markdown("---")
    _show_maintenance_actions(embeddings, faces_meta, persons)


def _show_basic_metrics(embeddings, faces_meta, persons):
    """Zeige grundlegende Metriken."""
    st.subheader("📈 Übersicht")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "👥 Personen", 
            len(persons),
            help="Anzahl der registrierten Personen"
        )
    
    with col2:
        st.metric(
            "🎯 Trainierte Gesichter", 
            len(faces_meta),
            help="Anzahl der trainierten Gesichtsbilder"
        )
    
    with col3:
        # Durchschnittliche Gesichter pro Person
        if len(persons) > 0:
            avg_faces = len(faces_meta) / len(persons)
            st.metric(
                "📊 Ø Gesichter/Person", 
                f"{avg_faces:.1f}",
                help="Durchschnittliche Anzahl Gesichter pro Person"
            )
        else:
            st.metric("📊 Ø Gesichter/Person", "0")
    
    with col4:
        # Speicherverbrauch
        total_size = _calculate_total_size()
        size_mb = total_size / (1024 * 1024)
        st.metric(
            "💾 Speicherverbrauch", 
            f"{size_mb:.1f} MB",
            help="Gesamter Speicherverbrauch der Datenbank"
        )


def _calculate_total_size():
    """Berechne den gesamten Speicherverbrauch."""
    total_size = 0
    
    for file_path in [EMBEDDINGS_FILE, FACES_META_FILE, PERSONS_FILE]:
        if os.path.exists(file_path):
            total_size += os.path.getsize(file_path)
    
    # Gesichtsbilder
    for filename in os.listdir(FACES_FOLDER):
        if filename.startswith('face_'):
            total_size += os.path.getsize(os.path.join(FACES_FOLDER, filename))
    
    return total_size


def _show_person_statistics(persons, faces_meta):
    """Zeige Personen-Statistiken."""
    st.subheader("👥 Personen-Details")
    
    # Tabelle mit Personen und ihren Gesichtern
    person_stats = []
    for person in persons:
        person_faces = [m for m in faces_meta if m.get("person_id") == person["person_id"]]
        person_stats.append({
            "Name": f"{person.get('vorname', '')} {person.get('nachname', '')}".strip(),
            "Label": person.get('name', ''),
            "Anzahl Gesichter": len(person_faces),
            "Erstellt": person.get('created_at', 'Unbekannt'),
            "ID": person["person_id"][:8]
        })
    
    # Sortiere nach Anzahl Gesichter
    person_stats.sort(key=lambda x: x["Anzahl Gesichter"], reverse=True)
    
    # Zeige Tabelle
    if person_stats:
        _render_person_table(person_stats)
        _render_face_distribution_chart(person_stats)
        _show_person_metrics(person_stats)


def _render_person_table(person_stats):
    """Rendere die Personen-Tabelle."""
    st.markdown("**Personen-Übersicht:**")
    
    # Header
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    with col1:
        st.markdown("**Name**")
    with col2:
        st.markdown("**Label**")
    with col3:
        st.markdown("**Gesichter**")
    with col4:
        st.markdown("**ID**")
    
    st.markdown("---")
    
    # Zeilen
    for stat in person_stats[:10]:  # Zeige max 10 Personen
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            st.write(stat["Name"])
        with col2:
            st.write(stat["Label"] or "-")
        with col3:
            st.write(f"**{stat['Anzahl Gesichter']}**")
        with col4:
            st.write(f"`{stat['ID']}`")
    
    if len(person_stats) > 10:
        st.info(f"... und {len(person_stats) - 10} weitere Personen")


def _render_face_distribution_chart(person_stats):
    """Rendere das Gesichter-Verteilungsdiagramm."""
    st.subheader("📊 Verteilung der Gesichter")
    
    if len(person_stats) > 0:
        # Daten für Diagramm vorbereiten
        chart_data = {}
        for p in person_stats[:8]:  # Max 8 für bessere Darstellung
            name = p["Name"] if p["Name"] else f"Person {p['ID']}"
            chart_data[name[:15]] = p["Anzahl Gesichter"]  # Kürze Namen
        
        # Einfaches Balkendiagramm mit Streamlit
        st.bar_chart(chart_data, height=400)


def _show_person_metrics(person_stats):
    """Zeige Person-spezifische Metriken."""
    col1, col2, col3 = st.columns(3)
    
    # Daten für Statistiken extrahieren
    face_counts = [p["Anzahl Gesichter"] for p in person_stats]
    names = [p["Name"] if p["Name"] else f"Person {p['ID']}" for p in person_stats]
    
    with col1:
        max_faces = max(face_counts) if face_counts else 0
        max_person = names[face_counts.index(max_faces)] if face_counts else "N/A"
        st.metric(
            "🏆 Meiste Gesichter", 
            f"{max_faces}",
            delta=f"Person: {max_person[:20]}..."
        )
    
    with col2:
        min_faces = min(face_counts) if face_counts else 0
        min_person = names[face_counts.index(min_faces)] if face_counts else "N/A"
        st.metric(
            "📉 Wenigste Gesichter", 
            f"{min_faces}",
            delta=f"Person: {min_person[:20]}..."
        )
    
    with col3:
        # Personen ohne Gesichter
        no_faces = len([p for p in person_stats if p["Anzahl Gesichter"] == 0])
        st.metric(
            "⚠️ Ohne Gesichter", 
            no_faces,
            delta="Personen ohne Training"
        )


def _show_system_information():
    """Zeige System-Informationen."""
    st.subheader("🔧 System-Informationen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📁 Dateipfade:**")
        st.code(f"""
Embeddings: {EMBEDDINGS_FILE}
Metadaten: {FACES_META_FILE}
Personen: {PERSONS_FILE}
Gesichter: {FACES_FOLDER}
Uploads: {IMAGE_FOLDER}
        """)
    
    with col2:
        st.markdown("**⚙️ Konfiguration:**")
        st.code(f"""
Distanz-Schwellwert: {DIST_THRESHOLD}
Max. Anzeige-Ergebnisse: {MAX_DISPLAY_RESULTS}
HEIC/HEIF Support: {'✅' if HEIF_SUPPORT else '❌'}
        """)


def _show_database_quality(faces_meta, persons):
    """Zeige Datenbank-Qualitätsprüfung."""
    st.subheader("🔍 Datenbank-Qualität")
    
    # Überprüfe auf Probleme
    issues, warnings = _check_database_issues(faces_meta, persons)
    
    # Zeige Ergebnisse
    if not issues and not warnings:
        st.success("✅ Datenbank ist in einem guten Zustand!")
    else:
        if issues:
            st.error(f"❌ {len(issues)} kritische Probleme gefunden:")
            for issue in issues[:5]:  # Zeige nur die ersten 5
                st.write(f"• {issue}")
            if len(issues) > 5:
                st.write(f"... und {len(issues) - 5} weitere")
        
        if warnings:
            st.warning(f"⚠️ {len(warnings)} Warnungen:")
            for warning in warnings[:5]:  # Zeige nur die ersten 5
                st.write(f"• {warning}")
            if len(warnings) > 5:
                st.write(f"... und {len(warnings) - 5} weitere")


def _check_database_issues(faces_meta, persons):
    """Prüfe die Datenbank auf Probleme."""
    issues = []
    warnings = []
    
    # Prüfe auf fehlende Dateien
    for meta in faces_meta:
        face_path = os.path.join(FACES_FOLDER, meta.get('filename', ''))
        if not os.path.exists(face_path):
            issues.append(f"Fehlende Datei: {meta.get('filename', 'Unbekannt')}")
    
    # Prüfe auf Personen ohne Gesichter
    for person in persons:
        person_faces = [m for m in faces_meta if m.get("person_id") == person["person_id"]]
        if len(person_faces) == 0:
            warnings.append(f"Person ohne Gesichter: {person.get('vorname', '')} {person.get('nachname', '')}")
    
    # Prüfe auf verwaiste Gesichter
    for meta in faces_meta:
        person_exists = any(p["person_id"] == meta.get("person_id") for p in persons)
        if not person_exists:
            issues.append(f"Verwaistes Gesicht: {meta.get('face_id', 'Unbekannt')[:8]}")
    
    return issues, warnings


def _show_maintenance_actions(embeddings, faces_meta, persons):
    """Zeige Datenbank-Wartungsaktionen."""
    st.subheader("🛠️ Datenbank-Wartung")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Datenbank bereinigen", help="Entfernt verwaiste Einträge und fehlende Dateien"):
            _clean_database(embeddings, faces_meta, persons)
    
    with col2:
        if st.button("📊 Embeddings neu berechnen", help="Berechnet alle Gesichts-Embeddings neu"):
            _recalculate_embeddings(faces_meta, persons)
    
    with col3:
        if st.button("📤 Datenbank exportieren", help="Exportiert alle Daten als JSON"):
            _export_database(embeddings, faces_meta, persons)


def _clean_database(embeddings, faces_meta, persons):
    """Bereinige die Datenbank."""
    cleaned_faces_meta = []
    cleaned_count = 0
    
    for meta in faces_meta:
        face_path = os.path.join(FACES_FOLDER, meta.get('filename', ''))
        person_exists = any(p["person_id"] == meta.get("person_id") for p in persons)
        
        if os.path.exists(face_path) and person_exists:
            cleaned_faces_meta.append(meta)
        else:
            cleaned_count += 1
    
    if cleaned_count > 0:
        save_data(embeddings, cleaned_faces_meta, persons)
        st.success(f"✅ {cleaned_count} Einträge bereinigt!")
        st.rerun()
    else:
        st.info("ℹ️ Keine Bereinigung erforderlich.")


def _recalculate_embeddings(faces_meta, persons):
    """Berechne alle Embeddings neu."""
    if len(faces_meta) > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        new_embeddings = []
        valid_meta = []
        
        for i, meta in enumerate(faces_meta):
            status_text.text(f"Berechne Embedding {i+1}/{len(faces_meta)}")
            progress_bar.progress((i + 1) / len(faces_meta))
            
            face_path = os.path.join(FACES_FOLDER, meta.get('filename', ''))
            if os.path.exists(face_path):
                try:
                    face_img = face_recognition.load_image_file(face_path)
                    face_encs = face_recognition.face_encodings(face_img)
                    if face_encs:
                        new_embeddings.append(face_encs[0])
                        valid_meta.append(meta)
                except Exception:
                    continue
        
        if new_embeddings:
            new_embeddings_array = np.array(new_embeddings)
            save_data(new_embeddings_array, valid_meta, persons)
            st.success(f"✅ {len(new_embeddings)} Embeddings neu berechnet!")
        else:
            st.error("❌ Keine gültigen Embeddings gefunden!")
        
        progress_bar.empty()
        status_text.empty()
        st.rerun()
    else:
        st.info("ℹ️ Keine Gesichter zum Neuberechnen vorhanden.")


def _export_database(embeddings, faces_meta, persons):
    """Exportiere die Datenbank."""
    export_data = {
        "export_date": time.time(),
        "persons": persons,
        "faces_meta": faces_meta,
        "embeddings_shape": embeddings.shape if embeddings.size > 0 else [0, 0],
        "config": {
            "dist_threshold": DIST_THRESHOLD,
            "max_display_results": MAX_DISPLAY_RESULTS,
            "heif_support": HEIF_SUPPORT
        }
    }
    
    export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    st.download_button(
        label="💾 JSON herunterladen",
        data=export_json,
        file_name=f"face_database_export_{int(time.time())}.json",
        mime="application/json",
        help="Lädt die Datenbank-Konfiguration als JSON herunter"
    )
