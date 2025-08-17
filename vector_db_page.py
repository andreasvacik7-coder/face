"""
Vector Database Verwaltungsseite für Streamlit App.
"""
import streamlit as st
import time
import os
from typing import Optional

try:
    from vector_db import get_vector_db, VectorFaceDatabase
    from config import ENABLE_VECTOR_DB, IMAGE_FOLDER, VECTOR_INDEX_FILE, VECTOR_METADATA_FILE
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

def show_vector_db_page():
    """Zeige Vector Database Verwaltungsseite."""
    st.header("🚀 Vector Database Verwaltung")
    
    if not VECTOR_DB_AVAILABLE:
        st.error("❌ Vector Database nicht verfügbar. Installiere FAISS: `pip install faiss-cpu`")
        return
    
    if not ENABLE_VECTOR_DB:
        st.warning("⚠️ Vector Database ist deaktiviert. Aktiviere sie in `config.py`:")
        st.code("ENABLE_VECTOR_DB = True")
        return
    
    # Initialisiere Vector DB
    vector_db = get_vector_db()
    
    if not vector_db or not vector_db.faiss:
        st.error("❌ Vector Database konnte nicht initialisiert werden")
        return
    
    # Zeige Statistiken
    st.subheader("📊 Aktuelle Statistiken")
    
    stats = vector_db.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Gesichter im Index", stats['total_faces'])
    
    with col2:
        st.metric("Einzigartige Bilder", stats['total_images'])
    
    with col3:
        st.metric("Index Größe", f"{stats['index_size_mb']:.1f} MB")
    
    with col4:
        st.metric("Metadata Größe", f"{stats['metadata_size_mb']:.1f} MB")
    
    # Status Informationen
    st.subheader("🔍 Status")
    
    # Prüfe ob Dateien existieren
    index_exists = os.path.exists(VECTOR_INDEX_FILE)
    metadata_exists = os.path.exists(VECTOR_METADATA_FILE)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if index_exists:
            st.success("✅ Index-Datei existiert")
        else:
            st.error("❌ Index-Datei fehlt")
    
    with col2:
        if metadata_exists:
            st.success("✅ Metadata-Datei existiert")
        else:
            st.error("❌ Metadata-Datei fehlt")
    
    # Rebuild Status
    needs_rebuild = vector_db.need_rebuild()
    
    if needs_rebuild:
        st.warning("🔄 Index-Update erforderlich")
        
        # Zeige Bildanzahl
        all_images = vector_db._discover_all_images()
        st.info(f"📁 Gefundene Bilder: {len(all_images)}")
        
        if len(all_images) == 0:
            st.warning(f"⚠️ Keine Bilder in `{IMAGE_FOLDER}` gefunden")
        
    else:
        st.success("✅ Index ist aktuell")
    
    # Aktionen
    st.subheader("⚙️ Aktionen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Index neu aufbauen", key="rebuild_index"):
            if len(vector_db._discover_all_images()) == 0:
                st.error("❌ Keine Bilder zum Indizieren gefunden")
            else:
                rebuild_index_with_progress(vector_db)
    
    with col2:
        if st.button("🗑️ Index löschen", key="delete_index"):
            delete_index_files()
    
    # Erweiterte Informationen
    if st.checkbox("🔍 Erweiterte Informationen anzeigen"):
        show_advanced_info(vector_db, stats)

def rebuild_index_with_progress(vector_db: VectorFaceDatabase):
    """Baue Index mit Progress Bar neu auf."""
    all_images = vector_db._discover_all_images()
    
    if len(all_images) == 0:
        st.error("❌ Keine Bilder gefunden")
        return
    
    # Warnung bei vielen Bildern
    if len(all_images) > 1000:
        st.warning(f"⚠️ {len(all_images)} Bilder gefunden. Dies kann mehrere Minuten dauern.")
        
        if not st.checkbox("Ich verstehe, dass dies lange dauern kann"):
            return
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(progress: float):
        progress_bar.progress(progress)
        percent = int(progress * 100)
        status_text.text(f"Verarbeite Bilder... {percent}%")
    
    try:
        status_text.text("🔄 Starte Index-Aufbau...")
        start_time = time.time()
        
        success = vector_db.rebuild_index(progress_callback)
        
        duration = time.time() - start_time
        
        if success:
            final_stats = vector_db.get_stats()
            st.success(f"✅ Index erfolgreich aufgebaut in {duration:.1f}s")
            st.info(f"📊 {final_stats['total_faces']} Gesichter aus {final_stats['total_images']} Bildern indexiert")
            
            # Seite neu laden um Statistiken zu aktualisieren
            st.rerun()
        else:
            st.error("❌ Index-Aufbau fehlgeschlagen")
    
    except Exception as e:
        st.error(f"❌ Fehler beim Index-Aufbau: {e}")
    
    finally:
        progress_bar.empty()
        status_text.empty()

def delete_index_files():
    """Lösche Index-Dateien."""
    st.warning("⚠️ Möchten Sie wirklich den kompletten Index löschen?")
    
    if st.button("🗑️ Ja, Index löschen", key="confirm_delete"):
        try:
            files_deleted = 0
            
            if os.path.exists(VECTOR_INDEX_FILE):
                os.remove(VECTOR_INDEX_FILE)
                files_deleted += 1
            
            if os.path.exists(VECTOR_METADATA_FILE):
                os.remove(VECTOR_METADATA_FILE)
                files_deleted += 1
            
            if files_deleted > 0:
                st.success(f"✅ {files_deleted} Index-Dateien gelöscht")
                # Seite neu laden
                st.rerun()
            else:
                st.info("ℹ️ Keine Index-Dateien zum Löschen gefunden")
        
        except Exception as e:
            st.error(f"❌ Fehler beim Löschen: {e}")

def show_advanced_info(vector_db: VectorFaceDatabase, stats: dict):
    """Zeige erweiterte Informationen."""
    st.subheader("🔍 Erweiterte Informationen")
    
    # Datei-Pfade
    st.text("📁 Dateipfade:")
    st.code(f"Index: {VECTOR_INDEX_FILE}")
    st.code(f"Metadata: {VECTOR_METADATA_FILE}")
    st.code(f"Bilder: {IMAGE_FOLDER}")
    
    # FAISS Index Info
    if vector_db.index:
        st.text("🔢 FAISS Index Details:")
        st.code(f"Dimension: {vector_db.dimension}")
        st.code(f"Index Typ: {type(vector_db.index).__name__}")
        st.code(f"Anzahl Vektoren: {vector_db.index.ntotal}")
        st.code(f"Index trainiert: {vector_db.index.is_trained}")
    
    # Letzte Metadata Einträge
    if vector_db.metadata and len(vector_db.metadata) > 0:
        st.text("📝 Letzte Metadata Einträge:")
        
        # Zeige letzte 5 Einträge
        recent_entries = vector_db.metadata[-5:]
        
        for entry in recent_entries:
            st.text(f"  • {entry['path']} (Gesicht {entry['face_index']})")
    
    # Performance Hinweise
    st.subheader("⚡ Performance Tipps")
    
    if stats['total_faces'] > 10000:
        st.info("💡 Bei >10.000 Gesichtern: Verwende IndexIVFFlat für bessere Performance")
    
    if stats['index_size_mb'] > 100:
        st.info("💡 Großer Index: Aktiviere Index-Kompression für Speicherersparnis")
    
    st.info("💡 Die Vector Database beschleunigt Suchen um das 10-100fache bei vielen Bildern")

# Für Integration in die Haupt-App
if __name__ == "__main__":
    show_vector_db_page()
