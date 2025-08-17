#!/usr/bin/env python3
"""
Vector Database Index-Aufbau über Terminal.
Direkte Kontrolle und detaillierte Logs ohne Frontend.
"""
import os
import sys
import time
import argparse
from pathlib import Path

# Füge das aktuelle Verzeichnis zum Python Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Vector Database Index Management')
    parser.add_argument('action', choices=['build', 'rebuild', 'update', 'status', 'delete'], 
                       help='Aktion: build/rebuild (Index aufbauen), update (nur neue Bilder), status (Statistiken), delete (Index löschen)')
    parser.add_argument('--force', action='store_true', 
                       help='Ohne Nachfrage ausführen')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch-Größe für Verarbeitung (auto-detect wenn nicht angegeben)')
    parser.add_argument('--delete-no-faces', action='store_true',
                       help='Lösche Bilder ohne Gesichter automatisch (spart Speicherplatz)')
    
    args = parser.parse_args()
    
    try:
        from vector_db import get_vector_db, VectorFaceDatabase
        from config import ENABLE_VECTOR_DB, IMAGE_FOLDER, VECTOR_INDEX_FILE, VECTOR_METADATA_FILE
        
        if not ENABLE_VECTOR_DB:
            print("❌ Vector Database ist in config.py deaktiviert")
            print("   Setze ENABLE_VECTOR_DB = True um sie zu aktivieren")
            return 1
        
        # Erstelle Vector DB Instanz
        print("📚 Initialisiere Vector Database...")
        vector_db = get_vector_db()
        
        if not vector_db:
            print("❌ Vector Database konnte nicht erstellt werden")
            return 1
        
        if not vector_db.faiss:
            print("❌ FAISS nicht verfügbar - installiere mit: pip install faiss-cpu")
            return 1
        
        # Führe gewählte Aktion aus
        if args.action == 'status':
            show_status(vector_db)
            
        elif args.action == 'delete':
            delete_index(vector_db, args.force)
            
        elif args.action == 'update':
            update_index(vector_db, args.force, args.batch_size, args.delete_no_faces)
            
        elif args.action in ['build', 'rebuild']:
            build_index(vector_db, args.force, args.batch_size, args.delete_no_faces)
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        print("   Installiere fehlende Pakete mit:")
        print("   pip install faiss-cpu numpy streamlit")
        return 1
    
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1

def show_status(vector_db):
    """Zeige detaillierte Status-Informationen."""
    from config import VECTOR_INDEX_FILE, VECTOR_METADATA_FILE
    
    print("\n" + "="*60)
    print("📊 VECTOR DATABASE STATUS")
    print("="*60)
    
    # Basis-Statistiken
    stats = vector_db.get_stats()
    
    print(f"📈 Indexierte Gesichter:     {stats['total_faces']:,}")
    print(f"📁 Einzigartige Bilder:      {stats['total_images']:,}")
    print(f"💾 Index Größe:             {stats['index_size_mb']:.1f} MB")
    print(f"📄 Metadata Größe:          {stats['metadata_size_mb']:.1f} MB")
    
    # Datei-Status
    print(f"\n📂 Dateien:")
    index_exists = os.path.exists(VECTOR_INDEX_FILE)
    metadata_exists = os.path.exists(VECTOR_METADATA_FILE)
    
    print(f"   Index-Datei:    {'✅ Existiert' if index_exists else '❌ Fehlt'}")
    print(f"   Metadata-Datei: {'✅ Existiert' if metadata_exists else '❌ Fehlt'}")
    
    # Rebuild-Status
    needs_rebuild = vector_db.need_rebuild()
    print(f"\n🔄 Index-Status: {'❌ Update nötig' if needs_rebuild else '✅ Aktuell'}")
    
    # Verfügbare Bilder
    available_images = len(vector_db._discover_all_images())
    print(f"📷 Verfügbare Bilder: {available_images:,}")
    
    # Performance-Schätzung
    if stats['total_faces'] > 0:
        print(f"\n⚡ Performance-Schätzung:")
        print(f"   Normale Suche:    ~{available_images * 0.1:.0f}s bei {available_images:,} Bildern")
        print(f"   Vector DB Suche:  ~0.1s (bis zu {available_images//1000}00x schneller!)")
    
    print("="*60)

def delete_index(vector_db, force=False):
    """Lösche Index-Dateien."""
    from config import VECTOR_INDEX_FILE, VECTOR_METADATA_FILE
    
    print("\n🗑️ INDEX LÖSCHEN")
    print("="*40)
    
    if not force:
        print("⚠️ WARNUNG: Dies löscht den kompletten Vector Index!")
        print("   Alle indexierten Gesichter gehen verloren.")
        response = input("\nMöchten Sie fortfahren? [j/N]: ").strip().lower()
        
        if response not in ['j', 'ja', 'y', 'yes']:
            print("❌ Abgebrochen")
            return
    
    try:
        files_deleted = 0
        
        if os.path.exists(VECTOR_INDEX_FILE):
            os.remove(VECTOR_INDEX_FILE)
            print(f"✅ Index-Datei gelöscht: {VECTOR_INDEX_FILE}")
            files_deleted += 1
        
        if os.path.exists(VECTOR_METADATA_FILE):
            os.remove(VECTOR_METADATA_FILE)
            print(f"✅ Metadata-Datei gelöscht: {VECTOR_METADATA_FILE}")
            files_deleted += 1
        
        if files_deleted > 0:
            print(f"\n🎉 {files_deleted} Dateien erfolgreich gelöscht")
        else:
            print("ℹ️ Keine Index-Dateien zum Löschen gefunden")
    
    except Exception as e:
        print(f"❌ Fehler beim Löschen: {e}")

def get_optimal_batch_size(num_images, system_memory_gb=None):
    """Berechne optimale Batch-Size basierend auf Bildanzahl und System-RAM."""
    import psutil
    
    if system_memory_gb is None:
        try:
            # Hole verfügbares System-RAM in GB
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
        except:
            system_memory_gb = 8  # Conservative fallback
    
    # Batch-Size Logik für CNN (reduziert gegenüber HOG):
    # - CNN braucht mehr GPU/RAM, daher kleinere Batches
    # - Wenig RAM (< 8GB): Sehr kleine Batches
    # - Mittleres RAM (8-16GB): Kleine Batches  
    # - Viel RAM (> 16GB): Moderate Batches
    
    if system_memory_gb < 8:
        base_batch = 5   # CNN auf schwachen Systemen
    elif system_memory_gb < 16:
        base_batch = 10  # CNN auf mittleren Systemen
    elif system_memory_gb < 32:
        base_batch = 18  # CNN auf starken Systemen (Ihr System!)
    else:
        base_batch = 25  # CNN auf sehr starken Systemen
    
    # Anpassung an Bildanzahl
    if num_images < 100:
        batch_size = min(base_batch, max(3, num_images // 4))  # Minimum 3 für CNN
    elif num_images < 1000:
        batch_size = base_batch
    else:
        # Bei vielen Bildern etwas größere Batches, aber begrenzt für CNN
        batch_size = min(30, base_batch + 5)  # Erhöht für starke Systeme
    
    print(f"💡 Auto-Batch-Size für CNN: {batch_size} (RAM: {system_memory_gb:.1f}GB, Bilder: {num_images})")
    return batch_size

def update_index(vector_db, force=False, batch_size=None, delete_images_without_faces=False):
    """Aktualisiere Index inkrementell (nur neue Bilder)."""
    from config import IMAGE_FOLDER
    
    print("\n🔄 INKREMENTELLER INDEX-UPDATE")
    print("="*50)
    
    if delete_images_without_faces:
        print("🗑️ Automatisches Löschen von Bildern ohne Gesichter aktiviert")
        print("   ⚠️ Bilder ohne erkennbare Gesichter werden permanent gelöscht!")
    
    # Prüfe verfügbare Bilder
    all_images = vector_db._discover_all_images()
    
    if len(all_images) == 0:
        print("❌ Keine Bilder gefunden")
        print(f"   Prüfe den Ordner: {IMAGE_FOLDER}")
        return
    
    # Prüfe ob Update nötig ist
    if not vector_db.need_rebuild():
        print("✅ Index ist bereits aktuell - kein Update nötig")
        print("💡 Verwende 'rebuild' für kompletten Neuaufbau")
        return
    
    # Bestimme optimale Batch-Size
    if batch_size is None:
        batch_size = get_optimal_batch_size(len(all_images))
    
    print(f"📊 Gefundene Bilder: {len(all_images):,}")
    print(f"🔧 Batch-Größe: {batch_size}")
    
    if delete_images_without_faces and not force:
        print(f"\n⚠️ WARNUNG: Bilder ohne Gesichter werden permanent gelöscht!")
        print("   Dies kann nicht rückgängig gemacht werden.")
        response = input("\nMöchten Sie fortfahren? [j/N]: ").strip().lower()
        
        if response not in ['j', 'ja', 'y', 'yes']:
            print("❌ Abgebrochen")
            return
    elif not force and len(all_images) > 500:
        print(f"\n⚠️ Dies wird neue Bilder aus {len(all_images):,} Gesamtbildern verarbeiten.")
        response = input("\nMöchten Sie fortfahren? [j/N]: ").strip().lower()
        
        if response not in ['j', 'ja', 'y', 'yes']:
            print("❌ Abgebrochen")
            return
    
    print(f"\n🔄 Starte inkrementellen Update...")
    print("="*50)
    
    def progress_callback(progress: float):
        # Terminal Progress Bar
        pass
    
    start_time = time.time()
    
    try:
        # Setze Batch-Größe
        original_batch_size = getattr(vector_db, '_batch_size', 25)
        vector_db._batch_size = batch_size
        
        # Verwende add_new_images_to_index mit delete_images_without_faces Parameter
        success = vector_db.add_new_images_to_index(progress_callback, delete_images_without_faces)
        
        # Restore original batch size
        vector_db._batch_size = original_batch_size
        
        duration = time.time() - start_time
        
        if success:
            final_stats = vector_db.get_stats()
            print("\n" + "="*50)
            print("🎉 INKREMENTELLER UPDATE ERFOLGREICH!")
            print("="*50)
            print(f"⏱️ Gesamtzeit:        {duration/60:.1f} Minuten")
            print(f"👥 Indexierte Gesichter: {final_stats['total_faces']:,}")
            print(f"📁 Verarbeitete Bilder:  {final_stats['total_images']:,}")
            print(f"💾 Index-Größe:       {final_stats['index_size_mb']:.1f} MB")
            print(f"\n🚀 Vector Database ist aktuell!")
            if delete_images_without_faces:
                print("🗑️ Bilder ohne Gesichter wurden automatisch entfernt")
        else:
            print("\n❌ Index-Update fehlgeschlagen")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Index-Update durch Benutzer unterbrochen")
        print("💡 Teilweise verarbeitete Daten wurden gespeichert")
    
    except Exception as e:
        print(f"\n❌ Fehler beim Index-Update: {e}")
        import traceback
        traceback.print_exc()

def build_index(vector_db, force=False, batch_size=None, delete_no_faces=False):
    """Baue Vector Index auf (komplett neu oder smart)."""
    from config import IMAGE_FOLDER
    
    print("\n🚀 VECTOR INDEX AUFBAU")
    print("="*50)
    
    # Prüfe verfügbare Bilder
    all_images = vector_db._discover_all_images()
    
    if len(all_images) == 0:
        print("❌ Keine Bilder gefunden")
        print(f"   Prüfe den Ordner: {IMAGE_FOLDER}")
        return
    
    # Intelligente Entscheidung: Update vs. Rebuild
    needs_rebuild = vector_db.need_rebuild()
    has_existing_index = vector_db.index and vector_db.index.ntotal > 0
    
    if has_existing_index and not needs_rebuild:
        print("✅ Index ist bereits aktuell!")
        print("💡 Verwende 'update' für neue Bilder oder 'rebuild' für kompletten Neuaufbau")
        return
    elif has_existing_index and needs_rebuild:
        print("🔄 Existierender Index gefunden, aber Update nötig")
        print("💡 Verwende inkrementellen Update für bessere Performance...")
        
        if not force:
            response = input("Inkrementeller Update statt kompletter Rebuild? [J/n]: ").strip().lower()
            if response not in ['n', 'no', 'nein']:
                # Führe inkrementellen Update durch
                update_index(vector_db, force=True, batch_size=batch_size, delete_images_without_faces=delete_no_faces)
                return
    
    # Bestimme optimale Batch-Size
    if batch_size is None:
        batch_size = get_optimal_batch_size(len(all_images))
    
    print(f"📊 Gefundene Bilder: {len(all_images):,}")
    
    # Zeitschätzung
    estimated_time = len(all_images) * 2  # ~2 Sekunden pro Bild
    estimated_minutes = estimated_time / 60
    
    print(f"⏱️ Geschätzte Zeit: ~{estimated_minutes:.0f} Minuten")
    print(f"🔧 Batch-Größe: {batch_size}")
    
    if not force and len(all_images) > 100:
        print(f"\n⚠️ Dies wird {len(all_images):,} Bilder komplett neu verarbeiten.")
        print("   Der Vorgang kann nicht unterbrochen werden.")
        response = input("\nMöchten Sie fortfahren? [j/N]: ").strip().lower()
        
        if response not in ['j', 'ja', 'y', 'yes']:
            print("❌ Abgebrochen")
            return
    
    print(f"\n🔄 Starte kompletten Index-Aufbau...")
    print("="*50)
    
    def progress_callback_rebuild(progress: float):
        # Terminal Progress Bar (wird von vector_db.py überschrieben)
        pass
    
    start_time = time.time()
    
    try:
                # Set batch size for this operation
        original_batch_size = getattr(vector_db, '_batch_size', 25)
        vector_db._batch_size = batch_size
        
        # Für kompletten Rebuild: Lösche alte Daten und erstelle neu
        print("🗑️ Lösche alte Vector Database Dateien...")
        vector_db._delete_existing_files()
        
        print("🆕 Erstelle neue Vector Database...")
        vector_db._create_new_index()
        
        print("📊 Verarbeite alle Bilder...")
        success = vector_db.add_new_images_to_index(progress_callback_rebuild, delete_images_without_faces=delete_no_faces)
        
        # Restore original batch size
        vector_db._batch_size = original_batch_size
        
        duration = time.time() - start_time
        
        if success:
            final_stats = vector_db.get_stats()
            print("\n" + "="*50)
            print("🎉 INDEX-AUFBAU ERFOLGREICH ABGESCHLOSSEN!")
            print("="*50)
            print(f"⏱️ Gesamtzeit:        {duration/60:.1f} Minuten")
            print(f"👥 Indexierte Gesichter: {final_stats['total_faces']:,}")
            print(f"📁 Verarbeitete Bilder:  {final_stats['total_images']:,}")
            print(f"💾 Index-Größe:       {final_stats['index_size_mb']:.1f} MB")
            print(f"\n🚀 Vector Database ist bereit!")
            print("   Suchen sind jetzt bis zu 100x schneller!")
        else:
            print("\n❌ Index-Aufbau fehlgeschlagen")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Index-Aufbau durch Benutzer unterbrochen")
        print("💡 Teilweise verarbeitete Daten wurden gespeichert")
        print("   Sie können den Aufbau später fortsetzen")
    
    except Exception as e:
        print(f"\n❌ Fehler beim Index-Aufbau: {e}")
        import traceback
        traceback.print_exc()
    """Baue Vector Index auf."""
    from config import IMAGE_FOLDER
    
    print("\n🚀 VECTOR INDEX AUFBAU")
    print("="*50)
    
    # Prüfe verfügbare Bilder
    all_images = vector_db._discover_all_images()
    
    if len(all_images) == 0:
        print("❌ Keine Bilder gefunden")
        print(f"   Prüfe den Ordner: {IMAGE_FOLDER}")
        return
    
    print(f"📊 Gefundene Bilder: {len(all_images):,}")
    
    # Zeitschätzung
    estimated_time = len(all_images) * 2  # ~2 Sekunden pro Bild
    estimated_minutes = estimated_time / 60
    
    print(f"⏱️ Geschätzte Zeit: ~{estimated_minutes:.0f} Minuten")
    print(f"🔧 Batch-Größe: {batch_size}")
    
    if not force and len(all_images) > 100:
        print(f"\n⚠️ Dies wird {len(all_images):,} Bilder verarbeiten.")
        print("   Der Vorgang kann nicht unterbrochen werden.")
        response = input("\nMöchten Sie fortfahren? [j/N]: ").strip().lower()
        
        if response not in ['j', 'ja', 'y', 'yes']:
            print("❌ Abgebrochen")
            return
    
    print(f"\n🔄 Starte Index-Aufbau...")
    print("="*50)
    
    def progress_callback(progress: float):
        # Terminal Progress Bar (wird von vector_db.py überschrieben)
        pass
    
    start_time = time.time()
    
    try:
        # Setze Batch-Größe falls angepasst
        original_batch_size = getattr(vector_db, '_batch_size', 25)
        vector_db._batch_size = batch_size
        
        # Für kompletten Rebuild: Lösche alte Daten und erstelle neu
        print("🗑️ Lösche alte Vector Database Dateien...")
        vector_db._delete_existing_files()
        
        print("🆕 Erstelle neue Vector Database...")
        vector_db._create_new_index()
        
        print("📊 Verarbeite alle Bilder...")
        success = vector_db.add_new_images_to_index(progress_callback)
        
        # Restore original batch size
        vector_db._batch_size = original_batch_size
        
        duration = time.time() - start_time
        
        if success:
            final_stats = vector_db.get_stats()
            print("\n" + "="*50)
            print("🎉 INDEX-AUFBAU ERFOLGREICH ABGESCHLOSSEN!")
            print("="*50)
            print(f"⏱️ Gesamtzeit:        {duration/60:.1f} Minuten")
            print(f"👥 Indexierte Gesichter: {final_stats['total_faces']:,}")
            print(f"📁 Verarbeitete Bilder:  {final_stats['total_images']:,}")
            print(f"💾 Index-Größe:       {final_stats['index_size_mb']:.1f} MB")
            print(f"\n🚀 Vector Database ist bereit!")
            print("   Suchen sind jetzt bis zu 100x schneller!")
        else:
            print("\n❌ Index-Aufbau fehlgeschlagen")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Index-Aufbau durch Benutzer unterbrochen")
        print("💡 Teilweise verarbeitete Daten wurden gespeichert")
        print("   Sie können den Aufbau später fortsetzen")
    
    except Exception as e:
        print(f"\n❌ Fehler beim Index-Aufbau: {e}")
        import traceback
        traceback.print_exc()

def show_help():
    """Zeige Verwendungshinweise."""
    print("""
🚀 Vector Database Index Management

VERWENDUNG:
    python build_index.py <action> [optionen]

AKTIONEN:
    status              - Zeige Index-Status und Statistiken
    build               - Intelligenter Aufbau (inkrementell wenn möglich)
    update              - Nur neue Bilder hinzufügen (empfohlen!)
    rebuild             - Kompletter Neuaufbau (alle Bilder)
    delete              - Lösche Index komplett

OPTIONEN:
    --force             - Ohne Nachfrage ausführen
    --batch-size N      - Batch-Größe (auto-detect wenn nicht angegeben)
    --delete-no-faces   - Lösche Bilder ohne Gesichter automatisch

BEISPIELE:
    python build_index.py status
    python build_index.py update        # Empfohlen für neue Bilder!
    python build_index.py update --delete-no-faces  # Mit automatischem Löschen
    python build_index.py build --force
    python build_index.py rebuild --batch-size 30
    python build_index.py delete

EMPFOHLENER WORKFLOW:
1. Erste Einrichtung:     build_index.py build
2. Nach neuen Bildern:    build_index.py update --delete-no-faces  # 🚀 Optimal!
3. Bei Problemen:        build_index.py rebuild

AUTOMATISCHES LÖSCHEN (--delete-no-faces):
- Entfernt Bilder ohne erkennbare Gesichter
- Spart Speicherplatz (oft 20-50% der Bilder)
- Verbessert Performance bei zukünftigen Scans
- WARNUNG: Kann nicht rückgängig gemacht werden!

BATCH-SIZE TIPPS:
- Auto-Detection basiert auf RAM und Bildanzahl
- < 8GB RAM: ~10-15 Bilder pro Batch
- 8-16GB RAM: ~20-25 Bilder pro Batch  
- > 16GB RAM: ~35+ Bilder pro Batch
- Bei Speicherproblemen: --batch-size 5

VORTEILE DER VECTOR DATABASE:
- 10-100x schnellere Gesichtssuchen
- Inkrementelle Updates (nur neue Bilder)
- Skaliert linear mit Bildanzahl
- Automatische Batch-Size-Optimierung
""")

if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help', 'help']):
        show_help()
        sys.exit(0)
    
    exit_code = main()
    sys.exit(exit_code)
