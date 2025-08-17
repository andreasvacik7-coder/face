#!/usr/bin/env python3
"""
Vector Database Test und Setup für Face Recognition App.
"""
import os
import sys
import time
import numpy as np
from typing import Optional

# Füge das aktuelle Verzeichnis zum Python Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vector_db():
    """Teste die Vector Database Funktionalität."""
    print("🚀 Vector Database Test gestartet...")
    
    try:
        from vector_db import get_vector_db, VectorFaceDatabase
        from config import ENABLE_VECTOR_DB, IMAGE_FOLDER
        
        if not ENABLE_VECTOR_DB:
            print("❌ Vector Database ist in config.py deaktiviert")
            print("   Setze ENABLE_VECTOR_DB = True um sie zu aktivieren")
            return False
        
        # Erstelle Vector DB Instanz
        print("📚 Erstelle Vector Database...")
        vector_db = get_vector_db()
        
        if not vector_db:
            print("❌ Vector Database konnte nicht erstellt werden")
            return False
        
        if not vector_db.faiss:
            print("❌ FAISS nicht verfügbar - installiere mit: pip install faiss-cpu")
            return False
        
        print("✅ Vector Database erfolgreich initialisiert")
        
        # Zeige Statistiken
        stats = vector_db.get_stats()
        print(f"📊 Aktuelle Statistiken:")
        print(f"   - Gesichter im Index: {stats['total_faces']}")
        print(f"   - Einzigartige Bilder: {stats['total_images']}")
        print(f"   - Index Größe: {stats['index_size_mb']:.1f} MB")
        print(f"   - Metadata Größe: {stats['metadata_size_mb']:.1f} MB")
        
        # Prüfe ob Rebuild nötig ist
        if vector_db.need_rebuild():
            print("🔄 Index-Update erforderlich...")
            
            # Zeige verfügbare Bilder
            image_count = len(vector_db._discover_all_images())
            print(f"📁 Gefundene Bilder: {image_count}")
            
            if image_count == 0:
                print(f"⚠️ Keine Bilder in {IMAGE_FOLDER} gefunden")
                return True
            
            # Frage ob Index aufgebaut werden soll
            print(f"\n🤔 Soll der Index für {image_count} Bilder aufgebaut werden?")
            print("   Dies kann bei vielen Bildern mehrere Minuten dauern...")
            response = input("   Fortfahren? [j/N]: ").strip().lower()
            
            if response in ['j', 'ja', 'y', 'yes']:
                print("🔄 Starte Index-Aufbau...")
                
                def progress_callback(progress: float):
                    percent = int(progress * 100)
                    bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
                    print(f"\r   [{bar}] {percent}%", end="", flush=True)
                
                start_time = time.time()
                success = vector_db.rebuild_index(progress_callback)
                duration = time.time() - start_time
                
                print()  # Neue Zeile nach Progress Bar
                
                if success:
                    final_stats = vector_db.get_stats()
                    print(f"✅ Index erfolgreich aufgebaut in {duration:.1f}s")
                    print(f"   - {final_stats['total_faces']} Gesichter indexiert")
                    print(f"   - {final_stats['total_images']} Bilder verarbeitet")
                else:
                    print("❌ Index-Aufbau fehlgeschlagen")
                    return False
            else:
                print("⏭️ Index-Aufbau übersprungen")
        else:
            print("✅ Index ist aktuell")
        
        # Test einer einfachen Suche (wenn Index existiert)
        if stats['total_faces'] > 0:
            print("\n🔍 Teste Suchfunktionalität...")
            
            # Erstelle Test-Encoding (zufällig)
            test_encoding = np.random.rand(128).astype(np.float32)
            test_encoding = test_encoding / np.linalg.norm(test_encoding)
            
            results = vector_db.search_similar_faces(test_encoding, k=5)
            print(f"   Test-Suche ergab {len(results)} Treffer")
            
            if results:
                print("   Beste Treffer:")
                for i, result in enumerate(results[:3]):
                    print(f"     {i+1}. {result['filename']} (Distanz: {result['dist']:.3f})")
        
        print("\n🎉 Vector Database Test erfolgreich abgeschlossen!")
        return True
        
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        print("   Installiere fehlende Pakete mit:")
        print("   pip install faiss-cpu numpy")
        return False
    
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage():
    """Zeige Verwendungshinweise."""
    print("""
🚀 Vector Database für Face Recognition

Verwendung:
    python test_vector_db.py       - Teste Vector Database
    
Die Vector Database beschleunigt Gesichtssuchen drastisch bei vielen Bildern.
Vorteile:
- 10-100x schneller als normale Suche
- Skaliert linear mit Bildanzahl  
- Persistente Indizierung
- Automatische Updates

Konfiguration in config.py:
- ENABLE_VECTOR_DB = True/False
- VECTOR_INDEX_FILE = Pfad zur Index-Datei
- VECTOR_METADATA_FILE = Pfad zur Metadata-Datei
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
    else:
        success = test_vector_db()
        sys.exit(0 if success else 1)
