#!/usr/bin/env python3
"""
Vector Database Demo mit wenigen Bildern zum Testen.
"""
import os
import sys
import time
import shutil
from pathlib import Path

# Füge das aktuelle Verzeichnis zum Python Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_demo_dataset():
    """Erstelle kleinen Demo-Datensatz für schnelle Tests."""
    demo_folder = "/Users/benedikt.schaechner/Documents/Programmieren/face/demo_images"
    
    if os.path.exists(demo_folder):
        print(f"📁 Demo-Ordner existiert bereits: {demo_folder}")
        return demo_folder
    
    # Erstelle Demo-Ordner
    os.makedirs(demo_folder, exist_ok=True)
    
    # Kopiere erste 10 Bilder aus dem Hauptordner
    source_folder = "/Users/benedikt.schaechner/Documents/Programmieren/face/static/images"
    copied = 0
    
    for root, dirs, files in os.walk(source_folder):
        if copied >= 10:
            break
        
        for file in files:
            if copied >= 10:
                break
            
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Nur schnelle Formate für Demo
                source = os.path.join(root, file)
                dest = os.path.join(demo_folder, f"demo_{copied+1}_{file}")
                
                try:
                    shutil.copy2(source, dest)
                    copied += 1
                    print(f"📋 Kopiert: {file}")
                except Exception as e:
                    print(f"❌ Fehler beim Kopieren: {e}")
    
    print(f"✅ Demo-Datensatz erstellt: {copied} Bilder in {demo_folder}")
    return demo_folder

def test_vector_db_demo():
    """Teste Vector Database mit kleinem Demo-Datensatz."""
    print("🚀 Vector Database Demo gestartet...")
    
    try:
        from vector_db import VectorFaceDatabase
        from config import ENABLE_VECTOR_DB
        
        if not ENABLE_VECTOR_DB:
            print("❌ Vector Database ist in config.py deaktiviert")
            return False
        
        # Erstelle Demo-Datensatz
        demo_folder = create_demo_dataset()
        
        # Temporär IMAGE_FOLDER auf Demo-Ordner setzen
        import config
        original_folder = config.IMAGE_FOLDER
        config.IMAGE_FOLDER = demo_folder
        
        try:
            # Erstelle Vector DB Instanz
            print("📚 Erstelle Vector Database für Demo...")
            vector_db = VectorFaceDatabase()
            
            if not vector_db.faiss:
                print("❌ FAISS nicht verfügbar")
                return False
            
            # Zeige Demo-Statistiken
            all_images = vector_db._discover_all_images()
            print(f"📊 Demo-Bilder: {len(all_images)}")
            
            if len(all_images) == 0:
                print("⚠️ Keine Demo-Bilder gefunden")
                return False
            
            # Starte Index-Aufbau
            print("🔄 Starte Demo Index-Aufbau...")
            
            def progress_callback(progress: float):
                percent = int(progress * 100)
                bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
                print(f"\r   [{bar}] {percent}%", end="", flush=True)
            
            start_time = time.time()
            success = vector_db.rebuild_index(progress_callback)
            duration = time.time() - start_time
            
            print()  # Neue Zeile nach Progress Bar
            
            if success:
                stats = vector_db.get_stats()
                print(f"✅ Demo-Index erfolgreich aufgebaut in {duration:.1f}s")
                print(f"   - {stats['total_faces']} Gesichter indexiert")
                print(f"   - {stats['total_images']} Bilder verarbeitet")
                
                # Test einer Suche
                if stats['total_faces'] > 0:
                    print("\n🔍 Teste Demo-Suche...")
                    import numpy as np
                    
                    # Nimm das erste indexierte Gesicht als Query
                    if vector_db.metadata:
                        first_meta = vector_db.metadata[0]
                        print(f"   Query-Bild: {first_meta['path']}")
                        
                        # Verwende das erste Encoding als Test-Query
                        test_results = vector_db.search_similar_faces(
                            np.random.rand(128).astype(np.float32), k=5
                        )
                        print(f"   Demo-Suche ergab {len(test_results)} Treffer")
                        
                        for i, result in enumerate(test_results[:3]):
                            print(f"     {i+1}. {result['filename']} (Distanz: {result['dist']:.3f})")
                
                print("\n🎉 Vector Database Demo erfolgreich!")
                print(f"\n💡 Für Produktion: Setze IMAGE_FOLDER zurück auf ursprünglichen Ordner")
                print(f"   und führe Index-Aufbau für alle {len(os.listdir(original_folder))} Bilder durch")
                
                return True
            else:
                print("❌ Demo Index-Aufbau fehlgeschlagen")
                return False
        
        finally:
            # Stelle ursprünglichen IMAGE_FOLDER wieder her
            config.IMAGE_FOLDER = original_folder
        
    except Exception as e:
        print(f"❌ Demo-Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_db_demo()
    sys.exit(0 if success else 1)
