#!/usr/bin/env python3
"""
Test-Script um zu prüfen, welche Bilder keine Gesichter haben.
"""
import os
import sys
from pathlib import Path

# Füge das aktuelle Verzeichnis zum Python Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_face_detection():
    """Teste Gesichtserkennung an ein paar Bildern."""
    try:
        from vector_db import get_vector_db
        from config import ENABLE_VECTOR_DB, IMAGE_FOLDER
        
        if not ENABLE_VECTOR_DB:
            print("❌ Vector Database ist in config.py deaktiviert")
            return
        
        print("🔍 TESTE GESICHTSERKENNUNG")
        print("="*50)
        
        vector_db = get_vector_db()
        if not vector_db or not vector_db.faiss:
            print("❌ Vector Database nicht verfügbar")
            return
        
        # Finde alle Bilder
        all_images = vector_db._discover_all_images()
        print(f"📁 Gefundene Bilder: {len(all_images)}")
        
        # Teste ersten 20 Bilder
        test_images = all_images[:20]
        
        results = {
            "faces_found": [],
            "no_faces": [],
            "errors": []
        }
        
        print(f"\n🧪 Teste {len(test_images)} Bilder:")
        
        for i, image_path in enumerate(test_images):
            filename = os.path.basename(image_path)
            print(f"\n{i+1:2d}. {filename}")
            
            try:
                result = vector_db._process_single_image(image_path)
                print(f"    Ergebnis: {result}")
                
                if result == "NO_FACES_FOUND" or result is None:
                    results["no_faces"].append(filename)
                    print("    🔴 Keine Gesichter erkannt")
                elif result and not result.startswith("SKIP"):
                    results["faces_found"].append(filename)
                    print("    ✅ Gesichter gefunden")
                else:
                    results["errors"].append(filename)
                    print(f"    ⚠️ Verarbeitungsfehler: {result}")
                    
            except Exception as e:
                results["errors"].append(filename)
                print(f"    ❌ Ausnahme: {e}")
        
        # Zusammenfassung
        print(f"\n📊 ZUSAMMENFASSUNG:")
        print(f"✅ Mit Gesichtern:    {len(results['faces_found'])}")
        print(f"🔴 Ohne Gesichter:    {len(results['no_faces'])}")
        print(f"⚠️ Fehler:           {len(results['errors'])}")
        
        if results["no_faces"]:
            print(f"\n🔴 Bilder ohne Gesichter:")
            for img in results["no_faces"][:10]:
                print(f"   - {img}")
            if len(results["no_faces"]) > 10:
                print(f"   ... und {len(results['no_faces']) - 10} weitere")
        
        if results["errors"]:
            print(f"\n⚠️ Bilder mit Fehlern:")
            for img in results["errors"][:5]:
                print(f"   - {img}")
        
        # Test mit delete_images_without_faces Parameter
        if results["no_faces"]:
            print(f"\n🗑️ TEST LÖSCHFUNKTION:")
            test_image = os.path.join(IMAGE_FOLDER, results["no_faces"][0]) if results["no_faces"] else None
            if test_image and os.path.exists(test_image):
                print(f"Teste Löschfunktion an: {os.path.basename(test_image)}")
                
                # Teste ohne wirklich zu löschen
                deletion_result = vector_db._handle_no_face_image(test_image, delete_images_without_faces=False)
                print(f"Ergebnis (ohne Löschen): {deletion_result}")
                
                print("💡 Führe 'python3 build_index.py update --delete-no-faces' aus zum tatsächlichen Löschen")
        
    except Exception as e:
        print(f"❌ Fehler beim Test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_detection()
