#!/usr/bin/env python3
"""
Debug-Script für Vector Database Suche.
Analysiert warum zu viele falsche Ergebnisse zurückgegeben werden.
"""
import os
import sys
import numpy as np
from pathlib import Path

# Füge das aktuelle Verzeichnis zum Python Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_search():
    """Debugge die Vector Database Suche."""
    try:
        from vector_db import get_vector_db
        from config import ENABLE_VECTOR_DB, DIST_THRESHOLD
        import face_recognition
        
        if not ENABLE_VECTOR_DB:
            print("❌ Vector Database ist deaktiviert")
            return
        
        print("🔍 DEBUG VECTOR DATABASE SUCHE")
        print("="*60)
        
        vector_db = get_vector_db()
        if not vector_db or not vector_db.faiss:
            print("❌ Vector Database nicht verfügbar")
            return
        
        # Zeige aktuelle Einstellungen
        print(f"⚙️ KONFIGURATION:")
        print(f"   DIST_THRESHOLD: {DIST_THRESHOLD}")
        print(f"   Total Gesichter: {vector_db.index.ntotal}")
        print(f"   Metadata Einträge: {len(vector_db.metadata)}")
        
        # Lade ein Test-Bild
        all_images = vector_db._discover_all_images()
        if not all_images:
            print("❌ Keine Bilder gefunden")
            return
        
        test_image = all_images[0]
        print(f"\n📷 TESTE MIT: {os.path.basename(test_image)}")
        
        # Lade und analysiere Bild
        try:
            image = face_recognition.load_image_file(test_image)
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if not face_locations:
                print("❌ Keine Gesichter im Test-Bild gefunden")
                return
            
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if not face_encodings:
                print("❌ Keine Encodings erstellt")
                return
            
            query_encoding = face_encodings[0]
            print(f"✅ Query-Encoding erstellt: Shape {query_encoding.shape}")
            
        except Exception as e:
            print(f"❌ Fehler beim Laden des Test-Bildes: {e}")
            return
        
        # Teste verschiedene k-Werte
        for k_value in [10, 20, 50]:
            print(f"\n🔍 SUCHE MIT k={k_value}:")
            results = vector_db.search_similar_faces(query_encoding, k=k_value)
            
            print(f"   Ergebnisse: {len(results)}")
            
            if results:
                print(f"   Beste Distanz: {results[0]['dist']:.4f}")
                print(f"   Schlechteste Distanz: {results[-1]['dist']:.4f}")
                
                # Zeige erste 5 Ergebnisse
                print(f"   Top 5 Ergebnisse:")
                for i, result in enumerate(results[:5]):
                    filename = os.path.basename(result['filename'])
                    dist = result['dist']
                    sim_score = result.get('similarity_score', 0)
                    print(f"      {i+1}. {filename} - Dist: {dist:.4f}, Sim: {sim_score:.4f}")
                
                # Prüfe auf verdächtige Ergebnisse
                suspicious = [r for r in results if r['dist'] > DIST_THRESHOLD * 0.8]
                if suspicious:
                    print(f"   ⚠️ {len(suspicious)} verdächtige Ergebnisse nahe Threshold")
            
            print(f"   Threshold: {DIST_THRESHOLD} ({'STRENG' if DIST_THRESHOLD < 0.4 else 'NORMAL'})")
        
        # Teste rohe FAISS Suche
        print(f"\n🔬 ROHE FAISS ANALYSE:")
        try:
            query_norm = query_encoding / np.linalg.norm(query_encoding)
            query_vector = query_norm.reshape(1, -1).astype('float32')
            
            # Suche mehr Kandidaten
            raw_similarities, raw_indices = vector_db.index.search(query_vector, 100)
            
            print(f"   Raw Similarities (erste 10):")
            for i in range(min(10, len(raw_similarities[0]))):
                sim = raw_similarities[0][i]
                idx = raw_indices[0][i]
                dist = 1.0 - sim
                status = "✅ UNTER" if dist <= DIST_THRESHOLD else "❌ ÜBER"
                print(f"      {i+1}. Index {idx}: Sim={sim:.4f}, Dist={dist:.4f} {status} Threshold")
            
            # Zeige Statistiken
            valid_count = sum(1 for sim in raw_similarities[0] if (1.0 - sim) <= DIST_THRESHOLD)
            print(f"   Gültige Treffer: {valid_count}/100 ({valid_count}%)")
            
        except Exception as e:
            print(f"❌ Fehler bei roher FAISS Analyse: {e}")
        
        # Empfehlungen
        print(f"\n💡 EMPFEHLUNGEN:")
        if DIST_THRESHOLD > 0.4:
            print(f"   - Threshold zu hoch! Empfohlen: 0.3-0.35 (aktuell: {DIST_THRESHOLD})")
        
        print(f"   - Für exakte Suche: DIST_THRESHOLD = 0.25")
        print(f"   - Für ähnliche Gesichter: DIST_THRESHOLD = 0.35")
        print(f"   - Für lockere Suche: DIST_THRESHOLD = 0.45")
        
    except Exception as e:
        print(f"❌ Debug-Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_search()
