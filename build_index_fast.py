#!/usr/bin/env python3
"""
Schneller Vector Database Index-Aufbau (nur JPG/PNG, keine HEIC).
Für schnelle erste Tests und bessere Performance.
"""
import os
import sys
import time
import argparse
from pathlib import Path

# Füge das aktuelle Verzeichnis zum Python Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Schneller Vector Database Index-Aufbau')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch-Größe für Verarbeitung (default: 10)')
    parser.add_argument('--max-images', type=int, default=1000,
                       help='Maximale Anzahl Bilder (default: 1000)')
    parser.add_argument('--force', action='store_true',
                       help='Ohne Nachfrage ausführen')
    
    args = parser.parse_args()
    
    try:
        from vector_db import VectorFaceDatabase
        from config import ENABLE_VECTOR_DB, IMAGE_FOLDER
        
        if not ENABLE_VECTOR_DB:
            print("❌ Vector Database ist in config.py deaktiviert")
            return 1
        
        print("🚀 SCHNELLER VECTOR INDEX AUFBAU")
        print("="*50)
        print("⚡ Nur JPG/PNG Dateien (keine HEIC)")
        print("🎯 Optimiert für Performance")
        print("="*50)
        
        # Erstelle Vector DB Instanz
        vector_db = VectorFaceDatabase()
        
        if not vector_db.faiss:
            print("❌ FAISS nicht verfügbar")
            return 1
        
        # Sammle nur schnelle Dateiformate
        fast_images = []
        for root, dirs, files in os.walk(IMAGE_FOLDER):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    try:
                        # Prüfe Dateigröße (max 10MB)
                        if os.path.getsize(full_path) < 10 * 1024 * 1024:
                            fast_images.append(full_path)
                    except:
                        continue
        
        # Begrenze auf max_images
        if len(fast_images) > args.max_images:
            fast_images = fast_images[:args.max_images]
        
        print(f"📊 Gefundene schnelle Bilder: {len(fast_images)}")
        print(f"🔧 Batch-Größe: {args.batch_size}")
        print(f"⏱️ Geschätzte Zeit: ~{len(fast_images) * 0.5 / 60:.1f} Minuten")
        
        if not args.force:
            response = input(f"\nMöchten Sie {len(fast_images)} Bilder verarbeiten? [j/N]: ").strip().lower()
            if response not in ['j', 'ja', 'y', 'yes']:
                print("❌ Abgebrochen")
                return 0
        
        # Schneller Index-Aufbau
        build_fast_index(vector_db, fast_images, args.batch_size)
        
        return 0
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1

def build_fast_index(vector_db, image_list, batch_size):
    """Schneller Index-Aufbau mit optimierten Einstellungen."""
    
    print("\n🔄 Starte schnellen Index-Aufbau...")
    start_time = time.time()
    
    # Erstelle neuen Index
    vector_db._create_new_index()
    
    processed = 0
    faces_found = 0
    skipped = 0
    
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        batch_start = time.time()
        
        print(f"\n📦 Batch {i//batch_size + 1}/{(len(image_list)-1)//batch_size + 1}")
        
        for image_path in batch:
            filename = os.path.basename(image_path)
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            
            print(f"   🔍 {filename} ({file_size_mb:.1f}MB)...", end="", flush=True)
            
            try:
                faces_before = vector_db.index.ntotal if vector_db.index else 0
                process_image_fast(vector_db, image_path)
                faces_after = vector_db.index.ntotal if vector_db.index else 0
                faces_in_image = faces_after - faces_before
                
                if faces_in_image > 0:
                    print(f" ✅ {faces_in_image}")
                    faces_found += faces_in_image
                else:
                    print(f" ⚪")
                    skipped += 1
                
            except Exception as e:
                print(f" ❌ {str(e)[:30]}")
                skipped += 1
            
            processed += 1
        
        # Speichere nach jedem Batch
        vector_db._save_index()
        batch_time = time.time() - batch_start
        print(f"   ⏱️ {batch_time:.1f}s | 💾 {vector_db.index.ntotal if vector_db.index else 0} Gesichter")
    
    duration = time.time() - start_time
    
    print(f"\n🎉 Schneller Index-Aufbau abgeschlossen!")
    print(f"   ⏱️ Zeit: {duration:.1f}s ({duration/60:.1f}min)")
    print(f"   📊 Bilder: {processed}")
    print(f"   👥 Gesichter: {faces_found}")
    print(f"   ⚪ Übersprungen: {skipped}")
    print(f"   🚀 Durchschnitt: {processed/max(1,duration):.1f} Bilder/Sekunde")

def process_image_fast(vector_db, image_path):
    """Schnelle Bildverarbeitung mit optimierten Einstellungen."""
    import face_recognition
    import numpy as np
    from config import IMAGE_FOLDER
    
    # Lade Bild
    image = face_recognition.load_image_file(image_path)
    
    # Nur HOG für Performance
    face_locations = face_recognition.face_locations(
        image, 
        number_of_times_to_upsample=0,
        model="hog"
    )
    
    if not face_locations:
        return
    
    # Schnelle Encodings (num_jitters=1)
    face_encodings = face_recognition.face_encodings(
        image, 
        known_face_locations=face_locations,
        num_jitters=1
    )
    
    if not face_encodings:
        return
    
    # Relativer Pfad
    rel_path = os.path.relpath(image_path, IMAGE_FOLDER)
    
    # Füge Gesichter zum Index hinzu
    for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
        if encoding is not None and len(encoding) > 0:
            # Normalisiere für Cosine Similarity
            norm_encoding = encoding / np.linalg.norm(encoding)
            
            # Füge zu FAISS Index hinzu
            vector_db.index.add(norm_encoding.reshape(1, -1).astype('float32'))
            
            # Speichere Metadata
            vector_db.metadata.append({
                'path': rel_path,
                'full_path': image_path,
                'face_index': i,
                'bbox': location,
                'file_hash': vector_db._get_file_hash(image_path),
                'timestamp': time.time()
            })

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
