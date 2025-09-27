#!/usr/bin/env python3
"""
Clear All Caches Tool

Löscht alle Anwendungs-Caches nach Duplikat-Bereinigung oder anderen Änderungen.
Behebt "Image not found" Probleme im Streamlit Frontend.

Verwendung:
    python3 clear_caches.py
"""

import sys
from pathlib import Path
import json
import os


def clear_vector_store_cache():
    """Clear Vector Store cache by creating reset marker"""
    try:
        cache_reset_file = Path("data/.cache_reset_marker")
        cache_reset_file.parent.mkdir(parents=True, exist_ok=True)
        cache_reset_file.write_text("cache_reset_requested")
        print("✅ Vector Store Cache-Reset markiert")
        return True
    except Exception as e:
        print(f"❌ Vector Store Cache-Reset Fehler: {e}")
        return False


def clear_streamlit_cache():
    """Try to clear Streamlit cache"""
    try:
        # Delete Streamlit cache directory if exists
        streamlit_cache_dirs = [
            Path.home() / ".streamlit" / "cache",
            Path(".streamlit") / "cache", 
            Path("cache"),
            Path(".cache")
        ]
        
        cleared = False
        for cache_dir in streamlit_cache_dirs:
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print(f"✅ Streamlit Cache Verzeichnis gelöscht: {cache_dir}")
                cleared = True
        
        if not cleared:
            print("✅ Streamlit Cache Verzeichnisse nicht gefunden (bereits sauber)")
        
        return True
    except Exception as e:
        print(f"❌ Streamlit Cache Fehler: {e}")
        return False


def clear_system_caches():
    """Clear system level caches"""
    try:
        # Sync filesystem
        os.sync()
        print("✅ Filesystem Cache synchronisiert")
        
        # Clear Python bytecode cache
        import py_compile
        
        # Find and remove __pycache__ directories
        cache_dirs_removed = 0
        for pycache_dir in Path(".").rglob("__pycache__"):
            if pycache_dir.is_dir():
                import shutil
                shutil.rmtree(pycache_dir)
                cache_dirs_removed += 1
        
        if cache_dirs_removed > 0:
            print(f"✅ {cache_dirs_removed} __pycache__ Verzeichnisse entfernt")
        
        return True
    except Exception as e:
        print(f"❌ System Cache Fehler: {e}")
        return False


def validate_metadata_consistency():
    """Check if metadata is consistent with actual files"""
    try:
        metadata_file = Path("data/images/image_metadata.json")
        if not metadata_file.exists():
            print("⚠️ Keine Metadaten-Datei gefunden")
            return True
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        missing_files = []
        for file_path in metadata.keys():
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"⚠️ {len(missing_files)} verwaiste Metadaten-Einträge gefunden")
            print("💡 Führe 'python3 repair_metadata.py --fix' aus")
            return False
        else:
            print("✅ Metadaten sind konsistent")
            return True
        
    except Exception as e:
        print(f"❌ Metadaten-Validierung Fehler: {e}")
        return False


def main():
    print("🧹 Cache-Bereinigung gestartet")
    print("=" * 50)
    
    success_count = 0
    total_operations = 4
    
    # 1. Clear Vector Store Cache
    print("1️⃣ Vector Store Cache...")
    if clear_vector_store_cache():
        success_count += 1
    
    # 2. Clear Streamlit Cache  
    print("\n2️⃣ Streamlit Cache...")
    if clear_streamlit_cache():
        success_count += 1
    
    # 3. Clear System Caches
    print("\n3️⃣ System Caches...")
    if clear_system_caches():
        success_count += 1
    
    # 4. Validate Metadata
    print("\n4️⃣ Metadaten-Konsistenz...")
    if validate_metadata_consistency():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 CACHE-BEREINIGUNG ABGESCHLOSSEN")
    print("=" * 50)
    print(f"✅ Erfolgreich: {success_count}/{total_operations} Operationen")
    
    if success_count == total_operations:
        print("\n🎉 Alle Caches erfolgreich geleert!")
        print("💡 Lade jetzt die Streamlit-Seite im Browser neu (F5 oder Strg+R)")
        print("🔄 Die 'Image not found' Fehler sollten verschwunden sein")
    else:
        print(f"\n⚠️ {total_operations - success_count} Operationen fehlgeschlagen")
        print("💡 Versuche die Streamlit-Seite trotzdem neu zu laden")
    
    print("\n📋 Nächste Schritte:")
    print("   1. Streamlit App neu laden (Browser F5)")
    print("   2. Falls Probleme bestehen: Streamlit neu starten")
    print("   3. Bei anhaltenden Problemen: 'python3 repair_metadata.py'")
    
    return 0 if success_count == total_operations else 1


if __name__ == "__main__":
    sys.exit(main())