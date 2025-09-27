#!/usr/bin/env python3
"""
Clean Image Duplicates - Utility Script

Einfaches Kommandozeilen-Tool zum Bereinigen von Bilderduplikaten.
Erkennt automatisch Bilder in verschiedenen Größen (z.B. -300x200.jpg, -scaled.jpg, -150x150.jpg)
und behält nur die beste Qualitätsversion bei.

Verwendung:
    python3 clean_duplicates.py                 # Zeigt Bericht ohne zu löschen
    python3 clean_duplicates.py --clean         # Bereinigt Duplikate  
    python3 clean_duplicates.py --report        # Nur Bericht anzeigen
    python3 clean_duplicates.py --dir "pfad"    # Anderes Verzeichnis
"""

import argparse
import sys
from pathlib import Path
from image_quality_manager import clean_image_duplicates, generate_duplicate_report


def clear_all_caches():
    """Clear all application caches after duplicate cleanup"""
    success = 0
    total = 3
    
    # Clear Vector Store cache by creating a cache reset marker
    try:
        cache_reset_file = Path("data/.cache_reset_marker")
        cache_reset_file.parent.mkdir(parents=True, exist_ok=True)
        cache_reset_file.write_text("cache_reset_requested")
        print("   ✅ Vector Store Cache-Reset markiert")
        success += 1
    except Exception as e:
        print(f"   ❌ Vector Store Cache-Reset: {e}")
    
    # Clear potential file system caches
    try:
        import os
        os.sync()  # Sync filesystem
        print("   ✅ Filesystem Cache synchronisiert")
        success += 1
    except Exception as e:
        print(f"   ❌ Filesystem Cache: {e}")
    
    # Clear Python bytecode caches
    try:
        cache_dirs_removed = 0
        for pycache_dir in Path(".").rglob("__pycache__"):
            if pycache_dir.is_dir():
                import shutil
                shutil.rmtree(pycache_dir)
                cache_dirs_removed += 1
        
        if cache_dirs_removed > 0:
            print(f"   ✅ {cache_dirs_removed} __pycache__ Verzeichnisse entfernt")
        else:
            print("   ✅ Python Cache bereits sauber")
        success += 1
    except Exception as e:
        print(f"   ❌ Python Cache: {e}")
    
    # Summary
    if success == total:
        print("   � Alle Caches erfolgreich geleert!")
    else:
        print(f"   ⚠️ {total - success}/{total} Cache-Operationen fehlgeschlagen")
    
    print("   💡 Lade die Streamlit-Seite im Browser neu (F5 oder Strg+R)")
    print("   🔄 'Image not found' Fehler sollten verschwunden sein")


def main():
    parser = argparse.ArgumentParser(
        description='Bereinige Bilderduplikate in verschiedenen Größen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python3 clean_duplicates.py                     # Bericht anzeigen
  python3 clean_duplicates.py --clean             # Duplikate entfernen
  python3 clean_duplicates.py --dir "bilder/"     # Anderes Verzeichnis
  python3 clean_duplicates.py --report --clean    # Bericht + bereinigen

Das Tool erkennt automatisch:
  • Größenvarianten: -300x200, -1024x683, -2048x1365, etc.
  • WordPress-Größen: -scaled, -thumbnail, -medium, -large
  • Behält die beste Qualität (höchste Auflösung oder -scaled)
  • Entfernt niedrigere Qualitäten automatisch
        """
    )
    
    parser.add_argument(
        '--dir', 
        default='data/images',
        help='Bildverzeichnis zum bereinigen (Standard: data/images)'
    )
    
    parser.add_argument(
        '--clean', 
        action='store_true',
        help='Duplikate tatsächlich entfernen (sonst nur Bericht)'
    )
    
    parser.add_argument(
        '--report', 
        action='store_true',
        help='Detaillierten Bericht anzeigen'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Nur wichtige Informationen anzeigen'
    )
    
    args = parser.parse_args()
    
    # Standard-Verhalten: Bericht anzeigen wenn weder --clean noch --report angegeben
    if not args.clean and not args.report:
        args.report = True
    
    image_dir = Path(args.dir)
    
    # Prüfe ob Verzeichnis existiert
    if not image_dir.exists():
        print(f"❌ Fehler: Verzeichnis '{image_dir}' existiert nicht!")
        return 1
    
    metadata_file = image_dir / "image_metadata.json"
    if not metadata_file.exists():
        print(f"❌ Fehler: Metadaten-Datei '{metadata_file}' nicht gefunden!")
        print("   Das Verzeichnis scheint keine heruntergeladenen Bilder zu enthalten.")
        return 1
    
    if not args.quiet:
        print(f"🔍 Analysiere Bilder in: {image_dir}")
        print(f"📄 Metadaten-Datei: {metadata_file}")
        print()
    
    try:
        # Bericht generieren
        if args.report and not args.quiet:
            print("📊 Generiere Duplikat-Bericht...")
            report = generate_duplicate_report(str(image_dir))
            print(report)
            print()
        
        # Bereinigung durchführen
        if args.clean:
            if not args.quiet:
                print("🧹 Starte Duplikat-Bereinigung...")
                
            # Sicherheitsabfrage wenn nicht im quiet mode
            if not args.quiet:
                response = input("Möchten Sie die Duplikate wirklich entfernen? (j/N): ").lower().strip()
                if response not in ['j', 'ja', 'y', 'yes']:
                    print("❌ Abgebrochen.")
                    return 0
            
            # Bereinigung durchführen
            results = clean_image_duplicates(str(image_dir), dry_run=False)
            stats = results['removal_stats']
            
            print("\n" + "="*50)
            print("✅ BEREINIGUNG ABGESCHLOSSEN")
            print("="*50)
            
            print(f"📊 Gruppen verarbeitet: {stats['total_groups']}")
            print(f"✅ Dateien behalten: {stats['files_to_keep']}")
            print(f"🗑️  Dateien entfernt: {stats['files_to_remove']}")
            
            if stats['bytes_to_free'] > 0:
                mb_freed = stats['bytes_to_free'] / (1024 * 1024)
                gb_freed = mb_freed / 1024
                if gb_freed >= 1:
                    print(f"💾 Speicher freigegeben: {gb_freed:.2f} GB")
                else:
                    print(f"💾 Speicher freigegeben: {mb_freed:.2f} MB")
            
            # Show orphan cleanup info if available
            if 'orphan_cleanup' in stats:
                orphan_stats = stats['orphan_cleanup']
                if orphan_stats['orphaned_entries_removed'] > 0:
                    print(f"🧹 Metadaten bereinigt: {orphan_stats['orphaned_entries_removed']} verwaiste Einträge entfernt")
            
            # Clear caches after cleanup  
            print("\n🧹 Lösche Caches...")
            clear_all_caches()
            
            if stats['errors']:
                print(f"⚠️ Fehler: {len(stats['errors'])}")
                if not args.quiet:
                    for error in stats['errors'][:5]:  # Nur erste 5 Fehler anzeigen
                        print(f"   • {error}")
                    if len(stats['errors']) > 5:
                        print(f"   ... und {len(stats['errors']) - 5} weitere Fehler")
            
            if not args.quiet:
                print("\n🎉 Ihre Bildsammlung wurde optimiert!")
                print("   • Nur die besten Qualitätsversionen wurden behalten")
                print("   • Duplikate in niedrigerer Auflösung wurden entfernt")
                print("   • Die Gesichtserkennung sollte nun sauberere Ergebnisse liefern")
        
        else:
            # Nur Dry-Run für finale Statistiken
            results = clean_image_duplicates(str(image_dir), dry_run=True)
            stats = results['removal_stats']
            
            if stats['files_to_remove'] > 0:
                mb_to_free = stats['bytes_to_free'] / (1024 * 1024)
                
                print("💡 Zusammenfassung:")
                print(f"   • {stats['files_to_remove']} Duplikate gefunden")
                print(f"   • {mb_to_free:.2f} MB können freigegeben werden")
                print(f"   • {stats['files_to_keep']} einzigartige Bilder bleiben erhalten")
                print("\n🚀 Führe mit --clean aus, um die Duplikate zu entfernen:")
                print(f"   python3 {sys.argv[0]} --clean --dir \"{args.dir}\"")
            else:
                print("✨ Perfekt! Keine Duplikate gefunden.")
                print("   Ihre Bildsammlung ist bereits optimal organisiert.")
    
    except Exception as e:
        print(f"❌ Fehler: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())