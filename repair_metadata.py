#!/usr/bin/env python3
"""
Metadata Repair Tool

Repariert Metadaten nach dem Löschen von Duplikaten.
Entfernt verwaiste Einträge für Dateien die nicht mehr existieren.

Verwendung:
    python3 repair_metadata.py                 # Zeigt Bericht
    python3 repair_metadata.py --fix           # Repariert Metadaten
    python3 repair_metadata.py --dir "pfad"    # Anderes Verzeichnis
"""

import argparse
import sys
import json
from pathlib import Path


def repair_metadata(image_dir: str, fix: bool = False) -> dict:
    """
    Repariere Metadaten-Datei nach dem Löschen von Duplikaten
    
    Args:
        image_dir: Pfad zum Bildverzeichnis
        fix: Ob die Reparatur durchgeführt werden soll
    
    Returns:
        Statistiken über die Reparatur
    """
    image_path = Path(image_dir)
    metadata_file = image_path / "image_metadata.json"
    
    if not metadata_file.exists():
        return {"error": f"Metadaten-Datei nicht gefunden: {metadata_file}"}
    
    # Lade Metadaten
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        return {"error": f"Fehler beim Laden der Metadaten: {e}"}
    
    # Prüfe welche Dateien existieren
    existing_files = []
    missing_files = []
    
    print(f"🔍 Prüfe {len(metadata)} Metadaten-Einträge...")
    
    for file_path in metadata.keys():
        physical_path = Path(file_path)
        if physical_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    stats = {
        "total_entries": len(metadata),
        "existing_files": len(existing_files),
        "missing_files": len(missing_files),
        "missing_examples": missing_files[:10]  # Erste 10 als Beispiele
    }
    
    if fix and missing_files:
        # Backup erstellen
        backup_file = metadata_file.with_suffix('.json.backup_repair')
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"✅ Backup erstellt: {backup_file}")
        except Exception as e:
            return {"error": f"Fehler beim Backup erstellen: {e}"}
        
        # Entferne fehlende Dateien aus Metadaten
        for missing_file in missing_files:
            del metadata[missing_file]
        
        # Speichere reparierte Metadaten
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            stats["repaired"] = True
            print(f"✅ Metadaten repariert: {len(missing_files)} verwaiste Einträge entfernt")
        except Exception as e:
            return {"error": f"Fehler beim Speichern: {e}"}
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Repariere Metadaten nach Duplikat-Bereinigung',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python3 repair_metadata.py                    # Zeigt Problem-Bericht
  python3 repair_metadata.py --fix              # Repariert Metadaten
  python3 repair_metadata.py --dir "bilder/"    # Anderes Verzeichnis

Diese Tool behebt das Problem:
  ERROR:utils:Image file does not exist: [datei]

Das passiert wenn:
  • Duplikate gelöscht wurden (physische Dateien weg)
  • Aber Metadaten noch auf gelöschte Dateien verweisen
  • Face Recognition versucht auf nicht-existierende Dateien zuzugreifen
        """
    )
    
    parser.add_argument(
        '--dir', 
        default='data/images',
        help='Bildverzeichnis (Standard: data/images)'
    )
    
    parser.add_argument(
        '--fix', 
        action='store_true',
        help='Metadaten tatsächlich reparieren (sonst nur Bericht)'
    )
    
    args = parser.parse_args()
    
    print(f"🔧 Metadaten-Reparatur für: {args.dir}")
    print("=" * 50)
    
    # Führe Reparatur durch
    stats = repair_metadata(args.dir, fix=args.fix)
    
    if "error" in stats:
        print(f"❌ {stats['error']}")
        return 1
    
    # Zeige Statistiken
    print(f"📊 Metadaten-Analyse:")
    print(f"   • Gesamt Einträge: {stats['total_entries']}")
    print(f"   • Existierende Dateien: {stats['existing_files']}")
    print(f"   • Fehlende Dateien: {stats['missing_files']}")
    
    if stats['missing_files'] > 0:
        print(f"\n⚠️ Problem gefunden:")
        print(f"   {stats['missing_files']} Metadaten-Einträge verweisen auf gelöschte Dateien")
        
        if stats.get('missing_examples'):
            print(f"\n📋 Beispiele fehlender Dateien:")
            for example in stats['missing_examples'][:5]:
                filename = Path(example).name
                print(f"   • {filename}")
            if len(stats['missing_examples']) > 5:
                print(f"   ... und {len(stats['missing_examples']) - 5} weitere")
        
        if args.fix:
            if stats.get('repaired'):
                print(f"\n✅ REPARATUR ERFOLGREICH")
                print(f"   • {stats['missing_files']} verwaiste Einträge entfernt")
                print(f"   • Metadaten-Datei bereinigt")
                print(f"   • Face Recognition Fehler sollten verschwunden sein")
            else:
                print(f"\n❌ Reparatur fehlgeschlagen")
        else:
            print(f"\n💡 Reparatur ausführen:")
            print(f"   python3 {sys.argv[0]} --fix --dir \"{args.dir}\"")
    else:
        print(f"\n✅ Keine Probleme gefunden!")
        print(f"   Alle Metadaten-Einträge verweisen auf existierende Dateien")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())