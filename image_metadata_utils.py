"""
Utility-Funktionen für Bild-Metadaten.
Ermöglicht das Zuordnen von lokalen Bildpfaden zu ihren ursprünglichen URLs.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional
from config import IMAGE_FOLDER

METADATA_FILE = "image_metadata.json"

def load_image_metadata() -> Dict:
    """Lade Bild-Metadaten aus JSON-Datei."""
    metadata_path = Path(IMAGE_FOLDER) / METADATA_FILE
    
    if not metadata_path.exists():
        return {}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Fehler beim Laden der Metadaten: {e}")
        return {}

def get_source_url_for_image(image_path: str) -> Optional[str]:
    """Hole ursprüngliche URL für ein lokales Bild."""
    metadata = load_image_metadata()
    
    # Versuche verschiedene Pfad-Varianten
    search_paths = [
        image_path,
        os.path.abspath(image_path),
        os.path.relpath(image_path, IMAGE_FOLDER) if image_path.startswith(IMAGE_FOLDER) else image_path
    ]
    
    for path in search_paths:
        if path in metadata:
            return metadata[path].get('source_url')
    
    return None

def get_image_info(image_path: str) -> Dict:
    """Hole vollständige Informationen für ein Bild."""
    metadata = load_image_metadata()
    
    # Versuche verschiedene Pfad-Varianten
    search_paths = [
        image_path,
        os.path.abspath(image_path),
        os.path.relpath(image_path, IMAGE_FOLDER) if image_path.startswith(IMAGE_FOLDER) else image_path
    ]
    
    for path in search_paths:
        if path in metadata:
            info = metadata[path].copy()
            info['local_path'] = image_path
            return info
    
    # Fallback: nur lokaler Pfad
    return {
        'local_path': image_path,
        'source_url': None,
        'website': 'unknown',
        'download_date': None
    }

def save_image_metadata(metadata: Dict):
    """Speichere Bild-Metadaten."""
    metadata_path = Path(IMAGE_FOLDER) / METADATA_FILE
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Fehler beim Speichern der Metadaten: {e}")

def add_image_metadata(local_path: str, source_url: str, website: str, **kwargs):
    """Füge Metadaten für ein Bild hinzu."""
    metadata = load_image_metadata()
    
    metadata[local_path] = {
        'source_url': source_url,
        'website': website,
        'download_date': kwargs.get('download_date'),
        'file_size': kwargs.get('file_size'),
        **kwargs
    }
    
    save_image_metadata(metadata)

def get_metadata_stats() -> Dict:
    """Hole Statistiken über die Metadaten."""
    metadata = load_image_metadata()
    
    if not metadata:
        return {
            'total_images': 0,
            'websites': [],
            'with_urls': 0,
            'without_urls': 0
        }
    
    websites = set()
    with_urls = 0
    
    for info in metadata.values():
        if info.get('website'):
            websites.add(info['website'])
        if info.get('source_url'):
            with_urls += 1
    
    return {
        'total_images': len(metadata),
        'websites': sorted(list(websites)),
        'with_urls': with_urls,
        'without_urls': len(metadata) - with_urls
    }

def format_image_info_for_display(image_path: str) -> str:
    """Formatiere Bild-Informationen für die Anzeige."""
    info = get_image_info(image_path)
    
    lines = []
    lines.append(f"📁 Lokaler Pfad: {os.path.basename(image_path)}")
    
    if info.get('source_url'):
        lines.append(f"🌐 Ursprüngliche URL: {info['source_url']}")
    
    if info.get('website'):
        lines.append(f"🏠 Website: {info['website']}")
    
    if info.get('download_date'):
        lines.append(f"📅 Download: {info['download_date']}")
    
    if info.get('file_size'):
        size_mb = info['file_size'] / (1024 * 1024)
        lines.append(f"💾 Größe: {size_mb:.1f} MB")
    
    return "\n".join(lines)
