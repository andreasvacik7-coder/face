# Image Quality Manager - Duplikat-Bereinigung

## Problem

Beim Herunterladen von Bildern aus Websites (besonders WordPress) entstehen oft mehrere Versionen des gleichen Bildes in verschiedenen Größen:

- `IMG_1234-scaled.jpg` (beste Qualität)
- `IMG_1234-2048x1365.jpg` (hohe Auflösung)
- `IMG_1234-1024x683.jpg` (mittlere Auflösung) 
- `IMG_1234-768x512.jpg` (niedrige Auflösung)
- `IMG_1234-300x200.jpg` (sehr kleine Version)
- `IMG_1234-150x150.jpg` (Thumbnail)

Dies führt zu:
- **Duplikaten in der Gesichtserkennung**: Dasselbe Gesicht wird 6x in verschiedenen Qualitäten angezeigt
- **Verschwendung von Speicherplatz**: Gigabytes unnötiger Duplikate
- **Langsamere Performance**: Mehr Dateien zu verarbeiten

## Lösung

Das **Image Quality Manager** System erkennt automatisch Bilder mit identischem Inhalt aber unterschiedlichen Größen und behält nur die **beste Qualitätsversion**.

### Intelligente Qualitätsbewertung

Das System bewertet Bilder nach:

1. **Format-Priorität** (höchster Score gewinnt):
   - `-scaled`: 1000 Punkte (WordPress "scaled" = beste Qualität)
   - `original` (ohne Suffix): 900 Punkte
   - `2048x+`: 800-850 Punkte (sehr hohe Auflösung)
   - `1536x+`: 700-750 Punkte (hohe Auflösung)
   - `1024x+`: 600-650 Punkte (mittlere Auflösung)
   - `768x+`: 500-550 Punkte (niedrige Auflösung)
   - `400x+`: 400-450 Punkte (kleine Auflösung)
   - `300x+`: 300-350 Punkte (sehr kleine Auflösung)
   - `150x+`: 200-250 Punkte (Thumbnail)

2. **Dateigröße**: Bei gleichem Score wird die größere Datei bevorzugt

3. **Spezielle Formate**:
   - `-thumbnail`, `-thumb`: 100 Punkte
   - `-medium`: 250 Punkte  
   - `-large`: 350 Punkte

## Verwendung

### 1. Einfache Bereinigung

```bash
# Bericht über Duplikate anzeigen
python3 clean_duplicates.py

# Duplikate tatsächlich entfernen
python3 clean_duplicates.py --clean

# Anderes Verzeichnis
python3 clean_duplicates.py --clean --dir "meine_bilder/"
```

### 2. Detaillierte Analyse

```bash
# Vollständiger Bericht mit Details
python3 image_quality_manager.py --action report

# Test der Erkennungslogik
python3 image_quality_manager.py --action test

# Bereinigung mit vollem Bericht
python3 image_quality_manager.py --action clean --no-dry-run
```

### 3. Python Integration

```python
from image_quality_manager import clean_image_duplicates, generate_duplicate_report

# Bericht generieren
report = generate_duplicate_report("data/images")
print(report)

# Duplikate bereinigen (Dry Run)
results = clean_image_duplicates("data/images", dry_run=True)
print(f"Würde {results['removal_stats']['files_to_remove']} Dateien entfernen")

# Tatsächlich bereinigen
results = clean_image_duplicates("data/images", dry_run=False)
```

### 4. Automatische Integration

Die Bereinigung ist bereits in `image-download.py` integriert und läuft automatisch nach jedem Download.

## Beispiel-Ergebnisse

### Vor der Bereinigung:
```
IMG_1234-scaled.jpg        (2.5 MB) ✅ BESTE QUALITÄT
IMG_1234-2048x1365.jpg     (1.8 MB) ❌ zu entfernen  
IMG_1234-1536x1024.jpg     (1.2 MB) ❌ zu entfernen
IMG_1234-1024x683.jpg      (800 KB) ❌ zu entfernen
IMG_1234-768x512.jpg       (450 KB) ❌ zu entfernen
IMG_1234-400x267.jpg       (180 KB) ❌ zu entfernen
IMG_1234-300x200.jpg       (120 KB) ❌ zu entfernen
IMG_1234-150x150.jpg       (45 KB)  ❌ zu entfernen
```

### Nach der Bereinigung:
```
IMG_1234-scaled.jpg        (2.5 MB) ✅ BEHALTEN
```

**Ergebnis**: 7 Duplikate entfernt, 4.6 MB gespart pro Bildgruppe!

## Aktuelle Statistiken

Bei der Oxford High School Bildsammlung:

- **11,121 Duplikat-Gruppen** gefunden
- **64,940 doppelte Dateien** können entfernt werden  
- **8.27 GB Speicherplatz** können freigegeben werden
- **84.5% Speicherplatz-Einsparung** möglich

## Sicherheit

- **Backup**: Automatisches Backup der Metadaten vor Änderungen
- **Dry-Run**: Standard-Modus zeigt nur Vorschau ohne zu löschen
- **Sichere Auswahl**: Algorithmus bevorzugt immer höchste Qualität
- **Fehlerbehandlung**: Ausführliche Logging und Error-Recovery

## Flexibilität

Das System erkennt verschiedene Namenskonventionen:

- WordPress: `bild-300x200.jpg`, `bild-scaled.jpg`
- Drupal: `bild_300x200.jpg`, `bild_large.jpg`  
- Custom: `bild-thumb.jpg`, `bild-2k.jpg`
- Position: `300x200-bild.jpg` (Größe am Anfang)

## Installation

Alle Dateien sind bereits verfügbar:
- `image_quality_manager.py` - Hauptsystem
- `clean_duplicates.py` - Einfaches Kommandozeilen-Tool  
- `test_image_quality.py` - Test und Demonstration

## Automatische Integration

Die Duplikat-Bereinigung läuft automatisch:

1. Nach jedem `image-download.py` Durchlauf
2. Beim Ausführen von `clean_duplicates.py --clean`
3. Über die Python-API in eigenen Scripts

## Ergebnis für Gesichtserkennung

Nach der Bereinigung:
- ✅ **Keine doppelten Gesichter** mehr in den Suchergebnissen
- ✅ **Beste Bildqualität** für optimale Gesichtserkennung
- ✅ **Schnellere Verarbeitung** durch weniger Dateien
- ✅ **Saubere Bildsammlung** ohne Duplikate

Die Gesichtserkennung wird nun viel präziser arbeiten und keine redundanten Ergebnisse mehr anzeigen!