# 🔍 Gesichts-Bildsuche mit KI

Eine moderne Streamlit-Anwendung für Gesichtserkennung und -verwaltung mit erweiterten KI-Funktionen.

## 🚀 Features

- **🎯 Training**: Neue Gesichter hinzufügen und Personen zuordnen
- **🔍 Live-Suche**: Ähnliche Gesichter in Echtzeit finden
- **🧠 KI-Analyse**: Erweiterte Gesichtsattribut-Analyse (Alter, Geschlecht, Emotionen)
- **📊 Statistiken**: Detaillierte Datenbank-Analytics
- **🎨 Moderne UI**: Benutzerfreundliche Oberfläche

## 🛠️ Installation

1. **Abhängigkeiten installieren**
   ```bash
   pip install -r requirements.txt
   ```

2. **Anwendung starten**
   ```bash
   streamlit run app_main.py
   ```

## 📁 Projektstruktur

```
face/
├── app_main.py          # 🆕 Hauptanwendung
├── config.py            # ⚙️ Konfiguration
├── utils.py             # 🔧 Hilfsfunktionen
├── search.py            # 🔍 Suchlogik
├── search_ui.py         # 🎨 Such-UI
├── training.py          # 🎯 Training-Logik
├── ai_analysis_page.py  # 🧠 KI-Analyse
├── deepface_analyzer.py # 🤖 DeepFace Integration
├── stats_page.py        # 📊 Statistiken
├── ui_components.py     # 🧩 UI-Komponenten
├── requirements.txt     # 📦 Abhängigkeiten
├── static/
│   ├── faces/          # 🎯 Trainierte Gesichter
│   └── images/         # 📷 Upload-Bilder
├── face_embeddings.npy # 🧠 KI-Embeddings
├── faces_meta.json     # 📋 Gesichts-Metadaten
└── persons.json        # 👥 Personen-Datenbank
```

## 🔧 Technologie

- **Streamlit**: Web-Framework
- **face_recognition**: Gesichtserkennung
- **DeepFace**: KI-Attributanalyse
- **NumPy/Pillow**: Bildverarbeitung

## 💡 Verwendung

### 1. 🎯 Training
- Neue Personen anlegen
- Mehrere Bilder pro Person hochladen
- Automatische Gesichtserkennung

### 2. 🔍 Suche
- Suchbild hochladen
- Live-Ergebnisse in Echtzeit
- Genauigkeits-Anzeige

### 3. 🧠 KI-Analyse
- Gesichtsattribut-Analyse
- Alter, Geschlecht, Emotionen
- Statistische Auswertungen

### 4. 📊 Statistiken
- Datenbank-Übersicht
- Qualitätsprüfung
- Export-Funktionen

---

**Entwickelt mit ❤️ und KI-Technologie**
