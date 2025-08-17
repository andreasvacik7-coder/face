"""
Konfiguration und Konstanten für die Gesichtserkennungsanwendung.
"""
import os

# Entfernt: API-Konfiguration - wird nicht verwendet

# Pfad-Konfiguration
BASE_DIR = os.path.dirname(__file__)
IMAGE_FOLDER = os.path.join(BASE_DIR, 'static', 'images')
FACES_FOLDER = os.path.join(BASE_DIR, 'static', 'faces')
EMBEDDINGS_FILE = os.path.join(BASE_DIR, 'face_embeddings.npy')
FACES_META_FILE = os.path.join(BASE_DIR, 'faces_meta.json')
PERSONS_FILE = os.path.join(BASE_DIR, 'persons.json')

# Vector Database Konfiguration
VECTOR_INDEX_FILE = os.path.join(BASE_DIR, 'face_index.faiss')
VECTOR_METADATA_FILE = os.path.join(BASE_DIR, 'face_vector_meta.json')
ENABLE_VECTOR_DB = True  # Aktiviere Vektor-Datenbank für bessere Performance

# Verhalten-Konfiguration - VERSCHÄRFTE Schwellwerte für präzise Ergebnisse
DIST_THRESHOLD = 0.065  # Strenger Threshold basierend auf User-Feedback (0.068 war zu ungenau)
MAX_DISPLAY_RESULTS = 150   # Erhöht für alle passenden Gesichter
IMAGES_PER_PAGE = 16        # Anzahl Bilder pro Seite (4x4 Grid)
MAX_SEARCH_IMAGES = 2000    # Maximale Anzahl Bilder für die Suche

# HEIC/HEIF Support prüfen
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

# Ordner erstellen
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)
