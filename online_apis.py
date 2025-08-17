"""
Erweiterte Online-Bildquellen mit echten API-Integrationen.
"""
import os
import requests
import json
from typing import List, Dict, Optional
import streamlit as st


class CelebrityImageAPI:
    """API-Wrapper für Berühmtheiten-Bilder."""
    
    def __init__(self):
        self.apis_available = {
            'tmdb': 'TMDB_API_KEY' in os.environ,
            'google': 'GOOGLE_SEARCH_API_KEY' in os.environ,
            'bing': 'BING_SEARCH_API_KEY' in os.environ,
        }
    
    def search_tmdb_person(self, person_name: str) -> List[str]:
        """Suche Person in TMDB und hole Bilder."""
        if not self.apis_available['tmdb']:
            st.warning("⚠️ TMDB API-Key nicht konfiguriert")
            return []
        
        api_key = os.environ['TMDB_API_KEY']
        base_url = "https://api.themoviedb.org/3"
        
        try:
            # Person suchen
            search_url = f"{base_url}/search/person"
            params = {
                'api_key': api_key,
                'query': person_name
            }
            
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data['results']:
                return []
            
            person = data['results'][0]  # Ersten Treffer nehmen
            person_id = person['id']
            
            # Bilder der Person holen
            images_url = f"{base_url}/person/{person_id}/images"
            response = requests.get(images_url, params={'api_key': api_key})
            response.raise_for_status()
            images_data = response.json()
            
            # Bild-URLs zusammenstellen
            image_urls = []
            base_image_url = "https://image.tmdb.org/t/p/w500"
            
            for profile in images_data.get('profiles', [])[:10]:  # Max 10 Bilder
                if profile.get('file_path'):
                    url = f"{base_image_url}{profile['file_path']}"
                    image_urls.append(url)
            
            return image_urls
            
        except Exception as e:
            st.error(f"❌ TMDB API Fehler: {e}")
            return []
    
    def search_google_images(self, query: str, num_results: int = 5) -> List[str]:
        """Suche Bilder über Google Custom Search API."""
        if not self.apis_available['google']:
            st.warning("⚠️ Google Search API-Key nicht konfiguriert")
            return []
        
        api_key = os.environ['GOOGLE_SEARCH_API_KEY']
        search_engine_id = os.environ.get('GOOGLE_SEARCH_ENGINE_ID', '')
        
        if not search_engine_id:
            st.warning("⚠️ Google Search Engine ID nicht konfiguriert")
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': search_engine_id,
                'q': query,
                'searchType': 'image',
                'num': min(num_results, 10),
                'imgType': 'photo',
                'imgSize': 'medium',
                'safe': 'active'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            image_urls = []
            for item in data.get('items', []):
                if 'link' in item:
                    image_urls.append(item['link'])
            
            return image_urls
            
        except Exception as e:
            st.error(f"❌ Google Search API Fehler: {e}")
            return []


class WikipediaImageExtractor:
    """Wikipedia-Bilder Extraktor."""
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1"
    
    def get_person_images(self, person_name: str) -> List[str]:
        """Hole Bilder einer Person von Wikipedia."""
        try:
            # Wikipedia-Artikel finden
            search_url = f"{self.base_url}/page/summary/{person_name.replace(' ', '_')}"
            response = requests.get(search_url)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            images = []
            
            # Hauptbild
            if 'thumbnail' in data:
                images.append(data['thumbnail']['source'])
            
            # Zusätzliche Bilder über MediaWiki API
            wiki_api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': person_name.replace(' ', '_'),
                'prop': 'images',
                'imlimit': 10
            }
            
            response = requests.get(wiki_api_url, params=params)
            if response.status_code == 200:
                wiki_data = response.json()
                pages = wiki_data.get('query', {}).get('pages', {})
                
                for page_id, page_data in pages.items():
                    for image in page_data.get('images', []):
                        image_title = image['title']
                        if any(ext in image_title.lower() for ext in ['.jpg', '.jpeg', '.png']):
                            # Bild-URL über imageinfo API holen
                            img_params = {
                                'action': 'query',
                                'format': 'json',
                                'titles': image_title,
                                'prop': 'imageinfo',
                                'iiprop': 'url',
                                'iiurlwidth': 500
                            }
                            
                            img_response = requests.get(wiki_api_url, params=img_params)
                            if img_response.status_code == 200:
                                img_data = img_response.json()
                                img_pages = img_data.get('query', {}).get('pages', {})
                                for img_page_id, img_page_data in img_pages.items():
                                    img_info = img_page_data.get('imageinfo', [])
                                    if img_info and 'thumburl' in img_info[0]:
                                        images.append(img_info[0]['thumburl'])
            
            return images[:10]  # Max 10 Bilder
            
        except Exception as e:
            st.error(f"❌ Wikipedia API Fehler: {e}")
            return []


def render_api_config():
    """Rendere API-Konfiguration."""
    st.subheader("⚙️ API-Konfiguration")
    
    st.info("""
    **Für erweiterte Funktionen konfigurieren Sie folgende APIs:**
    
    **TMDB (Berühmtheiten aus Filmen/TV):**
    - Registrierung: https://www.themoviedb.org/settings/api
    - Umgebungsvariable: `TMDB_API_KEY`
    
    **Google Custom Search (Allgemeine Bildsuche):**
    - Google Cloud Console: https://console.cloud.google.com/
    - Umgebungsvariablen: `GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`
    
    **Bing Image Search:**
    - Azure Cognitive Services: https://azure.microsoft.com/de-de/services/cognitive-services/
    - Umgebungsvariable: `BING_SEARCH_API_KEY`
    """)
    
    # API-Status anzeigen
    celebrity_api = CelebrityImageAPI()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        tmdb_status = "✅ Aktiv" if celebrity_api.apis_available['tmdb'] else "❌ Nicht konfiguriert"
        st.metric("TMDB API", tmdb_status)
    
    with col2:
        google_status = "✅ Aktiv" if celebrity_api.apis_available['google'] else "❌ Nicht konfiguriert"
        st.metric("Google Search", google_status)
    
    with col3:
        bing_status = "✅ Aktiv" if celebrity_api.apis_available['bing'] else "❌ Nicht konfiguriert"
        st.metric("Bing Search", bing_status)
    
    # Einfache Wikipedia-Suche (ohne API-Key)
    st.markdown("---")
    st.subheader("📚 Wikipedia-Bildsuche (kostenlos)")
    
    wiki_extractor = WikipediaImageExtractor()
    person_name = st.text_input("Person für Wikipedia-Suche", placeholder="z.B. Albert Einstein")
    
    if st.button("🔍 Wikipedia durchsuchen"):
        if person_name:
            with st.spinner("🔍 Suche in Wikipedia..."):
                wiki_images = wiki_extractor.get_person_images(person_name)
                
                if wiki_images:
                    st.success(f"✅ {len(wiki_images)} Bilder in Wikipedia gefunden!")
                    
                    # Vorschau der ersten paar Bilder
                    cols = st.columns(min(len(wiki_images), 4))
                    for i, img_url in enumerate(wiki_images[:4]):
                        with cols[i]:
                            try:
                                st.image(img_url, caption=f"Wikipedia Bild {i+1}", use_container_width=True)
                            except:
                                st.text(f"Bild {i+1}")
                    
                    # URLs für manuellen Import
                    st.text_area(
                        "🔗 URLs für Import kopieren:",
                        "\n".join(wiki_images),
                        height=150
                    )
                else:
                    st.warning(f"❌ Keine Bilder für '{person_name}' in Wikipedia gefunden")
        else:
            st.warning("⚠️ Bitte geben Sie einen Namen ein")


if __name__ == "__main__":
    # Test der APIs
    api = CelebrityImageAPI()
    wiki = WikipediaImageExtractor()
    print("APIs bereit!")
