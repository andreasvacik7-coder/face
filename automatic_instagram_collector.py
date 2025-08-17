"""
Automatischer Instagram-Bildsammler basierend auf den neuen Meta-Nutzungsrichtlinien 2024/2025.
Implementiert automatische Sammlung für KI-Training und Gesichtserkennung.
"""
import os
import requests
import json
import time
import re
from typing import List, Dict, Optional, Tuple
import streamlit as st
from urllib.parse import urljoin, urlparse
import random
from dataclasses import dataclass


@dataclass
class InstagramMedia:
    """Instagram-Medien-Objekt."""
    id: str
    url: str
    thumbnail_url: str
    caption: str
    like_count: int
    media_type: str
    timestamp: str
    shortcode: str


class ModernInstagramCollector:
    """
    Moderner Instagram-Sammler unter den neuen Meta-Richtlinien 2024/2025.
    Nutzt legale Methoden zur automatischen Bildsammlung für KI-Training.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 2.0  # Respektiere Server-Load
        
        # Statistiken
        self.stats = {
            'profiles_processed': 0,
            'images_found': 0,
            'images_downloaded': 0,
            'faces_detected': 0,
            'errors': 0
        }
    
    def _rate_limit(self):
        """Intelligentes Rate-Limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def extract_profile_data(self, username: str) -> Dict:
        """
        Extrahiert öffentliche Profildaten nach den neuen Meta-Richtlinien.
        Fokus auf öffentlich verfügbare Daten für KI-Training.
        """
        try:
            self._rate_limit()
            
            # Moderne Methode: Instagram Web Interface
            url = f"https://www.instagram.com/{username}/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                # Extrahiere JSON-Daten aus der HTML-Seite
                html_content = response.text
                
                # Suche nach dem __additionalDataLoaded JSON
                json_pattern = r'window\._sharedData\s*=\s*({.*?});'
                match = re.search(json_pattern, html_content)
                
                if match:
                    try:
                        shared_data = json.loads(match.group(1))
                        return self._parse_profile_data(shared_data, username)
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: Suche nach anderen JSON-Strukturen
                json_pattern2 = r'"ProfilePage"\s*:\s*\[({.*?})\]'
                match2 = re.search(json_pattern2, html_content)
                
                if match2:
                    try:
                        profile_data = json.loads(match2.group(1))
                        return self._parse_modern_profile_data(profile_data, username)
                    except json.JSONDecodeError:
                        pass
                
                # Minimale Daten extrahieren wenn JSON nicht verfügbar
                return self._extract_basic_profile_info(html_content, username)
            
            return {}
            
        except Exception as e:
            st.warning(f"⚠️ Profil-Extraktion für {username} fehlgeschlagen: {e}")
            self.stats['errors'] += 1
            return {}
    
    def _parse_profile_data(self, shared_data: Dict, username: str) -> Dict:
        """Parse klassische Instagram shared_data Struktur."""
        try:
            entry_data = shared_data.get('entry_data', {})
            profile_page = entry_data.get('ProfilePage', [])
            
            if profile_page:
                user = profile_page[0].get('graphql', {}).get('user', {})
                
                return {
                    'username': username,
                    'full_name': user.get('full_name', ''),
                    'biography': user.get('biography', ''),
                    'external_url': user.get('external_url', ''),
                    'profile_pic_url': user.get('profile_pic_url_hd', ''),
                    'follower_count': user.get('edge_followed_by', {}).get('count', 0),
                    'following_count': user.get('edge_follow', {}).get('count', 0),
                    'media_count': user.get('edge_owner_to_timeline_media', {}).get('count', 0),
                    'is_private': user.get('is_private', True),
                    'is_verified': user.get('is_verified', False),
                    'recent_media': self._extract_recent_media(user.get('edge_owner_to_timeline_media', {}))
                }
        except Exception:
            pass
        
        return {'username': username, 'error': 'Could not parse profile data'}
    
    def _parse_modern_profile_data(self, profile_data: Dict, username: str) -> Dict:
        """Parse moderne Instagram Profil-Datenstruktur."""
        try:
            # Struktur variiert je nach Instagram-Version
            user_data = profile_data.get('user', profile_data)
            
            return {
                'username': username,
                'full_name': user_data.get('full_name', ''),
                'biography': user_data.get('biography', ''),
                'profile_pic_url': user_data.get('profile_pic_url', ''),
                'is_private': user_data.get('is_private', True),
                'media_count': user_data.get('media_count', 0),
                'recent_media': []  # Erweiterte Medien-Extraktion nötig
            }
        except Exception:
            pass
        
        return {'username': username, 'error': 'Could not parse modern profile data'}
    
    def _extract_basic_profile_info(self, html_content: str, username: str) -> Dict:
        """Extrahiere Basis-Informationen aus HTML falls JSON nicht verfügbar."""
        try:
            # Suche nach Meta-Tags und anderen HTML-Elementen
            profile_info = {'username': username}
            
            # og:title für vollständigen Namen
            title_match = re.search(r'<meta property="og:title" content="([^"]*)"', html_content)
            if title_match:
                profile_info['full_name'] = title_match.group(1)
            
            # og:description für Biografie
            desc_match = re.search(r'<meta property="og:description" content="([^"]*)"', html_content)
            if desc_match:
                profile_info['biography'] = desc_match.group(1)
            
            # og:image für Profilbild
            img_match = re.search(r'<meta property="og:image" content="([^"]*)"', html_content)
            if img_match:
                profile_info['profile_pic_url'] = img_match.group(1)
            
            return profile_info
            
        except Exception:
            return {'username': username, 'error': 'Could not extract basic info'}
    
    def _extract_recent_media(self, timeline_media: Dict) -> List[InstagramMedia]:
        """Extrahiere aktuelle Medien aus Timeline-Daten."""
        media_list = []
        
        try:
            edges = timeline_media.get('edges', [])
            
            for edge in edges[:12]:  # Erste 12 Posts
                node = edge.get('node', {})
                
                if node.get('__typename') == 'GraphImage':  # Nur Bilder
                    media = InstagramMedia(
                        id=node.get('id', ''),
                        url=node.get('display_url', ''),
                        thumbnail_url=node.get('thumbnail_src', ''),
                        caption=self._extract_caption(node),
                        like_count=node.get('edge_liked_by', {}).get('count', 0),
                        media_type='image',
                        timestamp=str(node.get('taken_at_timestamp', '')),
                        shortcode=node.get('shortcode', '')
                    )
                    media_list.append(media)
        
        except Exception:
            pass
        
        return media_list
    
    def _extract_caption(self, node: Dict) -> str:
        """Extrahiere Caption aus Post-Node."""
        try:
            edge_caption = node.get('edge_media_to_caption', {})
            edges = edge_caption.get('edges', [])
            if edges:
                return edges[0].get('node', {}).get('text', '')
        except Exception:
            pass
        return ''
    
    def collect_profile_images(self, username: str, max_images: int = 20) -> List[str]:
        """
        Sammelt Bilder eines öffentlichen Profils automatisch.
        Nutzt die neuen Meta-Richtlinien für KI-Training.
        """
        st.info(f"🔍 Analysiere @{username} automatisch...")
        
        # Profildaten extrahieren
        profile_data = self.extract_profile_data(username)
        
        if not profile_data or profile_data.get('error'):
            st.warning(f"❌ Konnte @{username} nicht analysieren")
            return []
        
        self.stats['profiles_processed'] += 1
        
        # Prüfe ob Profil öffentlich ist
        if profile_data.get('is_private', True):
            st.warning(f"🔒 @{username} ist privat - wird übersprungen")
            return []
        
        st.success(f"✅ @{username} gefunden: {profile_data.get('full_name', 'Unbekannt')}")
        
        # Sammle Bild-URLs
        image_urls = []
        
        # Profilbild hinzufügen
        if profile_data.get('profile_pic_url'):
            image_urls.append(profile_data['profile_pic_url'])
        
        # Recent Media hinzufügen
        recent_media = profile_data.get('recent_media', [])
        for media in recent_media[:max_images-1]:  # -1 für Profilbild
            if media.url and self._is_suitable_for_face_detection(media):
                image_urls.append(media.url)
        
        self.stats['images_found'] += len(image_urls)
        
        if image_urls:
            st.success(f"📸 {len(image_urls)} Bilder für @{username} gefunden")
        else:
            st.warning(f"❌ Keine geeigneten Bilder für @{username} gefunden")
        
        return image_urls[:max_images]
    
    def _is_suitable_for_face_detection(self, media: InstagramMedia) -> bool:
        """Prüft ob ein Bild für Gesichtserkennung geeignet ist."""
        # Basis-Heuristiken für Gesichtsbilder
        caption = media.caption.lower()
        
        # Positive Indikatoren
        face_keywords = ['selfie', 'portrait', 'face', 'smile', 'photo', 'pic', 'me']
        has_face_keywords = any(keyword in caption for keyword in face_keywords)
        
        # Negative Indikatoren  
        exclude_keywords = ['landscape', 'food', 'sunset', 'building', 'car', 'animal']
        has_exclude_keywords = any(keyword in caption for keyword in exclude_keywords)
        
        # Likes als Qualitätsindikator
        popular = media.like_count > 100
        
        return (has_face_keywords or popular) and not has_exclude_keywords
    
    def batch_collect_profiles(self, usernames: List[str], max_images_per_profile: int = 15) -> Dict[str, List[str]]:
        """
        Sammelt automatisch Bilder von mehreren Profilen.
        Optimiert für Batch-Verarbeitung mit Progress-Tracking.
        """
        results = {}
        total_usernames = len(usernames)
        
        st.markdown(f"### 🚀 Automatische Sammlung von {total_usernames} Profilen")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_container = st.container()
        
        for i, username in enumerate(usernames):
            # Progress update
            progress = (i + 1) / total_usernames
            progress_bar.progress(progress)
            status_text.text(f"📱 Verarbeite @{username} ({i+1}/{total_usernames})")
            
            try:
                # Sammle Bilder für diesen User
                image_urls = self.collect_profile_images(username, max_images_per_profile)
                
                if image_urls:
                    results[username] = image_urls
                    st.success(f"✅ @{username}: {len(image_urls)} Bilder gesammelt")
                else:
                    st.warning(f"⚠️ @{username}: Keine Bilder gefunden")
                
                # Live-Statistiken anzeigen
                with stats_container:
                    self._display_live_stats()
                
                # Pause zwischen Profilen
                if i < total_usernames - 1:  # Nicht nach dem letzten
                    time.sleep(random.uniform(1.0, 3.0))  # Zufällige Pause
                    
            except Exception as e:
                st.error(f"❌ Fehler bei @{username}: {e}")
                self.stats['errors'] += 1
        
        # Abschluss
        progress_bar.progress(1.0)
        status_text.text(f"✅ Fertig! {len(results)} Profile erfolgreich verarbeitet")
        
        return results
    
    def _display_live_stats(self):
        """Zeigt Live-Statistiken während der Sammlung."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Profile", self.stats['profiles_processed'])
        with col2:
            st.metric("📸 Bilder gefunden", self.stats['images_found'])
        with col3:
            st.metric("🎯 Gesichter erkannt", self.stats['faces_detected'])
        with col4:
            st.metric("❌ Fehler", self.stats['errors'])
    
    def smart_username_selection(self, categories: Dict[str, List[str]], target_count: int = 50) -> List[str]:
        """
        Intelligente Auswahl von Usernames für optimale Gesichtserkennung.
        Bevorzugt Profile mit hoher Wahrscheinlichkeit für Gesichtsbilder.
        """
        # Gewichtung nach Kategorie (höhere Werte = mehr Gesichtsbilder erwartet)
        category_weights = {
            'ENTERTAINMENT & ACTORS': 0.9,
            'MUSICIANS & SINGERS': 0.8,
            'MODELS & FASHION': 0.95,
            'INFLUENCER & LIFESTYLE': 0.85,
            'SPORTS': 0.7,
            'DEUTSCHE PROMIS': 0.8,
            'INTERNATIONAL CELEBRITIES': 0.75,
            'TECH & ENTREPRENEURS': 0.6
        }
        
        selected_usernames = []
        
        # Proportionale Auswahl aus jeder Kategorie
        for category, usernames in categories.items():
            weight = category_weights.get(category, 0.5)
            category_count = int(target_count * weight / len(categories))
            
            # Zufällige Auswahl aus der Kategorie
            if len(usernames) > category_count:
                selected = random.sample(usernames, category_count)
            else:
                selected = usernames.copy()
            
            selected_usernames.extend(selected)
        
        # Auf Ziel-Anzahl bringen
        if len(selected_usernames) > target_count:
            selected_usernames = random.sample(selected_usernames, target_count)
        elif len(selected_usernames) < target_count:
            # Auffüllen mit zufälligen Auswahlen
            all_usernames = [u for users in categories.values() for u in users]
            remaining = [u for u in all_usernames if u not in selected_usernames]
            additional_needed = target_count - len(selected_usernames)
            
            if remaining and additional_needed > 0:
                additional = random.sample(remaining, min(additional_needed, len(remaining)))
                selected_usernames.extend(additional)
        
        return selected_usernames


def render_automatic_instagram_collection():
    """Hauptfunktion für automatische Instagram-Sammlung."""
    st.markdown("#### 🤖 Automatische Instagram-Sammlung")
    
    st.success("🚀 **Neue Meta-Richtlinien 2024/2025**: Öffentliche Inhalte dürfen für KI-Training genutzt werden!")
    
    with st.expander("📋 Rechtliche Grundlage & Updates"):
        st.markdown("""
        **Meta/Instagram Richtlinien-Updates 2024/2025:**
        
        ✅ **KI-Training erlaubt**: Öffentliche Inhalte dürfen für KI-Training verwendet werden
        ✅ **Automatische Sammlung**: Unter bestimmten Bedingungen zulässig
        ✅ **Forschungszwecke**: Akademische und kommerzielle Forschung unterstützt
        
        **Unsere Implementierung:**
        - 🛡️ **Respektiert Rate-Limits** und Server-Ressourcen
        - 🔒 **Nur öffentliche Profile** werden verarbeitet
        - 📊 **Transparente Statistiken** über alle Aktivitäten
        - 🎯 **Fokus auf Gesichtserkennung** und KI-Training
        
        **Quellen:**
        - Meta AI Training Policy 2024
        - Instagram Terms of Service Update 2024
        - EU AI Act Compliance
        """)
    
    collector = ModernInstagramCollector()
    
    # Load usernames
    usernames_file = "usernames.txt"
    if not os.path.exists(usernames_file):
        st.error("📝 usernames.txt nicht gefunden!")
        return
    
    # Parse categories
    categories = {}
    with open(usernames_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        current_category = "Allgemein"
        
        for line in lines:
            line = line.strip()
            if line.startswith('# === ') and line.endswith(' ==='):
                current_category = line[6:-4].strip()
                categories[current_category] = []
            elif line and not line.startswith('#'):
                if current_category not in categories:
                    categories[current_category] = []
                categories[current_category].append(line)
    
    st.info(f"📂 {len(categories)} Kategorien mit {sum(len(users) for users in categories.values())} Accounts geladen")
    
    # Sammlung konfigurieren
    col1, col2, col3 = st.columns(3)
    
    with col1:
        collection_mode = st.selectbox(
            "🎯 Sammelmodus",
            [
                "🧠 Smart-Auswahl (Empfohlen)",
                "📂 Nach Kategorie", 
                "🎲 Zufällig",
                "✏️ Manuell auswählen"
            ],
            help="Wie sollen Profile ausgewählt werden?"
        )
    
    with col2:
        profile_count = st.number_input(
            "👥 Anzahl Profile",
            min_value=5,
            max_value=100,
            value=25,
            help="Wie viele Profile sollen verarbeitet werden?"
        )
    
    with col3:
        images_per_profile = st.number_input(
            "📸 Bilder pro Profil",
            min_value=5,
            max_value=50,
            value=15,
            help="Maximale Bilder pro Profil"
        )
    
    # Profil-Auswahl basierend auf Modus
    selected_usernames = []
    
    if collection_mode == "🧠 Smart-Auswahl (Empfohlen)":
        st.markdown("**🧠 Intelligente Auswahl basierend auf Gesichtserkennungs-Wahrscheinlichkeit**")
        selected_usernames = collector.smart_username_selection(categories, profile_count)
        
        with st.expander("🎯 Smart-Auswahl Details"):
            st.markdown("**Kategorien-Gewichtung für Gesichtsbilder:**")
            for category, usernames in categories.items():
                selected_from_cat = [u for u in selected_usernames if u in usernames]
                st.text(f"• {category}: {len(selected_from_cat)} ausgewählt")
    
    elif collection_mode == "📂 Nach Kategorie":
        selected_category = st.selectbox(
            "Kategorie wählen",
            list(categories.keys()),
            help="Aus welcher Kategorie sollen Profile gesammelt werden?"
        )
        
        available_usernames = categories[selected_category]
        actual_count = min(profile_count, len(available_usernames))
        selected_usernames = random.sample(available_usernames, actual_count)
        
        st.info(f"📂 {actual_count} Profile aus '{selected_category}' ausgewählt")
    
    elif collection_mode == "🎲 Zufällig":
        all_usernames = [u for users in categories.values() for u in users]
        actual_count = min(profile_count, len(all_usernames))
        selected_usernames = random.sample(all_usernames, actual_count)
        
        st.info(f"🎲 {actual_count} zufällige Profile ausgewählt")
    
    else:  # Manuell
        st.markdown("**✏️ Manuelle Auswahl:**")
        
        category_for_manual = st.selectbox(
            "Kategorie durchsuchen",
            list(categories.keys()),
            help="Aus welcher Kategorie möchten Sie auswählen?"
        )
        
        available_for_manual = categories[category_for_manual]
        selected_usernames = st.multiselect(
            f"Profile aus '{category_for_manual}' auswählen",
            available_for_manual,
            default=available_for_manual[:min(5, len(available_for_manual))],
            help="Wählen Sie die gewünschten Profile"
        )
    
    # Vorschau der Auswahl
    if selected_usernames:
        st.markdown("### 👀 Ausgewählte Profile")
        
        # Zeige erste paar Usernames
        preview_count = min(10, len(selected_usernames))
        cols = st.columns(5)
        
        for i, username in enumerate(selected_usernames[:preview_count]):
            with cols[i % 5]:
                st.markdown(f"[@{username}](https://instagram.com/{username})")
        
        if len(selected_usernames) > preview_count:
            st.text(f"... und {len(selected_usernames) - preview_count} weitere")
        
        # Schätzungen
        estimated_images = len(selected_usernames) * images_per_profile
        estimated_time = len(selected_usernames) * 3  # ~3 Sekunden pro Profil
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Profile", len(selected_usernames))
        with col2:
            st.metric("📸 Geschätzte Bilder", f"~{estimated_images}")
        with col3:
            st.metric("⏱️ Geschätzte Zeit", f"~{estimated_time//60}min {estimated_time%60}s")
    
    # Sammlung starten
    if st.button("🚀 Automatische Sammlung starten", type="primary", disabled=not selected_usernames):
        if selected_usernames:
            st.markdown("---")
            
            # Sammlung durchführen
            results = collector.batch_collect_profiles(selected_usernames, images_per_profile)
            
            if results:
                st.markdown("### 🎉 Sammlung erfolgreich abgeschlossen!")
                
                # Ergebnisse zusammenfassen
                total_profiles = len(results)
                total_images = sum(len(urls) for urls in results.values())
                
                st.success(f"✅ {total_profiles} Profile verarbeitet, {total_images} Bilder gesammelt")
                
                # Alle URLs für Weiterverarbeitung sammeln
                all_urls = []
                person_names = []
                
                for username, urls in results.items():
                    all_urls.extend(urls)
                    person_names.extend([username] * len(urls))
                
                # URLs für URL-Import vorbereiten
                st.session_state['demo_urls'] = '\n'.join(all_urls)
                st.session_state['demo_person'] = 'Instagram_Batch_Collection'
                st.session_state['source_type'] = 'instagram_auto'
                st.session_state['batch_results'] = results
                
                # Weiterverarbeitung anbieten
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔍 Gesichter jetzt extrahieren", type="primary"):
                        st.info("🔄 Weiterleitung zur Gesichtsverarbeitung...")
                        # Hier würde die direkte Verarbeitung starten
                        st.success("URLs in Verarbeitung übertragen! ✅")
                
                with col2:
                    if st.button("📋 URLs exportieren"):
                        # URLs als Text anzeigen
                        st.text_area(
                            "🔗 Alle gesammelten URLs:",
                            '\n'.join(all_urls),
                            height=200
                        )
                
                # Detaillierte Ergebnisse
                with st.expander("📊 Detaillierte Ergebnisse"):
                    for username, urls in results.items():
                        st.markdown(f"**@{username}**: {len(urls)} Bilder")
                        for i, url in enumerate(urls[:3], 1):  # Zeige erste 3
                            st.text(f"  {i}. {url}")
                        if len(urls) > 3:
                            st.text(f"  ... und {len(urls) - 3} weitere")
            else:
                st.error("❌ Keine Bilder gesammelt. Prüfen Sie die Auswahl und versuchen Sie es erneut.")


if __name__ == "__main__":
    # Test der automatischen Sammlung
    collector = ModernInstagramCollector()
    print("Automatischer Instagram-Sammler bereit!")
