#!/usr/bin/env python3
"""
Image Crawler

Vereint:
- Playwright-basiertes Crawlen (JS-rendering, lazy-load, network capture)  
- Recursion auf interne Links (bis max_depth)
- Bruteforce für typische Upload-Verzeichnisse (mit Wortliste)
- Speichert alle gefundenen Bild-URLs in 'bilder_alles.txt'

NUTZUNG: Nur auf Seiten einsetzen, für die du die Erlaubnis hast!
"""

import asyncio
import re
import time
import os
import hashlib
from urllib.parse import urljoin, urlparse, urldefrag
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np

from playwright.async_api import async_playwright

# ------------- KONFIG -------------
BASE_URL = "https://koenig-karlmann-gymnasium.de/"   # <-- anpassen
MAX_DEPTH = 4
CONCURRENT_PAGES = 3
NAV_TIMEOUT = 20000  # ms
OUTPUT_FILE = "bilder_alles.txt"

# Download-Konfiguration
DOWNLOAD_DIR = "downloaded_images"
MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50MB max pro Bild
CONCURRENT_DOWNLOADS = 5

# Globale Listen
found_images = set()
visited = set()
errors = []

# Wortliste für Upload-Verzeichnisse
UPLOAD_KEYWORDS = [
    "uploads", "upload", "media", "files", "images", "img", "pics", "pictures", 
    "photos", "gallery", "assets", "content", "wp-content", "resources", "static",
    "data", "storage", "documents", "docs", "downloads", "public", "shared",
    "user", "users", "member", "members", "profile", "profiles", "avatar", "avatars",
    "news", "articles", "blog", "post", "posts", "thumb", "thumbnails", "cache",
    "temp", "tmp", "archive", "backup", "old", "new", "current", "latest"
]

def is_internal(url):
    """Prüft, ob URL zur gleichen Domain gehört"""
    return urlparse(url).netloc in ['', urlparse(BASE_URL).netloc]

def normalize(url):
    """Normalisiert URL (entfernt Fragment)"""
    return urldefrag(url)[0]

def is_allowed_image(url):
    """Prüft, ob URL ein erlaubtes Bildformat hat"""
    return url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'))

async def crawl_playwright():
    """Hauptcrawl-Funktion mit Playwright"""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # Semaphore für gleichzeitige Seiten
        sem = asyncio.Semaphore(CONCURRENT_PAGES)
        
        # Startseite zur queue hinzufügen
        queue = [(BASE_URL, 0)]
        workers = [worker(browser, sem) for _ in range(CONCURRENT_PAGES)]
        
        await asyncio.gather(*workers)
        await browser.close()

async def worker(browser, sem):
    """Worker für parallele Verarbeitung"""
    global queue
    
    while True:
        try:
            url, depth = queue.pop(0)
        except IndexError:
            break
        
        if url in visited or depth > MAX_DEPTH:
            continue
            
        await process_page(browser, url, depth)

async def process_page(browser, url, depth):
    """Verarbeitet eine einzelne Seite"""
    global found_images, visited, errors, queue
    
    if url in visited:
        return
    
    print(f"[{depth}] {url}")
    visited.add(url)
    
    def on_response(response):
        """Interceptor für alle Netzwerk-Requests"""
        resp_url = response.url
        if is_allowed_image(resp_url) and is_internal(resp_url):
            found_images.add(resp_url)
            print(f"    🖼️  {resp_url}")
    
    try:
        page = await browser.new_page()
        page.on("response", on_response)
        
        # Seite laden
        await page.goto(url, timeout=NAV_TIMEOUT, wait_until='networkidle')
        
        # Auto-scroll für Lazy Loading
        await auto_scroll(page)
        
        # HTML parsen
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Alle img-Tags sammeln
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src:
                full_url = urljoin(url, src)
                if is_allowed_image(full_url) and is_internal(full_url):
                    found_images.add(full_url)
                    print(f"    🖼️  {full_url}")
        
        # Links für weitere Crawling sammeln (nur wenn nicht zu tief)
        if depth < MAX_DEPTH:
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = normalize(urljoin(url, href))
                
                if is_internal(full_url) and full_url not in visited:
                    queue.append((full_url, depth + 1))
        
        await page.close()
        
    except Exception as e:
        errors.append(f"{url}: {e}")
        print(f"    ❌ Fehler: {e}")

async def auto_scroll(page, pause=0.2, max_scrolls=30):
    """Automatisches Scrollen für Lazy Loading"""
    for i in range(max_scrolls):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(pause)

def brute_force_directories():
    """Brute-Force für typische Upload-Verzeichnisse"""
    global found_images
    
    print("[*] Starte Bruteforce-Scan...")
    base_parsed = urlparse(BASE_URL)
    base_clean = f"{base_parsed.scheme}://{base_parsed.netloc}"
    
    for keyword in UPLOAD_KEYWORDS:
        test_urls = [
            f"{base_clean}/{keyword}/",
            f"{base_clean}/wp-content/{keyword}/",
            f"{base_clean}/assets/{keyword}/",
            f"{base_clean}/static/{keyword}/",
            f"{base_clean}/media/{keyword}/",
            f"{base_clean}/content/{keyword}/",
            f"{base_clean}/files/{keyword}/",
            f"{base_clean}/public/{keyword}/",
            f"{base_clean}/resources/{keyword}/",
            f"{base_clean}/data/{keyword}/",
        ]
        
        for test_url in test_urls:
            try:
                response = requests.get(test_url, timeout=10, allow_redirects=False)
                if response.status_code == 200:
                    print(f"    ✅ Gefunden: {test_url}")
                    
                    # Parse Directory Listing
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if is_allowed_image(href):
                            full_url = urljoin(test_url, href)
                            found_images.add(full_url)
                            print(f"        🖼️  {full_url}")
                            
            except Exception:
                pass  # Stille Fehler für Bruteforce

def save_results():
    """Speichert alle gefundenen URLs"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for url in sorted(found_images):
            f.write(f"{url}\n")
    
    print(f"\n[✓] {len(found_images)} URLs in '{OUTPUT_FILE}' gespeichert")

def get_filename_from_url(url):
    """Extrahiert Dateinamen aus URL"""
    path = urlparse(url).path
    filename = os.path.basename(path)
    
    if not filename or '.' not in filename:
        # Fallback: Hash der URL verwenden
        hash_obj = hashlib.md5(url.encode())
        filename = f"image_{hash_obj.hexdigest()[:8]}.jpg"
    
    return filename

async def download_image(session, url, download_dir):
    """Lädt ein einzelnes Bild herunter"""
    try:
        filename = get_filename_from_url(url)
        filepath = Path(download_dir) / filename
        
        # Skip wenn bereits existiert
        if filepath.exists():
            return filepath
        
        async with session.get(url) as response:
            if response.status == 200:
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
                    print(f"[!] Überspringe {url} (zu groß: {content_length} bytes)")
                    return None
                
                content = await response.read()
                
                with open(filepath, 'wb') as f:
                    f.write(content)
                
                print(f"[✓] {filename}")
                return filepath
            else:
                print(f"[!] HTTP {response.status}: {url}")
                return None
                
    except Exception as e:
        print(f"[!] Download-Fehler {url}: {e}")
        return None

async def download_and_filter_images():
    """Lädt alle Bilder herunter"""
    
    # Erstelle Verzeichnisse
    Path(DOWNLOAD_DIR).mkdir(exist_ok=True)
    
    # Lade URLs aus Datei
    if not Path(OUTPUT_FILE).exists():
        print(f"[!] {OUTPUT_FILE} nicht gefunden!")
        return
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print("[!] Keine URLs in der Datei gefunden!")
        return
    
    print(f"[*] Starte Download von {len(urls)} Bildern...")
    
    # Asynchroner Download mit Semaphore
    import aiohttp
    
    connector = aiohttp.TCPConnector(limit=CONCURRENT_DOWNLOADS)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        
        sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
        
        async def download_with_semaphore(url):
            async with sem:
                return await download_image(session, url, DOWNLOAD_DIR)
        
        download_tasks = [download_with_semaphore(url) for url in urls]
        downloaded_files = await asyncio.gather(*download_tasks, return_exceptions=True)
    
    # Filtere nach erfolgreichen Downloads
    successful_downloads = [f for f in downloaded_files if isinstance(f, Path) and f and f.exists()]
    print(f"[*] {len(successful_downloads)} Bilder erfolgreich heruntergeladen")
    
    if successful_downloads:
        print("\n" + "="*50)
        print("DOWNLOAD ABGESCHLOSSEN")
        print("="*50)
        print(f"Heruntergeladene Bilder: {len(successful_downloads)}")
        print(f"Verzeichnis: {DOWNLOAD_DIR}")
        print("="*50)
    else:
        print("[!] Keine Bilder heruntergeladen.")
    
    # Leeres Download-Verzeichnis aufräumen (falls leer)
    try:
        if Path(DOWNLOAD_DIR).exists() and not any(Path(DOWNLOAD_DIR).iterdir()):
            Path(DOWNLOAD_DIR).rmdir()
            print(f"[*] Leeres Verzeichnis '{DOWNLOAD_DIR}' aufgeräumt")
    except:
        pass

async def main():
    start = time.time()
    
    # Prüfe ob bereits URLs vorhanden sind
    if Path(OUTPUT_FILE).exists():
        print(f"[*] {OUTPUT_FILE} gefunden.")
        choice = input("Möchten Sie die URLs herunterladen? (j/n): ").lower().strip()
        if choice in ['j', 'ja', 'y', 'yes', '']:
            print("[*] Starte Download...")
            await download_and_filter_images()
            print(f"Fertig in {time.time()-start:.1f}s")
            return
    
    print("[*] Playwright-Crawl startet...")
    await crawl_playwright()
    print("[*] Playwright-Crawl fertig.")
    print("[*] Bruteforce-Scan startet...")
    brute_force_directories()
    print("[*] Bruteforce fertig.")
    save_results()
    
    # Automatisch Download starten wenn URLs gefunden
    if found_images:
        print(f"\n[*] {len(found_images)} Bild-URLs gefunden.")
        choice = input("Möchten Sie die Bilder jetzt herunterladen? (j/n): ").lower().strip()
        if choice in ['j', 'ja', 'y', 'yes', '']:
            await download_and_filter_images()
    
    print(f"Fertig in {time.time()-start:.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
