"""
UI-Komponenten und Seitenleisten-Funktionen.
"""
import datetime
import streamlit as st


def render_sidebar():
    """Rendere die Seitenleiste mit täglichem Tipp."""
    with st.sidebar:
        st.markdown("---")
        
        # Tipps-Liste  
        tips = [
            "🎯 Verwenden Sie mehrere Bilder pro Person für bessere Erkennung - 3-10 Bilder sind optimal!",
            "⚙️ CNN-Modus ist genauer, aber langsamer als HOG - nutzen Sie ihn bei wichtigen Suchen.",
            "💡 Gute Beleuchtung verbessert die Erkennungsqualität erheblich - vermeiden Sie starke Schatten.",
            "🧹 Regelmäßige Datenbank-Bereinigung hält das System optimal - prüfen Sie die Statistiken-Seite.",
            "👤 Frontale Gesichter werden am besten erkannt - 45° Winkel funktionieren aber auch gut.",
            "📷 Hochauflösende Bilder (min. 800px) liefern bessere Ergebnisse als kleine Fotos.",
            "🔍 Bei schlechten Suchergebnissen: Versuchen Sie verschiedene Blickwinkel desselben Gesichts.",
            "🎨 Verschiedene Lichtverhältnisse beim Training verbessern die Robustheit der Erkennung.",
            "📊 Nutzen Sie die Statistiken-Seite, um die Qualität Ihrer Datenbank zu überwachen.",
            "⚡ Live-Suche zeigt Ergebnisse sofort an - Sie müssen nicht auf das Ende warten!",
            "🏷️ Aussagekräftige Labels helfen bei der Organisation Ihrer Personen-Datenbank.",
            "🔄 Bei Änderungen an vielen Gesichtern: Embeddings neu berechnen für optimale Leistung.",
            "📁 HEIC/HEIF-Dateien werden unterstützt - installieren Sie pillow-heif für beste Kompatibilität.",
            "🎭 Verschiedene Gesichtsausdrücke beim Training machen die Erkennung flexibler.",
            "🔒 Exportieren Sie regelmäßig Ihre Datenbank als Backup über die Statistiken-Seite.",
            "🌟 Distanz-Werte unter 0.4 sind excellent, unter 0.6 sind noch sehr brauchbar.",
            "📐 Quadratische Gesichtsausschnitte funktionieren oft besser als rechteckige.",
            "🎪 Vermeiden Sie verdeckte Gesichter - Brillen und leichte Bärte sind meist kein Problem.",
            "🔢 Die Face-ID (8-stelliger Code) hilft bei der eindeutigen Identifikation von Gesichtern.",
            "💾 Große Datenbanken: Nutzen Sie den Export, um Datenverlust zu vermeiden.",
            "🌅 Tageslicht-Aufnahmen funktionieren meist besser als Kunstlicht-Bilder.",
            "👥 Bei Gruppenfotos: Wählen Sie das gewünschte Gesicht sorgfältig aus.",
            "🔧 Probleme mit der Erkennung? Überprüfen Sie die Datenbank-Qualität in den Statistiken.",
            "🎯 Training mehrerer Winkel (frontal, seitlich, halbseitig) macht die KI flexibler.",
            "📱 Handy-Selfies funktionieren genauso gut wie professionelle Fotos - Hauptsache scharf!"
        ]
        
        # Berechne täglichen Index basierend auf Datum
        today = datetime.date.today()
        day_of_year = today.timetuple().tm_yday
        tip_index = day_of_year % len(tips)
        
        # Zeige nur den täglichen Tipp
        st.markdown("### 💡 Tipp des Tages")
        st.info(tips[tip_index])
        
        st.markdown("---")


def render_footer():
    """Rendere den minimalen Footer."""
    st.markdown("---")
    st.markdown("🔍 **Gesichts-Bildsuche mit KI** - Moderne Gesichtserkennung mit Deep Learning")