"""
Hauptanwendung für die Gesichtserkennungs-App.
Modular aufgeteilte Streamlit-Anwendung mit Vector Database Support.
"""
import streamlit as st
from config import HEIF_SUPPORT
from ui_components import render_sidebar, render_footer
from training import render_training_page
from search_ui import render_search_page
from stats_page import render_statistics_page
from ai_analysis_page import render_ai_analysis_page

# Vector Database Import (optional)
try:
    from vector_db_page import show_vector_db_page
    VECTOR_DB_PAGE_AVAILABLE = True
except ImportError:
    VECTOR_DB_PAGE_AVAILABLE = False

# Simple Face Search Import (alternative to complex vector DB)
try:
    from simple_face_search import render_simple_search_page
    SIMPLE_SEARCH_AVAILABLE = True
except ImportError:
    SIMPLE_SEARCH_AVAILABLE = False


def main():
    """Hauptfunktion der Anwendung."""
    # Streamlit App Setup
    st.set_page_config(
        page_title="Gesichts-Bildsuche", 
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.title("🔍 Gesichts-Bildsuche mit KI")
    st.markdown("---")

    # Sidebar Menu - mit alternativen Such-Optionen
    menu_options = ["🔍 Suche", "🎯 Training", "🧠 KI-Analyse", "📊 Statistiken"]
    
    # Füge einfache Suche hinzu (robuste Alternative)
    if SIMPLE_SEARCH_AVAILABLE:
        menu_options.append("🎯 Einfache Suche")
    
    # Füge Vector DB hinzu (komplexe Option)
    if VECTOR_DB_PAGE_AVAILABLE:
        menu_options.append("🚀 Vector DB")
    
    menu = st.sidebar.radio(
        "📋 Aktion wählen", 
        menu_options,
        help="Wählen Sie zwischen verschiedenen Such-Methoden und Verwaltungsoptionen"
    )

    # Seiten-Routing
    if menu == "🎯 Training":
        render_training_page()
    elif menu == "🔍 Suche":
        render_search_page()
    elif menu == "🧠 KI-Analyse":
        render_ai_analysis_page()
    elif menu == "📊 Statistiken":
        render_statistics_page()
    elif menu == "🎯 Einfache Suche" and SIMPLE_SEARCH_AVAILABLE:
        render_simple_search_page()
    elif menu == "🚀 Vector DB" and VECTOR_DB_PAGE_AVAILABLE:
        show_vector_db_page()

    # Sidebar und Footer rendern
    render_sidebar()
    render_footer()


if __name__ == "__main__":
    main()
