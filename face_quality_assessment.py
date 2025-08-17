"""
Gesichtsqualitäts-Bewertung und Filterung für Online-Bilder.
"""
import numpy as np
import face_recognition
from typing import Tuple, List, Dict
from PIL import Image
import cv2


def assess_face_quality(face_crop: np.ndarray, face_encoding: np.ndarray) -> Dict:
    """
    Bewerte die Qualität eines Gesichtsausschnitts.
    
    Returns:
        Dict mit Qualitätsbewertungen
    """
    try:
        height, width = face_crop.shape[:2]
        
        # Grundlegende Metriken
        size_score = min(width, height) / 150.0  # Normalisiert auf 150px als "gut"
        size_score = min(size_score, 1.0)
        
        # Schärfe-Bewertung (Laplacian-Varianz)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalisiert
        
        # Helligkeit und Kontrast
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0  # Optimal bei 128
        
        std_dev = np.std(gray)
        contrast_score = min(std_dev / 50.0, 1.0)  # Normalisiert
        
        # Gesichtssymmetrie (einfache Näherung)
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        
        # Größen angleichen
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        symmetry_score = max(0, 1.0 - symmetry_diff / 100.0)
        
        # Encoding-Qualität (Distanz zum Mittelwert)
        # Ein "typischeres" Gesicht hat oft bessere Qualität
        avg_distance = np.linalg.norm(face_encoding)
        encoding_score = max(0, 1.0 - (avg_distance - 1.0) / 2.0)
        
        # Gesamtscore berechnen
        weights = {
            'size': 0.25,
            'sharpness': 0.25,
            'brightness': 0.15,
            'contrast': 0.15,
            'symmetry': 0.10,
            'encoding': 0.10
        }
        
        overall_score = (
            weights['size'] * size_score +
            weights['sharpness'] * sharpness_score +
            weights['brightness'] * brightness_score +
            weights['contrast'] * contrast_score +
            weights['symmetry'] * symmetry_score +
            weights['encoding'] * encoding_score
        )
        
        return {
            'overall_score': overall_score,
            'size_score': size_score,
            'sharpness_score': sharpness_score,
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'symmetry_score': symmetry_score,
            'encoding_score': encoding_score,
            'metrics': {
                'width': width,
                'height': height,
                'sharpness': laplacian_var,
                'brightness': mean_brightness,
                'contrast': std_dev,
                'symmetry_diff': symmetry_diff,
                'encoding_norm': avg_distance
            }
        }
        
    except Exception as e:
        # Fallback bei Fehlern
        return {
            'overall_score': 0.5,
            'size_score': 0.5,
            'sharpness_score': 0.5,
            'brightness_score': 0.5,
            'contrast_score': 0.5,
            'symmetry_score': 0.5,
            'encoding_score': 0.5,
            'metrics': {'error': str(e)}
        }


def filter_best_faces(faces_data: List[Dict], max_faces: int = 5) -> List[Dict]:
    """
    Filtere die besten Gesichter basierend auf Qualitätsbewertung.
    
    Args:
        faces_data: Liste von Gesichtsdaten mit quality_assessment
        max_faces: Maximale Anzahl zurückzugebender Gesichter
        
    Returns:
        Gefilterte und sortierte Liste der besten Gesichter
    """
    if not faces_data:
        return []
    
    # Nach Gesamtscore sortieren
    sorted_faces = sorted(
        faces_data, 
        key=lambda x: x.get('quality_assessment', {}).get('overall_score', 0),
        reverse=True
    )
    
    return sorted_faces[:max_faces]


def get_quality_description(score: float) -> Tuple[str, str]:
    """
    Gebe eine menschenlesbare Beschreibung der Qualität zurück.
    
    Returns:
        (description, emoji)
    """
    if score >= 0.8:
        return "Ausgezeichnet", "🌟"
    elif score >= 0.7:
        return "Sehr gut", "🟢"
    elif score >= 0.6:
        return "Gut", "🟡"
    elif score >= 0.5:
        return "Akzeptabel", "🟠"
    else:
        return "Schlecht", "🔴"


def render_quality_analysis(faces_data: List[Dict], show_details: bool = False):
    """Rendere Qualitätsanalyse in Streamlit."""
    import streamlit as st
    
    if not faces_data:
        return
    
    st.markdown("### 🎯 Gesichtsqualitäts-Analyse")
    
    # Übersicht
    scores = [f.get('quality_assessment', {}).get('overall_score', 0) for f in faces_data]
    avg_score = np.mean(scores) if scores else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Durchschnittsscore", f"{avg_score:.2f}")
    with col2:
        best_score = max(scores) if scores else 0
        st.metric("🏆 Bester Score", f"{best_score:.2f}")
    with col3:
        desc, emoji = get_quality_description(avg_score)
        st.metric("💎 Qualität", f"{emoji} {desc}")
    
    if show_details:
        # Detaillierte Analyse
        st.markdown("#### 📋 Detaillierte Bewertungen")
        
        for i, face_data in enumerate(faces_data[:5]):  # Zeige Top 5
            quality = face_data.get('quality_assessment', {})
            overall = quality.get('overall_score', 0)
            desc, emoji = get_quality_description(overall)
            
            with st.expander(f"{emoji} Gesicht {i+1} - {desc} ({overall:.2f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Einzelbewertungen:**")
                    for metric in ['size', 'sharpness', 'brightness', 'contrast', 'symmetry', 'encoding']:
                        score = quality.get(f'{metric}_score', 0)
                        st.text(f"{metric.title()}: {score:.2f}")
                
                with col2:
                    metrics = quality.get('metrics', {})
                    if metrics and 'error' not in metrics:
                        st.markdown("**Technische Daten:**")
                        st.text(f"Größe: {metrics.get('width', 0)}x{metrics.get('height', 0)}")
                        st.text(f"Schärfe: {metrics.get('sharpness', 0):.1f}")
                        st.text(f"Helligkeit: {metrics.get('brightness', 0):.1f}")
                        st.text(f"Kontrast: {metrics.get('contrast', 0):.1f}")


if __name__ == "__main__":
    # Test der Qualitätsbewertung
    print("Gesichtsqualitäts-Bewertung bereit!")
