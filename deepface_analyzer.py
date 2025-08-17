"""
Optimierte DeepFace Gesichtsanalyse mit verbesserter Genauigkeit und Robustheit.
"""
import os
import io
import numpy as np
import tempfile
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple

# Versuche DeepFace zu importieren
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    DeepFace = None

from config import HEIF_SUPPORT


def analyze_facial_attributes(
    image_input: Union[str, np.ndarray, bytes], 
    actions: List[str] = ['age', 'gender', 'race', 'emotion'],
    enforce_detection: bool = False,
    detector_backend: str = 'opencv'
) -> Optional[List[Dict]]:
    """
    Analysiere Gesichtsattribute mit verbesserter DeepFace-Integration.
    
    Args:
        image_input: Bildpfad, numpy array oder bytes
        actions: Liste der zu analysierenden Attribute
        enforce_detection: Ob Gesichtserkennung erzwungen werden soll
        detector_backend: Gesichtserkennungs-Backend
        
    Returns:
        Liste von Analyseergebnissen für jedes erkannte Gesicht
    """
    if not DEEPFACE_AVAILABLE or DeepFace is None:
        return None
    
    try:
        # Bereite Eingabe vor
        prepared_input = _prepare_image_input(image_input)
        if prepared_input is None:
            return None
        
        # Verbesserte Analyse mit Fallback-Strategien
        results = _analyze_with_fallback(prepared_input, actions, enforce_detection, detector_backend)
        
        if results:
            # Validiere und verbessere Ergebnisse
            validated_results = _validate_and_improve_results(results)
            return _format_analysis_results(validated_results)
            
    except Exception as e:
        # Stille Behandlung von Fehlern - keine UI-Störungen
        return None
    
    return None


def _prepare_image_input(image_input: Union[str, np.ndarray, bytes]) -> Optional[str]:
    """
    Bereite verschiedene Eingabetypen für DeepFace vor.
    """
    try:
        # Bereits ein Pfad
        if isinstance(image_input, str) and os.path.exists(image_input):
            return image_input
        
        # Numpy Array
        if isinstance(image_input, np.ndarray):
            return _save_temp_image(image_input)
            
        # Bytes
        if isinstance(image_input, bytes):
            # Konvertiere zu PIL Image und dann zu numpy
            pil_image = Image.open(io.BytesIO(image_input))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            np_image = np.array(pil_image)
            return _save_temp_image(np_image)
            
    except Exception:
        pass
    
    return None


def _save_temp_image(image_array: np.ndarray) -> Optional[str]:
    """
    Speichere numpy Array als temporäres Bild.
    """
    try:
        pil_image = Image.fromarray(image_array.astype('uint8'))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        pil_image.save(temp_file.name, 'JPEG')
        return temp_file.name
    except Exception:
        return None


def _analyze_with_fallback(image_path: str, actions: List[str], enforce_detection: bool, detector_backend: str) -> Optional[List[Dict]]:
    """
    Analysiere mit mehreren Fallback-Strategien für bessere Robustheit.
    """
    if not DEEPFACE_AVAILABLE or DeepFace is None:
        return None
        
    # Strategie 1: Standard-Analyse
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=actions,
            enforce_detection=enforce_detection,
            detector_backend=detector_backend,
            silent=True
        )
        if isinstance(results, dict):
            results = [results]
        return results
    except:
        pass
    
    # Strategie 2: Ohne enforce_detection mit opencv
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=actions,
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        if isinstance(results, dict):
            results = [results]
        return results
    except:
        pass
    
    # Strategie 3: Nur wichtigste Attribute
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        if isinstance(results, dict):
            results = [results]
        return results
    except:
        pass
    
    return None


def _validate_and_improve_results(results: List[Dict]) -> List[Dict]:
    """
    Validiere und verbessere DeepFace-Ergebnisse für bessere Genauigkeit.
    """
    improved_results = []
    
    for result in results:
        improved_result = result.copy()
        
        # Age Validation und Verbesserung
        if 'age' in result:
            age = result['age']
            # Plausibilitätsprüfung für Alter (5-95 Jahre)
            if age < 5:
                improved_result['age'] = max(5, age)
                improved_result['age_confidence'] = 'low'
            elif age > 95:
                improved_result['age'] = min(95, age)
                improved_result['age_confidence'] = 'low'
            else:
                improved_result['age_confidence'] = 'medium' if 10 <= age <= 85 else 'low'
            
            # Altersgruppen für bessere Interpretation
            improved_result['age_group'] = _get_age_group(improved_result['age'])
        
        # Emotion Validation
        if 'emotion' in result:
            emotion_scores = result['emotion']
            if isinstance(emotion_scores, dict):
                # Prüfe auf unrealistische Emotionserkennung
                max_score = max(emotion_scores.values())
                if max_score < 40:
                    improved_result['emotion_confidence'] = 'very_low'
                elif max_score < 60:
                    improved_result['emotion_confidence'] = 'low'
                elif max_score < 80:
                    improved_result['emotion_confidence'] = 'medium'
                else:
                    improved_result['emotion_confidence'] = 'high'
                
                # Kombiniere ähnliche Emotionen für bessere Interpretation
                improved_result['emotion_simplified'] = _simplify_emotion(emotion_scores)
        
        # Gender Validation
        if 'gender' in result:
            gender_scores = result['gender']
            if isinstance(gender_scores, dict):
                max_score = max(gender_scores.values())
                if max_score < 60:
                    improved_result['gender_confidence'] = 'low'
                elif max_score < 80:
                    improved_result['gender_confidence'] = 'medium'
                else:
                    improved_result['gender_confidence'] = 'high'
        
        # Race Validation (falls vorhanden)
        if 'race' in result:
            race_scores = result['race']
            if isinstance(race_scores, dict):
                max_score = max(race_scores.values())
                if max_score < 50:
                    improved_result['race_confidence'] = 'very_low'
                elif max_score < 70:
                    improved_result['race_confidence'] = 'low'
                else:
                    improved_result['race_confidence'] = 'medium'
        
        improved_results.append(improved_result)
    
    return improved_results


def _get_age_group(age: int) -> str:
    """Bestimme Altersgruppe für bessere Kategorisierung."""
    if age < 13:
        return 'Kind'
    elif age < 20:
        return 'Jugendlich'
    elif age < 30:
        return 'Jung Erwachsen'
    elif age < 50:
        return 'Erwachsen'
    elif age < 65:
        return 'Mittleres Alter'
    else:
        return 'Senior'


def _simplify_emotion(emotion_scores: Dict[str, float]) -> str:
    """
    Vereinfache Emotionserkennung durch Gruppierung ähnlicher Emotionen.
    """
    # Gruppiere positive vs negative Emotionen
    positive_emotions = ['happy']
    negative_emotions = ['sad', 'angry', 'fear', 'disgust']
    neutral_emotions = ['neutral', 'surprise']
    
    # Berechne Gruppenwerte
    positive_score = sum(emotion_scores.get(emotion, 0) for emotion in positive_emotions)
    negative_score = sum(emotion_scores.get(emotion, 0) for emotion in negative_emotions)
    neutral_score = sum(emotion_scores.get(emotion, 0) for emotion in neutral_emotions)
    
    # Bestimme dominante Gruppe
    if positive_score >= negative_score and positive_score >= neutral_score:
        return 'Positiv'
    elif negative_score >= neutral_score:
        return 'Negativ'
    else:
        return 'Neutral'


def _format_analysis_results(results: List[Dict]) -> List[Dict]:
    """
    Formatiere Analyseergebnisse für konsistente Ausgabe.
    """
    formatted_results = []
    
    for result in results:
        formatted_result = {}
        
        # Age
        if 'age' in result:
            formatted_result['age'] = result['age']
            formatted_result['age_group'] = result.get('age_group', 'Unbekannt')
            formatted_result['age_confidence'] = result.get('age_confidence', 'medium')
        
        # Gender
        if 'gender' in result:
            if isinstance(result['gender'], dict):
                dominant_gender = max(result['gender'].items(), key=lambda x: x[1])[0]
                confidence = result['gender'][dominant_gender]
                formatted_result['gender'] = {
                    'dominant': dominant_gender,
                    'confidence': confidence
                }
                formatted_result['gender_confidence'] = result.get('gender_confidence', 'medium')
        
        # Emotion
        if 'emotion' in result:
            if isinstance(result['emotion'], dict):
                dominant_emotion = max(result['emotion'].items(), key=lambda x: x[1])[0]
                confidence = result['emotion'][dominant_emotion]
                formatted_result['emotion'] = {
                    'dominant': dominant_emotion,
                    'confidence': confidence,
                    'all_scores': result['emotion']
                }
                formatted_result['emotion_simplified'] = result.get('emotion_simplified', 'Neutral')
                formatted_result['emotion_confidence'] = result.get('emotion_confidence', 'medium')
        
        # Race (falls vorhanden)
        if 'race' in result:
            if isinstance(result['race'], dict):
                dominant_race = max(result['race'].items(), key=lambda x: x[1])[0]
                confidence = result['race'][dominant_race]
                formatted_result['race'] = {
                    'dominant': dominant_race,
                    'confidence': confidence
                }
                formatted_result['race_confidence'] = result.get('race_confidence', 'low')
        
        formatted_results.append(formatted_result)
    
    return formatted_results


def analyze_uploaded_image_with_attributes(
    file_bytes: bytes, 
    filename: str, 
    use_improved_analysis: bool = False,
    show_optimization_messages: bool = False
) -> Tuple[Optional[np.ndarray], Optional[List[Dict]], str]:
    """
    Kombinierte Bildverarbeitung und Attributanalyse für Upload-Bilder.
    
    Returns:
        Tuple: (processed_image, analysis_results, error_message)
    """
    try:
        # Bildverarbeitung
        from utils import process_image_for_faces
        
        result = process_image_for_faces(file_bytes, filename)
        if result is None or (isinstance(result, tuple) and result[0] is None):
            return None, None, "Bildverarbeitung fehlgeschlagen"
        
        # Extrahiere das Bild aus dem Ergebnis
        if isinstance(result, tuple):
            processed_image = result[0]
        else:
            processed_image = result
        
        # KI-Analyse mit ALLEN Attributen
        actions = ['age', 'gender', 'race', 'emotion']  # Alle verfügbaren Attribute
        
        if processed_image is not None:
            analysis_results = analyze_facial_attributes(
                processed_image,
                actions=actions,
                enforce_detection=not use_improved_analysis  # Weniger streng bei verbesserter Analyse
            )
        else:
            analysis_results = None
        
        return processed_image, analysis_results, ""
        
    except Exception as e:
        return None, None, f"Fehler bei der Analyse: {str(e)}"


def render_attribute_analysis(analysis_results: List[Dict], show_details: bool = True):
    """
    Rendere ALLE Analyseergebnisse in der Streamlit-UI mit vollständiger DeepFace-Darstellung.
    """
    # Dynamischer Import, um Circular Imports zu vermeiden
    import streamlit as st
    
    if not analysis_results:
        st.warning("⚠️ Keine Analyseergebnisse verfügbar")
        return
    
    st.markdown("### 🧠 Vollständige DeepFace Gesichtsanalyse")
    st.markdown("*Alle verfügbaren Attribute: Alter, Geschlecht, Ethnizität, Emotionen*")
    
    for i, result in enumerate(analysis_results):
        with st.expander(f"👤 Gesicht {i+1} - Vollständige Analyse", expanded=True):
            
            # Vier Spalten für alle Attribute
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Age Analysis
                if 'age' in result:
                    age = result['age']
                    age_group = result.get('age_group', 'Unbekannt')
                    confidence = result.get('age_confidence', 'medium')
                    
                    confidence_emoji = _get_confidence_emoji(confidence)
                    st.metric(
                        f"🎂 Alter {confidence_emoji}",
                        f"{age} Jahre",
                        delta=f"{age_group}"
                    )
                    
                    if confidence == 'low':
                        st.caption("⚠️ Unsichere Altersschätzung")
            
            with col2:
                # Gender Analysis
                if 'gender' in result:
                    gender_info = result['gender']
                    confidence = result.get('gender_confidence', 'medium')
                    
                    confidence_emoji = _get_confidence_emoji(confidence)
                    gender_emoji = "👨" if gender_info['dominant'].lower() == 'man' else "👩"
                    
                    st.metric(
                        f"{gender_emoji} Geschlecht {confidence_emoji}",
                        gender_info['dominant'].title(),
                        delta=f"{gender_info['confidence']:.1f}%"
                    )
                    
                    # Detaillierte Gender-Scores
                    if show_details and isinstance(result.get('gender'), dict):
                        with st.container():
                            st.caption("**Detaillierte Scores:**")
                            # Filtere nur numerische Werte
                            gender_scores = {k: v for k, v in result['gender'].items() 
                                           if k not in ['dominant', 'confidence'] and isinstance(v, (int, float))}
                            for gender, score in gender_scores.items():
                                st.progress(score / 100, text=f"{gender.title()}: {score:.1f}%")
            
            with col3:
                # Emotion Analysis
                if 'emotion' in result:
                    emotion_info = result['emotion']
                    simplified = result.get('emotion_simplified', 'Neutral')
                    confidence = result.get('emotion_confidence', 'medium')
                    
                    confidence_emoji = _get_confidence_emoji(confidence)
                    emotion_emoji = _get_emotion_emoji(emotion_info['dominant'])
                    
                    st.metric(
                        f"{emotion_emoji} Emotion {confidence_emoji}",
                        f"{emotion_info['dominant'].title()}",
                        delta=f"Gruppe: {simplified}"
                    )
                    
                    # Detaillierte Emotion-Scores
                    if show_details and 'all_scores' in emotion_info:
                        with st.container():
                            st.caption("**Alle Emotionen:**")
                            emotion_scores = emotion_info['all_scores']
                            # Filtere nur numerische Werte für das Sortieren
                            filtered_emotions = {k: v for k, v in emotion_scores.items() 
                                               if isinstance(v, (int, float))}
                            for emotion, score in sorted(filtered_emotions.items(), key=lambda x: x[1], reverse=True):
                                emoji = _get_emotion_emoji(emotion)
                                st.progress(score / 100, text=f"{emoji} {emotion.title()}: {score:.1f}%")
            
            with col4:
                # Race/Ethnicity Analysis
                if 'race' in result:
                    race_info = result['race']
                    race_confidence = result.get('race_confidence', 'low')
                    
                    confidence_emoji = _get_confidence_emoji(race_confidence)
                    
                    st.metric(
                        f"🌍 Ethnizität {confidence_emoji}",
                        race_info['dominant'].title(),
                        delta=f"{race_info['confidence']:.1f}%"
                    )
                    
                    # Detaillierte Race-Scores
                    if show_details and isinstance(result.get('race'), dict):
                        with st.container():
                            st.caption("**Alle Kategorien:**")
                            # Filtere nur numerische Werte für das Sortieren
                            race_scores = {k: v for k, v in result['race'].items() 
                                         if k not in ['dominant', 'confidence'] and isinstance(v, (int, float))}
                            for race, score in sorted(race_scores.items(), key=lambda x: x[1], reverse=True):
                                st.progress(score / 100, text=f"{race.title()}: {score:.1f}%")
                    
                    # Wichtiger Hinweis zu Ethnizität
                    st.caption("⚠️ Statistisches Merkmal - nicht definitiv!")
                else:
                    st.info("🌍 Ethnizität nicht analysiert")
            
            # Zusätzliche Informationen
            st.markdown("---")
            
            # Rohdaten anzeigen (optional)
            if show_details:
                st.markdown("**🔍 Technische Details & Rohdaten:**")
                with st.container():
                    st.json(result)


def create_attribute_summary(analysis_results: List[Dict]) -> str:
    """
    Erstelle eine kompakte Zusammenfassung der Analyseergebnisse.
    
    Args:
        analysis_results: Liste von Analyseergebnissen
        
    Returns:
        Kompakte String-Zusammenfassung
    """
    if not analysis_results:
        return "Keine Analyse verfügbar"
    
    summaries = []
    for i, result in enumerate(analysis_results):
        parts = []
        
        # Alter
        if 'age' in result:
            age = result['age']
            age_group = result.get('age_group', '')
            if age_group:
                parts.append(f"{age}J ({age_group})")
            else:
                parts.append(f"{age}J")
        
        # Geschlecht
        if 'gender' in result and isinstance(result['gender'], dict):
            gender = result['gender']['dominant']
            confidence = result['gender']['confidence']
            gender_short = "M" if gender.lower() == 'man' else "W"
            parts.append(f"{gender_short}({confidence:.0f}%)")
        
        # Emotion
        if 'emotion' in result and isinstance(result['emotion'], dict):
            emotion = result['emotion']['dominant']
            emotion_simplified = result.get('emotion_simplified', emotion)
            emotion_short = emotion_simplified[:3] if emotion_simplified else emotion[:3]
            parts.append(f"{emotion_short}")
        
        if parts:
            face_summary = f"Gesicht {i+1}: " + ", ".join(parts)
            summaries.append(face_summary)
    
    return " | ".join(summaries) if summaries else "Keine Details verfügbar"


def _get_confidence_emoji(confidence: str) -> str:
    """Gib Emoji basierend auf Konfidenz zurück."""
    confidence_map = {
        'very_low': '❓',
        'low': '⚠️',
        'medium': '🎲',
        'high': '🎯'
    }
    return confidence_map.get(confidence, '🎲')


def _get_emotion_emoji(emotion: str) -> str:
    """Gib Emoji basierend auf Emotion zurück."""
    emotion_map = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😡',
        'surprise': '😲',
        'fear': '😨',
        'disgust': '🤢',
        'neutral': '😐'
    }
    return emotion_map.get(emotion.lower(), '😐')


def _translate_emotion(emotion: str) -> str:
    """Übersetze Emotion ins Deutsche."""
    translation_map = {
        'happy': 'Glücklich',
        'sad': 'Traurig',
        'angry': 'Wütend',
        'surprise': 'Überrascht',
        'fear': 'Ängstlich',
        'disgust': 'Angeekelt',
        'neutral': 'Neutral'
    }
    return translation_map.get(emotion.lower(), emotion.title())


def _translate_ethnicity(ethnicity: str) -> str:
    """Übersetze Ethnizität ins Deutsche."""
    translation_map = {
        'asian': 'Asiatisch',
        'indian': 'Indisch',
        'black': 'Schwarz',
        'white': 'Weiß',
        'middle eastern': 'Naher Osten',
        'latino hispanic': 'Latino/Hispanisch'
    }
    return translation_map.get(ethnicity.lower(), ethnicity.title())
