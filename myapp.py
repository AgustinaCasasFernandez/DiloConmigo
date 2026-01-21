# myapp.py

# --- 1. PARCHE DE COMPATIBILIDAD (CRÍTICO: MANTENER AL INICIO) ---
import sys
try:
    import numpy
    # Engaño para leer archivos creados con NumPy 2.0 en versiones 1.x
    if not hasattr(numpy, '_core'):
        from numpy import core
        sys.modules['numpy._core'] = core
        sys.modules['numpy._core.multiarray'] = core.multiarray
        print("INFO: Parche de compatibilidad NumPy 2.0 aplicado correctamente.")
except ImportError:
    pass
except Exception as e:
    print(f"ADVERTENCIA: Falló el parche de NumPy: {e}")
# -----------------------------------------------------------------

# --- Imports Estándar y Web ---
from flask import Flask, request, render_template, send_file, make_response
from werkzeug.utils import secure_filename
import os
import io
import json
import string
import pickle
import warnings
import unicodedata
from datetime import datetime
from typing import Tuple, Optional, Dict
from pathlib import Path
from waitress import serve

# --- Imports Científicos ---
import numpy as np
import librosa
import librosa.display
import jiwer
import whisper
import soundfile as sf 
import parselmouth   
import joblib 
from scipy.stats import chi2, linregress 

# --- Imports Gráficos para PDF ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import base64

# --- Imports de PDF ---
try:
    import weasyprint
except ImportError:
    print("ADVERTENCIA: WeasyPrint no está instalado. La descarga de PDF fallará.")
    weasyprint = None

# --- Imports Locales ---
from signal_processing.pre_processing import (
    main_prepro, 
    clean_audio_advanced, 
    analyze_audio_file_web
)

# --- Configuración de la App ---
app = Flask(__name__)

# [CAMBIO 1] Guardar en 'static/uploads' para que sean accesibles desde el historial del Home
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

REPORTS_FOLDER = os.path.join(app.static_folder, 'reports')
os.makedirs(REPORTS_FOLDER, exist_ok=True)
HISTORY_FILE = 'history.json'
CONTACT_FILE = 'contact_messages.json' 
TEMPLATE_AUDIO_ROOT = Path("static/audios")

## --- CONSTANTES Y CARGA DE MODELOS --- ##

FEATURE_COLUMNS_NUCLEO = [
    'meanF0', 'stdevF0', 'hnr', 'localJitter', 'localShimmer',
    'f1_median', 'f2_median', 'f3_median'
]
NUM_FEATURES_NUCLEO = len(FEATURE_COLUMNS_NUCLEO) 

# Cargar el modelo Whisper
try:
    print("Cargando modelo Whisper (base)...")
    WHISPER_MODEL = whisper.load_model("base")
    print("Modelo Whisper cargado.")
except Exception as e:
    print(f"Error cargando Whisper: {e}")
    WHISPER_MODEL = None

# Cargar los modelos Mahalanobis
model_filename = 'mahalanobis_models_final.pkl'
MAHALANOBIS_MODELS = {}

try:
    print(f"Cargando modelos desde {model_filename}...")
    with open(model_filename, 'rb') as f:
        RAW_MODELS = joblib.load(f)
    
    print(f"Archivo cargado. Reestructurando {len(RAW_MODELS)} llaves...")

    for key, value in RAW_MODELS.items():
        if isinstance(key, tuple) and len(key) == 2:
            gender_code, label_raw = key
            label = str(label_raw).strip().lower()
            if gender_code == 'f': gender_key = 'female'
            elif gender_code == 'm': gender_key = 'male'
            else: gender_key = 'neutral'
            if label not in MAHALANOBIS_MODELS: MAHALANOBIS_MODELS[label] = {}
            MAHALANOBIS_MODELS[label][gender_key] = value
        else:
            label = str(key).strip().lower()
            if label not in MAHALANOBIS_MODELS: MAHALANOBIS_MODELS[label] = {}
            MAHALANOBIS_MODELS[label]['neutral'] = value
    print(f"ÉXITO: Modelos listos. Sonidos disponibles: {len(MAHALANOBIS_MODELS)}")

except Exception as e:
    print(f"Error CRÍTICO cargando '{model_filename}': {e}")
    MAHALANOBIS_MODELS = {}

# Estadísticas
SPANISH_PUNCTUATION = string.punctuation + "¡¿"
ALPHA_MAHALANOBIS = 0.001 
UMBRAL_DISTANCIA_CHI2 = chi2.ppf(1 - ALPHA_MAHALANOBIS, df=NUM_FEATURES_NUCLEO)
print(UMBRAL_DISTANCIA_CHI2)

## --- MÓDULOS DE UTILIDAD --- ##

def normalize_text_for_comparison(text: str, practice_id: str = "unknown") -> str:
    try:
        text = text.lower()
        nfkd_form = unicodedata.normalize('NFD', text)
        text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        text = text.translate(str.maketrans('', '', SPANISH_PUNCTUATION))
        text = text.strip()
        if practice_id == 'practica1': text = text.rstrip('h')
        return text
    except: return text

def find_template_path(label: str) -> Optional[str]:
    filename = f"{label}.wav"
    for folder in ["fonemas", "silabas", "palabras", "frases"]:
        template_path = TEMPLATE_AUDIO_ROOT / folder / filename
        if template_path.exists(): return str(template_path)
    return None

def clean_b64_string(b64_str):
    if not b64_str: return ""
    if "," in b64_str: return b64_str.split(",", 1)[1]
    return b64_str

def generate_template_plots(label):
    template_path = find_template_path(label)
    if not template_path or not os.path.exists(template_path): return None, None
    try:
        sound_clean_template, results_dict = analyze_audio_file_web(template_path, gender='neutral')
        wav_data_uri = results_dict['plots']['waveform_clean']
        img_wav_b64 = clean_b64_string(wav_data_uri)

        y = sound_clean_template.values[0]
        sr = int(sound_clean_template.sampling_frequency)
        plt.figure(figsize=(12, 6))
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='inferno')
        plt.colorbar(format="%+2.0f dB", label='Densidad espectral de potencia')
        plt.ylim(0, 5000)
        plt.xlabel("Tiempo (s)", fontsize=12)
        plt.ylabel("Frecuencia (Hz)", fontsize=12)
        plt.title(f"Espectrograma Patrón (Procesado: {label})", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf_spec = io.BytesIO()
        plt.savefig(buf_spec, format='png', bbox_inches='tight')
        plt.close()
        buf_spec.seek(0)
        img_spec_b64 = base64.b64encode(buf_spec.read()).decode('utf-8')
        return img_wav_b64, img_spec_b64
    except Exception as e:
        print(f"Error gráficos plantilla '{label}': {e}")
        return None, None

## --- LÓGICA DE SCORING --- ##

def get_mahalanobis_p_value(new_features: np.ndarray, mean_vec: np.ndarray, inv_cov: np.ndarray, df: int) -> Tuple[float, float]:
    if hasattr(mean_vec, 'values'): mean_vec = mean_vec.values
    if hasattr(inv_cov, 'values'): inv_cov = inv_cov.values
    x_minus_mu = new_features - mean_vec
    mahalanobis_sq_dist = x_minus_mu.dot(inv_cov).dot(x_minus_mu.T)
    p_value = 1 - chi2.cdf(mahalanobis_sq_dist, df=df)
    return mahalanobis_sq_dist, p_value

# --- REEMPLAZA ESTA FUNCIÓN COMPLETA ---
def get_speech_normality_score(new_features: np.ndarray, sound_label: str, gender: str, models_dict: dict) -> Optional[float]:
    if sound_label not in models_dict: return None
    sound_profile_by_gender = models_dict[sound_label] 
    
    profile = None
    if gender in sound_profile_by_gender: profile = sound_profile_by_gender[gender]
    elif 'neutral' in sound_profile_by_gender: profile = sound_profile_by_gender['neutral'] 
    elif 'male' in sound_profile_by_gender: profile = sound_profile_by_gender['male']
    elif 'female' in sound_profile_by_gender: profile = sound_profile_by_gender['female']
    else: return None

    try:
        if 'mean' in profile: mu = profile['mean']
        elif 'mean_vector' in profile: mu = profile['mean_vector']
        else: return None
        if 'inv_cov' in profile: inv = profile['inv_cov']
        elif 'inv_cov_matrix' in profile: inv = profile['inv_cov_matrix']
        else: return None
        
        # AQUÍ ESTÁ EL CAMBIO: Solo calculamos la distancia, no el p-value
        distancia_sq, _ = get_mahalanobis_p_value(new_features, mu, inv, NUM_FEATURES_NUCLEO)
        
        # Devolvemos la raíz cuadrada para tener la Distancia de Mahalanobis (no al cuadrado), 
        # que es más lineal y fácil de entender, o mantenemos al cuadrado si prefieres.
        # Por consistencia con tu umbral (que es chi2, o sea distancia al cuadrado), devolvamos la cuadrada:
        return distancia_sq
    except: return None
    
def get_intelligibility_score(audio_path: str, target_text: str, whisper_model, metric_type: str = 'wer', practice_id: str = "unknown") -> Tuple[Optional[float], Optional[str]]:
    if whisper_model is None: return None, "Whisper no cargado"
    try:
        result = whisper_model.transcribe(audio_path, language="es", task="transcribe", fp16=False)
        transcription_raw = result["text"].strip()
        target_norm = normalize_text_for_comparison(target_text, practice_id=practice_id)
        transcription_norm = normalize_text_for_comparison(transcription_raw, practice_id=practice_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if metric_type == 'cer': error_rate = jiwer.cer(target_norm, transcription_norm)
            else: error_rate = jiwer.wer(target_norm, transcription_norm)
        score = max(0.0, 1.0 - error_rate)
        return score, transcription_raw
    except: return None, "Error en transcripción"

def _get_prosody_features(audio_path: str) -> Optional[Dict[str, float]]:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25) 
        if len(y_trimmed) < 500: return None
        sound = parselmouth.Sound(audio_path)
        duration_voz = librosa.get_duration(y=y_trimmed, sr=sr)
        rms = librosa.feature.rms(y=y_trimmed)[0]
        rms_t = librosa.times_like(rms, sr=sr)
        loc_pico_intensidad_pct = 0.5 
        if len(rms_t) > 0: loc_pico_intensidad_pct = rms_t[np.argmax(rms)] / duration_voz 
        pitch = sound.to_pitch(pitch_floor=75.0, pitch_ceiling=500.0)
        f0_values = pitch.selected_array['frequency']
        f0_values_voiced = f0_values[f0_values > 0]
        if len(f0_values_voiced) < 5: return None
        rango_f0 = np.max(f0_values_voiced) - np.min(f0_values_voiced)
        slope_result = linregress(np.linspace(0, 1, len(f0_values_voiced)), f0_values_voiced)
        return {"rango_f0": rango_f0, "pendiente_f0": slope_result.slope, "duracion_voz": duration_voz, "loc_pico_intensidad_pct": loc_pico_intensidad_pct}
    except: return None

def get_prosody_score(new_audio_path: str, template_audio_path: str) -> Tuple[float, Dict[str, float]]:
    default_errors = {"rango": 0, "pendiente": 0, "duracion": 0, "pico": 0}
    if not template_audio_path or not os.path.exists(template_audio_path): return 999.0, default_errors
    if not new_audio_path or not os.path.exists(new_audio_path): return 999.0, default_errors
    feat_user = _get_prosody_features(new_audio_path)
    feat_templ = _get_prosody_features(template_audio_path)
    if feat_user is None or feat_templ is None: return 999.0, default_errors
    error_rango = abs(feat_templ["rango_f0"] - feat_user["rango_f0"]) / (abs(feat_templ["rango_f0"]) + 1e-6)
    error_dur = abs(feat_templ["duracion_voz"] - feat_user["duracion_voz"]) / (abs(feat_templ["duracion_voz"]) + 1e-6)
    error_pico = abs(feat_templ["loc_pico_intensidad_pct"] - feat_user["loc_pico_intensidad_pct"])
    if np.sign(feat_templ["pendiente_f0"]) != np.sign(feat_user["pendiente_f0"]): error_pend = 1.0
    else: error_pend = abs(feat_templ["pendiente_f0"] - feat_user["pendiente_f0"]) / (abs(feat_templ["pendiente_f0"]) + 1e-6)
    costo_total = (error_pend * 0.40) + (error_dur * 0.30) + (error_rango * 0.10) + (error_pico * 0.20)
    return max(0.0, min(2.0, costo_total)), {"rango": error_rango, "pendiente": error_pend, "duracion": error_dur, "pico": error_pico}

def normalize_gender(g):
    g = (g or 'neutral').lower()
    if g in ['m', 'male']: return 'male'
    if g in ['f', 'female']: return 'female'
    return 'neutral'

PESO_CALIDAD, PESO_INTEL, PESO_FORMA = 0.25, 0.45, 0.30
CALIDAD_UMBRAL_CONSEJO = ALPHA_MAHALANOBIS * 100.0 
FORMA_COSTO_MAXIMO, FORMA_COSTO_MINIMO = 0.70, 0.10

# --- REEMPLAZA ESTA FUNCIÓN COMPLETA ---
def generate_feedback_advice(scores: dict) -> dict:
    try:
        # [CAMBIO] Leemos la distancia
        distancia_usuario = float(scores.get("distancia_mahalanobis", 999.0))
        score_intel = float(scores["score_intel"])
        costo_forma = float(scores["score_forma"]) 
        prosody_errors = scores.get("prosody_errors", {})
        transcripcion_raw = scores.get("transcripcion") or "..."
        objetivo_raw = scores.get("objetivo") or "..."
    except: return {"weighted_score_pct": "0", "overall_phrase": "Error", "improvement_list": ["Error calculando puntaje."]}

    # 1. Cálculo del Score de Forma (Prosodia) - Se mantiene igual
    temp_forma = 1.0 - ((costo_forma - FORMA_COSTO_MINIMO) / (FORMA_COSTO_MAXIMO - FORMA_COSTO_MINIMO))
    score_forma_pct = max(0.0, min(100.0, temp_forma * 100.0)) 
    
    limite_aceptable = UMBRAL_DISTANCIA_CHI2 # El valor 26.12
    
    if distancia_usuario <= limite_aceptable:
        score_calidad_pond = 100.0
    else:
        exceso = distancia_usuario - limite_aceptable
        score_calidad_pond = max(0.0, 100.0 - (exceso * 0.2))

    # Debug para que lo veas en consola
    print(f"Distancia: {distancia_usuario:.2f} | Umbral: {limite_aceptable:.2f} | Nota Calidad: {score_calidad_pond:.1f}")

    # 3. Ponderación Final
    if costo_forma == 999.0: weighted_score = score_intel
    else: weighted_score = (score_calidad_pond * PESO_CALIDAD) + (score_intel * PESO_INTEL) + (score_forma_pct * PESO_FORMA)

    # Frases de feedback
    if weighted_score >= 80: overall_phrase = "¡Excelente trabajo!"
    elif weighted_score >= 60: overall_phrase = "¡Ya casi lo tienes!"
    elif weighted_score >= 40: overall_phrase = "¡Sigue practicando!"
    else: overall_phrase = "Consulta con tu fonoaudióloga para más soporte."

    improvement_list = []
    if score_intel < 100: improvement_list.append(f"Pronunciación: Se entendió '{transcripcion_raw}' en vez de '{objetivo_raw}'.")
    
    # [CAMBIO] Feedback basado en distancia
    if score_calidad_pond < 50.0:
        improvement_list.append(f"Calidad Vocal: Tu voz se aleja del modelo. Revisa tu entorno de grabación e intenta nuevamente.")
    
    if costo_forma != 999.0 and costo_forma > 0.40:
        errs = []
        if prosody_errors.get("duracion", 0) > 0.3: errs.append("el ritmo")
        if prosody_errors.get("pendiente", 0) > 0.5: errs.append("la entonación")
        if errs: improvement_list.append("Prosodia: Revisa " + " y ".join(errs) + ".")
        
    if not improvement_list: improvement_list.append("¡Todo muy bien!")
    
    return {"weighted_score_pct": f"{weighted_score:.0f}", "overall_phrase": overall_phrase, "improvement_list": improvement_list}

# --- PERSISTENCIA SIMPLE (HISTORIAL) ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    return []

def load_contact_messages():
    if os.path.exists(CONTACT_FILE):
        with open(CONTACT_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    return []

def save_contact_messages(msgs):
    with open(CONTACT_FILE, 'w', encoding='utf-8') as f: json.dump(msgs, f, indent=2, ensure_ascii=False)

# [CAMBIO 2] Función para guardar y ordenar el Top 10
# [CORRECCIÓN] Función para guardar SOLO EL MEJOR PUNTAJE por ejercicio
def update_history(new_record):
    """
    Actualiza el historial manteniendo solo el MEJOR puntaje único por etiqueta.
    Luego ordena y se queda con el Top 10 global.
    """
    try:
        hist = load_history()
        
        # 1. Buscar si ya existe este ejercicio en el historial
        label_buscado = new_record.get('label')
        nuevo_puntaje = float(new_record.get('score', 0))
        
        encontrado = False
        
        for i, item in enumerate(hist):
            if item.get('label') == label_buscado:
                encontrado = True
                puntaje_anterior = float(item.get('score', 0))
                
                # 2. Si existe, comparamos: ¿El nuevo es mejor?
                if nuevo_puntaje > puntaje_anterior:
                    # ¡Récord batido! Reemplazamos la entrada vieja con la nueva
                    hist[i] = new_record
                # Si es menor o igual, no hacemos nada (nos quedamos con el récord anterior)
                break
        
        # 3. Si no existía, lo agregamos como nuevo
        if not encontrado:
            hist.append(new_record)
            
        # 4. Ordenar: Mayor puntaje primero
        hist.sort(key=lambda x: float(x.get('score', 0)), reverse=True)
        
        # 5. Recortar al Top 10
        hist = hist[:10]
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(hist, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error actualizando historial: {e}")

## --- RUTAS DE LA APP --- ##

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_file = request.files.get('audio_data')
    if not audio_file: return "No se envió archivo", 400

    practice = request.form.get('practice', 'practica1')
    gender_raw = request.form.get('gender', 'neutral')
    label = request.form.get('label', 'sin_label').strip().lower()
    
    # [CAMBIO] Detectar si es el caso especial /sh/
    clean_label = label.replace("/", "").strip()
    is_sh_case = (clean_label == 'sh')
    
    analysis_type = request.form.get('analysis_type', 'full')
    gender_norm = normalize_gender(gender_raw)
    metric_para_jiwer = 'wer' if practice == 'practica4' else 'cer'
    
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    upload_dir = os.path.join(UPLOAD_FOLDER, secure_filename(label.split(" ")[0]))
    os.makedirs(upload_dir, exist_ok=True)
    wav_path = os.path.join(upload_dir, f"grabacion_{ts}.wav")
    audio_file.save(wav_path) 

    # Cargar audio
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(wav_path, sr=None) 
        sf.write(wav_path, y, sr, subtype='PCM_16')
    except: return "Error audio format", 500

    img_tpl_wav = None
    img_tpl_pitch = None
    img_tpl_formants = None
    
    # [CAMBIO] Lógica condicional: Si es Whisper_only O Práctica 4 O el caso /sh/
    if analysis_type == 'whisper_only' or practice == 'practica4' or is_sh_case:
        target_text = label.replace("_", " ").replace("/", "")
        
        score_intel_raw, transcripcion = get_intelligibility_score(
            wav_path, target_text, WHISPER_MODEL, metric_para_jiwer, practice
        )
        score_intel = score_intel_raw * 100.0 if score_intel_raw is not None else 0.0
        
        # [CAMBIO] Si es /sh/, normalizar la transcripción para mostrarla limpia
        if is_sh_case:
            transcripcion = normalize_text_for_comparison(transcripcion, practice)

        feedback_scores = {
            "score_calidad": "NaN", 
            "score_intel": f"{score_intel:.1f}",
            "score_forma": "999.0", 
            "prosody_errors": {}, 
            "transcripcion": transcripcion, 
            "objetivo": target_text
        }
        results = {}
        
        # Inicializar imágenes en None
        img_wav, img_wav_clean, img_pitch, img_formants = None, None, None, None

        # [CAMBIO] Si es /sh/, generar SOLO la forma de onda manualmente
        if is_sh_case:
            try:
                plt.figure(figsize=(10, 3))
                librosa.display.waveshow(y, sr=sr, color='#4F46E5', alpha=0.8) # Un azul bonito
                plt.title(f"Forma de Onda: {label}", fontsize=10)
                plt.xlabel("Tiempo (s)")
                plt.ylabel("Amplitud")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                img_wav = base64.b64encode(buf.read()).decode('utf-8')
            except Exception as e:
                print(f"Error generando plot sh: {e}")

    else:
        # --- BLOQUE ANÁLISIS COMPLETO (MAHALANOBIS) ---
        sound_clean, results, img_wav, img_wav_clean, img_pitch, img_formants = main_prepro(wav_path, gender=gender_norm)
        distancia_raw = 999.0
        try:
            features_8 = np.array([
                results['audio_info']['meanF0'], results['audio_info']['stdevF0'],
                results['audio_info']['hnr'], results['audio_info']['localJitter'],
                results['audio_info']['localShimmer'], results['formants']['f1_median'],
                results['formants']['f2_median'], results['formants']['f3_median']
            ])
            val_distancia = get_speech_normality_score(features_8, label, gender_norm, MAHALANOBIS_MODELS)
            if val_distancia is not None:
                distancia_raw = val_distancia
                
        except Exception as e: print(f"Error Mahalanobis: {e}")
        
        target_text = label.replace("_", " ") 
        score_intel_raw, transcripcion = get_intelligibility_score(
            wav_path, target_text, WHISPER_MODEL, 'cer', practice
        )
        score_intel = score_intel_raw * 100.0 if score_intel_raw is not None else 0.0

        user_clean_path = wav_path.replace('.wav', '_cleaned_for_prosody.wav')
        try: sound_clean.save(user_clean_path, "WAV") 
        except: user_clean_path = wav_path 
            
        template_path = find_template_path(label)
        template_clean_path = None
        
        if template_path:
            template_clean_path = template_path.replace('.wav', '_clean.wav')
            if not os.path.exists(template_clean_path):
                try:
                    sound_t = parselmouth.Sound(template_path)
                    clean_audio_advanced(sound_t, output_file=template_clean_path)
                except: template_clean_path = template_path
            try:
                _, tpl_results = analyze_audio_file_web(template_path, gender='neutral')
                img_tpl_wav = clean_b64_string(tpl_results['plots']['waveform_clean'])
                img_tpl_pitch = tpl_results['plots']['pitch']
                img_tpl_formants = clean_b64_string(tpl_results['plots']['formants'])
            except Exception as e: print(f"Error generando plots plantilla web: {e}")
        costo_forma, prosody_errors = get_prosody_score(user_clean_path, template_clean_path)
        
        feedback_scores = {
            "distancia_mahalanobis": f"{distancia_raw:.2f}",
            "score_intel": f"{score_intel:.1f}",
            "score_forma": f"{costo_forma:.3f}", 
            "prosody_errors": prosody_errors,
            "transcripcion": transcripcion, 
            "objetivo": target_text
        }
    
    results['feedback'] = feedback_scores
    results['advice'] = generate_feedback_advice(feedback_scores)

    # Guardar en Historial (TOP 10)
    try:
        practice_names = {"practica1": "Fonemas", "practica2": "Sílabas", "practica3": "Palabras", "practica4": "Frases"}
        rel_path = os.path.relpath(wav_path, app.static_folder).replace("\\", "/")
        
        history_item = {
            "score": int(float(results['advice']['weighted_score_pct'])),
            "label": label.capitalize(),
            "practice_name": practice_names.get(practice, practice),
            "static_path": rel_path,
            "date": datetime.now().strftime('%d/%m/%Y')
        }
        update_history(history_item)
    except Exception as e:
        print(f"No se pudo guardar historial: {e}")
    
    return render_template(f"{practice}.html", results=results, 
                           img_wav=img_wav, img_wav_clean=img_wav_clean, 
                           img_pitch=img_pitch, img_formants=img_formants,
                           img_tpl_wav=img_tpl_wav, img_tpl_pitch=img_tpl_pitch, 
                           img_tpl_formants=img_tpl_formants, active_page=practice)
    
@app.route('/download_report', methods=['POST'])
def download_report():
    if not weasyprint: return "WeasyPrint no instalado", 500
    try:
        data = request.form
        label = data.get('label', 'Ejercicio')
        p_id_raw = data.get('practice_id', 'practica1')
        practice_number = p_id_raw.replace('practica', '') 
        
        # [CAMBIO] Limpiar label y detectar si es sh
        clean_label = label.lower().replace("/", "").strip()
        
        # [CAMBIO] Condición para usar PDF Lite (Frases O Whisper Only O caso sh)
        if p_id_raw == 'practica4' or data.get('analysis_type') == 'whisper_only' or clean_label == 'sh':
             # Usar plantilla simple
             html_template = 'pdf_lite.html'
             tpl_wav, tpl_spec = None, None
        else:
             html_template = 'pdf.html'
             tpl_wav, tpl_spec = generate_template_plots(label)

        user_img_clean = data.get('img_clean_src', '') 
        user_img_pitch = data.get('img_pitch_src', '')
        user_img_formants = clean_b64_string(data.get('img_formants_src', ''))

        html_string = render_template(html_template, 
            label=label, practice_number=practice_number, practice_name=f"Práctica {practice_number}",
            gender=data.get('gender', 'N/A'),
            current_time=datetime.now().strftime('%d/%m/%Y %H:%M'),
            score_pct=data.get('score_pct', '0'),
            advice_phrase=data.get('advice_phrase', ''),
            advice_list=json.loads(data.get('advice_list', '[]')),
            mean_f0=data.get('mean_f0', '-'), jitter=data.get('jitter', '-'), shimmer=data.get('shimmer', '-'), hnr=data.get('hnr', '-'),
            f1=data.get('f1', '-'), f2=data.get('f2', '-'), f3=data.get('f3', '-'), f4=data.get('f4', '-'),
            img_clean_src=user_img_clean, img_pitch_src=user_img_pitch, img_formants_src=user_img_formants,
            img_template_wav=tpl_wav, img_template_spec=tpl_spec
        )
        pdf_bytes = weasyprint.HTML(string=html_string).write_pdf()
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        safe_label = secure_filename(label)
        response.headers['Content-Disposition'] = f'attachment; filename=Reporte_{safe_label}.pdf'
        return response
    except Exception as e:
        print(f"Error PDF: {e}")
        import traceback
        traceback.print_exc() 
        return f"Error generando PDF: {e}", 500
    
@app.route("/")
def home(): 
    # Muestra el historial completo (que ya está limitado a 10 por update_history)
    return render_template("home.html", active_page="home", history=load_history())

@app.route("/practica1")
def practica1(): return render_template("practica1.html", active_page="practica1")
@app.route("/practica2")
def practica2(): return render_template("practica2.html", active_page="practica2")
@app.route("/practica3")
def practica3(): return render_template("practica3.html", active_page="practica3")
@app.route("/practica4")
def practica4(): return render_template("practica4.html", active_page="practica4")
@app.route("/contacto")
def contacto(): return render_template("contacto.html", active_page="contacto")

@app.route("/enviar_contacto", methods=['POST'])
def enviar_contacto():
    try:
        msgs = load_contact_messages()
        msgs.append(dict(request.form))
        save_contact_messages(msgs)
        return render_template("gracias.html")
    except: return "Error", 500

if __name__ == "__main__":
    print("Iniciando servidor en http://127.0.0.1:8080")
    serve(app, host="127.0.0.1", port=8080)