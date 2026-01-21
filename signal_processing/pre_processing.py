# ================= Preprocessing para integración web =================
import numpy as np
import pandas as pd
import parselmouth
import librosa
import statistics
import noisereduce as nr
import librosa.display 
import matplotlib.pyplot as plt
from parselmouth.praat import call
import base64
import io
import os
# (Opcional: estos imports no son necesarios si este módulo no define rutas Flask)
# from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para web

# ------------ Utilidades de género / F0 ------------
def normalize_gender(g):
    g = (g or "neutral").lower()
    return g if g in ("male", "female", "neutral") else "neutral"

def get_f0_bounds(gender: str):
    g = normalize_gender(gender)
    if g == "male":
        return 75, 300
    if g == "female":
        return 100, 500
    return 75, 400  # neutral

# ------------ Medidas de fuente de voz ------------
def measure_pitch(sound, f0min, f0max, unit):
    """
    Mide características de la fuente de voz:
    - Duración, F0 (media, desvío), HNR
    - Jitter (local/abs/rap/ppq5/ddp)
    - Shimmer (local, dB, apq3/5/11, dda)
    """
    duration = call(sound, "Get total duration")
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    meanF0 = call(pitch, "Get mean", 0, 0, unit)
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)
    
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return {
        'duration': duration,
        'meanF0': meanF0,
        'stdevF0': stdevF0,
        'hnr': hnr,
        'localJitter': localJitter,
        'localabsoluteJitter': localabsoluteJitter,
        'rapJitter': rapJitter,
        'ppq5Jitter': ppq5Jitter,
        'ddpJitter': ddpJitter,
        'localShimmer': localShimmer,
        'localdbShimmer': localdbShimmer,
        'apq3Shimmer': apq3Shimmer,
        'apq5Shimmer': apq5Shimmer,
        'apq11Shimmer': apq11Shimmer,
        'ddaShimmer': ddaShimmer
    }

# ------------ Formantes ------------
def measure_formants(sound, f0min, f0max):
    """
    Mide F1..F4 en pulsos glóticos y calcula medias/medianas.
    """
    _ = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list, f2_list, f3_list, f4_list = [], [], [], []
    for idx in range(1, numPoints + 1):
        t = call(pointProcess, "Get time from index", idx)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        if str(f1) != 'nan': f1_list.append(f1)
        if str(f2) != 'nan': f2_list.append(f2)
        if str(f3) != 'nan': f3_list.append(f3)
        if str(f4) != 'nan': f4_list.append(f4)

    f1_mean = statistics.mean(f1_list) if f1_list else 0
    f2_mean = statistics.mean(f2_list) if f2_list else 0
    f3_mean = statistics.mean(f3_list) if f3_list else 0
    f4_mean = statistics.mean(f4_list) if f4_list else 0

    f1_median = statistics.median(f1_list) if f1_list else 0
    f2_median = statistics.median(f2_list) if f2_list else 0
    f3_median = statistics.median(f3_list) if f3_list else 0
    f4_median = statistics.median(f4_list) if f4_list else 0

    return {
        'f1_mean': f1_mean, 'f2_mean': f2_mean, 'f3_mean': f3_mean, 'f4_mean': f4_mean,
        'f1_median': f1_median, 'f2_median': f2_median, 'f3_median': f3_median, 'f4_median': f4_median,
        'f1_points': f1_list, 'f2_points': f2_list, 'f3_points': f3_list, 'f4_points': f4_list
    }

# ------------ Estimaciones de VTL ------------
def calculate_vtl_estimates(formant_data):
    """
    Estimaciones de longitud del tracto vocal a partir de formantes.
    """
    results = {}
    f1 = formant_data['f1_median']
    f2 = formant_data['f2_median']
    f3 = formant_data['f3_median']
    f4 = formant_data['f4_median']

    if all([f1, f2, f3, f4]):
        results['fdisp'] = (f4 - f1) / 3
        results['avgFormant'] = (f1 + f2 + f3 + f4) / 4
        results['mff'] = (f1 * f2 * f3 * f4) ** 0.25
        results['fitch_vtl'] = ((1 * (35000 / (4 * f1))) +
                                (3 * (35000 / (4 * f2))) + 
                                (5 * (35000 / (4 * f3))) + 
                                (7 * (35000 / (4 * f4)))) / 4
        xysum = (0.5 * f1) + (1.5 * f2) + (2.5 * f3) + (3.5 * f4)
        xsquaredsum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
        results['delta_f'] = xysum / xsquaredsum
        results['vtl_delta_f'] = 35000 / (2 * results['delta_f'])
    return results

# ------------ Limpieza de ruido ------------
def clean_audio_advanced(sound, noise_sample_start=0.0, noise_sample_duration=0.30, 
                         prop_decrease=0.8, output_file=None):
    """
    Reduce ruido usando una muestra del propio audio como referencia.
    """
    sampling_rate = int(sound.get_sampling_frequency())
    duration = sound.get_total_duration()
    audio_data = sound.values[0]

    start_sample = int(noise_sample_start * sampling_rate)
    end_sample = int((noise_sample_start + noise_sample_duration) * sampling_rate)
    end_sample = min(end_sample, len(audio_data))

    print(f"Audio original: {duration:.2f}s, {sampling_rate}Hz")
    print(f"Muestra de ruido: {noise_sample_start:.1f}s - {noise_sample_start + noise_sample_duration:.1f}s")

    audio_clean = nr.reduce_noise(
        y=audio_data,
        sr=sampling_rate,
        y_noise=audio_data[start_sample:end_sample],
        prop_decrease=prop_decrease,
        stationary=False
    )

    sound_clean = parselmouth.Sound(
        audio_clean, 
        sampling_frequency=sampling_rate,
        start_time=sound.get_start_time()
    )
    if output_file:
        sound_clean.save(output_file, "WAV")
        print(f"Audio limpio guardado en: {output_file}")

    print("Limpieza avanzada completada")
    return sound_clean

# ------------ Plots a base64 (para web) ------------
def plot_waveform_efficient_web(sound, max_points=10000):
    duration = sound.get_total_duration()
    sampling_rate = sound.get_sampling_frequency()
    values = sound.values[0]
    times = sound.xs()

    num_samples = len(values)
    step = max(1, num_samples // max_points)
    if step > 1:
        times = times[::step]
        values = values[::step]

    plt.figure(figsize=(12, 6))
    plt.plot(times, values, linewidth=0.5, color='blue')
    plt.grid(True, alpha=0.3)
    plt.ylabel("Amplitud")
    plt.xlabel("Tiempo (s)")
    plt.title("Forma de onda")
    plt.xlim(0, duration)

    info_text = f"Duración: {duration:.2f}s | Freq. muestreo: {sampling_rate:.0f}Hz | Muestras: {num_samples}"
    if step > 1:
        info_text += f" | Submuestreo: 1:{step}"
    plt.figtext(0.02, 0.02, info_text, fontsize=8, ha='left')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{img_base64}"

def plot_pitch_web(sound, f0min, f0max):
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    num_frames = pitch.get_number_of_frames()
    times, values = [], []

    for i in range(1, num_frames + 1):
        t = pitch.get_time_from_frame_number(i)
        v = pitch.get_value_in_frame(i)
        times.append(t)
        values.append(v if v is not None else float('nan'))

    import numpy as np
    times = np.array(times)
    values = np.array(values)

    plt.figure(figsize=(10, 5))
    valid_mask = ~np.isnan(values)
    plt.plot(times[valid_mask], values[valid_mask], 'o', markersize=2, color='blue', alpha=0.7)

    if np.sum(valid_mask) > 1:
        from scipy.interpolate import interp1d
        try:
            f_interp = interp1d(times[valid_mask], values[valid_mask], kind='linear',
                                bounds_error=False, fill_value=np.nan)
            times_interp = np.linspace(times[0], times[-1], len(times)*2)
            values_interp = f_interp(times_interp)
            plt.plot(times_interp, values_interp, '-', linewidth=1, color='blue', alpha=0.5)
        except Exception:
            pass

    plt.xlim(times[0], times[-1])
    plt.grid(True, alpha=0.3)
    plt.ylabel("Frecuencia fundamental (Hz)")
    plt.xlabel("Tiempo (s)")
    plt.title("Contorno de Pitch")
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    total_duration = float(times[-1] - times[0]) if len(times) > 0 else 0.0
    voiced_percentage = (float(np.sum(valid_mask)) / len(times) * 100) if len(times) > 0 else 0.0

    return f"data:image/png;base64,{img_base64}", {
        "duration": total_duration,
        "voiced_percentage": voiced_percentage
    }

def plot_formants_web(sound, file_path=None): # file_path ahora es opcional/ignorado
    # --- CAMBIO CRÍTICO: Usar el audio en memoria (Trimmed), no el archivo en disco ---
    
    # Extraemos los datos numéricos (numpy array) del objeto Parselmouth
    y = sound.values[0] 
    sr = int(sound.sampling_frequency)
    
    # ----------------------------------------------------------------------------------

    # Calcular Espectrograma (Fondo)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))

    plt.figure(figsize=(12, 6))
    # Usamos 'hz' y 'inferno' para coincidir con el estilo del PDF
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='inferno')
    plt.colorbar(format="%+2.0f dB", label='Densidad espectral de potencia')

    # Calcular Formantes (Líneas)
    formants = sound.to_formant_burg(time_step=0.0025, max_number_of_formants=5,
                                     maximum_formant=5000, window_length=0.025, pre_emphasis_from=50)
    n_formant_frames = formants.get_number_of_frames()

    colors = ['red', 'lime', 'cyan', 'magenta']
    formant_names = ['F1', 'F2', 'F3', 'F4']

    for i in range(1, 5):
        times, values = [], []
        for frame in range(1, n_formant_frames + 1):
            t = formants.get_time_from_frame_number(frame)
            f = formants.get_value_at_time(i, t, parselmouth.FormantUnit.HERTZ)
            if f is not None and f > 0:
                times.append(t)
                values.append(f)

        if times and values:
            import numpy as np
            from scipy.interpolate import interp1d
            
            # Filtrar datos para plotear
            sorted_pairs = sorted(zip(times, values))
            t_sorted = [p[0] for p in sorted_pairs]
            v_sorted = [p[1] for p in sorted_pairs]

            if len(t_sorted) > 3:
                try:
                    f_interp = interp1d(t_sorted, v_sorted, kind='cubic',
                                        bounds_error=False, fill_value='extrapolate')
                    # Generar puntos suaves para la línea
                    t_cont = np.linspace(min(t_sorted), max(t_sorted), 200)
                    v_cont = f_interp(t_cont)
                    
                    # Limpieza visual de outliers
                    mask = (v_cont > 0) & (v_cont < 5000)
                    t_cont = t_cont[mask]; v_cont = v_cont[mask]
                    
                    avg_value = np.mean(v_sorted)
                    plt.plot(t_cont, v_cont, color=colors[i-1], linewidth=2.5,
                             label=f'{formant_names[i-1]} ({avg_value:.0f} Hz)', alpha=0.8)
                except Exception:
                    avg_value = np.mean(v_sorted)
                    plt.plot(t_sorted, v_sorted, color=colors[i-1], linewidth=2.5,
                             label=f'{formant_names[i-1]} ({avg_value:.0f} Hz)', alpha=0.8)

    plt.ylim(0, 5000)
    plt.xlabel("Tiempo (s)", fontsize=12)
    plt.ylabel("Frecuencia (Hz)", fontsize=12)
    plt.title("Espectrograma con Formantes (Usuario)", fontsize=14, fontweight='bold')
    
    # Leyenda
    legend = plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                        fontsize=10, title="Formantes")
    legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)

    # Guardar a base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close() # Cerrar para liberar memoria
    
    return img_base64

# ------------ Función principal para la web (ORDEN CORREGIDO: CLEAN -> TRIM) ------------
def analyze_audio_file_web(file_path, gender="neutral"):
    """
    Analiza un archivo:
    1. Carga audio completo.
    2. Limpia ruido (usando los primeros 0.3s como perfil).
    3. Recorta silencios (Trim) sobre el audio ya limpio.
    4. Calcula métricas y gráficos.
    """
    f0min, f0max = get_f0_bounds(gender)

    # 1. Cargar Audio Original (Parselmouth)
    # Cargamos todo para tener el ruido ambiente del inicio disponible.
    sound_raw = parselmouth.Sound(file_path)

    # 2. Limpiar Ruido (CLEAN FIRST)
    # Usamos 0.3s del inicio. Como no hemos hecho trim, aquí suele estar el ruido ambiente.
    sound_clean_full = clean_audio_advanced(
        sound_raw, 
        noise_sample_start=0.0, 
        noise_sample_duration=0.3
    )

    # 3. Recortar Silencios (TRIM SECOND)
    # Extraemos los datos del audio limpio para pasarlos a Librosa
    y_clean = sound_clean_full.values[0]
    sr = int(sound_clean_full.sampling_frequency)

    # Aplicamos Trim con Librosa (top_db=25 es estándar para voz hablada)
    y_trimmed, _ = librosa.effects.trim(y_clean, top_db=25)

    # Re-empaquetamos en Parselmouth para el análisis final
    # Este 'sound_final' es el que usaremos para métricas y gráficos limpios
    sound_final = parselmouth.Sound(y_trimmed, sampling_frequency=sr)

    # --- MÉTICAS (Usando sound_final) ---
    pitch_results = measure_pitch(sound_final, f0min, f0max, "Hertz")
    formant_results = measure_formants(sound_final, f0min, f0max)
    vtl_results = calculate_vtl_estimates(formant_results)
    print("analyze characteristics done")

    # --- PLOTS ---
    # waveform_original: Muestra el crudo (con ruido y silencios largos)
    waveform_original = plot_waveform_efficient_web(sound_raw)
    
    # waveform_clean: Muestra el resultado final (limpio y recortado)
    waveform_clean = plot_waveform_efficient_web(sound_final)
    
    # Pitch y Formantes sobre el final
    pitch_plot, pitch_info = plot_pitch_web(sound_final, f0min, f0max)
    formants_plot = plot_formants_web(sound_final, file_path) # Nota: plot_formants_web carga file_path interno, eso es solo visual.
    print("plots done")

    return sound_final, {
        'filename': os.path.basename(file_path),
        'audio_info': {
            'duration': pitch_results['duration'],
            'meanF0': pitch_results['meanF0'],
            'stdevF0': pitch_results['stdevF0'],
            'hnr': pitch_results['hnr'],
            'localJitter': pitch_results['localJitter'],
            'localShimmer': pitch_results['localShimmer']
        },
        'formants': {
            'f1_median': formant_results['f1_median'],
            'f2_median': formant_results['f2_median'],
            'f3_median': formant_results['f3_median'],
            'f4_median': formant_results['f4_median']
        },
        'vtl_estimates': vtl_results,
        'plots': {
            'waveform_original': waveform_original,
            'waveform_clean': waveform_clean,
            'pitch': pitch_plot,
            'formants': formants_plot
        },
        'pitch_info': pitch_info
    }
# ------------ Helper sin input() para backend ------------
def main_prepro(file_path, gender='neutral'):
    gender = normalize_gender(gender)
    sound_clean, results = analyze_audio_file_web(file_path, gender)
    img_wav = results['plots']['waveform_original']
    img_wav_clean = results['plots']['waveform_clean']
    img_pitch = results['plots']['pitch']
    img_formants = results['plots']['formants']
    return sound_clean,results, img_wav, img_wav_clean, img_pitch, img_formants
