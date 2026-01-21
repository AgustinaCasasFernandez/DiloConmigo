import os
import base64
import parselmouth
from pathlib import Path

# Importamos tu función de ploteo desde tu archivo pre_processing
# Asegúrate de que la carpeta se llame signal_processing y tenga el __init__.py
try:
    from signal_processing.pre_processing import plot_waveform_efficient_web
except ImportError:
    print("ERROR: No encuentro el módulo signal_processing. Asegúrate de ejecutar esto desde la raíz del proyecto.")
    exit()

# --- CONFIGURACIÓN ---
# Carpeta donde están los audios originales (ajusta si es 'static/fonemas' directo)
INPUT_ROOT = Path("static/audios") 

# Carpeta donde guardaremos las imágenes
OUTPUT_ROOT = Path("static/img")

# Las subcarpetas que queremos procesar
CATEGORIAS = ["fonemas", "silabas", "palabras", "frases"]

def guardar_imagen_desde_base64(b64_string, output_path):
    """Convierte el string data:image... de tu función a un archivo real PNG"""
    try:
        # El string viene como "data:image/png;base64,iVBORw0KGgo..."
        # Separamos por la coma para quedarnos solo con la parte codificada
        if "," in b64_string:
            base64_data = b64_string.split(",", 1)[1]
        else:
            base64_data = b64_string

        # Decodificamos y escribimos el archivo binario
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        return True
    except Exception as e:
        print(f"Error guardando imagen {output_path}: {e}")
        return False

def procesar_todo():
    print("--- INICIANDO GENERACIÓN DE IMÁGENES ---")

    for cat in CATEGORIAS:
        input_dir = INPUT_ROOT / cat
        output_dir = OUTPUT_ROOT / cat
        
        # Crear carpeta de destino si no existe (ej: static/img/fonemas)
        os.makedirs(output_dir, exist_ok=True)

        # Verificar que existe la carpeta de origen
        if not input_dir.exists():
            print(f"AVISO: La carpeta {input_dir} no existe. Saltando...")
            continue

        print(f"\nProcesando categoría: {cat.upper()}...")
        
        # Buscar todos los .wav
        archivos_wav = list(input_dir.glob("*.wav"))
        
        if not archivos_wav:
            print(f"  -> No hay archivos .wav en {cat}")
            continue

        for wav_path in archivos_wav:
            nombre_archivo = wav_path.stem # 'hola' (sin .wav)
            png_path = output_dir / f"{nombre_archivo}.png"

            try:
                # 1. Cargar audio con Parselmouth
                sound = parselmouth.Sound(str(wav_path))
                
                # 2. Generar el plot (Obtenemos el string base64)
                # Usamos tu función existente
                b64_plot = plot_waveform_efficient_web(sound)

                # 3. Guardar en disco
                if guardar_imagen_desde_base64(b64_plot, png_path):
                    print(f"  [OK] Generado: {png_path.name}")
                
            except Exception as e:
                print(f"  [ERROR] Falló {nombre_archivo}: {e}")

    print("\n--- PROCESO TERMINADO ---")

if __name__ == "__main__":
    procesar_todo()