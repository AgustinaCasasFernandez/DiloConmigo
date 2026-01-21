# DiloConmigo
# Dilo Conmigo
Plataforma Open Source Interactiva para la Rehabilitación del Habla en Pacientes Hipoacúsicos

## Resumen Ejecutivo

**Dilo Conmigo** es una aplicación web de biofeedback acústico diseñada para asistir en la rehabilitación de la inteligibilidad del habla en usuarios de implante coclear. El sistema actúa como un complemento a la terapia fonoaudiológica tradicional, permitiendo la práctica autónoma mediante el análisis objetivo y en tiempo real de la voz.

El software utiliza algoritmos de Procesamiento Digital de Señales (DSP) y modelos de Inteligencia Artificial para evaluar tres dimensiones críticas del habla: calidad vocal, inteligibilidad y prosodia.

Este desarrollo constituye el **Proyecto Final de Carrera de Bioingeniería** en el Instituto Tecnológico de Buenos Aires (ITBA), finalizado en diciembre de 2025.

---

## Funcionalidades Principales

* **Feedback Visual Instantáneo:** Visualización simultánea de la forma de onda, contorno de pitch y espectrograma del usuario frente al modelo ideal.
* **Evaluación Objetiva:** Sistema de puntuación porcentual basado en métricas acústicas, eliminando la subjetividad en la autoevaluación.
* **Módulos de Práctica Jerárquicos:** Ejercicios segmentados en Fonemas, Sílabas, Palabras y Frases.
* **Reportes Clínicos Automáticos:** Generación de informes en PDF detallados para el seguimiento asincrónico por parte del profesional tratante.
* **Adaptabilidad:** Configuración de rangos de frecuencia fundamental (F0) según el perfil del usuario.

---

## Instalación y Despliegue Local

Para ejecutar la plataforma en un entorno local, siga los pasos detallados a continuación.

### 1. Requisitos del Sistema (ver requirements.txt)
* **Python 3.10+**
* **FFmpeg:** Requerido para la manipulación de audio (librerías `pydub` y `librosa`).
* **Navegador Web:** Chrome, Firefox o Edge (con soporte para AudioContext API).

### 2. Obtención del Código Fuente
Puede descargar el código fuente clonando el repositorio o descargando el archivo ZIP.

**Vía Git:**
bash
git clone [https://github.com/AgustinaCasasFernandez/DiloConmigo.git](https://github.com/AgustinaCasasFernandez/DiloConmigo.git)
cd DiloConmigo
