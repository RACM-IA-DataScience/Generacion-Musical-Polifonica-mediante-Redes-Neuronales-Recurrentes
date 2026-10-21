# 🎵 Proyecto Final: Generación Musical con Deep Learning (LSTM) y Reproductor Interactivo MIDI Studio

**Materia:** Diseño, Visualización y Comprensión de Deep Neural Networks  
**Autor:** Roberto Alejandro Chagra Martinez  
**Código:** 208447128  
**Institución:** Universidad de Guadalajara (UDG)

---

## 📋 Descripción General

Este repositorio contiene el desarrollo completo del **Proyecto Final**, dividido en dos módulos complementarios enfocados en el procesamiento simbólico de música en formato **MIDI**, el entrenamiento de **Redes Neuronales Recurrentes de Memoria a Corto y Largo Plazo (LSTM)** en PyTorch para la composición musical autónoma, y la visualización e interpretación interactiva de secuencias MIDI mediante síntesis de audio e interfaces gráficas web embebidas.

1. **Parte 1 — Pipeline de Datos MIDI y Generación Musical con LSTM (`Parte_1_Reproducir_Nin_Video_Game_MIDIs`):**
   Implementa un flujo de ingeniería de datos de 6 fases sobre el dataset de videojuegos de Nintendo (*Super Mario 3D World*), transforma los archivos `.mid` en matrices binarias de *Piano Roll*, entrena un modelo `GeneradorMIDI_LSTM` y genera nuevas composiciones musicales de forma autoregresiva (`results/mi_primera_composicion_ia.mid`).
2. **Parte 2 — Reproductor Interactivo MIDI Studio (`Parte_2_Reproductor_MIDI`):**
   Desarrolla una aplicación interactiva con `ipywidgets`, síntesis de audio mediante `FluidSynth` (`assets/FluidR3_GM.sf2`) y un reproductor visual basado en `HTML5 Canvas` + `Web Audio API` que dibuja el *Piano Roll* en tiempo real sincronizado con el audio y permite reasignar cualquiera de los 128 instrumentos del estándar *General MIDI*.

---

## 📂 Estructura del Proyecto

```text
Proyecto_final/
├── src/
│   ├── Entregable_de_Proyecto_final,_parte_1_Visualizar_archivos_MIDI.ipynb
│   └── results/mi_primera_composicion_ia.mid
├── src/
│   ├── examples/
│   │   ├── Coldplay_Viva_La_Vida.mid
│   │   ├── La_bella_y_la_bestia.mid
│   │   └── Titanic.mid
│   ├── assets/FluidR3_GM.sf2
│   └── Entregable_de_Proyecto_final,_parte_2_Reproductor_MIDI.ipynb
├── TABLA_DE_FUNCIONES.md         # Catálogo detallado en tablas de todas las funciones utilizadas
├── DOCUMENTACION_TECNICA.md      # Documentación técnica de arquitectura, pipeline y diseño
├── requirements.txt              # Dependencias de Python del proyecto
├── .gitignore                    # Exclusión de entornos virtuales, cachés y binarios pesados
└── README.md                     # Guía principal del proyecto
```

---

## ⚙️ Requisitos e Instalación

### 1. Requisitos del Sistema (Sintetizador FluidSynth)

Para renderizar archivos MIDI a ondas de audio `.wav`, el proyecto utiliza **FluidSynth** junto con el banco de sonidos **assets/FluidR3_GM.sf2**:

- **Linux / Ubuntu / Google Colab:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y fluidsynth fluid-soundfont-gm
  ```
- **Windows:**
  Descargar el ejecutable de [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) y asegurarse de agregarlo a la variable de entorno `PATH`. El banco de sonidos `assets/FluidR3_GM.sf2` se incluye en `src/assets/FluidR3_GM.sf2`.

### 2. Instalación de Dependencias en Python

Se recomienda crear y activar un entorno virtual antes de instalar los paquetes listados en `requirements.txt`:

```bash
# Crear entorno virtual
python -m venv .venv

# Activar en Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activar en Linux / macOS
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Guía de Uso

### Ejecución de la Parte 1: Entrenamiento LSTM y Composición con IA
1. Abrir el notebook `src/Entregable_de_Proyecto_final,_parte_1_Visualizar_archivos_MIDI.ipynb` en Jupyter o Google Colab.
2. Ejecutar las celdas de preparación e importación de librerías (`pretty_midi`, `torch`, `kagglehub`, `midi2audio`).
3. Ejecutar la descarga automática del dataset `hansespinosa2/nin-video-game-midis` (filtrado para las 21 pistas de *Super Mario 3D World*).
4. Ejecutar las **Fases 1 a 6** del pipeline de limpieza y extracción de matrices binarias (`fs=5`, `longitud_secuencia=50`), que producen **11,251 secuencias** de entrenamiento con forma `[50, 128]`.
5. Entrenar el modelo `GeneradorMIDI_LSTM` durante **35 épocas** con `BCEWithLogitsLoss`, optimizador `Adam (lr=0.001)` y recorte de gradientes (`max_norm=1.0`).
6. Generar música nueva con `generar_musica()` y reconstruir el archivo estándar `.mid` mediante `tensor_a_midi()`, obteniendo `results/mi_primera_composicion_ia.mid`.

### Ejecución de la Parte 2: MIDI Studio (Visualizador y Reproductor)
1. Abrir el notebook `src/Entregable_de_Proyecto_final,_parte_2_Reproductor_MIDI.ipynb`.
2. Asegurarse de contar con los archivos `.mid` en la carpeta configurada (`/content/MIDIS` en Colab o `examples/` en entorno local).
3. Ejecutar la celda de la interfaz **MIDI Studio** para desplegar:
   - Selector de Dataset y Canción (`Coldplay_Viva_La_Vida.mid`, `La_bella_y_la_bestia.mid`, `Titanic.mid`).
   - Selector de cualquiera de los **128 instrumentos General MIDI** o casilla para conservar los instrumentos originales.
   - Reproductor interactivo en tiempo real con visualización de **Piano Roll en Canvas HTML5**, barra de progreso y controles de reproducción/pausa/reinicio.

---

## 📊 Resultados Principales

| Métrica / Componente | Resultado Obtenido |
| :--- | :--- |
| **Archivos MIDI auditados (Super Mario 3D World)** | 21 / 21 archivos válidos procesados sin corrupción |
| **Ventanas deslizantes extraídas (`fs=5`, `L=50`)** | 11,251 pares de secuencias `(X, Y)` de dimensión `[50, 128]` |
| **Reducción de pérdida (`BCEWithLogitsLoss`)** | De `0.0917` (Época 1) a `0.0026` (Época 30–35) |
| **Composición generada por IA** | `results/mi_primera_composicion_ia.mid` (600 pasos temporales generados) |
| **Visualización interactiva (Parte 2)** | Reproductor web embebido a 60 FPS con barra celeste sincronizada y cambio de timbre en 128 instrumentos GM |

---

## 📚 Documentación Adicional

- **[TABLA_DE_FUNCIONES.md](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto_final/TABLA_DE_FUNCIONES.md):** Tablas exhaustivas con todas las funciones propias y de librerías empleadas en el proyecto, incluyendo descripción, sintaxis exacta, resultados obtenidos y justificación técnica.
- **[DOCUMENTACION_TECNICA.md](file:///g:/Mi%20unidad/Drive.alumno.udg.mx/Proyectos/Proyecto_final/DOCUMENTACION_TECNICA.md):** Especificación técnica detallada de la arquitectura de la red neuronal LSTM, formulación matemática del preprocesamiento MIDI, flujo de síntesis de audio y arquitectura del visualizador interactivo.
