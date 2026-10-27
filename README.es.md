# ⚡ Motor de Búsqueda Semántica de Alto Rendimiento con FAISS y Sentence-Transformers

🌐 **Idiomas / Languages:** [🇪🇸 Versión en Español](README.es.md) | [🇬🇧 English Version](README.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FAISS](https://img.shields.io/badge/FAISS-Búsqueda%20Vectorial-green)
![HuggingFace](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?logo=huggingface)
![Licencia](https://img.shields.io/badge/Licencia-MIT-lightgrey)

Implementación lista para producción de indexación por similitud semántica y búsqueda vectorial densa utilizando **FAISS (Facebook AI Similarity Search)** de Meta AI y embeddings preentrenados basados en Transformers.

---

## 📌 Arquitectura del Sistema

A diferencia de los motores de búsqueda léxicos tradicionales (como BM25 o TF-IDF), que dependen de la coincidencia exacta de palabras clave, este proyecto implementa un pipeline de recuperación vectorial densa capaz de capturar el significado, los sinónimos y el contexto semántico:

```
[Ingesta de Texto] ➡️ [Embeddings con Transformers] ➡️ [Casting FP32] ➡️ [Índice FAISS IndexFlatL2] ➡️ [Recuperación Sub-milisegundo]
```

1. **Generación de Embeddings:** Utiliza el modelo `all-MiniLM-L6-v2` de Sentence-Transformers para mapear texto a un espacio vectorial continuo de 384 dimensiones.
2. **Indexación Vectorial Densa:** Implementa `faiss.IndexFlatL2` para el cálculo exacto de distancia euclídea ($L_2$), optimizado nativamente en C++ para máxima velocidad de cómputo.
3. **Resolución de Consultas en Tiempo Real:** Vectoriza la consulta del usuario al vuelo, calcula los vecinos más cercanos ($k$-NN) y entrega los resultados semánticos con latencia inferior a 5 milisegundos.

---

## 🚀 Características Principales

* **Búsqueda Sub-milisegundo:** Búsqueda exacta de vecinos más cercanos en $< 5\text{ ms}$ en hardware estándar (CPU).
* **Integración con Transformers:** Compatible con modelos multilingües y específicos de dominio de Hugging Face.
* **Gestión Eficiente de Memoria:** Conversión estricta a formato flotante de 32 bits (`float32`) para alineación directa con el motor C++ de FAISS, evitando sobrecarga de memoria.
* **Interfaz de Consola Interactiva (CLI):** Búsqueda interactiva en tiempo real con reporte inmediato de distancia matemática y tiempo exacto de ejecución en milisegundos.

---

## 🛠️ Tecnologías y Requisitos

* **Python 3.10+**
* **FAISS-CPU** (`faiss-cpu==1.8.0`)
* **Sentence-Transformers** (`sentence-transformers>=2.2.0`)
* **NumPy** (`numpy>=1.24.0`)
* **PyTorch**

---

## ⚙️ Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/JulioRubio6767/faiss-semantic-search-engine.git
cd faiss-semantic-search-engine
```

### 2. Crear y activar el entorno virtual
```bash
python -m venv .venv
# En Linux/macOS:
source .venv/bin/activate
# En Windows:
.venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 💻 Ejecución

Inicia el motor de búsqueda ejecutando:
```bash
python main.py
```

### Ejemplo de Búsqueda en Vivo:
```text
--- SISTEMA DE BÚSQUEDA SEMÁNTICA (FAISS 1.8) ---
Vectorizando base de conocimiento...
Índice listo con 5 frases cargadas.

------------------------------------------------------------
¿Qué quieres buscar?: computación del futuro y física
Buscando en la base vectorial...

RESULTADO:
-> Los algoritmos cuánticos prometen cambiar la forma en que resolvemos problemas complejos.
Distancia: 0.8412 | Tiempo: 2.14 ms
```

---

## 📈 Hoja de Ruta de Escalabilidad (De Prototipo a RAG Empresarial)

* [ ] Implementar índices con particionado por clusters (`faiss.IndexIVFFlat`) para escalar a millones de documentos.
* [ ] Exponer el motor a través de una API REST de alto rendimiento con **FastAPI**.
* [ ] Integrar división semántica de documentos extensos (PDFs, TXT) mediante chunking recursivo.
* [ ] Soporte para similitud por coseno mediante producto escalar normalizado (`faiss.IndexFlatIP`).

---

## 👤 Autor
**Julio Andrés Rubio Echeverría**  
*Analytics & Machine Learning Engineer*  
* [LinkedIn](https://www.linkedin.com/in/juliorubio-data/)
* [GitHub](https://github.com/JulioRubio6767)
