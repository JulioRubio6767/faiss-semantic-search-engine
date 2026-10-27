import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import os

# PROYECTO: Búsqueda Semántica con FAISS
# UNIR - Integración y operativización de modelos generativos

def limpiar_consola():
    # A veces falla en algunos terminales, pero para el entorno local de UNIR va bien
    os.system('cls' if os.name == 'nt' else 'clear')

def ejecutar_sistema():
    limpiar_consola()
    print("--- SISTEMA DE BÚSQUEDA SEMÁNTICA (FAISS 1.8) ---")
    print("Iniciando componentes del sistema...\n")
    
    # Base de conocimiento oficial de la actividad
    # Al principio pensé en usar un archivo .txt externo, pero para 5 frases no valía la pena.
    frases = [
        "El amanecer en la montaña revela colores que no existen en la ciudad.",
        "Los algoritmos cuánticos prometen cambiar la forma en que resolvemos problemas complejos.",
        "Un perro corre feliz por la playa mientras las olas rompen suavemente.",
        "La historia del arte está llena de secretos que aún no hemos descubierto.",
        "El café colombiano tiene un aroma que despierta recuerdos de infancia."
    ]
    
    # Usamos all-MiniLM-L6-v2 porque es el más ligero y rápido para este ejercicio
    print("Cargando modelo SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generación de Embeddings
    # Tuve un problema aquí porque FAISS me pedía float32 obligatoriamente
    print("Vectorizando base de conocimiento...")
    embeddings = model.encode(frases)
    embeddings = np.array(embeddings).astype('float32')
    
    # Construcción del índice FAISS
    # IndexFlatL2 es lo más sencillo para buscar por distancia euclídea
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print(f"Índice listo con {index.ntotal} frases cargadas.\n")

    while True:
        print("------------------------------------------------------------")
        consulta = input("¿Qué quieres buscar? (escribe 'salir' para terminar): ")
        
        if consulta.lower() in ['salir', 'exit', 'quit']:
            print("Cerrando el programa. Éxito con la actividad.")
            break

        if not consulta.strip():
            continue

        print(f"Buscando en la base vectorial...")
        start = time.time()
        
        # Vectorizamos lo que puso el usuario
        q_vector = model.encode([consulta]).astype('float32')
        
        # Buscamos el k=1 (el más parecido)
        dist, ids = index.search(q_vector, k=1)
        end = time.time()
        
        res_idx = ids[0][0]
        score = dist[0][0]
        t_ms = (end - start) * 1000

        print("\n" + "RESULTADO:")
        print(f"-> {frases[res_idx]}")
        print(f"Distancia: {score:.4f} | Tiempo: {t_ms:.2f} ms\n")

if __name__ == "__main__":
    try:
        ejecutar_sistema()
    except Exception as e:
        # Error genérico por si falla la carga del modelo o la memoria
        print(f"Algo salió mal: {e}")
