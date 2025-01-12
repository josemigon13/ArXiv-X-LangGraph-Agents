# Proyecto: ArXiv & X LangGraph Agents

Prototipo basado en agentes de LangGraph para buscar, resumir y publicar artículos científicos de ArXiv en hilos de X (Twitter), utilizando Cohere, Gradio, ChromaDB y modelos de lenguaje avanzados.

## Descripción
**ArXiv & X Agents** es un sistema diseñado para interactuar con artículos científicos de arXiv, permitiendo:
1. **Búsqueda** de artículos relevantes basados en un tema.
2. **Reranking** de resultados utilizando Cohere y ChromaDB para mejorar la relevancia.
3. **Resumir** los artículos usando Pegasus de Hugging Face.
4. **Publicar hilos** en Twitter/X basados en los artículos seleccionados.

El proyecto utiliza un enfoque modular basado en un grafo de estados para gestionar la interacción del usuario y los procesos del sistema.

---

## Cómo usar el proyecto

### **Requisitos previos**
1. Clona el repositorio:
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configura un archivo `.env` en el directorio principal con las siguientes claves:
   ```
   COHERE_API_KEY=<tu_clave_cohere>
   TWITTER_API_KEY=<tu_clave_api_twitter>
   TWITTER_API_SECRET=<tu_secreto_api_twitter>
   TWITTER_ACCESS_TOKEN=<tu_token_acceso_twitter>
   TWITTER_ACCESS_SECRET=<tu_secreto_token_acceso_twitter>
   ```
4. Verifica que tienes las claves necesarias para Cohere y Twitter/X.

### **Ejecutar el proyecto**
1. Ejecuta el script principal:
   ```bash
   python main.py
   ```
2. Abre la interfaz gráfica en el navegador desde el enlace que se mostrará en la consola (por defecto: `http://127.0.0.1:7860`).

---

## Características principales

### **1. Búsqueda de artículos**
- Introduce un tema de interés en la interfaz gráfica.
- El sistema buscará en arXiv y ChromaDB, aplicando un proceso de **reranking** para ofrecer los resultados más relevantes.

### **2. Selección de artículos**
- Tras la búsqueda, selecciona los artículos que deseas incluir en el hilo. Puedes hacerlo:
  - Por números (por ejemplo: `1, 2, 3`).
  - Por descripción textual (por ejemplo: "el primero y el tercero").
  - Por títulos parciales (por ejemplo: "Machine Learning").

### **3. Resumen de artículos**
- El sistema utiliza el modelo **Pegasus-XSum** para generar un resumen breve de los artículos seleccionados.

### **4. Publicación en X/Twitter**
- Revisa el borrador del hilo generado por el sistema.
- Publica directamente o realiza modificaciones antes de publicar.

---

## Estructura de agentes en el grafo

### **Flujo principal**
1. **`handle_instruction`**: Interpreta las instrucciones del usuario (buscar, seleccionar, publicar).
2. **`handle_search`**: Realiza búsquedas en arXiv y ChromaDB.
3. **`handle_selection`**: Gestiona la selección de artículos para generar un hilo.
4. **`handle_feedback`**: Permite al usuario modificar, cancelar o publicar el hilo generado.

### **Conexiones**
- **Inicio (`START`) → `handle_instruction`**: Interpreta la consulta inicial del usuario.
- **`handle_instruction` → `handle_search`**: Si la acción es buscar o publicar.
- **`handle_instruction` → `handle_selection`**: Si se espera selección de artículos listados.
- **`handle_instruction` → `handle_feedback`**: Si se espera retroalimentación por parte del usuario.
- **Todos los nodos → `END`**: Termina el flujo actual y guarda el estado.

---

## Estructura del proyecto

```plaintext
.
├── main.py                # Archivo principal para orquestar el sistema
├── .env                   # Archivo con claves de API (excluido del repositorio)
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Este archivo
```

---

## Notas adicionales
1. **Limpieza del estado:** Al finalizar la ejecución, el archivo `state.json` se elimina automáticamente para evitar conflictos entre ejecuciones.
2. **Modularidad:** El sistema está diseñado para ser extensible; puedes agregar nuevos nodos al grafo según sea necesario.
