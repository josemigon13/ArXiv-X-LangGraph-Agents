# Proyecto: ArXiv & X Agents

# Dependencias necesarias:
import langgraph as lg
import gradio as gr
import cohere
from requests_oauthlib import OAuth1Session  # Para autenticación OAuth 1.0a
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Cargar variables desde .env
load_dotenv()

# Acceder a las variables de entorno

# Claves de X para OAuth 1.0a
API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
# Cohere API KEY
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Inicializar la API de Cohere
cohere_client = cohere.ClientV2(COHERE_API_KEY)

# Método para publicar los tweets en un hilo
def publish_tweet(content_list):
    """
    Publica una lista de tweets como un hilo en X utilizando OAuth 1.0a.
    """
    if not content_list or len(content_list) == 0:
        return "Error: No hay contenido para publicar."

    url = "https://api.twitter.com/2/tweets"
    in_reply_to_status_id = None  # Inicializar el ID del tweet anterior

    for content in content_list:
        payload = {"text": content}
        # Agregar el ID del tweet anterior si no es el primero
        if in_reply_to_status_id:
            payload["reply"] = {"in_reply_to_tweet_id": in_reply_to_status_id}

        try:
            # Crear una sesión autenticada
            twitter = OAuth1Session(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
            response = twitter.post(url, json=payload)
            response.raise_for_status()  # Lanza excepción para códigos HTTP 4xx y 5xx

            # Obtener el ID del tweet recién publicado
            in_reply_to_status_id = response.json().get("data", {}).get("id")
            print("Tweet publicado con éxito:", response.json())

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return f"Error al publicar en X: {str(http_err)}"
        except requests.exceptions.RequestException as req_err:
            print(f"Error de conexión: {req_err}")
            return f"Error de conexión: {str(req_err)}"
        except Exception as e:
            print("Error inesperado:", e)
            return f"Error inesperado: {str(e)}"

    return "Hilo publicado con éxito"
 
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

# Método para dividir texto en tweets de 280 caracteres como máx
def create_thread(content, max_chars=280):
    """
    Divide el contenido en partes de máximo `max_chars` caracteres.
    """
    tweets = []
    while len(content) > max_chars:
        # Encuentra el último espacio antes del límite de caracteres
        split_point = content[:max_chars].rfind(" ")
        if split_point == -1:
            split_point = max_chars
        tweets.append(content[:split_point].strip())
        content = content[split_point:].strip()
    tweets.append(content)
    return tweets


from torch.utils.data import DataLoader
# Método para resumir artículos utilizando Hugging Face
def summarize_articles(articles, batch_size=8):
    """
    Genera un resumen breve de los artículos encontrados usando Pegasus.
    """
    if not articles or len(articles) == 0:
        return "No hay artículos disponibles para resumir."

    try:
        # Inicializar modelo y tokenizer
        model_name = "google/pegasus-xsum"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

        summaries = []

        # Preparamos los datos en un formato para DataLoader
        data = [article['summary'] for article in articles]  # Sólo tomamos el resumen
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        for batch in loader:
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                max_length=512,
                return_tensors="pt",
            ).to(device)
            outputs = model.generate(
                inputs.input_ids, max_length=50, num_beams=5, early_stopping=True
            )
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(batch_summaries)

        return summaries  # Sólo devolvemos el resumen generado
    except Exception as e:
        return [f"Error al generar resúmenes: {str(e)}"]

# Método de consulta a ChromaDB / ArXiv
def get_articles(parameters, state):

    collection = state["collection"]
    query = parameters.get("topic", "")
    
    if not query or len(query.strip()) == 0:
        return ["Error: La consulta no puede estar vacía."]

    try:
        results = search_and_rerank(collection, query)
        if results:
            return results

        # Consultar ArXiv si no hay resultados válidos en ChromaDB
        import requests
        from xml.etree import ElementTree as ET

        url = (
            f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0"
            f"&max_results=10&sortBy=relevance&sortOrder=descending"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        articles = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
            pdf_link = next(
                (link.attrib.get("href") for link in entry.findall("{http://www.w3.org/2005/Atom}link")
                if link.attrib.get("rel") == "related" and link.attrib.get("title") == "pdf"),
                None
            )
            articles.append({"title": title, "summary": summary, "pdf_link": pdf_link})

        if articles:
            add_articles_to_chromadb(collection, articles, query)
            return articles
        else:
            return ["No se encontraron resultados en ArXiv."]
    except Exception as e:
        return [f"Error: {str(e)}"]

# Configuración de la base de datos de vectores con ChromaDB
def setup_chromadb():
    """
    Configura una base de datos de vectores usando ChromaDB con la nueva arquitectura.
    """
    import chromadb

    # Crear un cliente persistente con la nueva API
    try:
        client = chromadb.PersistentClient(path="./chroma_storage")
        # Crear o recuperar la colección donde almacenaremos los datos
        collection = client.get_or_create_collection(name="arxiv_articles")
        return collection
    except Exception as e:
        print("Error al configurar ChromaDB:", str(e))
        raise

import hashlib
import json

# Método para generar IDs únicos para los artículos almacenados en la BBDD
def generate_unique_id(document):
    hash_object = hashlib.sha256(json.dumps(document, sort_keys=True).encode('utf-8'))
    return hash_object.hexdigest()

# Método para agregar artículos a la base de datos de vectores
def add_articles_to_chromadb(collection, articles, query):
    """
    Agrega artículos a la base de datos de vectores para futuras búsquedas.
    """
    try:
        # Inicializar modelo para generar embeddings
        model = SentenceTransformer("sentence-transformers/LaBSE")
        
        # Crear una lista de texto concatenado para los embeddings
        documents = [
            {"title": article["title"], "summary": article["summary"], "pdf_link": article["pdf_link"]}
            for article in articles
        ]
        embeddings = model.encode([f"{doc['title']} - {doc['summary']}" for doc in documents])

        # Generar IDs únicos para cada artículo
        ids = [generate_unique_id(doc) for doc in documents]

        # Crear metadata estructurada para cada documento
        metadatas = [
            {
                "query": query,
                "title": article["title"],
                "summary": article["summary"],
                "pdf_link": article["pdf_link"]
            }
            for article in articles
        ]

        # Agregar documentos, metadatas e IDs a ChromaDB
        collection.add(
            documents=[f"{doc['title']} - {doc['summary']}" for doc in documents],
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

        print(f"Artículos añadidos a ChromaDB con éxito.")
        print(f"Tamaño de la colección actual: {collection.count()}.")
    except Exception as e:
        print("Error al añadir artículos a ChromaDB:", str(e))
        raise

# Función de búsqueda y reranking de artículos almacenados en ChromaDB con la API de Cohere
def search_and_rerank(collection, query, chroma_threshold=1.5, cohere_threshold=0.25):
    """
    Busca en ChromaDB los documentos que cumplen el threshold y realiza un reranking con Cohere.
    """
    try:
        # Generar embedding de la consulta
        model = SentenceTransformer("sentence-transformers/LaBSE")
        query_embedding = model.encode([query])[0]

        # Solicitar hasta 50 resultados a ChromaDB
        results = collection.query(query_embeddings=[query_embedding], n_results=20)
        distances = results["distances"][0]  # Distancias devueltas por ChromaDB
        documents = results["documents"][0]  # Documentos devueltos
        metadatas = results["metadatas"][0]  # Metadatos devueltos

        # Encontrar el índice del primer resultado que no cumple el threshold
        valid_indices = [i for i, distance in enumerate(distances) if distance < chroma_threshold]
        if not valid_indices:
            print("No se encontraron artículos relacionados en la BBDD.")
            return False

        # Obtener solo los documentos válidos con todos los metadatos
        valid_documents = [
            {
                "title": metadatas[i]["title"],
                "summary": metadatas[i]["summary"],
                "pdf_link": metadatas[i]["pdf_link"],
            }
            for i in valid_indices
        ]

        # Preparar documentos para Cohere
        formatted_documents = [
            {"id": f"doc_{i}", "text": f"{doc['title']} - {doc['summary']}"} for i, doc in enumerate(valid_documents)
        ]
        
        # Usar Cohere para rerankear
        response = cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=formatted_documents,
            top_n=10,
        )

        reranked_results = sorted(response.results, key=lambda x: x.relevance_score, reverse=True)

        # Filtrar documentos según el threshold de Cohere
        reranked_docs = [
            valid_documents[result.index]  # Usar el índice para mapear de nuevo a valid_documents
            for result in reranked_results if result.relevance_score > cohere_threshold
        ]

        if not reranked_docs:
            print("No se encontraron artículos relevantes tras el reranking.")
            return False

        return reranked_docs

    except Exception as e:
        print("Error en el proceso de búsqueda y reranking:", str(e))
        raise

# Método para interpretar el mensaje del usuario
def interpret_instruction(input_output, state):
    if state.get("awaiting_selection"):
        return {"action": "awaiting_selection", "parameters": {"input_output": input_output}}
    
    if state.get("awaiting_feedback"):
        return {"action": "awaiting_feedback", "parameters": {"input_output": input_output}}
    
    # Procesar instrucciones principales
    try:
        prompt = f"""
        Eres un asistente para procesar artículos científicos. El usuario puede realizar estas acciones:
        - 'search': Busca artículos basados en un tema.
        - 'publish': Publica artículos seleccionados en X (Twitter).

        Devuelve un JSON con dos campos, sin ningún comentario adicional:
        - 'action': la acción principal (search, publish, none).
        - 'parameters': los parámetros relevantes (como el tema de búsqueda o artículos seleccionados).
        
        Siempre utiliza la clave 'topic' para el parámetro que representa el tema de búsqueda o publicación.
        
        Si el usuario menciona el verbo publicar o similar, devuelve siempre 'publish'; si sólo menciona buscar
        o similar, y no menciona explícitamente publicar, devuelve 'search', y si no menciona ninguna de estas dos
        acciones/verbos, devuelve 'none'.

        Entrada del usuario: '{input_output}'
        """

        # API de Chat de Cohere
        response = cohere_client.chat(
            model="command-r7b-12-2024",
            messages=[{"role": "user", "content": prompt}]
        )

        # Accede al contenido del mensaje del asistente
        # Procesa la lista y extrae el texto del primer elemento
        content_text = response.message.content[0].text.strip()  # Extraer y limpiar el texto
            
        # Elimina delimitadores de código como ```json y ```
        if "```" in content_text:
            content_text = content_text.split("\n", 1)[-1].rsplit("\n", 1)[0]
        
        # Parsear el contenido como JSON
        return json.loads(content_text)
    except Exception as e:
        print("Excepción interpretando la instrucción del usuario: ", str(e))
        return {"action": "error", "parameters": {"message": str(e)}}

# Método auxiliar para interpretar la selección de artículos de una lista por parte del usuario
def interpret_selection(input_output, articles):
    """
    Utiliza un modelo de lenguaje para interpretar la selección de artículos del usuario.
    Args:
        input_output (str): Entrada del usuario.
        articles (list): Lista de artículos disponibles.
    Returns:
        dict: Contiene los índices de los artículos seleccionados, entre otros.
    """
    
    try:
        articles_list = "\n".join([
            f"{i+1}. {article['title']}" if isinstance(article, dict) and 'title' in article else f"{i+1}. Sin título"
            for i, article in enumerate(articles)
        ])

        prompt = f"""
        Eres un asistente para procesar artículos científicos. Tienes una lista de artículos numerados del 1 al {len(articles)}.
        Cada artículo tiene un título y un resumen breve.

        Lista de artículos:
        {articles_list}

        El usuario puede seleccionar artículos usando las siguientes formas:
        - Por números: "1", "1, 2 y 3", etc.
        - Por descripciones como "El primero y el segundo".
        - Por títulos exactos o parciales, como "El artículo cuyo título comienza por Machine Learning".
        - Puede cancelar escribiendo "cancelar" o similar.

        Entrada del usuario: {json.dumps(input_output)}

        Devuelve estrictamente un JSON con la siguiente estructura, sin ningún comentario adicional:
        {{
            "indices": [números de los artículos seleccionados basados en su posición en la lista original, comenzando en 1],
            "cancel": true si el usuario decide cancelar, de lo contrario false
            "warning_flag": true si el usuario no cancela pero tampoco elige ningún artículo para publicar, false en caso
            contrario (cancela explícitamente o elige algún artículo)
        }}
        """
        # API de Chat de Cohere
        response = cohere_client.chat(
            model="command-r7b-12-2024",
            messages=[{"role": "user", "content": prompt}]
        )

        # Accede al contenido del mensaje del asistente
        # Procesa la lista y extrae el texto del primer elemento
        content_text = response.message.content[0].text.strip()  # Extraer y limpiar el texto

        # Elimina delimitadores de código como ```json y ```
        if "```" in content_text:
            content_text = content_text.split("\n", 1)[-1].rsplit("\n", 1)[0]
        
        # Parsear el contenido como JSON
        result = json.loads(content_text)
        
        # Si se cancela, devolver inmediatamente
        if result["cancel"]:
            return {"cancel": True}

        # Si el usuario no selecciona ningún artículo o cancela el proceso,
        # devolver una bandera de advertencia
        if result["warning_flag"]:
            return {"warning_flag": True}
            
        # Convertir índices a 0-based y retornar
        return {"indices": [int(index) - 1 for index in result["indices"]]}
    except Exception as e:
        print("Excepción:", str(e))
        raise ValueError(f"Error al interpretar la selección: {str(e)}")

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from typing import Any, Dict

class State(TypedDict):
    input_output: str         # Entrada del usuario / salida del sistema
    awaiting_selection: bool  # Si el sistema espera selección de artículos
    awaiting_feedback: bool   # Si el sistema espera retroalimentación del usuario
    collection: object        # Base de datos de vectores
    articles: list[dict]      # Lista de artículos encontrados
    tweets: list[str]         # Lista de tweets generados
    action: str               # Próxima acción a realizar
    parameters: Dict[str, Any]# Parámetros de las instrucciones

def setup_state_graph():
    """
    Configura el grafo de estados según el flujo deseado.
    """
    graph = StateGraph(State)

    # Estado inicial
    def instruction_handler(state: State):
        """
        Nodo inicial que interpreta la instrucción del usuario y actualiza el estado.
        """
        instruction = interpret_instruction(state["input_output"], state)
        instruction_action = instruction.get("action")
        action = (
            "handle_search" if instruction_action in ["search", "publish"]
            else "handle_selection" if instruction_action == "awaiting_selection"
            else "handle_feedback" if instruction_action == "awaiting_feedback"
            else None
        )
        return {
            **state,
            "action": action,
            "parameters": instruction.get("parameters", {}),
            "awaiting_selection": instruction_action=="publish" or instruction_action=="awaiting_selection",
            "input_output": (
                "¡Hola!\nSoy un bot encargado de ayudarte a buscar artículos académicos en arXiv, y a publicarlos en X si lo deseas.\n"
                "Por favor, aségurate de que tu consulta sea válida."
                if action is None
                else state["input_output"]
            ),
        }

    # Buscar artículos en ChromaDB / ArXiv
    def search_handler(state: State):
        """
        Nodo de búsqueda: Consulta artículos en ArXiv o ChromaDB.
        """
        parameters = state.get("parameters", {})
        articles = get_articles(parameters, state)
        
        if articles:
            # Formatear los artículos encontrados
            formatted_articles = "\n\n".join(
                [
                    f"- {i + 1} - Título: {article['title']}\n"
                    f"Resumen: {article.get('summary', 'Sin resumen disponible')}\n"
                    f"Enlace: {article.get('pdf_link', 'Sin enlace disponible')}"
                    for i, article in enumerate(articles)
                ]
            )
            input_output = f"Artículos encontrados:\n\n{formatted_articles}"
        else:
            input_output = "Error: No se encontraron artículos."
        return {
            **state,
            "articles": articles or [],
            "input_output": input_output,
        }
    
    # Seleccionar artículos mostrados para publicar
    def selection_handler(state: State):
        """
        Nodo que maneja la selección de artículos por parte del usuario.
        Genera un borrador del hilo de tweets basados en los artículos seleccionados.

        Args:
            state (State): Estado actual del sistema.

        Returns:
            State: Estado actualizado después de manejar la selección.
        """
        input_output = state["input_output"]
        articles = state["articles"]
        # Interpretar la selección del usuario
        result = interpret_selection(input_output, articles)
        
        if "cancel" in result and result["cancel"]:
            return {
                **state,
                "awaiting_selection": False,
                "input_output": "Selección cancelada.",
            }

        if "warning_flag" in result and result["warning_flag"]:
            return {
                **state,
                "input_output": (
                    "Por favor, selecciona algún artículo para publicar o cancela la selección:\n\n"
                    + "\n\n".join(
                        [
                            f"- {i + 1} - Título: {article['title']}\n"
                            f"Resumen: {article.get('summary', 'Sin resumen disponible')}\n"
                            f"Enlace: {article.get('pdf_link', 'Sin enlace disponible')}"
                            for i, article in enumerate(articles)
                        ]
                    )
                ),
            }
            
        # Obtener los índices seleccionados
        indices = result["indices"]
        selected_articles = [articles[i] for i in indices]
        # Generar resúmenes con Pegasus
        summaries = summarize_articles(selected_articles)
        # Preparar tweets combinando título, resumen y enlace
        combined_content = "\n\n".join(
            [
                f"Título: {article['title']}\nResumen: {summary}\nEnlace: {article['pdf_link']}"
                for article, summary in zip(selected_articles, summaries)
            ]
        )
        tweets = create_thread(combined_content)
        # Mensaje al usuario con el borrador
        return {
            **state,
            "awaiting_selection": False,
            "awaiting_feedback": True,
            "tweets": tweets,
            "input_output": (
                "He generado un borrador del hilo basado en tu selección. Revisa el contenido:\n\n"
                + "\n\n".join(tweets)
                + "\n\nEscribe 'OK' para publicar, 'Modificar [nuevo contenido]' o 'Cancelar'."
            ),
        }

    # Dar feedback
    def feedback_handler(state: State):
        """
        Nodo de retroalimentación: Permite al usuario modificar, cancelar o publicar los tweets.
        """
        input_output = state["input_output"]
        if input_output.lower() == "cancelar":
            return {
                **state,
                "awaiting_feedback": False,
                "input_output": "Publicación cancelada.",
            }
        elif input_output.lower() == "ok":
            publish_tweet(state["tweets"])
            return {
                **state,
                "awaiting_feedback": False,
                "input_output": "Publicación realizada con éxito.",
            }
        elif input_output.lower().startswith("modificar"):
            modification = input_output.split("modificar", 1)[-1].strip()
            tweets = create_thread(modification)
            publish_tweet(tweets)
            return {**state,"input_output":"Publicación realizada con éxito","tweets":tweets,"awaiting_feedback":False}
        else:
            return {
                **state,
                "input_output": "Entrada no válida. Escribe 'OK', 'Modificar [nuevo contenido]' o 'Cancelar'.",
            }

    # Agregar nodos al grafo
    graph.add_node("handle_instruction", instruction_handler)
    graph.add_node("handle_search", search_handler)
    graph.add_node("handle_selection", selection_handler)
    graph.add_node("handle_feedback", feedback_handler)

    # Conectar transiciones
    graph.add_edge(START, "handle_instruction")  # Inicio del flujo
    graph.add_conditional_edges("handle_instruction", lambda state: state["action"] if state["action"] else END)
    graph.add_edge("handle_search", END)
    graph.add_edge("handle_selection", END)
    graph.add_edge("handle_feedback", END)

    return graph.compile()

STATE_FILE_PATH = "./state.json"
def load_state():
    """
    Carga el estado desde un archivo JSON.
    Si el archivo no existe, inicializa un estado vacío.
    """
    if os.path.exists(STATE_FILE_PATH):
        with open(STATE_FILE_PATH, "r") as f:
            loaded_state = json.load(f)
            # Asegurar que los datos cargados sean compatibles con State
            return State(
                input_output=loaded_state.get("input_output", ""),
                awaiting_selection=loaded_state.get("awaiting_selection", False),
                awaiting_feedback=loaded_state.get("awaiting_feedback", False),
                collection=None,  # La colección siempre se inicializa al cargar
                articles=loaded_state.get("articles", []),
                tweets=loaded_state.get("tweets", []),
                action=loaded_state.get("action"),
                parameters=loaded_state.get("parameters", [])
            )
    else:
        # Retornar un objeto de la clase State vacío
        return State(
            input_output="",
            awaiting_selection=False,
            awaiting_feedback=False,
            collection=None,  # La colección siempre se inicializa al cargar
            articles=[],
            tweets=[],
            action=None,
            parameters=[]
        )

def save_state(state):
    """
    Guarda el estado en un archivo JSON.
    Excluye la colección (que no puede serializarse) y otros campos no persistentes.
    """
    state_to_save = {k: v for k, v in state.items() if k != "collection"}
    with open(STATE_FILE_PATH, "w") as f:
        json.dump(state_to_save, f)

def interact_with_system(state_graph, userInput):
    """
    Maneja la interacción del usuario con el sistema y actualiza el estado global.
    """
    # Cargar el estado desde el archivo
    state = load_state()

    # Actualizar los campos reinicializados
    state.update({
        "input_output": userInput,
        "collection": setup_chromadb()  # Reiniciar la colección
    })

    # Ejecutar el grafo de estados
    response = state_graph.invoke(state)
    state.update(response)  # Actualizar el estado

    # Guardar el estado actualizado
    save_state(state)

    return state["input_output"]  # Retornar el mensaje actualizado

def interface(state_graph):
    """
    Interfaz gráfica con Gradio.
    """
    def interact_with_ui(input_output):
        return interact_with_system(state_graph, input_output)

    with gr.Blocks() as ui:
        gr.Markdown("# ArXiv & X Agents - Interfaz de usuario")

        with gr.Row():
            query_input=gr.Textbox(label="Consulta",placeholder="Ejemplo: Buscar y publicar artículos sobre Machine Learning")
            ejecutar_btn=gr.Button("Enviar")

        resultados_output = gr.Textbox(label="Respuesta", lines=20)

        ejecutar_btn.click(interact_with_ui, inputs=[query_input], outputs=[resultados_output])

    return ui

if __name__ == "__main__":
    try:
        # Configurar el grafo de estados
        state_graph = setup_state_graph()

        # Lanzar la interfaz con el grafo de estados
        ui = interface(state_graph)
        ui.launch()
    finally:
        # Eliminar el archivo state.json al finalizar
        if os.path.exists(STATE_FILE_PATH):
            os.remove(STATE_FILE_PATH)
            print(f"Archivo {STATE_FILE_PATH} eliminado.")