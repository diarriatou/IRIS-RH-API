import streamlit as st
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import base64
from langchain_community.llms import Ollama
# from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Configuration de Chromadb
chroma_client = chromadb.PersistentClient(path="chromadb-vdb")
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_vdb = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_function,
)

# Configuration du modèle Ollama
llm = Ollama(
    model="llama2",
    base_url="http://18.130.190.88:11434",  # URL explicite
    timeout=120,  # Augmentation du timeout
)


# Configuration du text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Template pour le prompt
prompt_template = """
Contexte: {context}

Question: {question}

Répondez à la question en utilisant uniquement le contexte fourni.
Si vous ne pouvez pas répondre à la question à partir du contexte, dites-le clairement.

Réponse:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
parser = StrOutputParser()

# Chaîne de traitement
chain = prompt | llm | parser

def process_document(file):
    """Traite un document (PDF ou Word) et retourne son contenu."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.getvalue())
        file_path = temp_file.name

    try:
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = ' '.join([page.page_content for page in pages])
        elif file.name.endswith('.docx') or file.name.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
            text = loader.load()[0].page_content
        else:
            raise ValueError("Format de fichier non supporté")
        
        return text
    finally:
        os.unlink(file_path)

def add_document_to_vectorstore(text, document_id):
    """Ajoute un document à la base de données vectorielle."""
    chunks = text_splitter.split_text(text)
    
    # Ajout des chunks à Chroma
    chroma_vdb.add(
        documents=chunks,
        ids=[f"{document_id}-chunk-{i}" for i in range(len(chunks))],
        metadatas=[{"document_id": document_id} for _ in range(len(chunks))]
    )

def search_documents(query):
    """Recherche les documents pertinents."""
    results = chroma_vdb.query(
        query_texts=[query],
        n_results=3
    )
    return results

def main():
    st.set_page_config(layout="wide")
    st.title("Assistant documentaire RAG avec Ollama")
    
    # Zone de téléchargement de documents
    st.subheader("Téléchargement de documents")
    uploaded_file = st.file_uploader(
        "Téléchargez un document (PDF ou Word)",
        type=['pdf', 'docx', 'doc']
    )
    
    if uploaded_file:
        with st.spinner("Traitement du document..."):
            try:
                text = process_document(uploaded_file)
                document_id = uploaded_file.name
                add_document_to_vectorstore(text, document_id)
                st.success(f"Document '{document_id}' ajouté avec succès!")
            except Exception as e:
                st.error(f"Erreur lors du traitement du document: {str(e)}")
    
    # Zone de chat
    st.subheader("Zone de discussion")
    user_question = st.text_input("Posez votre question :")
    
    if user_question:
        with st.spinner("Recherche en cours..."):
            # Recherche des documents pertinents
            results = search_documents(user_question)
            
            if results["documents"]:
                # Préparation du contexte
                context = "\n".join(results["documents"][0])
                
                # Génération de la réponse
                response = chain.invoke({
                    "context": context,
                    "question": user_question
                })
                
                st.markdown("### Réponse:")
                st.markdown(response)
            else:
                st.warning("Aucun document pertinent trouvé pour votre question.")

if __name__ == "__main__":
    main()