# IRIS-RH API

**IRIS-RH** est une API intelligente dédiée à l'automatisation et à l'optimisation du processus de recrutement. Ce projet permet d'extraire automatiquement le contenu des CV (PDF ou DOCX), de générer des analyses d'entretien et des quiz techniques, et de stocker les données sous forme d'embeddings dans une base vectorielle (ChromaDB). Il intègre également des modèles de langage pour fournir des évaluations et recommandations personnalisées.

---

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Technologies utilisées](#technologies-utilisées)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Endpoints de l'API](#endpoints-de-lapi)
- [Structure du projet](#structure-du-projet)
- [Contributions](#contributions)
- [Licence](#licence)

---

## Fonctionnalités

- **Upload de CV**  
  Permet d'uploader des documents au format PDF ou DOCX pour être analysés.

- **Traitement de documents**  
  Extraction du texte via des loaders spécialisés (PyPDFLoader et Docx2txtLoader) et découpage en morceaux (avec RecursiveCharacterTextSplitter).

- **Stockage vectoriel**  
  Stocke les fragments de CV dans ChromaDB avec des embeddings générés par le modèle "all-MiniLM-L6-v2".

- **Préparation d'entretien**  
  Génère une analyse d'entretien et 10 questions techniques (avec réponses multiples) à partir du contenu du CV, en utilisant un modèle LLM via Ollama.

- **Évaluation de quiz**  
  Permet de soumettre les réponses au quiz et d'obtenir une évaluation globale intégrant l'analyse du CV et les performances au quiz.

- **Recherche de profils**  
  Recherche et classe les meilleurs candidats pour un poste donné en utilisant les données stockées dans ChromaDB et un prompt personnalisé.

---

## Technologies utilisées

### Backend & API
- **FastAPI** : Création d'une API performante pour le traitement des CV.
- **Uvicorn** : Serveur ASGI léger pour exécuter FastAPI.

### Extraction & Analyse de Documents
- **PyPDF2** : Extraction du texte à partir des fichiers PDF.
- **PyPDFLoader** et **Docx2txtLoader** : Chargement de documents PDF et DOCX.
- **RecursiveCharacterTextSplitter** : Découpage du texte en morceaux exploitables.

### Stockage Vectoriel
- **ChromaDB** : Base de données vectorielle pour stocker les embeddings.
- **SentenceTransformerEmbeddingFunction** : Génération d'embeddings via le modèle "all-MiniLM-L6-v2".

### Modèles de Langage (LLM)
- **Ollama** (via `langchain_community.llms`) : Génération d'analyses d'entretien et de quiz techniques.
- **ChatPromptTemplate** et **StrOutputParser** : Création et traitement de prompts pour l'LLM.

### Divers
- **Pydantic** : Validation et gestion des données (modèles de requêtes/réponses).
- **StaticFiles** : Service de fichiers statiques pour l'accès aux CV uploadés.
- **OS, shutil, glob, datetime, json** : Gestion des fichiers, opérations système et manipulation de données.

---

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip

### Étapes d'installation

1. **Cloner le dépôt :**

   ```bash
   git clone https://github.com/diarriatou/iris-rh.git
   cd iris-rh
