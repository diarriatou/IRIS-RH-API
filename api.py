from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import shutil
import tempfile
from typing import List, Dict, Optional, Any
import glob
import json
from datetime import datetime

app = FastAPI(title="IRIS-RH API")

# Configure static file serving for CV documents
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/cv-files", StaticFiles(directory=UPLOAD_DIR), name="cv-files")

# Initialize ChromaDB clients
chroma_client = chromadb.PersistentClient(path="chromadb-vdb")
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create collections
main_collection = chroma_client.get_or_create_collection(
    name="cv_database",
    embedding_function=embedding_function,
)

interview_collection = chroma_client.get_or_create_collection(
    name="interview_cv",
    embedding_function=embedding_function,
)

# Initialize LLM
llm = Ollama(
    model="llama2",
    base_url="http://18.130.190.88:11434",
    timeout=120,
)

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Enhanced prompt templates
profile_search_template = """
Context: {context}

Question: Looking for the best candidate for the position of {position}. Consider their skills, experience, and suitability for the role.

Based on the CV data provided in the context, identify and rank the best candidates for this position.
Provide a detailed analysis for each candidate including:
1. Relevant skills and technologies
2. Years of experience
3. Key achievements
4. Overall fit for the position
5. A score out of 100 for role suitability
6. Identify any personal or sensitive information
    -replace it with appropriate markers ([NAME], [EMAIL], etc.)
    -Provide a mapping of the replacements made

Response format should be structured and clear.
"""

interview_analysis_template = """
Based on the CV provided, please analyze the following aspects:

CV Content: {context}

Please provide a structured analysis including:
1. Detailed analysis of technical skills and expertise
2. Work experience summary and key achievements
3. Education and certifications
4. Strengths and potential areas for improvement
5. Overall evaluation score out of 100 with justification

Also generate 10 technical quiz questions based on their expertise, with:
- Question
- 4 multiple choice options
- Correct answer
- Difficulty level (Easy/Medium/Hard)

Format the questions as JSON.

Response:
"""

quiz_evaluation_template = """
Based on the candidate's responses to the technical quiz:

Original CV analysis: {cv_analysis}
Quiz questions and answers: {quiz_responses}

Please provide:
1. Number of correct answers
2. Analysis of knowledge areas where the candidate performed well
3. Areas that need improvement
4. Updated overall score considering both CV and quiz performance
5. Final recommendation

Response:
"""

# Pydantic models
class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    difficulty: str

class QuizResponse(BaseModel):
    question_id: int
    selected_answer: str

class InterviewAnalysis(BaseModel):
    cv_id: str
    cv_url: str
    analysis: str
    score: int
    quiz_questions: List[QuizQuestion]

class SearchResult(BaseModel):
    candidates: List[Dict[str, Any]]
    analysis: str

class QuizSubmission(BaseModel):
    cv_id: str
    responses: List[QuizResponse]

def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return its URL path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return f"/cv-files/{filename}"

def process_document(file_path: str) -> str:
    """Process a document (PDF or Word) and return its content."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return ' '.join([page.page_content for page in pages])
    elif file_path.endswith(('.docx', '.doc')):
        loader = Docx2txtLoader(file_path)
        return loader.load()[0].page_content
    else:
        raise ValueError("Unsupported file format")

def add_document_to_vectorstore(text: str, document_id: str, cv_url: str, collection):
    """Add a document to the specified vector store collection."""
    chunks = text_splitter.split_text(text)
    collection.add(
        documents=chunks,
        ids=[f"{document_id}-chunk-{i}" for i in range(len(chunks))],
        metadatas=[{
            "document_id": document_id,
            "cv_url": cv_url
        } for _ in range(len(chunks))]
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the application by processing all CVs in the documents folder."""
    cv_directory = "documents"
    if not os.path.exists(cv_directory):
        os.makedirs(cv_directory)
    
    # Process all CV files
    cv_files = glob.glob(os.path.join(cv_directory, "*.[pd][do][cf]*"))
    for cv_file in cv_files:
        try:
            # Copy file to uploads directory
            filename = os.path.basename(cv_file)
            cv_url = f"/cv-files/{filename}"
            shutil.copy2(cv_file, os.path.join(UPLOAD_DIR, filename))
            
            text = process_document(cv_file)
            add_document_to_vectorstore(text, filename, cv_url, main_collection)
        except Exception as e:
            print(f"Error processing {cv_file}: {str(e)}")

@app.post("/upload-cv/")
async def upload_cv(file: UploadFile):
    """Upload a CV to the main collection."""
    try:
        cv_url = save_uploaded_file(file)
        file_path = os.path.join(UPLOAD_DIR, os.path.basename(cv_url))
        
        text = process_document(file_path)
        add_document_to_vectorstore(text, file.filename, cv_url, main_collection)
        
        return {
            "message": f"CV {file.filename} uploaded successfully",
            "cv_url": cv_url
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/prepare-interview/")
async def prepare_interview(file: UploadFile) -> InterviewAnalysis:
    """Upload a CV for interview preparation and analyze it."""
    try:
        # Clear previous interview CV
        interview_collection.delete(where={})
        
        # Save and process CV
        cv_url = save_uploaded_file(file)
        file_path = os.path.join(UPLOAD_DIR, os.path.basename(cv_url))
        text = process_document(file_path)
        
        # Add to both collections
        add_document_to_vectorstore(text, file.filename, cv_url, interview_collection)
        add_document_to_vectorstore(text, file.filename, cv_url, main_collection)
        
        # Generate interview analysis
        prompt = ChatPromptTemplate.from_template(interview_analysis_template)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": text})
        
        # Parse response
        analysis_parts = response.split("JSON")
        analysis_text = analysis_parts[0]
        
        # Extract score
        score = 0
        for line in analysis_text.split('\n'):
            if "score" in line.lower():
                try:
                    score = int(line.split("/")[0].split()[-1])
                except:
                    continue
        
        # Parse quiz questions
        try:
            quiz_json = analysis_parts[1].strip()
            quiz_data = json.loads(quiz_json)
            quiz_questions = [
                QuizQuestion(
                    question=q["question"],
                    options=q["options"],
                    correct_answer=q["correct_answer"],
                    difficulty=q["difficulty"]
                )
                for q in quiz_data
            ]
        except:
            # Fallback if JSON parsing fails
            quiz_questions = []
        
        return InterviewAnalysis(
            cv_id=file.filename,
            cv_url=cv_url,
            analysis=analysis_text,
            score=score,
            quiz_questions=quiz_questions
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/submit-quiz/")
async def submit_quiz(submission: QuizSubmission):
    """Submit quiz responses and get evaluation."""
    try:
        # Retrieve original interview analysis
        results = interview_collection.query(
            query_texts=["interview analysis"],
            where={"document_id": submission.cv_id},
            n_results=1
        )
        
        if not results["documents"]:
            raise HTTPException(status_code=404, detail="Interview CV not found")
        
        # Generate evaluation
        prompt = ChatPromptTemplate.from_template(quiz_evaluation_template)
        chain = prompt | llm | StrOutputParser()
        
        evaluation = chain.invoke({
            "cv_analysis": results["documents"][0][0],
            "quiz_responses": json.dumps(submission.dict())
        })
        
        return {
            "evaluation": evaluation,
            "cv_url": results["metadatas"][0][0]["cv_url"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/search-profiles/{position}")
async def search_profiles(position: str) -> SearchResult:
    """Search for the best profiles for a given position."""
    try:
        results = main_collection.query(
            query_texts=[position],
            n_results=3
        )
        
        prompt = ChatPromptTemplate.from_template(profile_search_template)
        chain = prompt | llm | StrOutputParser()
        
        context = "\n".join(results["documents"][0])
        analysis = chain.invoke({
            "context": context,
            "position": position
        })
        
        candidates = [
            {
                "document_id": meta["document_id"],
                "cv_url": meta["cv_url"],
                "relevance": dist
            }
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]
        
        return SearchResult(
            candidates=candidates,
            analysis=analysis
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint for API health check."""
    return {"status": "OK", "message": "HR Automation API is running"}