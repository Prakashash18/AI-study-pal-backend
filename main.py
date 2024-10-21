from fastapi import FastAPI, HTTPException, Depends, Path, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor
# from langchain_community.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
# from langchain_community.tools.retriever import create_retriever_tool
from enum import Enum

import os
import json
import dotenv
import logging
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import langgraph components
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langgraph_p import app as langgraph_app  # Assuming you rename langgraph-pk.py to langgraph_p.py for import
import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


# Load environment variables
dotenv.load_dotenv()

# Initialize Firebase
firebase_credentials_str = os.getenv("FIREBASE_CREDENTIALS")
firebase_credentials = json.loads(firebase_credentials_str)
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'intelligent-study-buddy.appspot.com'
})

# Initialize Firestore
db = firestore.client()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Set up OpenAI and Google API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")

# Set up embeddings and LLM
embeddings = OpenAIEmbeddings()
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.environ['GEMINI_API_KEY'],
    max_tokens=800,
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
)

class QuestionRequest(BaseModel):
    question: str
    selectedText: str | None = None
    chat_history: List[Dict]

async def create_or_load_chroma_vectorstore(user_id: str, filename: str, persist_directory: str):
    vectorstore_path = os.path.join(persist_directory, f"{user_id}_{filename}")
    
    if os.path.exists(vectorstore_path):
        # Load existing vectorstore
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    else:
        # Create new vectorstore
        bucket = storage.bucket()
        blob = bucket.blob(f'users/{user_id}/pdfs/{filename}')
        
        # Download PDF to a temporary file
        _, temp_filename = tempfile.mkstemp(suffix='.pdf')
        blob.download_to_filename(temp_filename)
        
        # Load and split the PDF
        loader = PyPDFLoader(temp_filename)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create and persist the vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=vectorstore_path
        )
        vectorstore.persist()
        
        # Clean up temporary file
        os.remove(temp_filename)
    
    return vectorstore

# Add this function before the route handler
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@app.post("/ask/{file_name}")
async def ask_question(
    file_name: str = Path(..., description="The name of the file to ask about"),
    request: QuestionRequest = Body(...),
    user: str = Depends(get_current_user)
):
    return await gemini_response_generator(file_name, request.question, request.selectedText, request.chat_history, user)

async def gemini_response_generator(file_name: str, question: str, selected_text: str | None, chat_history: List[dict], user: str):
    user_id = user

    try:
        vectorstore = await create_or_load_chroma_vectorstore(user_id, file_name, "chroma_db")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing Chroma vectorstore: {e}")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Combine question and selected text if available
    full_query = f"{question}\n\nSelected Text: {selected_text}" if selected_text else question
    relevant_docs = retriever.get_relevant_documents(full_query)

    # Prepare chat history messages
    chat_history_messages = []
    for item in chat_history:
        if 'role' not in item or 'content' not in item:
            logger.warning(f"Skipping invalid chat history item: {item}")
            continue
        if item['role'] == MessageRole.USER.value:
            chat_history_messages.append(HumanMessage(content=item['content']))
        elif item['role'] == MessageRole.ASSISTANT.value:
            chat_history_messages.append(AIMessage(content=item['content']))
        else:
            logger.warning(f"Unknown role in chat history item: {item}")

    # Save user's question to chat history
    save_chat_history({
        'user_id': user_id,
        'filename': file_name,
        'role': MessageRole.USER.value,
        'content': question,
        'selectedText': selected_text
    })

    # Prepare variables to collect streamed response
    full_response = []
    web_answer = []
    web_links = []
    youtube_links = []

    # Prepare the state for langgraph
    state = {
        "question": full_query,
        "chat_history": [msg.content for msg in chat_history_messages],
        "notes": [doc.page_content for doc in relevant_docs],
        "num_steps": 0,
    }

    # Adjust this function to properly stream the data
    async def stream_response():
        nonlocal full_response, web_answer, web_links, youtube_links
        try:
            for result in langgraph_app.stream(state):
                for key, value in result.items():
                    logger.info(f"Streaming {key}: {value}")
                    if key == "student_question_cat":
                        yield f"data: {json.dumps({'type': 'category', 'data': value})}\n\n"
                    elif key == "notes_reference":
                        yield f"data: {json.dumps({'type': 'notes', 'data': value})}\n\n"
                    elif key == "web_reference":
                        yield f"data: {json.dumps({'type': 'web', 'data': value})}\n\n"
                    elif key == "youtube_reference":
                        yield f"data: {json.dumps({'type': 'youtube', 'data': value})}\n\n"
                    elif key == "summary_reference":
                        yield f"data: {json.dumps({'type': 'summary', 'data': value})}\n\n"

                    # Collect the streamed response
                    if key == 'summary_reference' and 'final_answer' in value:
                        final_answer = value['final_answer']
                        full_response.append(final_answer.get('Final Answer', ''))
                        web_answer.append(final_answer.get('Web Answer', ''))
                        web_links.extend(final_answer.get('Web Links', []))
                        youtube_links.extend(final_answer.get('YouTube Links', []))

            # Indicate the end of the stream
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # Create the streaming response with the correct media type
    streaming_response = StreamingResponse(stream_response(), media_type="text/event-stream")
    
    # Save AI's response to chat history after streaming is complete
    async def save_ai_response():
        try:
            combined_response = "".join(full_response)
            combined_web_answer = "".join(web_answer)
            
            if not combined_response:
                logger.warning("AI response is empty")
                combined_response = "I apologize, but I couldn't generate a response. Please try asking your question again."

            save_chat_history({
                'user_id': user_id,
                'filename': file_name,
                'role': MessageRole.ASSISTANT.value,
                'content': combined_response,
                'tags': ['AI Response'],
                'selectedText': selected_text,
                'finalAnswer': combined_response,
                'webAnswer': combined_web_answer,
                'webLinks': web_links,
                'youtubeLinks': youtube_links
            })
            logger.info(f"Saved AI response: {combined_response[:100]}...")
        except Exception as e:
            logger.error(f"Error in save_ai_response: {e}")

    # Add the background task to the streaming response
    streaming_response.background = save_ai_response

    return streaming_response

def clean_up_document_content(document_content):
    return document_content


def load_chat_history(user_id: str, filename: str):
    try:
        # Query the chat_history collection
        chat_history_ref = db.collection('chat_history').document(user_id).collection('files').document(filename)
        doc = chat_history_ref.get()

        if not doc.exists:
            return []

        chat_history = doc.to_dict().get('history', [])

        # Get only the latest 10 messages
        chat_history = chat_history[-10:]

        mapped_history = []
        for message in chat_history:
            mapped_message = {
                "id": message.get("id", str(len(mapped_history) + 1)),
                "sender": message.get("sender"),
                "selectedText": message.get("selectedText", ""),
                "tags": message.get("tags", [])
            }

            if message.get("sender") == MessageRole.ASSISTANT.value:
                mapped_message["content"] = {
                    "final_answer": {
                        "Final Answer": message.get("finalAnswer", ""),
                        "Web Answer": message.get("webAnswer", ""),
                        "Web Links": message.get("webLinks", []),
                        "YouTube Links": message.get("youtubeLinks", []),
                        "Notes Answer": message.get("notes_answer", "")  # Load notes_answer
                    }
                }
            else:
                mapped_message["content"] = message.get("content", "")

            mapped_history.append(mapped_message)

        # logger.info(f"Mapped chat history (latest 10): {mapped_history}")

        return mapped_history
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

# Define MessageRole enum
class MessageRole(Enum):
    USER = 'user'
    ASSISTANT = 'assistant'

def save_chat_history(chat_message):
    user_id = chat_message['user_id']
    file_name = chat_message['filename']
    
    doc_ref = db.collection('chat_history').document(user_id).collection('files').document(file_name)
    doc = doc_ref.get()
    existing_history = doc.to_dict().get('history', []) if doc.exists else []

    new_message = {
        "id": str(len(existing_history) + 1),
        "content": chat_message['content'],
        "sender": chat_message['role'],
    }

    # Add optional fields if they are present and not None
    optional_fields = ['selectedText', 'tags', 'finalAnswer', 'webAnswer', 'webLinks', 'youtubeLinks', 'notes_answer']
    for field in optional_fields:
        if field in chat_message and chat_message[field] is not None:
            new_message[field] = chat_message[field]

    existing_history.append(new_message)
    doc_ref.set({'history': existing_history})

    # Log the saved message for debugging
    logger.debug(f"Saved chat message: {new_message}")

@app.get("/chat_history/{file_name}")
async def load_chat_history_endpoint(
    file_name: str, 
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Extract the user ID from the token
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        user_id = decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    chat_history = load_chat_history(user_id, file_name)
    return JSONResponse(content={"chat_history": chat_history}, media_type='application/json; charset=utf-8')

@app.get("/pdf/{user_id}/{file_name}")
async def get_pdf(
    user_id: str = Path(..., description="The ID of the user"),
    file_name: str = Path(..., description="The name of the PDF file"),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify the token and extract the authenticated user's ID
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        authenticated_user_id = decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    # Verify that the authenticated user matches the user_id in the path
    if authenticated_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        bucket = storage.bucket()
        blob = bucket.blob(f'users/{user_id}/pdfs/{file_name}')

        if not blob.exists():
            raise HTTPException(status_code=404, detail="PDF not found")

        # Create a streaming response
        def iterfile():
            stream = blob.download_as_bytes()
            yield stream

        return StreamingResponse(iterfile(), media_type="application/pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDF: {str(e)}")

# Remove the openai_response_generator function
# Removed: async def openai_response_generator(...)

# Remove the /ask_openai endpoint
# Removed: @app.post("/ask_openai/{file_name}")

@app.post("/ask_stream/{file_name}")
async def ask_question_stream(
    file_name: str = Path(..., description="The name of the file to ask about"),
    request: QuestionRequest = Body(...),
    user: str = Depends(get_current_user)
):
    return StreamingResponse(
        gemini_response_generator_stream(file_name, request.question, request.selectedText, request.chat_history, user),
        media_type="text/event-stream"
    )

async def gemini_response_generator_stream(file_name: str, question: str, selected_text: str | None, chat_history: List[dict], user: str):
    user_id = user

    # Send "AI Study Pal is thinking" message
    yield f"data: {json.dumps({'type': 'thinking', 'data': 'AI Study Pal is thinking...'})}\n\n"

    try:
        vectorstore = await create_or_load_chroma_vectorstore(user_id, file_name, "chroma_db")
    except Exception as e:
        yield f"data: {json.dumps({'error': f'Error initializing Chroma vectorstore: {e}'})}\n\n"
        return

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    full_query = f"{question}\n\nSelected Text: {selected_text}" if selected_text else question
    relevant_docs = await asyncio.to_thread(retriever.get_relevant_documents, full_query)

    chat_history_messages = []
    for item in chat_history:
        if 'role' not in item or 'content' not in item:
            continue
        if item['role'] == MessageRole.USER.value:
            chat_history_messages.append(HumanMessage(content=item['content']))
        elif item['role'] == MessageRole.ASSISTANT.value:
            chat_history_messages.append(AIMessage(content=item['content']))

    await asyncio.to_thread(save_chat_history, {
        'user_id': user_id,
        'filename': file_name,
        'role': MessageRole.USER.value,
        'content': question,
        'selectedText': selected_text
    })

    # Prepare variables to collect streamed response
    full_response = []
    web_answer = []
    web_links = []
    youtube_links = []
    notes_answer = []

    state = {
        "question": full_query,
        "chat_history": [msg.content for msg in chat_history_messages],
        "notes": [doc.page_content for doc in relevant_docs],
        "num_steps": 0,
    }

    try:
        for result in langgraph_app.stream(state):
            for key, value in result.items():
                logger.info(f"Streaming {key}: {value}")
                if key == "student_question_cat":
                    yield f"data: {json.dumps({'type': 'category', 'data': value})}\n\n"
                elif key == "notes_reference":
                    # Include page_number and page_content in the streamed data
                    yield f"data: {json.dumps({'type': 'notes', 'data': { 'found_answer': value['found_answer'], 'notes_answer': value['notes_answer'], 'notes_explanation' : value['notes_explanation'], 'page_number': value['page_number'], 'page_content': value['page_content']}})}\n\n"
                elif key == "web_reference":
                    yield f"data: {json.dumps({'type': 'web', 'data': value})}\n\n"
                elif key == "youtube_reference":
                    yield f"data: {json.dumps({'type': 'youtube', 'data': value})}\n\n"
                elif key == "summary_reference":
                    yield f"data: {json.dumps({'type': 'summary', 'data': value})}\n\n"

                # Collect the streamed response
                if key == 'summary_reference' and 'final_answer' in value:
                    final_answer = value['final_answer']
                    full_response.append(str(final_answer.get('Final Answer', '')))
                    web_answer.append(str(final_answer.get('Web Answer', '')))
                    web_links.extend(final_answer.get('Web Links', []))
                    youtube_links.extend(final_answer.get('YouTube Links', []))
                    notes_answer.extend(final_answer.get("Notes Answer", []))

                await asyncio.sleep(0)  # Allow other tasks to run

        # Indicate the end of the stream with a valid JSON
        yield f"data: {json.dumps({'type': 'done', 'data': '[DONE]'})}\n\n"

        try:
            combined_response = "".join(full_response)
            combined_web_answer = "".join(web_answer)
            
            if not combined_response:
                logger.warning("AI response is empty")
                combined_response = "I apologize, but I couldn't generate a response. Please try asking your question again."

            await asyncio.to_thread(save_chat_history, {
                'user_id': user_id,
                'filename': file_name,
                'role': MessageRole.ASSISTANT.value,
                'content': combined_response,
                'tags': ['AI Response'],
                'selectedText': selected_text,
                'finalAnswer': combined_response,
                'notes_answer': notes_answer,
                'webAnswer': combined_web_answer,
                'webLinks': web_links,
                'youtubeLinks': youtube_links
            })
            logger.info(f"Saved AI response: {combined_response[:100]}...")
        except Exception as e:
            logger.error(f"Error in save_ai_response: {e}")

    except Exception as e:
        logger.error(f"Error in gemini_response_generator_stream: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"



# Add these imports at the top of the file
from fastapi import Query

# Add the QuizSettings model
class QuizSettings(BaseModel):
    quiz_type: str
    level: str
    question_count: int
    file_name: str
    file_content: str

@app.post("/quiz/mcq")
async def create_mcq_quiz(settings: QuizSettings):
    # Process the MCQ quiz settings
    # Here you can implement the logic to handle the MCQ quiz creation
    return {"message": "MCQ quiz created", "settings": settings}

@app.post("/quiz/short_answer")
async def create_short_answer_quiz(settings: QuizSettings):
    # Process the Short Answer quiz settings
    # Here you can implement the logic to handle the Short Answer quiz creation
    return {"message": "Short Answer quiz created", "settings": settings}

# Define the request and response models
class MCQRequest(BaseModel):
    filename: str
    num_questions: int
    difficulty: str

class MCQStructure(BaseModel):
    question: str = Field(description="The question generated")
    options: List[str] = Field(description="List of MCQ options with one correct answer and three incorrect options")
    correct_answer: str = Field(description="The correct option to the question")

class MCQList(BaseModel):
    questions: List[MCQStructure] = Field(description="List of MCQ questions")

@app.post("/generate-mcqs")
async def generate_mcqs_endpoint(request: MCQRequest, user: str = Depends(get_current_user)):
    try:
        mcqs = await generate_mcqs(request.filename, request.num_questions, request.difficulty, user)
        return JSONResponse(content=mcqs, media_type='application/json; charset=utf-8')
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))

async def generate_mcqs(filename, num_questions, difficulty, user):
    # Read the content of the file
    try:
        user_id = user
        vectorstore = await create_or_load_chroma_vectorstore(user_id, filename, "chroma_db")
        collections = vectorstore.get()

        # Ensure collections is a dictionary and contains 'documents'
        if isinstance(collections, dict) and 'documents' in collections:
            content = collections['documents']
        else:
            raise ValueError("Invalid structure for collections: expected a dictionary with 'documents' key.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading ChromaDB file: {str(e)}")

    question_query = "generate questions"

    parser = JsonOutputParser(pydantic_object=MCQList)

    prompt = PromptTemplate(
        template="""Based on the following content, generate {num_questions} multiple-choice questions 
                 with {difficulty} difficulty level. Each question should have one correct answer and three incorrect options. 
                 Return the correct answer at the end of each question. 
                 Content:{content}
                 {format_instructions}
                 """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions(), "content": content, "num_questions": num_questions, "difficulty": difficulty},
    )

    chain = prompt | llm_gemini | parser

    # Change this line to remove await if chain.invoke is not async
    response = chain.invoke({"query": question_query})  # {{ edit_1 }}

    return response

class ExplanationRequest(BaseModel):
    question: str
    options: List[str]
    correctAnswer: str


@app.post("/explain")
async def explain(request: ExplanationRequest, user: str = Depends(get_current_user)):
    """
    Provides an explanation for the given question, options, and correct answer using the existing LLM.

    Args:
        request: The request body containing the question, options, and correct answer.

    Returns:
        A dictionary containing the explanation.
    """
    question = request.question
    options = request.options
    correct_answer = request.correctAnswer

    # Generate the explanation using the existing LLM
    explanation = generate_explanation_with_llm(question, options, correct_answer, user)

    return JSONResponse(content={'explanation': explanation}, media_type='application/json; charset=utf-8')

def generate_explanation_with_llm(question, options, correct_answer, user):
    # Construct the prompt for the LLM
    prompt = PromptTemplate.from_template(f"""
    Question: {question}
    Options: {', '.join(options)}
    Correct Answer: {correct_answer}

    State the {correct_answer} in bold.

    Then in the next paragraph provide a concise and short explanation for why the correct answer is '{correct_answer}'.

    Be friendly, use emojis in your response.
    """)

    chain = prompt | llm_gemini

    response = chain.invoke({"question": question, "options": options, "correct_answer": correct_answer})

    print(response)
    return response.content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")





