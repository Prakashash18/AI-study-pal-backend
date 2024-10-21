import os

from langchain_community.tools import DuckDuckGoSearchResults, YouTubeSearchTool
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# prompt: create chat_llm using openai

chat_llm = ChatOpenAI(temperature=0, streaming=True)

#Categorize student question
prompt = PromptTemplate(
   template="""
    You are an AI assistant designed to categorize student inquiries based on their understanding and the context of their chat history.

    Analyze the following student question, chat history and notes to determine the appropriate response category.

    - If the student requires factual information that might be in their notes or previous explanations, set the output as "refer_to_notes".
    - If the student requires more information or a detailed explanation beyond what's likely in their notes, or if the question contains the web keyword, set the output as "web_search".
    - If the explanations provided do not seem sufficient, if the student explicitly requests videos, or if the topic would benefit from visual explanation, set the output as "video_search".

    Consider the chat history when making your decision. 

    Student Question: "{question}"

    Chat History:
    {chat_history}

    Notes: {notes}

    Based on this analysis, provide the output in JSON format as follows:
    {{
        "output": "<appropriate_category>",
        "reasoning": "<brief explanation of why this category was chosen>"
    }}
    """,
    input_variables=["question", "chat_history", "notes"],
)

tutor_category_generator = prompt | chat_llm | JsonOutputParser()

# QUESTION = """HI there, \n
# What is v=ir ? I still don't understand what it is. \n Any videos ?
# """

# result = tutor_category_generator.invoke({"question": QUESTION, "chat_history":[]})

# print(result)

#Notes Search Tool

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)


prompt_notes = PromptTemplate(
    template="""
    You are an AI assistant designed to carefully check the content from notes and chat history and the student question to see if they are related.

    Analyze the following student question, the chat history, and determine if there are relevant answers in the provided notes.

    Student Question: "{question}"

    Chat History:
    {chat_history}

    Notes: "{notes}"

    Based on this analysis, provide the output in JSON format as follows:
    {{
        "Found Answer": <true_or_false>,
        "Answer": "<brief explanation of how the answer was derived from notes and/or chat history>",
        "Explanation": "<brief explanation of how the answer was derived from notes and/or chat history>"
        "Page Number": <the page number of notes if answer is found, else strictly empty string>
        "Page Content":<the exact content used if answer is found, else strictly empty string>
    }}
    """,
    input_variables=["question", "chat_history", "notes"],
)

notes_reference_tool = prompt_notes | chat_llm | JsonOutputParser()

# QUESTION = """HI there, \n
# What is BJT ?

# """

retriever = index.as_retriever()
# nodes = retriever.retrieve(QUESTION)

# print(nodes)


# result = notes_reference_tool.invoke({"question": QUESTION, "chat_history":[], "notes" : nodes})

# print(result)

## Web search tool

from typing import List, Dict

# Define the prompt for retrieving answers from the web search
prompt_web_search = PromptTemplate(
    template="""
    You are an AI assistant designed to retrieve answers from the web based on student inquiries, while considering their chat history.

    Analyze the following student question, the chat history, and the web search results.

    Student Question: "{question}"

    Chat History:
    {chat_history}

    Web Search Results: "{web_results}"

    Based on this analysis, provide the output in JSON format as follows:
    {{
        "Found Answer": <true_or_false>,
        "Answer": "<comprehensive_answer_if_found, incorporating relevant information from chat history and web results, else empty string>",
        "Web Links": [
            {{
                "title": "<title_of_webpage>",
                "link": "<url_of_webpage>"
            }},
            ...
        ],
        "Explanation": "<brief explanation of how the answer relates to the chat history and current question>"
    }}
    """,
    input_variables=["question", "chat_history", "web_results"],
)

# Function to retrieve from DuckDuckGo
import re

def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    search = DuckDuckGoSearchResults()
    results_str = search.invoke(query, max_results=max_results)
    parsed_results = []
    
    # Define a regex pattern to match 'snippet', 'title', and 'link' fields
    pattern = r'snippet:\s*(.*?),\s*title:\s*(.*?),\s*link:\s*(https?://\S+)(?:,|$)'
    
    # Use finditer to get all matches
    matches = re.finditer(pattern, results_str, re.DOTALL)
    
    for match in matches:
        snippet = match.group(1).strip()
        title = match.group(2).strip()
        link = match.group(3).strip()
        parsed_results.append({"title": title, "link": link, "snippet": snippet})
    
    return parsed_results


# Web search tool
web_reference_tool = prompt_web_search | chat_llm | JsonOutputParser()

# # Student question
# QUESTION = """HI there, \n
# What is BJT?
# """

# # Perform DuckDuckGo web search
# web_results = search_web(QUESTION)

# # Use the prompt to generate the response
# result = web_reference_tool.invoke({
#     "question": QUESTION,
#     "chat_history": [],
#     "web_results": web_results
# })

## Youtube Search tool
from typing import List

# Define the prompt for retrieving answers from YouTube search results
prompt_youtube_search = PromptTemplate(
    template="""
    You are an AI assistant designed to retrieve answers from YouTube videos based on student inquiries, while considering their chat history.

    Analyze the following student question, chat history, and the YouTube video links.

    Student Question: "{question}"

    Chat History:
    {chat_history}

    YouTube Video Links:
    {youtube_links}

    Based on this information, provide the output in JSON format as follows:
    {{
        "Found Answer": <true_or_false>,
        "Answer": "<brief_summary_of_how_the_videos_relate_to_the_question_and_chat_history>",
        "YouTube Links": [
            {{
                "link": "<youtube_video_link>",
                "relevance": "<brief explanation of video relevance to question and chat history>"
            }},
            ...
        ]
    }}
    """,
    input_variables=["question", "chat_history", "youtube_links"],
)

# Function to retrieve YouTube search results
def search_youtube(query: str, max_results: int = 3) -> List[str]:
    youtube_tool = YouTubeSearchTool()
    results_str = youtube_tool.run(query)

    # Print the raw results to understand the format
    print("YouTube Results (raw):")
    print(results_str)

    # Assuming results_str is a newline-separated string of YouTube video links
    video_links = results_str.strip().split('\n')
    print("Parsed Video Links:", video_links)  # Debugging output

    # Check the type of video_links
    print("Type of Parsed Video Links:", type(video_links))  # Should be List[str]
    for link in video_links:
        print("Type of each link:", type(link))  # Should be str

    return video_links[:max_results]

# YouTube search tool
youtube_reference_tool = prompt_youtube_search | chat_llm | JsonOutputParser()

# # Student question
# QUESTION = """HI there
# What is BJT?
# """

# # Perform YouTube search
# youtube_results = search_youtube(QUESTION)

# # Use the prompt to generate the response
# result = youtube_reference_tool.invoke({
#     "question": QUESTION,
#     "chat_history": [],
#     "youtube_links": youtube_results
# })



# Define the final prompt template for summarizing the answer
prompt_summary = PromptTemplate(
    template="""
    You are an AI assistant designed to summarize and present answers from various sources, including notes, web search, and YouTube videos, in response to a student's question.

    Here is the student's inquiry, chat history, and the collected information from notes, web, and YouTube searches:

    Student Question: "{question}"

    Chat History: "{chat_history}"

    Answer from Notes: "{notes_answer}"

    Answer from Web Search: "{web_answer}"

    Web Links:
    {web_links}

    YouTube Video Links:
    {youtube_links}

    Based on this information, provide the output in JSON format as follows:
    {{
        "Final Answer": "<best_answer_based_on_sources>",
        "Notes Answer": "<answer_from_notes>",
        "Web Answer": "<answer_from_web>",
        "Web Links": [
            "<url_of_webpage>",
            ...
        ],
        "YouTube Links": [
            "<youtube_video_link>",
            ...
        ]
    }}
    """,
    input_variables=["question", "chat_history", "notes_answer", "web_answer", "web_links", "youtube_links"],
)

# Define the summary tool
summary_tool = prompt_summary | chat_llm | JsonOutputParser()

# # Gather results from previous tools
# notes_result = notes_reference_tool.invoke({
#     "question": QUESTION,
#     "chat_history": [],
#     "notes": nodes
# })

# web_result = web_reference_tool.invoke({
#     "question": QUESTION,
#     "chat_history": [],
#     "web_results": web_results
# })

# youtube_result = youtube_reference_tool.invoke({
#     "question": QUESTION,
#     "chat_history": [],
#     "youtube_links": youtube_results
# })

# # Prepare YouTube results for the final prompt
# youtube_links = [
#     {
#         "link": video["link"],
#         "explanation": video["explanation"]
#     }
#     for video in youtube_result.get("YouTube Links with Explanation", [])
# ]

# # Use the summary tool to generate the final output
# final_result = summary_tool.invoke({
#     "question": QUESTION,
#     "chat_history": [],
#     "notes_answer": notes_result.get("Answer", ""),
#     "web_answer": web_result.get("Answer", ""),
#     "web_links": web_result.get("Web Links", []),
#     "youtube_links": youtube_links
# })

#define states

from typing_extensions import TypedDict
from typing import List

### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: the student's question
        chat_history: the history of the student's questions and ai answers
        question_category: the category of the student's question
        found_answer: whether the student's question has been answered
        notes_answer: the answer to the student's question based on notes
        web_answer: the answer to the student's question based on web search
        web_links: the web links found during web search
        youtube_results: the search results from YouTube
        final_answer: the final answer to the student's question

    """
    question : str
    chat_history : List[str]
    question_category : str
    found_answer: bool
    notes_answer : str
    page_number: str
    page_content:str
    web_answer : str
    web_links : List[str]
    youtube_links : List[str]
    final_answer : str
    notes_explanation: str





def student_question_cat(state):
    print("Starting question categorization")
    question = state['question']
    chat_history = state['chat_history']
    notes = retriever.retrieve(question)
    feedback = tutor_category_generator.invoke({"question": question, "chat_history":chat_history, "notes":notes})
    print("From notes", notes)
    print("Question categorised", feedback)
    return {"question_category": feedback['output']}

def notes_reference(state):
    print("Starting notes reference")
    question = state['question']
    chat_history = state['chat_history']
    notes = retriever.retrieve(question)
    feedback = notes_reference_tool.invoke({"question": question, "chat_history":chat_history, "notes" : notes})
    print("Notes run completed", feedback)
    return {
        "found_answer": feedback['Found Answer'], 
        "notes_answer": feedback['Answer'], 
        "notes_explanation": feedback['Explanation'],
        "page_number": feedback.get("Page Number", ""),  # Use .get() with a default value
        "page_content": feedback.get("Page Content", "")  # Use .get() with a default value
    }


def web_reference(state):
    print("Starting web reference")
    question = state['question']
    chat_history = state['chat_history']
    web_results = search_web(question)
    feedback = web_reference_tool.invoke({
        "question": question, 
        "chat_history": chat_history, 
        "web_results": web_results
    })
    print("Web run completed", feedback)
    print("Web Links:", feedback['Web Links'])  # Debugging output
    print("Type of Web Links:", type(feedback['Web Links']))  # Check type

    # Ensure web_links is a list of strings
    web_links = [link['link'] for link in feedback['Web Links']]
    print("Processed Web Links:", web_links)  # Debugging output
    print("Type of Processed Web Links:", type(web_links))  # Check type

    return {
        "found_answer": feedback['Found Answer'], 
        "web_answer": feedback['Answer'],
        "web_links": web_links
    }

def youtube_reference(state):
    print("Starting youtube reference")
    question = state['question']
    chat_history = state['chat_history']
    youtube_links = search_youtube(question)
    # Convert youtube_links to string if it's a dictionary
    if isinstance(youtube_links, dict):
        youtube_links_str = "\n".join(f"{key}: {value}" for key, value in youtube_links.items())
    else:
        youtube_links_str = "\n".join(youtube_links)

    feedback = youtube_reference_tool.invoke({
        "question": question,
        "chat_history": chat_history,
        "youtube_links": youtube_links_str
    })
    print("Youtube run completed", feedback)
    return {
        "found_answer": feedback['Found Answer'],
        "youtube_links": feedback['YouTube Links']
    }

def summary_reference(state):
    print("Summary Reference started")
    question = state.get('question', "")
    chat_history = state.get('chat_history', [])
    notes_answer = state.get('notes_explanation', "")
    web_answer = state.get('web_answer', "")
    web_links = state.get('web_links', [])
    youtube_links = state.get('youtube_links', [])

    # Prepare web_links and youtube_links as strings
    web_links_str = "\n".join(web_links)
    
    # Ensure youtube_links is a list of strings
    if isinstance(youtube_links, list):
        youtube_links_str = "\n".join(link['link'] for link in youtube_links)  # Extract 'link' from each dict
    else:
        youtube_links_str = "\n".join(youtube_links)

    feedback = summary_tool.invoke({
        "question": question,
        "chat_history": chat_history,
        "notes_answer": notes_answer,
        "web_answer": web_answer,
        "web_links": web_links_str,
        "youtube_links": youtube_links_str
    })

    print("Summary Reference completed", feedback)

    return {"final_answer": feedback}


def route_to_category(state):
  if state['question_category'] == 'refer_to_notes':
    return "refer_to_notes"
  elif state['question_category'] == 'web_search':
    return "web_search"
  elif state['question_category'] == 'video_search':
    return "video_search"


def route_to_web_search(state):
    if not state['found_answer']:
        return "not_found_answer"
    else:
        return "found_answer"

from langchain.schema import Document
from langgraph.graph import StateGraph, END

workflow = StateGraph(GraphState)

workflow.add_node("student_question_cat", student_question_cat)
workflow.add_node("notes_reference", notes_reference)
workflow.add_node("web_reference", web_reference)
workflow.add_node("youtube_reference", youtube_reference)
workflow.add_node("summary_reference", summary_reference)

workflow.set_entry_point("student_question_cat")
workflow.add_conditional_edges("student_question_cat",
                              route_to_category,
                              {
                                  "refer_to_notes": "notes_reference",
                                  "web_search": "web_reference",
                                  "video_search": "youtube_reference"
                              })
workflow.add_conditional_edges("notes_reference",
                              route_to_web_search,
                              {
                                  "found_answer": "summary_reference",
                                  "not_found_answer": "web_reference"
                              })
workflow.add_edge("web_reference", "summary_reference")
workflow.add_edge("youtube_reference", "summary_reference")
workflow.add_edge("summary_reference", END)


# Compile
app = workflow.compile()

# from pprint import pprint

# inputs = {"question": "Explain whats an ammeter? Include video links", "chat_history": [], "num_steps":0}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")



















