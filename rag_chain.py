import patch_protobuf
import os
import time
from typing import Dict, Any, List, Tuple


from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

import config
from logger import get_logger

logger = get_logger(__name__)

def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """Formats standard session state chat history for context in LLM prompts."""
    formatted = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def needs_rephrasing(question: str) -> bool:
    """Detects whether a follow-up question references chat history.
    If it is self-contained, we can skip the rephrasing LLM call.
    """
    words = question.lower().translate(str.maketrans("", "", ".,?!:;")).split()
    # Very short queries (e.g. "why?", "explain further") almost certainly depend on context
    if len(words) < 4:
        return True
        
    reference_words = {
        "this", "that", "it", "them", "these", "those", "they", "their", "he", "him", "his", "she", "her",
        "who", "why", "how", "previous", "above", "before", "again", "also", "too", "other", "another",
        "yes", "no", "correct", "ok", "okay", "alternative", "detail", "details", "more", "explain",
        "describe", "summarize", "elaborate", "same"
    }
    
    for word in words:
        if word in reference_words:
            return True
            
    # Check for context-dependent phrasing
    q_lower = question.lower()
    phrases = ["what about", "tell me", "what does", "what did", "what was", "in contrast", "compare to"]
    for phrase in phrases:
        if phrase in q_lower:
            return True
            
    return False

def _extract_text(content: Any) -> str:
    """Safely extracts string content from model response, handling list-of-parts types."""
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts).strip()
    elif isinstance(content, str):
        return content.strip()
    return str(content).strip() if content is not None else ""


def rephrase_question(
    question: str,
    chat_history: List[Dict[str, Any]],
    primary_model: ChatGoogleGenerativeAI,
    fallback_model: ChatGoogleGenerativeAI
) -> str:
    """Rephrases a follow-up question into a standalone question using chat history."""
    if not chat_history or not needs_rephrasing(question):
        if chat_history:
            logger.info(f"Skipping rephrasing: question '{question}' appears self-contained.")
        return question

    formatted_history = format_chat_history(chat_history)
    rephrase_instruction = (
        "Given the following conversation history and a follow-up question, rephrase the follow-up question "
        "to be a standalone question (in English). Do NOT answer the question, just rephrase it.\n"
        "If the follow-up question is already standalone or doesn't refer to the history, "
        "return it exactly as it is."
    )
    
    prompt = f"{rephrase_instruction}\n\nChat History:\n{formatted_history}\n\nFollow-up Question: {question}\n\nStandalone Question:"
    
    try:
        logger.info(f"Attempting to rephrase question using primary model: {config.PRIMARY_MODEL}")
        response = primary_model.invoke(prompt)
        standalone = _extract_text(response.content)
        logger.info(f"Successfully rephrased question: '{standalone}'")
        return standalone
    except Exception as e:
        logger.warning(f"Primary model rephrasing failed: {str(e)}. Trying fallback model: {config.FALLBACK_MODEL}")
        try:
            response = fallback_model.invoke(prompt)
            standalone = _extract_text(response.content)
            logger.info(f"Successfully rephrased question with fallback model: '{standalone}'")
            return standalone
        except Exception as e_fallback:
            logger.error(f"Both models failed to rephrase question: {str(e_fallback)}. Using original question.")
            return question


def query_rag_chain(
    question: str,
    chat_history: List[Dict[str, Any]],
    db: Chroma
) -> Dict[str, Any]:
    """Executes the Conversational RAG pipeline:
    1. Rephrases follow-up questions using chat history.
    2. Retrieves relevant chunks from Chroma vector store.
    3. Feeds context chunks to LLM (with fallback).
    4. Returns the answer, source page numbers, and details on which model was used.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

    # Initialize chat models
    primary_model = ChatGoogleGenerativeAI(
        model=config.PRIMARY_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY
    )
    fallback_model = ChatGoogleGenerativeAI(
        model=config.FALLBACK_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY
    )

    # 1. Rephrase user question if chat history exists
    standalone_question = rephrase_question(question, chat_history, primary_model, fallback_model)

    # 2. Retrieve top 4 most relevant chunks
    logger.info(f"Retrieving chunks for query: '{standalone_question}'")
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(standalone_question)
    logger.info(f"Retrieved {len(docs)} chunks.")

    # Extract unique, sorted page numbers
    source_pages = sorted(list(set(
        doc.metadata.get("page_number")
        for doc in docs
        if doc.metadata.get("page_number") is not None
    )))

    # 3. Construct prompt
    system_instruction = (
        "You are a helpful assistant answering questions about the uploaded PDF document.\n"
        "Use ONLY the provided context chunks to answer the user's question.\n"
        "Always scan deeply for parenthetical details, footnotes, and regional terminology aliases "
        "(such as equivalent names, synonyms, or naming conventions) within the context text to ensure "
        "no hidden context is missed. Answer factually based on these details.\n"
        "If you cannot find the answer in the provided context, reply exactly with: "
        "\"I couldn't find this in the document\" - do not make up information.\n"
        "Do not use any external knowledge or assumptions. Keep your answer factual, direct, and concise."
    )

    # Reconstruct context from parent chunks for hierarchical RAG, fallback to child if not present
    seen_parents = set()
    unique_parent_docs = []
    
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")
        
        content = parent_content if parent_content else doc.page_content
        page_num = doc.metadata.get("page_number", "Unknown")
        
        if parent_id:
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                unique_parent_docs.append((page_num, content))
        else:
            unique_parent_docs.append((page_num, content))
            
    context_str = "\n\n".join([
        f"[Page {page_num}]:\n{content}"
        for page_num, content in unique_parent_docs
    ])

    user_content = f"Context:\n{context_str}\n\nQuestion: {standalone_question}"

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_content)
    ]

    # 4. Invoke LLM with fallback mechanism and model logging
    answer = None
    model_used = None
    
    try:
        logger.info(f"Invoking primary model: {config.PRIMARY_MODEL}")
        response = primary_model.invoke(messages)
        answer = _extract_text(response.content)
        model_used = config.PRIMARY_MODEL
        logger.info(f"Answer generated successfully using primary model: {config.PRIMARY_MODEL}")
    except Exception as e:
        logger.warning(f"Primary model {config.PRIMARY_MODEL} failed: {str(e)}. Retrying with fallback: {config.FALLBACK_MODEL}")
        try:
            response = fallback_model.invoke(messages)
            answer = _extract_text(response.content)
            model_used = config.FALLBACK_MODEL
            logger.info(f"Answer generated successfully using fallback model: {config.FALLBACK_MODEL}")
        except Exception as e_fallback:
            logger.error(f"Both primary and fallback models failed: {str(e_fallback)}", exc_info=True)
            raise e_fallback

    # Determine if answer indicates information wasn't found
    not_found_indicators = [
        "couldn't find this in the document",
        "could not find this in the document",
        "i couldn't find this"
    ]
    has_found = not any(indicator in answer.lower() for indicator in not_found_indicators)

    return {
        "answer": answer,
        "source_pages": source_pages if has_found else [],
        "model_used": model_used,
        "retrieved_docs": docs
    }

def query_rag_chain_stream(
    question: str,
    chat_history: List[Dict[str, Any]],
    db: Chroma,
    meta: Dict[str, Any]
):
    """Executes the Conversational RAG pipeline in streaming mode:
    1. Rephrases follow-up questions using chat history if needed.
    2. Retrieves relevant chunks from Chroma vector store.
    3. Feeds context chunks to LLM (with fallback).
    4. Yields chunks of text in real-time.
    5. Populates the meta dictionary with source page numbers and model details.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

    # Initialize chat models
    primary_model = ChatGoogleGenerativeAI(
        model=config.PRIMARY_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY
    )
    fallback_model = ChatGoogleGenerativeAI(
        model=config.FALLBACK_MODEL,
        temperature=0,
        google_api_key=config.GEMINI_API_KEY
    )

    # 1. Rephrase user question if needed
    standalone_question = rephrase_question(question, chat_history, primary_model, fallback_model)

    # 2. Retrieve top 4 most relevant chunks
    logger.info(f"Retrieving chunks for query: '{standalone_question}'")
    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(standalone_question)
    logger.info(f"Retrieved {len(docs)} chunks.")

    # Extract unique, sorted page numbers
    source_pages = sorted(list(set(
        doc.metadata.get("page_number")
        for doc in docs
        if doc.metadata.get("page_number") is not None
    )))

    # Store metadata (source pages are populated before streaming starts)
    meta["source_pages"] = source_pages
    meta["model_used"] = config.PRIMARY_MODEL

    # 3. Construct prompt
    system_instruction = (
        "You are a helpful assistant answering questions about the uploaded PDF document.\n"
        "Use ONLY the provided context chunks to answer the user's question.\n"
        "Always scan deeply for parenthetical details, footnotes, and regional terminology aliases "
        "(such as equivalent names, synonyms, or naming conventions) within the context text to ensure "
        "no hidden context is missed. Answer factually based on these details.\n"
        "If you cannot find the answer in the provided context, reply exactly with: "
        "\"I couldn't find this in the document\" - do not make up information.\n"
        "Do not use any external knowledge or assumptions. Keep your answer factual, direct, and concise."
    )

    # Reconstruct context from parent chunks for hierarchical RAG, fallback to child if not present
    seen_parents = set()
    unique_parent_docs = []
    
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")
        
        content = parent_content if parent_content else doc.page_content
        page_num = doc.metadata.get("page_number", "Unknown")
        
        if parent_id:
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                unique_parent_docs.append((page_num, content))
        else:
            unique_parent_docs.append((page_num, content))
            
    context_str = "\n\n".join([
        f"[Page {page_num}]:\n{content}"
        for page_num, content in unique_parent_docs
    ])

    user_content = f"Context:\n{context_str}\n\nQuestion: {standalone_question}"

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_content)
    ]

    # 4. Stream LLM chunks with fallback mechanism
    logger.info(f"Streaming model output for query: '{standalone_question}'")
    
    # Track generated text to verify if "not found" indicators are present
    full_text = []
    
    try:
        logger.info(f"Attempting stream with primary model: {config.PRIMARY_MODEL}")
        stream = primary_model.stream(messages)
        for chunk in stream:
            token = chunk.content
            
            # Extract raw text from structured content list or dictionary
            if isinstance(token, list):
                text_parts = []
                for part in token:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                token_str = "".join(text_parts)
            else:
                token_str = str(token) if token is not None else ""
                
            if token_str:
                full_text.append(token_str)
                for char in token_str:
                    yield char
                    time.sleep(0.005)
                
        meta["model_used"] = config.PRIMARY_MODEL
        logger.info(f"Stream completed using primary model: {config.PRIMARY_MODEL}")
    except Exception as e:
        logger.warning(f"Primary model stream failed: {str(e)}. Attempting stream with fallback model: {config.FALLBACK_MODEL}")
        full_text = []
        try:
            stream = fallback_model.stream(messages)
            meta["model_used"] = config.FALLBACK_MODEL
            for chunk in stream:
                token = chunk.content
                
                # Extract raw text from structured content list or dictionary
                if isinstance(token, list):
                    text_parts = []
                    for part in token:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    token_str = "".join(text_parts)
                else:
                    token_str = str(token) if token is not None else ""
                    
                if token_str:
                    full_text.append(token_str)
                    for char in token_str:
                        yield char
                        time.sleep(0.005)
                    
            logger.info(f"Stream completed using fallback model: {config.FALLBACK_MODEL}")
        except Exception as e_fallback:
            logger.error(f"Both streaming models failed: {str(e_fallback)}", exc_info=True)
            raise e_fallback

    # Post-process response to see if it failed to find answer
    answer = "".join(full_text).lower()
    not_found_indicators = [
        "couldn't find this in the document",
        "could not find this in the document",
        "i couldn't find this"
    ]
    if any(indicator in answer for indicator in not_found_indicators):
        # Clear sources if not found
        meta["source_pages"] = []

