import patch_protobuf
import os
import time
import re
import pickle
from typing import Dict, Any, List, Tuple
from langchain_core.retrievers import BaseRetriever


from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

import config
from logger import get_logger

logger = get_logger(__name__)

logger.info(f"RAG Chain Startup: RETRIEVAL_MODE is set to '{config.RETRIEVAL_MODE.upper()}'")

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        try:
            from langchain_community.retrievers.ensemble import EnsembleRetriever
        except ImportError:
            EnsembleRetriever = None

def is_greeting(query: str) -> bool:
    clean = query.lower().strip().translate(str.maketrans("", "", ".,?!:;()[]{}"))
    greetings = {
        "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "how is it going", "hows it going", "whats up", "what is up",
        "thank you", "thanks", "thank you very much", "thanks a lot", "thank you!",
        "bye", "goodbye", "see you", "clear history", "reset history",
        "who are you", "what are you", "tell me about yourself", "what can you do", "help"
    }
    if clean in greetings:
        return True
    words = clean.split()
    if len(words) <= 3 and any(w in greetings for w in words):
        return True
    return False

class LoggingHybridRetriever(BaseRetriever):
    ensemble_retriever: Any
    semantic_retriever: Any
    bm25_retriever: Any
    
    model_config = {"arbitrary_types_allowed": True}
        
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Get documents from semantic retriever
        semantic_docs = self.semantic_retriever.invoke(query)
        # Get documents from BM25 retriever
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # Get final merged docs from EnsembleRetriever
        final_docs = self.ensemble_retriever.invoke(query)
        
        # Count contributions
        final_contents = {d.page_content for d in final_docs}
        semantic_contrib = sum(1 for d in semantic_docs if d.page_content in final_contents)
        bm25_contrib = sum(1 for d in bm25_docs if d.page_content in final_contents)
        
        logger.debug(
            f"Hybrid retrieval contributions for query '{query}': "
            f"Semantic contributed {semantic_contrib}/{len(semantic_docs)} chunks, "
            f"BM25 contributed {bm25_contrib}/{len(bm25_docs)} chunks. "
            f"Final merged count: {len(final_docs)}"
        )
        return final_docs

def get_hybrid_retriever(db: Chroma, k: int) -> BaseRetriever:
    """Returns a retriever based on RETRIEVAL_MODE.
    If 'hybrid', returns an EnsembleRetriever combining semantic (Chroma) and BM25 search.
    If 'semantic_only', returns the standard Chroma retriever.
    """
    mode = config.RETRIEVAL_MODE
    
    semantic_retriever = db.as_retriever(search_kwargs={"k": k})
    
    if mode != "hybrid" or EnsembleRetriever is None:
        if mode == "hybrid" and EnsembleRetriever is None:
            logger.warning("EnsembleRetriever could not be imported. Falling back to semantic_only.")
        return semantic_retriever
        
    collection_name = db._collection.name
    bm25_retriever = None
    
    # Try session state cache first
    try:
        import streamlit as st
        if "bm25_retrievers" not in st.session_state:
            st.session_state["bm25_retrievers"] = {}
        if collection_name in st.session_state["bm25_retrievers"]:
            logger.debug(f"Retrieving cached BM25 retriever from session state for collection {collection_name}")
            bm25_retriever = st.session_state["bm25_retrievers"][collection_name]
            bm25_retriever.k = k
    except Exception as e:
        logger.warning(f"Session state not available or failed to access BM25 cache: {str(e)}")
        
    if bm25_retriever is None:
        bm25_path = os.path.join(config.VECTOR_STORE_PATH, f"bm25_{collection_name}.pkl")
        if os.path.exists(bm25_path):
            try:
                logger.info(f"Loading cached BM25 index from {bm25_path}")
                with open(bm25_path, "rb") as f:
                    bm25_retriever = pickle.load(f)
                bm25_retriever.k = k
                # Cache it in session state
                try:
                    import streamlit as st
                    st.session_state["bm25_retrievers"][collection_name] = bm25_retriever
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Failed to load cached BM25 index: {str(e)}")
                
    if bm25_retriever is None:
        logger.info("BM25 index cache missing or failed to load. Building from Chroma on the fly...")
        try:
            results = db.get()
            ids = results.get("ids", [])
            if ids:
                metadatas = results.get("metadatas", [])
                documents = results.get("documents", [])
                chunks = []
                for i in range(len(ids)):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    doc_text = documents[i] if i < len(documents) else ""
                    chunks.append(Document(page_content=doc_text, metadata=meta))
                
                from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
                bm25_retriever = LangChainBM25Retriever.from_documents(chunks)
                bm25_retriever.k = k
                
                # Persist it for future calls
                os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
                with open(bm25_path, "wb") as f:
                    pickle.dump(bm25_retriever, f)
                logger.info(f"Persisted rebuilt BM25 index to {bm25_path}")
                
                # Cache it in session state
                try:
                    import streamlit as st
                    st.session_state["bm25_retrievers"][collection_name] = bm25_retriever
                except Exception:
                    pass
        except Exception as build_err:
            logger.error(f"Failed to dynamically build BM25 retriever: {str(build_err)}", exc_info=True)
            
    if bm25_retriever is None:
        logger.warning("Falling back to semantic only search due to missing BM25 retriever.")
        return semantic_retriever
        
    # Combine semantic and BM25 retrievers using EnsembleRetriever
    weights = [config.SEMANTIC_SEARCH_WEIGHT, config.KEYWORD_SEARCH_WEIGHT]
    logger.info(f"Creating EnsembleRetriever with weights (Semantic: {weights[0]}, BM25: {weights[1]}) and k={k}")
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=weights
    )
    
    return LoggingHybridRetriever(
        ensemble_retriever=ensemble_retriever,
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever
    )


def format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """Formats standard session state chat history for context in LLM prompts."""
    formatted = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def needs_rephrasing(question: str) -> bool:
    """Detects whether a follow-up question references chat history.
    If it is self-contained, we can skip the rephrasing.
    """
    if is_greeting(question):
        return False
        
    words = question.lower().translate(str.maketrans("", "", ".,?!:;")).split()
    if not words:
        return False
        
    # Very short queries (e.g. "why?", "explain further") almost certainly depend on context
    if len(words) < 4:
        return True
        
    # Standard reference pronouns and adjectives
    reference_words = {
        "this", "that", "it", "them", "these", "those", "they", "their", "he", "him", "his", "she", "her",
        "previous", "above", "before", "again", "also", "too", "other", "another",
        "yes", "no", "correct", "ok", "okay", "alternative", "detail", "details", "more", "explain",
        "describe", "summarize", "elaborate", "same"
    }
    
    for word in words:
        if word in reference_words:
            return True
            
    # Check for context-dependent phrasing (removed standard question starters like 'what was')
    q_lower = question.lower().strip()
    context_phrases = [
        "what about", "how about", "tell me more", "in contrast", "compare to",
        "and for", "any other", "what of", "what else", "go deeper"
    ]
    for phrase in context_phrases:
        if phrase in q_lower:
            return True
            
    # Check if it starts with joining words implying continuation
    if q_lower.startswith("and ") or q_lower.startswith("or "):
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


def decompose_query(
    question: str,
    primary_model: ChatGoogleGenerativeAI,
    fallback_model: ChatGoogleGenerativeAI
) -> List[str]:
    """Uses the LLM to decompose a compound question into separate standalone sub-questions.
    Returns a list of sub-questions. If the question is simple, returns a list with only the original question.
    """
    prompt = (
        "Deconstruct the user's question into distinct, standalone search queries (in English) "
        "if it is a compound question asking for multiple separate pieces of information.\n"
        "If the question is simple and contains only one single ask, return the original question as the only item.\n"
        "Crucial Rules:\n"
        "1. Do NOT include the specific company name (e.g., 'Apple') or generic terms like 'the company' or 'registrant' in the sub-questions, "
        "as they are redundant and degrade vector search retrieval quality. Focus strictly on the core entity or concept.\n"
        "2. Expand common acronyms/abbreviations to their full forms (e.g., expand 'CEO' to 'Chief Executive Officer (CEO)' "
        "or 'CEO', and 'CFO' to 'Chief Financial Officer').\n"
        "3. Keep each query extremely concise, direct, and keyword-focused. Avoid conversational filler words like "
        "'How many', 'are there', 'what is', 'who is', 'find', etc. Focus on the entity and metadata (e.g. 'number of full-time employees' "
        "instead of 'How many full-time employees are there?').\n"
        "Format the output as a clean bulleted list starting with '- ' for each sub-question/query. Do not add any conversational text.\n\n"
        "Example 1 (Compound):\n"
        "Input: 'How many employees does Apple have, and who is the CEO?'\n"
        "Output:\n"
        "- number of full-time employees\n"
        "- Chief Executive Officer (CEO)\n\n"
        "Example 2 (Simple):\n"
        "Input: 'What is the revenue for 2023?'\n"
        "Output:\n"
        "- revenue for 2023\n\n"
        f"Now deconstruct the following question:\n"
        f"Input: '{question}'\n"
        f"Output:"
    )
    
    try:
        logger.info("Attempting query decomposition using primary model.")
        response = primary_model.invoke(prompt)
        raw_output = _extract_text(response.content)
    except Exception as e:
        logger.warning(f"Primary model query decomposition failed: {str(e)}. Trying fallback model.")
        try:
            response = fallback_model.invoke(prompt)
            raw_output = _extract_text(response.content)
        except Exception as e_fallback:
            logger.error(f"Both models failed query decomposition: {str(e_fallback)}. Falling back to original question.")
            return [question]
            
    # Parse bulleted list
    sub_queries = []
    for line in raw_output.split("\n"):
        line_str = line.strip()
        if line_str.startswith("-"):
            sub_q = line_str.lstrip("-").strip()
            if sub_q:
                sub_queries.append(sub_q)
                
    if not sub_queries:
        return [question]
        
    return sub_queries



def is_likely_compound(question: str) -> bool:
    """Detects whether a question is likely to be a compound question
    (asking for multiple separate pieces of information) using local heuristics.
    """
    # 1. Multiple question marks
    if question.count("?") > 1:
        return True
        
    # 2. Check for explicit multi-part phrases
    phrases = [
        "as well as", "along with", "also tell me", "in addition to",
        "both of", "as well", "also look up", "also find", "also show"
    ]
    q_lower = question.lower()
    for phrase in phrases:
        if phrase in q_lower:
            return True
            
    # 3. Whole-word check for joining words (and, or)
    # Using word boundaries to avoid matching substrings like "Andorra" or "organization"
    words = re.findall(r"\b\w+\b", q_lower)
    joining_words = {"and", "or"}
    
    for word in words:
        if word in joining_words:
            return True
            
    return False

def retrieve_and_combine_docs(
    standalone_question: str,
    db: Chroma,
    primary_model: ChatGoogleGenerativeAI,
    fallback_model: ChatGoogleGenerativeAI,
    status_callback=None
) -> Tuple[List[Document], bool]:
    """Retrieves relevant chunks directly for the question.
    Decomposition is disabled to optimize speed and API usage.
    """
    if is_greeting(standalone_question):
        logger.info("Greeting/conversational query detected. Bypassing document retrieval.")
        return [], False
        
    if status_callback:
        status_callback("Searching document...")
    # Retrieve top 6 chunks for direct, fast search
    retriever = get_hybrid_retriever(db, 6)
    docs = retriever.invoke(standalone_question)
    logger.info(f"Retrieved {len(docs)} chunks.")
    return docs, False


def filter_used_pages(
    answer: str,
    docs: List[Document],
    primary_model: ChatGoogleGenerativeAI,
    fallback_model: ChatGoogleGenerativeAI,
    question: str = None
) -> List[int]:
    """Identifies the unique page numbers from the retrieved documents that were
    actually relevant/used to formulate the answer based on content overlap.
    Matches strictly against chunk content (not parent content) to avoid false citations.
    """
    if "couldn't find this" in answer.lower():
        logger.info("Answer states info not found. Returning empty sources.")
        return []
        
    # Clean and tokenize the answer
    clean_answer = re.sub(r'[^\w\s-]', '', answer.lower())
    answer_words = set(w for w in clean_answer.split() if len(w) > 2)
    
    # Stop words
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "was", "were", "are", "been", "have", "has",
        "not", "but", "their", "will", "would", "about", "more", "than", "other", "some", "any", "such",
        "into", "only", "also", "its", "our", "your", "they", "them", "these", "those"
    }
    
    important_words = answer_words - stop_words
    
    # If question is provided, filter out words that were already in the question
    if question:
        clean_question = re.sub(r'[^\w\s-]', '', question.lower())
        question_words = set(w for w in clean_question.split() if len(w) > 2)
        important_words = important_words - question_words
            
    if not important_words:
        return []
        
    used_pages = set()
    scores = []
    
    for doc in docs:
        page_num = doc.metadata.get("page_number")
        if page_num is None:
            continue
            
        content = doc.page_content.lower()
        search_text = re.sub(r'[^\w\s-]', '', content)
        search_text = re.sub(r'\s+', ' ', search_text)
        
        # Calculate overlap score
        match_count = 0
        for word in important_words:
            if any(c.isdigit() for c in word):
                digit_only = "".join(c for c in word if c.isdigit())
                if digit_only:
                    val = int(digit_only)
                    weight = 1 if ((1900 <= val <= 2100) or (val < 100)) else 5
                else:
                    weight = 1
            else:
                weight = 1
                
            if f" {word} " in f" {search_text} ":
                match_count += weight
                
        scores.append((page_num, match_count))
        
    if not scores:
        return []
        
    max_score = max(score[1] for score in scores)
    if max_score == 0:
        return []
        
    # We require a minimum overlap of 3 score units, and at least 60% of the maximum score
    threshold = max(3, max_score * 0.6)
    for page_num, score in scores:
        if score >= threshold:
            used_pages.add(page_num)
            
    result = sorted(list(used_pages))
    logger.info(f"Filtered used pages: {scores} -> {result} (max score: {max_score}, threshold: {threshold:.2f})")
    return result



def rephrase_question(
    question: str,
    chat_history: List[Dict[str, Any]],
    primary_model: ChatGoogleGenerativeAI,
    fallback_model: ChatGoogleGenerativeAI
) -> str:
    """Combines the current question with key context words from history
    only if the question is detected as context-dependent.
    """
    if is_greeting(question):
        return question

    if not chat_history:
        return question

    # First check if the question actually needs context from history
    if not needs_rephrasing(question):
        logger.info("Question is self-contained. Skipping heuristic query expansion.")
        return question

    # Get the last user message from history to expand context keywords
    last_user_msg = None
    for msg in reversed(chat_history):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    if last_user_msg:
        # Simple heuristic context expansion (e.g., search queries are just bag of words)
        logger.info("Applying fast heuristic query expansion for context-dependent question.")
        return f"{last_user_msg} {question}"
    return question

_primary_model = None
_fallback_model = None

def get_chat_models() -> Tuple[ChatGoogleGenerativeAI, ChatGoogleGenerativeAI]:
    """Retrieves cached instances of ChatGoogleGenerativeAI models to avoid instantiation delay.
    Configures max_retries to 0 for instant fallback on rate limits.
    """
    global _primary_model, _fallback_model
    try:
        import streamlit as st
        # Use Streamlit's cache_resource to keep the models cached across sessions/reruns
        @st.cache_resource(show_spinner=False)
        def _get_cached_models(primary_model_name, fallback_model_name, api_key):
            logger.info("Initializing and caching ChatGoogleGenerativeAI models using Streamlit cache...")
            p_model = ChatGoogleGenerativeAI(
                model=primary_model_name,
                temperature=0,
                google_api_key=api_key,
                max_retries=0
            )
            f_model = ChatGoogleGenerativeAI(
                model=fallback_model_name,
                temperature=0,
                google_api_key=api_key,
                max_retries=0
            )
            return p_model, f_model
        
        return _get_cached_models(config.PRIMARY_MODEL, config.FALLBACK_MODEL, config.GEMINI_API_KEY)
    except Exception as e:
        # Fallback to standard Python global cache if Streamlit is not available or errors out
        if _primary_model is None or _fallback_model is None:
            logger.info("Initializing ChatGoogleGenerativeAI models (caching globally)...")
            _primary_model = ChatGoogleGenerativeAI(
                model=config.PRIMARY_MODEL,
                temperature=0,
                google_api_key=config.GEMINI_API_KEY,
                max_retries=0
            )
            _fallback_model = ChatGoogleGenerativeAI(
                model=config.FALLBACK_MODEL,
                temperature=0,
                google_api_key=config.GEMINI_API_KEY,
                max_retries=0
            )
        return _primary_model, _fallback_model



def query_rag_chain(
    question: str,
    chat_history: List[Dict[str, Any]],
    db: Chroma,
    status_callback=None
) -> Dict[str, Any]:
    """Executes the Conversational RAG pipeline:
    1. Rephrases follow-up questions using chat history.
    2. Retrieves relevant chunks from Chroma vector store.
    3. Feeds context chunks to LLM (with fallback).
    4. Returns the answer, source page numbers, and details on which model was used.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

    start_time = time.time()
    logger.info(f"RAG Chain query start (non-streaming): '{question}'")

    if status_callback:
        status_callback("Checking history...")

    # Initialize chat models (caching globally to prevent startup lag)
    primary_model, fallback_model = get_chat_models()

    # 1. Rephrase user question if chat history exists
    standalone_question = rephrase_question(question, chat_history, primary_model, fallback_model)
    rephrase_time = time.time() - start_time
    logger.info(f"Timing - Rephrasing completed: {rephrase_time:.4f}s elapsed (standalone: '{standalone_question}')")

    # 2. Retrieve chunks (decomposes the query if compound)
    retrieval_start = time.time()
    docs, is_compound = retrieve_and_combine_docs(standalone_question, db, primary_model, fallback_model, status_callback)
    retrieval_time = time.time() - retrieval_start
    logger.info(f"Timing - Retrieval completed: {retrieval_time:.4f}s (compound: {is_compound})")

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
        "Note: The user may refer to a section header or note number (e.g., 'Note 2' or 'Note 17') that exists on earlier pages of the document or spans multiple pages. The retrieved chunks are labeled with their page numbers. If the text in the context chunks contains the requested information on the specified topic (e.g., noncontrolling interests, profit-sharing, or segment tables), assume it is correct and answer the question even if the specific note number header itself is not explicitly visible in the retrieved text block.\n"
        "For general greetings (e.g., 'hi', 'hello', 'good morning', 'how are you') or polite conversational messages (e.g., 'thank you', 'bye'), respond friendly, politely, and naturally. Do not require document context or citations for these conversational messages, and write [Sources: None] at the end.\n"
        "If the user asks for a summary of the entire PDF, do not say you couldn't find it. Instead, explain that you can help query specific details of this document, identify the document name/type from the context (e.g., Tesla Form 10-K), and provide a brief high-level overview of the major sections covered in the context chunks, ending with [Sources: None] or pages if appropriate.\n"
        "If you cannot find the answer in the provided context and the query is not a greeting or a general summary request, reply exactly with: "
        "\"I couldn't find this in the document\" - do not make up information.\n"
        "If the user's question contains multiple distinct sub-questions/asks, you must address "
        "each part clearly in your answer (e.g., using separate sentences or a bulleted list).\n"
        "Do NOT include any inline page citations or bracketed page references (such as '[Page 27]' or 'Page 27') "
        "in your main text response. Just provide the textual answer directly.\n"
        "At the very end of your response, you MUST list the source page numbers you used to answer the question, formatted exactly like: "
        "[Sources: Page X, Page Y] "
        "If no sources were used, write [Sources: None]. Do not include any other text after this bracket.\n"
        "Do not use any external knowledge or assumptions. Keep your answer factual, direct, and concise."
    )

    # Reconstruct context from retrieved chunks
    seen_contents = set()
    unique_docs = []
    
    for doc in docs:
        content = doc.page_content
        page_num = doc.metadata.get("page_number", "Unknown")
        if content not in seen_contents:
            seen_contents.add(content)
            unique_docs.append((page_num, content))
            
    context_str = "\n\n".join([
        f"[Page {page_num}]:\n{content}"
        for page_num, content in unique_docs
    ])

    # Format chat history for prompt context
    history_str = ""
    if chat_history:
        history_str = "Conversation History:\n"
        for msg in chat_history[-5:]: # Keep last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        history_str += "\n"

    user_content = f"Context:\n{context_str}\n\n{history_str}Current Question: {question}"

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_content)
    ]

    # 4. Invoke LLM with fallback mechanism and model logging
    answer = None
    model_used = None
    
    llm_start = time.time()
    logger.info(f"Timing - Invoking LLM at {time.time() - start_time:.4f}s elapsed")
    if status_callback:
        status_callback("Generating answer...")

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

    llm_time = time.time() - llm_start
    logger.info(f"Timing - LLM call completed: {llm_time:.4f}s. Total pipeline time: {time.time() - start_time:.4f}s")

    # Determine if answer indicates information wasn't found
    not_found_indicators = [
        "couldn't find this in the document",
        "could not find this in the document",
        "i couldn't find this"
    ]
    has_found = not any(indicator in answer.lower() for indicator in not_found_indicators)
    
    # Parse source pages from the end of the answer
    source_pages = []
    if has_found:
        import re
        sources_match = re.search(r'\[Sources:\s*(.*?)\]', answer)
        if sources_match:
            sources_str = sources_match.group(1)
            answer = answer.replace(sources_match.group(0), "").strip()
            if "none" not in sources_str.lower():
                page_nums = re.findall(r'\d+', sources_str)
                source_pages = sorted(list(set(int(p) for p in page_nums)))
                
        if not source_pages:
            source_pages = filter_used_pages(answer, docs, primary_model, fallback_model, question=standalone_question)

    return {
        "answer": answer,
        "source_pages": source_pages,
        "model_used": model_used,
        "retrieved_docs": docs
    }

def query_rag_chain_stream(
    question: str,
    chat_history: List[Dict[str, Any]],
    db: Chroma,
    meta: Dict[str, Any],
    status_callback=None
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

    start_time = time.time()
    logger.info(f"RAG Chain query start (streaming): '{question}'")

    if status_callback:
        status_callback("Checking history...")

    # Initialize chat models (caching globally to prevent startup lag)
    primary_model, fallback_model = get_chat_models()

    # 1. Rephrase user question if needed
    standalone_question = rephrase_question(question, chat_history, primary_model, fallback_model)
    rephrase_time = time.time() - start_time
    logger.info(f"Timing - Rephrasing completed: {rephrase_time:.4f}s elapsed (standalone: '{standalone_question}')")

    # 2. Retrieve chunks (decomposes the query if compound)
    retrieval_start = time.time()
    docs, is_compound = retrieve_and_combine_docs(standalone_question, db, primary_model, fallback_model, status_callback)
    retrieval_time = time.time() - retrieval_start
    logger.info(f"Timing - Retrieval completed: {retrieval_time:.4f}s (compound: {is_compound})")

    # Store metadata
    meta["source_pages"] = []
    meta["model_used"] = config.PRIMARY_MODEL
    meta["retrieved_docs"] = docs
    meta["primary_model"] = primary_model
    meta["fallback_model"] = fallback_model

    # 3. Construct prompt
    system_instruction = (
        "You are a helpful assistant answering questions about the uploaded PDF document.\n"
        "Use ONLY the provided context chunks to answer the user's question.\n"
        "Always scan deeply for parenthetical details, footnotes, and regional terminology aliases "
        "(such as equivalent names, synonyms, or naming conventions) within the context text to ensure "
        "no hidden context is missed. Answer factually based on these details.\n"
        "Note: The user may refer to a section header or note number (e.g., 'Note 2' or 'Note 17') that exists on earlier pages of the document or spans multiple pages. The retrieved chunks are labeled with their page numbers. If the text in the context chunks contains the requested information on the specified topic (e.g., noncontrolling interests, profit-sharing, or segment tables), assume it is correct and answer the question even if the specific note number header itself is not explicitly visible in the retrieved text block.\n"
        "For general greetings (e.g., 'hi', 'hello', 'good morning', 'how are you') or polite conversational messages (e.g., 'thank you', 'bye'), respond friendly, politely, and naturally. Do not require document context or citations for these conversational messages, and write [Sources: None] at the end.\n"
        "If the user asks for a summary of the entire PDF, do not say you couldn't find it. Instead, explain that you can help query specific details of this document, identify the document name/type from the context (e.g., Tesla Form 10-K), and provide a brief high-level overview of the major sections covered in the context chunks, ending with [Sources: None] or pages if appropriate.\n"
        "If you cannot find the answer in the provided context and the query is not a greeting or a general summary request, reply exactly with: "
        "\"I couldn't find this in the document\" - do not make up information.\n"
        "If the user's question contains multiple distinct sub-questions/asks, you must address "
        "each part clearly in your answer (e.g., using separate sentences or a bulleted list).\n"
        "Do NOT include any inline page citations or bracketed page references (such as '[Page 27]' or 'Page 27') "
        "in your main text response. Just provide the textual answer directly.\n"
        "At the very end of your response, you MUST list the source page numbers you used to answer the question, formatted exactly like: "
        "[Sources: Page X, Page Y] "
        "If no sources were used, write [Sources: None]. Do not include any other text after this bracket.\n"
        "Do not use any external knowledge or assumptions. Keep your answer factual, direct, and concise."
    )

    # Reconstruct context from retrieved chunks
    seen_contents = set()
    unique_docs = []
    
    for doc in docs:
        content = doc.page_content
        page_num = doc.metadata.get("page_number", "Unknown")
        if content not in seen_contents:
            seen_contents.add(content)
            unique_docs.append((page_num, content))
            
    context_str = "\n\n".join([
        f"[Page {page_num}]:\n{content}"
        for page_num, content in unique_docs
    ])

    # Format chat history for prompt context
    history_str = ""
    if chat_history:
        history_str = "Conversation History:\n"
        for msg in chat_history[-5:]: # Keep last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        history_str += "\n"

    user_content = f"Context:\n{context_str}\n\n{history_str}Current Question: {question}"

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_content)
    ]

    # 4. Stream LLM chunks with fallback mechanism
    llm_start = time.time()
    logger.info(f"Timing - Invoking LLM stream at {time.time() - start_time:.4f}s elapsed")
    if status_callback:
        status_callback("Generating answer...")
    
    # Track generated text to verify if "not found" indicators are present
    full_text = []
    
    try:
        logger.info(f"Attempting stream with primary model: {config.PRIMARY_MODEL}")
        stream = primary_model.stream(messages)
        first_token = True
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
                if first_token:
                    first_token_time = time.time() - start_time
                    logger.info(f"Timing - First token received at {first_token_time:.4f}s elapsed (from start)")
                    first_token = False
                full_text.append(token_str)
                yield token_str
                
        meta["model_used"] = config.PRIMARY_MODEL
        logger.info(f"Stream completed using primary model: {config.PRIMARY_MODEL} in {time.time() - llm_start:.4f}s. Total pipeline time: {time.time() - start_time:.4f}s")
    except Exception as e:
        logger.warning(f"Primary model stream failed: {str(e)}. Attempting stream with fallback model: {config.FALLBACK_MODEL}")
        full_text = []
        try:
            stream = fallback_model.stream(messages)
            meta["model_used"] = config.FALLBACK_MODEL
            first_token = True
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
                    if first_token:
                        first_token_time = time.time() - start_time
                        logger.info(f"Timing - First token received (fallback) at {first_token_time:.4f}s elapsed (from start)")
                        first_token = False
                    full_text.append(token_str)
                    yield token_str
                    
            logger.info(f"Stream completed using fallback model: {config.FALLBACK_MODEL} in {time.time() - llm_start:.4f}s. Total pipeline time: {time.time() - start_time:.4f}s")
        except Exception as e_fallback:
            logger.error(f"Both streaming models failed: {str(e_fallback)}", exc_info=True)
            raise e_fallback

