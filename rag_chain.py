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
    bm25_path = os.path.join(config.VECTOR_STORE_PATH, f"bm25_{collection_name}.pkl")
    
    bm25_retriever = None
    if os.path.exists(bm25_path):
        try:
            logger.info(f"Loading cached BM25 index from {bm25_path}")
            with open(bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
            bm25_retriever.k = k
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
    fallback_model: ChatGoogleGenerativeAI
) -> Tuple[List[Document], bool]:
    """Decomposes the query if compound, runs retrieval for each sub-query,
    and returns a combined, deduplicated list of documents. Also returns a boolean
    indicating if the query was compound.
    """
    # Fast local pre-check to bypass decomposition for simple questions
    if not is_likely_compound(standalone_question):
        logger.info(f"Local heuristic detected simple question. Bypassing decomposition LLM call for: '{standalone_question}'")
        # Simple question: retrieve N=4 chunks
        retriever = get_hybrid_retriever(db, 4)
        docs = retriever.invoke(standalone_question)
        logger.info(f"Retrieved {len(docs)} chunks for simple question.")
        return docs, False
        
    logger.info(f"Local heuristic detected likely compound question. Proceeding to LLM decomposition for: '{standalone_question}'")
    # Decompose the question into sub-queries
    sub_queries = decompose_query(standalone_question, primary_model, fallback_model)
    
    is_compound = len(sub_queries) > 1
    
    if is_compound:
        logger.info(f"Compound question detected. Decomposed sub-queries: {sub_queries}")
        # When compound, retrieve N=5 chunks per sub-query to ensure adequate coverage
        k = 5
        all_docs = []
        for sub_q in sub_queries:
            logger.info(f"Retrieving top {k} chunks for sub-query: '{sub_q}'")
            retriever = get_hybrid_retriever(db, k)
            sub_docs = retriever.invoke(sub_q)
            logger.info(f"Retrieved {len(sub_docs)} chunks for sub-query: '{sub_q}'. Pages: {[d.metadata.get('page_number') for d in sub_docs]}")
            all_docs.extend(sub_docs)
            
        # Deduplicate retrieved child documents by page_content to avoid redundant text
        seen_contents = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
                
        logger.info(f"Combined and deduplicated retrieved chunks: {len(unique_docs)} unique chunks (out of {len(all_docs)} total retrieved)")
        return unique_docs, True
    else:
        # Simple question: retrieve N=4 chunks as before
        logger.info(f"LLM decomposition decided question is simple after all. Retrieving top 4 chunks.")
        retriever = get_hybrid_retriever(db, 4)
        docs = retriever.invoke(standalone_question)
        logger.info(f"Retrieved {len(docs)} chunks for simple question.")
        return docs, False


def filter_used_pages(
    answer: str,
    docs: List[Document],
    primary_model: ChatGoogleGenerativeAI,
    fallback_model: ChatGoogleGenerativeAI
) -> List[int]:
    """Uses the LLM to inspect the final answer and identify which of the retrieved
    document pages actually support the answer.
    """
    if "couldn't find this" in answer.lower():
        logger.info("Answer states info not found. Returning empty sources.")
        return []
        
    # Get all unique page numbers
    unique_pages = sorted(list(set(
        doc.metadata.get("page_number")
        for doc in docs
        if doc.metadata.get("page_number") is not None
    )))
    
    # Group chunk snippets by page number to ensure all facts retrieved on a page are visible to the LLM
    page_contents = {}
    for doc in docs:
        page_num = doc.metadata.get("page_number")
        if page_num is not None:
            if page_num not in page_contents:
                page_contents[page_num] = []
            page_contents[page_num].append(doc.page_content.strip().replace("\n", " "))
            
    candidates = []
    for page_num in sorted(page_contents.keys()):
        # Join snippets of the same page, limiting total text to 1000 characters per page
        full_page_snippet = " | ".join(page_contents[page_num])
        if len(full_page_snippet) > 1000:
            full_page_snippet = full_page_snippet[:1000] + "..."
        candidates.append(f"Page {page_num}: {full_page_snippet}")
            
    if not candidates:
        return []
        
    candidates_str = "\n".join(candidates)
    
    prompt = (
        "Identify which of the following candidate pages directly support the facts stated in the answer.\n"
        "Filter out any pages that do not contain information supporting any part of the answer.\n\n"
        f"Answer:\n{answer}\n\n"
        f"Candidate Pages:\n{candidates_str}\n\n"
        "Return only a comma-separated list of the page numbers (e.g., '9, 63') that actually supported the answer. "
        "Do not include any other text."
    )
    
    try:
        logger.info("Attempting to filter used pages using primary model.")
        response = primary_model.invoke(prompt)
        raw_output = _extract_text(response.content)
    except Exception as e:
        logger.warning(f"Failed to filter pages using primary model: {str(e)}. Trying fallback model.")
        try:
            response = fallback_model.invoke(prompt)
            raw_output = _extract_text(response.content)
        except Exception as e_fallback:
            logger.error(f"Both models failed to filter pages: {str(e_fallback)}. Returning all candidate pages.")
            return unique_pages
            
    # Parse numbers from raw_output
    used_pages = []
    numbers = re.findall(r'\d+', raw_output)
    for num_str in numbers:
        try:
            val = int(num_str)
            if val in unique_pages:
                used_pages.append(val)
        except ValueError:
            pass
            
    filtered = sorted(list(set(used_pages)))
    logger.info(f"Filtered source pages: {filtered} (from candidates: {unique_pages})")
    return filtered


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

    # 2. Retrieve chunks (decomposes the query if compound)
    docs, is_compound = retrieve_and_combine_docs(standalone_question, db, primary_model, fallback_model)

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
        "If the user's question contains multiple distinct sub-questions/asks, you must address "
        "each part clearly in your answer (e.g., using separate sentences or a bulleted list).\n"
        "Do NOT include any inline page citations or bracketed page references (such as '[Page 27]' or 'Page 27') "
        "in your final text response. Just provide the textual answer directly (e.g., you are allowed and encouraged "
        "to include names, dates, financial statistics, or employee counts normally in your text).\n"
        "If a specific sub-part of the question cannot be found in the context, state clearly for that part "
        "that it was not found, while still answering the other parts that are present in the context. "
        "Do not refuse the entire query if only one part is missing.\n"
        "Do not use any external knowledge or assumptions. Keep your answer factual, direct, and concise."
    )

    # Reconstruct context from parent chunks for hierarchical RAG, fallback to child if not present
    seen_parents = set()
    seen_contents = set()
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
            if content not in seen_contents:
                seen_contents.add(content)
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
    used_pages = filter_used_pages(answer, docs, primary_model, fallback_model) if has_found else []

    return {
        "answer": answer,
        "source_pages": used_pages,
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

    # 2. Retrieve chunks (decomposes the query if compound)
    docs, is_compound = retrieve_and_combine_docs(standalone_question, db, primary_model, fallback_model)

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
        "If you cannot find the answer in the provided context, reply exactly with: "
        "\"I couldn't find this in the document\" - do not make up information.\n"
        "If the user's question contains multiple distinct sub-questions/asks, you must address "
        "each part clearly in your answer (e.g., using separate sentences or a bulleted list).\n"
        "Do NOT include any inline page citations or bracketed page references (such as '[Page 27]' or 'Page 27') "
        "in your final text response. Just provide the textual answer directly (e.g., you are allowed and encouraged "
        "to include names, dates, financial statistics, or employee counts normally in your text).\n"
        "If a specific sub-part of the question cannot be found in the context, state clearly for that part "
        "that it was not found, while still answering the other parts that are present in the context. "
        "Do not refuse the entire query if only one part is missing.\n"
        "Do not use any external knowledge or assumptions. Keep your answer factual, direct, and concise."
    )

    # Reconstruct context from parent chunks for hierarchical RAG, fallback to child if not present
    seen_parents = set()
    seen_contents = set()
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
            if content not in seen_contents:
                seen_contents.add(content)
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
                yield token_str
                
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
                    yield token_str
                    
            logger.info(f"Stream completed using fallback model: {config.FALLBACK_MODEL}")
        except Exception as e_fallback:
            logger.error(f"Both streaming models failed: {str(e_fallback)}", exc_info=True)
            raise e_fallback

