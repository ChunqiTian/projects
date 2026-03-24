
from typing import List, Tuple, Dict, Any # Import typing helpers for clearer function type hints
from logging import Logger # Import Logger type so the type hint is clear

def log_retrieval_results(
        logger: Logger,
        results: List[Tuple[Dict[str, str], float]],
        max_items: int=3
) -> None:
    """
    Log the top retrieved chunks in a readable way. 
    Why this function exists:
    - retrieval results can be messy to inspect
    - centralize the formatting of retrieval logs
    - make debugging easier when the wrong chunks are returned
    Params: 
    - logger: configured logger 
    - results: list like [(chunk_dict, score), ...]
    - max_items: only log the top N chunks to avoid huge logs
    """
    # If no results were found, log that clearly and stop
    if not results: 
        logger.debug("Retrieval returned no chunks.")
        return
    # Log how many results came back in total
    logger.debug(f"Retrieval returned {len(results)} chunk(s).")
    # Loop through only the top few results for readability. 
    for rank, (chunk, score) in enumerate(results[:max_items], start=1):
        # Safely get chunk_id; use fallback if missing
        chunk_id = chunk.get("chunk_id", "unknown_chunk")
        # Safely get text; default to empty string if key missing
        text = chunk.get("text", "")
        # Make preview shorter so logs stay readable. 
        preview = text[:120].replace("\n", " ").strip()
        # Write one clean debug line per retrieved chunk
        logger.debug(
            f"Top {rank} | chunk_id={chunk_id} | score={score: 4f} | preview={preview}"
        )

def log_final_decision(logger: Logger, route: str, confidence: float, needs_human:bool) -> None:
    """
    Log the final system decision in one place. - Prints the retrieval results in a clean readable format. 
    Why this function exists: 
    - the final route is one of the most important observations
    - helps debugging the whole pipeline quickly
    - standardizes final summary logs
    Params:
    - logger: configured logger instance
    - route: final route/intention chosen by the bot
    - confidence: model/system confidence
    - needs_human: whether escalation is needed
    """
    # Log final summary in a consistent compact format
    logger.info(f"Final decision | route={route} | confidence={confidence:.2f} | needs_human={needs_human}")







