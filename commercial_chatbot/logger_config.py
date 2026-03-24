# Step 11 - Logging and debugging visibility
"""
Why it matters? Without logging, the bot feels like a black box. 
- When it gives a bad answer, you want to know: what intent it detected, whether retrieval ran...
- With logging, you can debug questions like: why did it refuse here? Why did escalation trigger?...
- It's observability. 
Coding thinking process:
1. What do I want to observe
    - bot important stages: user sends message->intent/routing logic runs->retrieval may run
        ->answer generation may run->refusal may happen->escalation may happen->final response is returned
    - what should I log: incoming user text->detected intent/route->selected tool if any
        ->retrieved chunks and scores->evidence strength->refusal reason->escalation reason->final output summary
2. Where should logging live? 
    - common mistake: write print() everywhere
    - Better: create one small logging utility file | configure logger once | import logger anywhere you need it
        - Instead of random prints, you get structured reusable logging.
3. What level of logging should I use? - Python logging usually has levels like:
    - Threshold levels: CRITICAL (severe failure) > ERROR (failure) > WARNING (caution)> INFO (normal info)> DEBUG (details)
    - DEBUG (very detailed internal info) - here for retrieved chunks and scores
    - INFO (normal workflow steps) - here for normal flow
    - WARNING (suspicious or weak evidence) - here for refusal/weak evidence/escalation
    - ERROR (actual failures exceptions) - here for crashes
4. Should logs go only to terminal? - At minimum, terminal is enough. 
    - For a stronger project, it's better to support: console logs | optional log file
    - That way you can: watch the bot live in terminal | save logs for debugging later
5. What should the log format look like? - Contains: timestamp | level | module name | message
    - eg.** 2026-03-22 17:20:10 | INFO | pipeline | Intent detected: order_status **

"""
import logging # structured logs in python
from pathlib import Path # import Path so we can create a logs folder/file

def setup_logger(name: str="chatbot", log_file:str="logs/chatbot.log")-> logging.Logger:
    """
    Create and return a configured logger.(create clean summary logs to terminal + detailed logs to file)
    Why this function exists:
    - centralize logging setup in one placce
    - avoid repeating formatter/handler code in many files
    - allow both console logging and file logging
    Parameters: - name: logger name shown in output | log_file: path to the log file
    Returns: configured logger object
    """
    # Create a logger object with the given name. (like the main logging channel for your app) 
    logger = logging.getLogger(name)

    # If handlers alreadly exist, return the logger immediately.
    # This avoids duplicate logs when setup_logger() is called multiple times. 
    if logger.handlers: return logger

    # Set the minimum level the logger will accept.
    # DEBUG means all elvels above it are allowed too: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger.setLevel(logging.DEBUG)

    # Make sure the parent folders for the log file exist.
    # Eg: if log_file is "logs/chatbot.log", this creates "logs/" if needed. 
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Define how each log line should look.
    # %(asctime)s = timestamp
    # %(levelname)s = INFO / DEBUG / WARNING / ERROR
    # %(name)s = logger name
    # %(message)s = actual log message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    # Create a handler for showing logs in the terminal.
    console_handler = logging.StreamHandler()
    # Console shows INFO and above by default. This keeps terminal cleaner than full DEBUG spam. 
    console_handler.setLevel(logging.INFO)
    # Apply the formatter to the console handler
    console_handler.setFormatter(formatter)
    # Create a handler for saving logs to a file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    # File gets DEBUG and above, so it stores more detailed info. 
    file_handler.setLevel(logging.DEBUG)
    # Apply the formatter to the file handler too
    file_handler.setFormatter(formatter)
    # Attach both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # Prevent logs from being passed upward to the root logger.
    # This avoids duplicate output in some environments.
    logger.propagate = False
    # Return the fully configured logger
    return logger



