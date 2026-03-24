# Step 10 - UI Layer with Streamlit
"""
Big-picture goal
- user types in a chat box
- app sends text to your backend
- backend returns structured output
- UI displays: answer | citations | confidence | escalation flag | optional debug info
mental model: User Input -> Streamlit UI -> handle_message() -> BotResponse -> UI rendering
Code thinking process:
1. What UI should do?
    - collect user input
    - preserve chat history
    - call backend logic
    - display response nicely
    - expose extra metadata
    - optionally show debug info 
2. What should remain backend vs frontend?
 - backend: routing | retrieval | answer generation | refusal | escalation | confidence scoring
 - UI handle: user interaction | layout | rendering output | storing session chat history | showing debug details
3. Structured output: answer | citation | confidence | needs_human | debug_chunks
4. Streamlit layout: page title | short proj description | chat history | chat input | 
        assistant response | metadata section | expandable debug section
5. How should chat histoy work? - In streamlit, every interaction reruns the script from top to bottom.
    So if you do not save conversation history, it disappears. 
    -> st.session_state # Store past user and assistant message
6. How should debug info be displayed?
    - debug info is helpful for: portfolio demo | testing retrieval | seeing to chunks | understanding why the answer was produced
    - keep normal answer visible and put debug details inside an st.expander()

Functions
1. initialize_session_state() - This function ensures chat history survives across Streamlit reruns
    - Streamlit re-executes the whole script whenever the user interacts with the page
    - Without session state: user message appears | page reruns | old messages disappear
    - st.session_state.messages # acts like temporary memory for the app session
2. render_header() - This func renders the static top section of the page
    - Separating header logic from chat logic keeps your code cleaner.
3. render_chat_history() - This func redraws all previous messages from session state
    - Bcz the script reruns every time, the UI must reconstruct the conversation from stored history. 
    - This func loops through stored messages and re-renders them one by one. 
    - It distinguishes: normal text content & structured assistant metadata
    - It's why assistant messages can show not only the answer, but also: confidence | citations| escalation flag | debug chunks
4. render_response_metadata(response) - This func displays the structured technical details under the assistant's answer
    - A bot proj becomes much more impressive when it is not just "chat text", but also shows inspectable signals
    - The func displays: confidence | needs human | citations | debug chunks
5. process_user_message(user_text) - This func is the bridge between fronted and backend
    - handle_message(user_text)
    - Later you can add: exception handling | timing measurements | logging | alternate models | tracing
6. add_user_message(user_text) - Saves a user message into chat history
    - It keeps session-state writing centralized and consistent.
    - st.session_state.messages.append(...) # over the file, you keep message insertion neat and reusable
7. add_assistant_message(response) - Stores the assistant response in history
    - You need more than just answer text and the structured response object. 
    - content -> answer text; response_obj -> full metadata dict
8. main() - controls the entire app flow. 
    - The orchestration func: setup -> render old msm -> get new input -> call backend -> render response -> save history
    - This makes the application predictable and easy to debug. 
Later upgrades: Sidebar controls | Clear chat button | Color-coded status badges | Retrieved chunk viewer | Source doc goruping
    
"""
import streamlit as st # Streamlit UI library
from main import handle_message # import your backend entry function
from schemas import BotResponse # import structured response model
from memory import ConversationState

# Set basic page configuration
st.set_page_config(
    page_title="Commercial Support Chatbot", # browser tab title
    page_icon="💬", # icon shown in browser tab
    layout="wide" # use wider layout for cleaner spacing
)

def initialize_session_state() -> None:
    """
    Initialize Streamlist session state variables
    Why - Streamlit reruns script each interaction, history would disappear without session state
        - centralizing setup keeps the app cleaner
    What it initializes: messages - stores the full chat history
    """
    if "messages" not in st.session_state: st.session_state.messages=[] 
        # check whether messages already exist, create an empty chat history list
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = ConversationState()   

def render_header() -> None:
    """
    Render the app title and short description. 
    Why this func exists:
    - keeps top-of-page UI separate from chat logic
    - improves readability of the main script
    - makes future edits easier
    What it shows: project title | short explanation of what the demo does
    """
    st.title("💬 Commercial Support Chatbot") # main app title
    st.write(
        "A portfolio demo of a support chatbot with retrieval, citatiions,"
        "confidence scoring, and escalation signals."
    ) # Short proj description 

def render_chat_history() -> None:
    """
    Render all previous chat messages stored in session state.
    Why this function exists:
    - shows persistent conversation history
    - keeps rendering logic separated from backend call logic
    Expected message format:
    each item in st.session_state.messages should look like:
    {
        "role": "user" or "assistant",
        "content": "message text",
        "response_obj": optional BotResponse data for assistant messages
    }
    """
    for message in st.session_state.messages: # loop through all saveed messages
        with st.chat_message(message["role"]): # create a-chat bubble for the role
            st.markdown(message["content"]) # show the main text content
            # Only assistant messages may have metadata to render
            if message["role"] == "assistant" and "response_obj" in message: 
                render_response_metadata(message["response_obj"]) # show citations/confidence/debug

def render_response_metadata(response: dict) -> None:
    """
    Render extra metadata for an assistant response. 
    Why this func exists:
    - separates main answer text from technical metadata
    - keeps the UI organized
    - allows a cleaner portfolio demo
    Metadata shown: citations | confidence | escalation flag | optional debug chunks
    """
    citations = response.get("citations", []) # get citations list safely
    confidence = response.get("confidence", 0.0) # get confidence safely
    needs_human = response.get("needs_human", False) # get escalation flag safely
    debug_chunks = response.get("debug_chunks") # get optional debug chunks

    st.markdown("---") # visual separator under answer
    col1, col2 = st.columns(2) # create two cols for compact metadata display
    with col1: st.write(f'**Confidence:** {confidence:.2f}') # show confidence rounded 0.00 in 1st col
    with col2: st.write(f"**Needs human:** {'Yes' if needs_human else 'No'}") # show escalation flag in 2nd col
    if citations: # only render citations if they exist
        st.write("**Citations:**") # citations section label
        for citation in citations: # loop through each citation
            st.code(citation) # show citation in code-style box for clarity

    if debug_chunks: # only render debug section if debug chunks exist
        with st.expander("Debug: Retrieved Chunks"): # collapsible debug area
            for i, chunk in enumerate(debug_chunks, start=1): # loop with numbering
                st.markdown(f"**Chunk {i}**") # show chunk number
                st.write(f"**Chunk ID:** {chunk.get('chunk_id', 'N/A')}") # Show chunk ID
                st.write(f"**Score:** {chunk.get('score', 'N/A')}") # show chunk score
                st.write(f"**Text:** {chunk.get('text', '')}") # show chunk text
                st.markdown("---") # separator between chunks
    
def process_user_message(user_text: str) -> BotResponse:
    """
    Send user input to the backend and return the structured response
    Why this func exists:
    - creates a clean boundary between UI and backend
    - makes it easier to swap backend implementations later
    - lets us add error handling in one place
    Input: user_text: the user's message string
    output: BotResponse obj returned by handle_message()
    """
    response = handle_message(user_text, st.session_state.conversation_state) # call your backend logic
    return response # return the backend result directly

def add_user_message(user_text: str) -> None:
    """
    Save a user message into session history. 
    Why this func exists: centralizes message formatting | keeps the main app flow cleaner
    """
    st.session_state.messages.append(
        {
            "role": "user", # mark this message as from the user
            "content": user_text # store the text content
        }
    )

def add_assistant_message(response: BotResponse) -> None:
    """
    Save an assistant response into session history. 
    Why this func exists: stores both answer & metadata | makes re-rendering chat history easy on later reruns
    input: response: BotResponse obj from the backend
    """
    st.session_state.messages.append(
        {
            "role": "assistant", # mark this message as from the assistant
            "content": response.answer, # store hte main answer text
            "response_obj": response.model_dump() # store full structured response as a dict
        }
    )

def main() -> None:
    """
    Main Streamlit application function.

    Why this function exists:
    - it controls the top-level UI flow
    - initializes state
    - renders previous messages
    - accepts new input
    - gets backend response
    - stores and displays results

    High-level flow:
    1. initialize session state
    2. render header
    3. render past chat history
    4. wait for user input
    5. send input to backend
    6. display and save response
    """
    initialize_session_state()  # make sure session variables exist
    render_header()  # show title and intro
    render_chat_history()  # display previous conversation

    user_text = st.chat_input("Ask a support question...")  # create chat input box

    if user_text:  # only continue if the user submitted something
        add_user_message(user_text)  # save the user message first

        with st.chat_message("user"):  # display the new user message immediately
            st.markdown(user_text)  # show user text

        response = process_user_message(user_text)  # call backend to get bot response

        with st.chat_message("assistant"):  # create assistant message bubble
            st.markdown(response.answer)  # show the assistant's answer
            render_response_metadata(response.model_dump())  # show citations/confidence/debug

        add_assistant_message(response)  # save assistant response into history


if __name__ == "__main__":
    main()  # run the Streamlit app


