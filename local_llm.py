import gradio as gr
from ollama import Client
import datetime
from pathlib import Path


# Replace with your actual model 
DEFAULT_MODEL = "pulled_model_name"  

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True) 

# Initialize Ollama client
client = Client()

# Helper: get per-session log file
def get_log_file(session_id: str) -> Path:
    return LOG_DIR / f"chat_{session_id}.txt"

# Initialize session: clear log file
# Note: Gradio automatically injects gr.Request into this function
def init_session(request: gr.Request) -> str:
    session_id = request.session_hash
    log_file = get_log_file(session_id)
    log_file.write_text("")
    return session_id

# Read conversation history from log file
def read_history(log_file: Path) -> list[dict]:
    history = []
    if log_file.exists():
        for line in log_file.read_text().splitlines():
            if "| USER:" in line:
                content = line.split("| USER:")[1].strip()
                history.append({"role": "user", "content": content})
            elif "| AI:" in line:
                content = line.split("| AI:")[1].strip()
                history.append({"role": "assistant", "content": content})
    return history

# Log a message to the log file with timestamp
def log_message(log_file: Path, role: str, text: str):
    timestamp = datetime.datetime.now().isoformat()
    entry = f"{timestamp} | {role}: {text}\n"
    log_file.open("a", encoding="utf-8").write(entry)

# New session handler: resets log and outputs clear UI
def new_session(request: gr.Request):
    session_id = request.session_hash
    log_file = get_log_file(session_id)
    log_file.write_text("")
    return session_id, "", ""

# Main chat function: rebuilds history, streams response, and logs
def chat_model(prompt: str, session_id: str):
    log_file = get_log_file(session_id)
    history = read_history(log_file)

    # Append user message and log it
    history.append({"role": "user", "content": prompt})
    log_message(log_file, "USER", prompt)

    response = ""
    try:
        for chunk in client.chat(model=DEFAULT_MODEL, messages=history, stream=True):
            delta = chunk["message"]["content"]
            response += delta
            yield response, log_file.read_text(), gr.update(value="")
    except Exception as e:
        # Yield a friendly error message
        error_msg = f"Error: {e}"
        yield error_msg, log_file.read_text(), gr.update(value="")
        return

    # After complete, log assistant reply
    log_message(log_file, "AI", response)

# Custom CSS for UI styling. You can define your own styles here too. 
# I only wanted smaller font for the log to not take too much space.
css = r"""
#log_out textarea      { font-size: 10px; }
"""

# Build Gradio interface
demo = gr.Blocks(css=css)
with demo:
    gr.Markdown("# Local LLM Chat Interface")
    session_state = gr.State()
    demo.load(fn=init_session, outputs=[session_state])

    # Control buttons
    with gr.Row():
        new_button = gr.Button("New Session")

    # Chat components
    prompt_in = gr.Textbox(
        lines=1,
        placeholder="Enter prompt…",
        label="Prompt",
        autofocus=True
    )
    response_out = gr.Textbox(
        label="Model Response (Streaming)",
        elem_id="response_out"
    )
    log_out = gr.Textbox(
        label="Session Log",
        lines=5,
        interactive=False,
        elem_id="log_out"
    )

    # Submit and new session events
    prompt_in.submit(
        fn=chat_model,
        inputs=[prompt_in, session_state],
        outputs=[response_out, log_out, prompt_in],
    )
    new_button.click(
        fn=new_session,
        inputs=None,
        outputs=[session_state, response_out, log_out]
    )

    demo.queue()
    # Replace "your_ip_address" with the actual IP address of your server
    # This will allow you to access the Gradio interface from other devices on the same network.
    # If you only wan to access locally use 0.0.0.0 
    demo.launch(server_name="your_ip_address", server_port=7860)
