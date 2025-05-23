# Local LLM Chat Interface

A lightweight Gradio application that connects to an Ollama-powered local LLM for live, streaming chat with per-session logging.

## Features

* **Session Logs**: All conversations are appended to timestamped log files in the `logs/` directory for future use.
* **Streaming Responses**: Model output is streamed chunk-by-chunk for a natural typing effect.
* **Error Handling**: Captures exceptions from the Ollama client and displays friendly error messages.
* **Local Access with Multiple Devices**: Use your own IP address to access the interface from all the devices on your local network.

## Setup

1. **Clone or download** this repository.
2. **Install dependencies**:

   ```bash
   pip install gradio ollama-client pathlib
   ```
3. **Configure** the script:

   * Open `local_llm.py` and set `DEFAULT_MODEL` to your Ollama model name.
   * In the `demo.launch()` call, replace `server_name` with `0.0.0.0` (external access) or `127.0.0.1` (local-only).
4. **Run**:

   ```bash
   python local_llm.py
   ```
5. **Access** the app at:

   ```
   http://<server_name>:7860
   ```

## Usage

* **Enter Prompt**: Type your query in the input box and press **Enter**.
* **View Stream**: Watch the model’s response appear in real-time.
* **New Session**: Click **New Session** to clear logs and start over.

---

> Feel free to customize and extend this interface to suit your project needs!

> Misagh