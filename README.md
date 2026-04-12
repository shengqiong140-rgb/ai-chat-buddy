# AI Chat Buddy

A local AI voice assistant that runs entirely on your Mac. Talk to it like a friend — it listens, thinks, and talks back. No cloud AI APIs needed, no subscription fees.

## Features

- **Continuous conversation** — Just speak, no buttons to press. Voice Activity Detection (VAD) automatically starts recording when you talk and stops when you pause
- **Chinese & English** — Supports both languages, auto-detects which one you're speaking
- **Runs locally** — AI model runs on your Mac via Ollama, no data sent to cloud for AI processing
- **Two TTS options** — High quality (edge-tts, needs internet) or offline (macOS say)
- **Kid-friendly** — Built-in safety prompt that filters inappropriate content
- **Conversation memory** — Remembers context within a session

## How It Works

```
You speak → VAD detects voice → Whisper transcribes → Ollama AI generates reply → TTS speaks back
```

| Component | What it does | Runs locally? |
|-----------|-------------|---------------|
| **Whisper** (medium) | Speech-to-text, auto Chinese/English | Yes |
| **Ollama** (qwen2.5:14b) | AI conversation | Yes |
| **edge-tts** | Text-to-speech (high quality) | Needs internet |
| **macOS say** | Text-to-speech (offline fallback) | Yes |

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **16GB+ RAM** (24GB recommended for qwen2.5:14b)
- **Microphone** (built-in, AirPods, or USB mic)
- **Python 3.11+**

## Setup

### 1. Install Ollama & download AI model

```bash
brew install ollama
ollama pull qwen2.5:14b
```

### 2. Clone this repo

```bash
git clone https://github.com/shengqiong140-rgb/ai-chat-buddy.git
cd ai-chat-buddy
```

### 3. Install Python dependencies

```bash
brew install uv
uv sync --python 3.12
uv pip install webrtcvad-wheels edge-tts
```

### 4. Download NLTK data

```bash
.venv/bin/python -c "import nltk; nltk.download('punkt_tab')"
```

### 5. Run

```bash
# High quality TTS (needs internet)
.venv/bin/python app.py

# Offline mode (macOS built-in TTS)
.venv/bin/python app.py --tts-engine say
```

That's it! Just start talking.

## Options

```bash
.venv/bin/python app.py [options]

--model           Ollama model name (default: qwen2.5:14b)
--whisper-model   Whisper model size (default: medium)
--tts-engine      TTS engine: edge or say (default: edge)
--silence-duration  Seconds of silence before stopping (default: 0.8)
--vad-aggressiveness  VAD sensitivity 0-3 (default: 2)
--temperature     LLM temperature (default: 0.7)
```

## Examples

```bash
# Use a smaller/faster model
.venv/bin/python app.py --model qwen2.5:7b

# More responsive (stops recording faster)
.venv/bin/python app.py --silence-duration 0.5

# Fully offline (no internet needed at all)
.venv/bin/python app.py --tts-engine say
```

## License

MIT
