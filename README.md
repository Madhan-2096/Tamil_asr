# 🎙️ Tamil Automatic Speech Recognition System

**Whisper Small + LoRA Fine-Tuned on Common Voice Tamil**

A production-ready Streamlit web application for Tamil speech-to-text. Upload a
recording (WAV / MP3 / M4A) and get a Tamil transcription powered by OpenAI
Whisper Small with a LoRA adapter trained on the Common Voice Tamil corpus.

## Model

| | |
|---|---|
| Base model | `openai/whisper-small` |
| LoRA adapter | `checkpoint-1128` |
| Fine-tuning method | LoRA (PEFT) |
| Language / task | Tamil · transcribe |
| Trainable parameters | 1,769,472 (0.73%) |
| Total parameters | 243,504,384 |
| Dataset | Common Voice Tamil · 5,000 samples · 22 speakers |

Generation uses beam search (`num_beams=5`, `max_new_tokens=128`,
`no_repeat_ngram_size=3`, `repetition_penalty=1.2`) with the Tamil decoder
prompt from `processor.get_decoder_prompt_ids(language="tamil", task="transcribe")`.

## Setup

### macOS / Linux

```bash
# 1. Create an environm
 venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. ffmpeg is required for MP3/M4A uploads
sudo apt install ffmpeg        # Debian/Ubuntu
```

### Windows PowerShell

```powershell
py -3 -m venv venv
. .\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> Note: `python3` and `source` are shell-specific commands for macOS/Linux. On Windows PowerShell, use `py -3` (or `python`) and the PowerShell activation script above.

If `Activate.ps1` is missing, the virtual environment was not created successfully. Make sure Python is installed and available on your PATH, then rerun the first command from the project root.

If `pip` is still not found after activation, use `python -m pip install -r requirements.txt`.

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

### Windows Command Prompt

```cmd
py -3 -m venv venv
venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

### Checkpoint location

The app looks for the LoRA adapter in this order:

1. `$TAMIL_ASR_CHECKPOINT` (environment variable, if set)
2. `models/checkpoint-1128` (conventional location — copy or symlink it here)
3. `../Tamil_asr_project/checkpoints/checkpoint_1128` (the training project)

```bash
# Example: symlink the trained adapter into the conventional path
mkdir -p models
ln -s ../../Tamil_asr_project/checkpoints/checkpoint_1128 models/checkpoint-1128
```

The directory must contain `adapter_config.json` and `adapter_model.safetensors`.
The base Whisper Small weights download automatically from the Hugging Face Hub
on first run (~1 GB, cached afterwards).

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501. The model loads once per server process
(`@st.cache_resource`); a CUDA GPU is used automatically when available, with
a clearly-indicated CPU fallback otherwise.

## Features

- Drag-and-drop upload (WAV / MP3 / M4A) with in-browser audio preview
- Tamil transcription with copy-to-clipboard and TXT download
- Live status indicators: model loaded · audio uploaded · transcription complete
- Word count, character count, processing time and realtime factor
- Model performance profile (parameters, method, dataset)
- Robust error handling: invalid file type, corrupted audio, missing
  checkpoint, GPU unavailable
- Dark, Tamil-typography-aware UI (Catamaran + Fraunces, kolam-dot texture)

## Project structure

```
tamil_asr_app/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml         # Dark theme + 50 MB upload limit
└── models/
    └── checkpoint-1128/    # LoRA adapter (you provide this)
```

## Notes

- Whisper processes a 30-second window; longer clips are transcribed for the
  first 30 seconds only (the UI warns when this applies).
- All uploaded audio is resampled to 16 kHz mono with librosa before feature
  extraction.
