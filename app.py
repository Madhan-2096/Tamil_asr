"""
Tamil Automatic Speech Recognition System — Streamlit front-end.

Whisper Small + LoRA (PEFT) fine-tuned on Common Voice Tamil.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Heavy ML imports are wrapped so a missing dependency produces a readable
# error inside the app instead of a blank Streamlit crash page.
try:
    import torch
    import librosa
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel

    IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover
    IMPORT_ERROR = _e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "openai/whisper-small"
TARGET_SAMPLE_RATE = 16_000
SUPPORTED_TYPES = ["wav", "mp3", "m4a"]
MAX_UPLOAD_MB = 50

APP_DIR = Path(__file__).resolve().parent

# Checkpoint resolution order: explicit env var first, then the conventional
# `models/` location, then the training project's own checkpoints directory.
CHECKPOINT_CANDIDATES = [
    os.environ.get("TAMIL_ASR_CHECKPOINT", ""),
    str(APP_DIR / "models" / "checkpoint-1128"),
    str(APP_DIR / "models" / "checkpoint_1128"),
    str(APP_DIR.parent / "Tamil_asr_project" / "checkpoints" / "checkpoint_1128"),
]

MODEL_CARD = {
    "Base Model": "openai/whisper-small",
    "LoRA Adapter": "checkpoint-1128",
    "Language": "Tamil (ta)",
    "Task": "Speech-to-Text",
    "Fine-Tuning Method": "LoRA (PEFT)",
    "Dataset": "Common Voice Tamil",
    "Samples": "5,000",
    "Speakers": "22",
    "Trainable Parameters": "1,769,472",
    "Total Parameters": "243,504,384",
}

GENERATION_KWARGS = dict(
    num_beams=5,
    max_new_tokens=128,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
)

# ---------------------------------------------------------------------------
# Page setup & theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tamil ASR · Whisper + LoRA",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Catamaran:wght@300;400;600;800&family=Spline+Sans+Mono:wght@400;500&display=swap');

:root {
    --ink:        #0a0d12;
    --ink-raised: #121722;
    --ink-border: #242c3b;
    --bronze:     #e2a33b;
    --bronze-dim: rgba(226, 163, 59, 0.14);
    --kumkum:     #c4453a;
    --leaf:       #5cb87a;
    --paper:      #e9e4d8;
    --paper-dim:  #9aa3b2;
}

/* Kolam-dot texture over the deep-ink canvas */
.stApp {
    background-color: var(--ink);
    background-image: radial-gradient(circle, rgba(226,163,59,0.05) 1px, transparent 1px);
    background-size: 26px 26px;
}

html, body, [class*="css"], .stMarkdown, p, label {
    font-family: 'Catamaran', 'Noto Sans Tamil', sans-serif;
    color: var(--paper);
}

h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--paper); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1119 0%, #0a0d12 100%);
    border-right: 1px solid var(--ink-border);
}
[data-testid="stSidebar"] hr { border-color: var(--ink-border); }

/* Hero */
.hero {
    position: relative;
    padding: 2.6rem 2.4rem 2.2rem;
    border: 1px solid var(--ink-border);
    border-radius: 18px;
    background:
        radial-gradient(1100px 380px at 12% -30%, rgba(226,163,59,0.16), transparent 60%),
        linear-gradient(160deg, #131927 0%, #0c1018 70%);
    overflow: hidden;
    margin-bottom: 1.4rem;
}
.hero::after {
    content: "த";
    position: absolute;
    right: 1.5rem; top: 50%;
    transform: translateY(-52%);
    font-family: 'Catamaran', sans-serif;
    font-weight: 800;
    font-size: 11rem;
    line-height: 1;
    color: rgba(226,163,59,0.07);
    pointer-events: none;
}
.hero .eyebrow {
    font-family: 'Spline Sans Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--bronze);
    margin-bottom: 0.7rem;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 0.45rem;
    letter-spacing: -0.01em;
}
.hero .subtitle { color: var(--paper-dim); font-size: 1.04rem; margin: 0; }

/* Status pills */
.status-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1.6rem; }
.pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.42rem 1rem;
    border-radius: 999px;
    border: 1px solid var(--ink-border);
    background: var(--ink-raised);
    font-family: 'Spline Sans Mono', monospace;
    font-size: 0.78rem;
    color: var(--paper-dim);
}
.pill.on {
    border-color: rgba(92,184,122,0.45);
    color: var(--leaf);
    box-shadow: 0 0 14px rgba(92,184,122,0.12) inset;
}
.pill.err { border-color: rgba(196,69,58,0.55); color: #e0786f; }
.pill .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.pill.on .dot { box-shadow: 0 0 8px currentColor; }

/* Section labels */
.section-label {
    font-family: 'Spline Sans Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color: var(--bronze);
    border-bottom: 1px solid var(--ink-border);
    padding-bottom: 0.45rem;
    margin: 1.6rem 0 1rem;
}

/* Uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--ink-raised);
    border: 1.5px dashed rgba(226,163,59,0.4);
    border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--bronze); }

/* Transcription output */
.stTextArea textarea {
    background: var(--ink-raised) !important;
    border: 1px solid var(--ink-border) !important;
    border-radius: 12px !important;
    color: var(--paper) !important;
    font-family: 'Catamaran', 'Noto Sans Tamil', sans-serif !important;
    font-size: 1.35rem !important;
    line-height: 1.85 !important;
}

/* Metric cards */
.metric-card {
    background: var(--ink-raised);
    border: 1px solid var(--ink-border);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    height: 100%;
}
.metric-card .k {
    font-family: 'Spline Sans Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--paper-dim);
    margin-bottom: 0.3rem;
}
.metric-card .v {
    font-family: 'Fraunces', serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--bronze);
}
.metric-card .s { font-size: 0.8rem; color: var(--paper-dim); }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(180deg, #f0b554, var(--bronze));
    color: #1a1206;
    font-family: 'Catamaran', sans-serif;
    font-weight: 800;
    letter-spacing: 0.04em;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    box-shadow: 0 4px 22px rgba(226,163,59,0.35);
    transform: translateY(-1px);
    color: #1a1206;
}

/* Sidebar info table */
.info-table { width: 100%; font-size: 0.86rem; border-collapse: collapse; }
.info-table td { padding: 0.32rem 0; border-bottom: 1px dotted var(--ink-border); vertical-align: top; }
.info-table td:first-child { color: var(--paper-dim); padding-right: 0.8rem; white-space: nowrap; }
.info-table td:last-child { color: var(--paper); font-weight: 600; text-align: right; }

footer, #MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached once per server process)
# ---------------------------------------------------------------------------


def resolve_checkpoint() -> Path | None:
    """Return the first existing LoRA checkpoint directory, if any."""
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate and (Path(candidate) / "adapter_config.json").is_file():
            return Path(candidate)
    return None


@st.cache_resource(show_spinner=False)
def load_asr_pipeline() -> dict:
    """
    Load processor, base Whisper model and the LoRA adapter exactly once.

    Returns a bundle dict; raises FileNotFoundError when the adapter
    checkpoint cannot be located.
    """
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError(
            "LoRA checkpoint not found. Place it at 'models/checkpoint-1128' "
            "or set the TAMIL_ASR_CHECKPOINT environment variable."
        )

    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(model, str(checkpoint))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Tamil transcription prompt; also clear any baked-in forced ids so the
    # prompt below is the single source of truth at generation time.
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="tamil",
        task="transcribe",
    )
    model.generation_config.forced_decoder_ids = None

    return {
        "processor": processor,
        "model": model,
        "device": device,
        "forced_decoder_ids": forced_decoder_ids,
        "checkpoint": str(checkpoint),
    }


# ---------------------------------------------------------------------------
# Audio handling & inference
# ---------------------------------------------------------------------------


def decode_audio(raw: bytes, suffix: str) -> tuple[np.ndarray, float]:
    """
    Decode uploaded bytes to a 16 kHz mono float waveform.

    Goes through a temp file so librosa's backends (soundfile / audioread +
    ffmpeg) can sniff the container format — required for m4a.
    """
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        audio, _ = librosa.load(tmp_path, sr=TARGET_SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Decoded waveform is empty or contains invalid samples.")

    duration = audio.shape[0] / TARGET_SAMPLE_RATE
    return audio, duration


def transcribe(bundle: dict, audio: np.ndarray) -> tuple[str, float]:
    """Run Whisper feature extraction + beam-search generation. Returns (text, seconds)."""
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]

    started = time.perf_counter()

    inputs = processor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        try:
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=bundle["forced_decoder_ids"],
                **GENERATION_KWARGS,
            )
        except (ValueError, TypeError):
            # Newer transformers releases reject forced_decoder_ids in favour
            # of explicit language/task arguments — same Tamil prompt either way.
            predicted_ids = model.generate(
                input_features,
                language="tamil",
                task="transcribe",
                **GENERATION_KWARGS,
            )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return text, time.perf_counter() - started


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def status_pill(label: str, on: bool, error: bool = False) -> str:
    state = "err" if error else ("on" if on else "")
    return f'<span class="pill {state}"><span class="dot"></span>{label}</span>'


def metric_card(key: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card"><div class="k">{key}</div>'
        f'<div class="v">{value}</div>{sub_html}</div>'
    )


def copy_button(text: str) -> None:
    """Clipboard copy button rendered in a component iframe (needs JS)."""
    payload = json.dumps(text)
    components.html(
        f"""
        <button id="copy" style="
            width:100%; padding:0.62rem 1rem; cursor:pointer;
            font-family:'Catamaran',sans-serif; font-weight:800; font-size:0.95rem;
            color:#e2a33b; background:#121722; border:1px solid #3a4458;
            border-radius:10px; letter-spacing:0.04em;
        ">📋 Copy Transcription</button>
        <script>
            const btn = document.getElementById("copy");
            btn.addEventListener("click", async () => {{
                await navigator.clipboard.writeText({payload});
                btn.textContent = "✓ Copied!";
                setTimeout(() => btn.textContent = "📋 Copy Transcription", 1600);
            }});
        </script>
        """,
        height=52,
    )


def render_sidebar(bundle: dict | None) -> None:
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Fraunces,serif;font-size:1.35rem;font-weight:700;">'
            '🎙️ Tamil ASR</div>'
            '<div style="color:#9aa3b2;font-size:0.85rem;margin-bottom:0.6rem;">'
            'Speech-to-Text · தமிழ்</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown('<div class="section-label">Project</div>', unsafe_allow_html=True)
        st.markdown(
            "End-to-end Tamil speech recognition built by parameter-efficient "
            "fine-tuning of OpenAI **Whisper Small** with **LoRA** adapters on "
            "the **Common Voice Tamil** corpus."
        )

        st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
        model_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in [
                ("Base model", "Whisper Small"),
                ("Adapter", "checkpoint-1128"),
                ("Training method", "LoRA"),
                ("Language", "Tamil"),
                ("Task", "Transcribe"),
            ]
        )
        st.markdown(f'<table class="info-table">{model_rows}</table>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
        data_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in [
                ("Corpus", "Common Voice Tamil"),
                ("Samples", "5,000"),
                ("Speakers", "22"),
                ("Sampling rate", "16 kHz"),
            ]
        )
        st.markdown(f'<table class="info-table">{data_rows}</table>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Runtime</div>', unsafe_allow_html=True)
        if bundle is not None:
            if bundle["device"] == "cuda":
                st.success(f"GPU · {torch.cuda.get_device_name(0)}")
            else:
                st.warning("GPU unavailable — running on CPU (slower inference).")
            st.caption(f"Adapter path: `{bundle['checkpoint']}`")
        else:
            st.error("Model not loaded.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def main() -> None:
    if IMPORT_ERROR is not None:
        st.error(
            "Missing ML dependencies — install them with "
            "`pip install -r requirements.txt`.\n\n"
            f"Details: {IMPORT_ERROR}"
        )
        st.stop()

    # ---- Model bootstrap -------------------------------------------------
    bundle: dict | None = None
    load_error: str | None = None
    with st.spinner("Loading Whisper Small + LoRA adapter…"):
        try:
            bundle = load_asr_pipeline()
        except FileNotFoundError as exc:
            load_error = str(exc)
        except Exception as exc:  # corrupted weights, OOM, etc.
            load_error = f"Failed to initialise the model: {exc}"

    render_sidebar(bundle)

    # ---- Hero ------------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Speech · தமிழ் · Recognition</div>
            <h1>Tamil Automatic Speech Recognition System</h1>
            <p class="subtitle">Whisper Small + LoRA Fine-Tuned on Common Voice Tamil</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Status indicators -------------------------------------------------
    audio_ready = st.session_state.get("audio_name") is not None
    done = bool(st.session_state.get("transcription"))
    st.markdown(
        '<div class="status-row">'
        + status_pill("Model Loaded", bundle is not None, error=bundle is None)
        + status_pill("Audio Uploaded", audio_ready)
        + status_pill("Transcription Complete", done)
        + "</div>",
        unsafe_allow_html=True,
    )

    if load_error:
        st.error(f"🚫 {load_error}")

    # ---- Upload ------------------------------------------------------------
    st.markdown('<div class="section-label">01 · Upload Audio</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag and drop a Tamil speech recording",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        help="Supported formats: WAV, MP3, M4A · Whisper transcribes the first 30 seconds.",
    )

    audio: np.ndarray | None = None
    duration = 0.0

    if uploaded is not None:
        suffix = uploaded.name.rsplit(".", 1)[-1].lower()
        if suffix not in SUPPORTED_TYPES:
            st.error(f"🚫 Unsupported file type: `.{suffix}`. Please upload WAV, MP3 or M4A.")
        elif uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"🚫 File exceeds the {MAX_UPLOAD_MB} MB limit.")
        else:
            raw = uploaded.getvalue()
            try:
                audio, duration = decode_audio(raw, suffix)
            except Exception:
                st.error(
                    "🚫 Could not decode this file — it appears corrupted or is not "
                    "valid audio. (M4A/MP3 decoding also requires ffmpeg on the host.)"
                )
            else:
                # New file invalidates the previous transcription.
                if st.session_state.get("audio_name") != uploaded.name:
                    st.session_state.pop("transcription", None)
                st.session_state["audio_name"] = uploaded.name

                st.audio(raw, format=f"audio/{'mp4' if suffix == 'm4a' else suffix}")
                st.caption(
                    f"**{uploaded.name}** · {duration:.1f}s · resampled to 16 kHz mono"
                )
                if duration > 30:
                    st.info("⏱️ Clip is longer than 30s — Whisper will transcribe the first 30 seconds.")

    # ---- Transcribe ----------------------------------------------------------
    st.markdown('<div class="section-label">02 · Transcribe</div>', unsafe_allow_html=True)
    if st.button("✨ Transcribe to Tamil", disabled=(bundle is None or audio is None)):
        with st.spinner("Running beam-search inference…"):
            try:
                text, elapsed = transcribe(bundle, audio)
            except Exception as exc:
                st.error(f"🚫 Inference failed: {exc}")
            else:
                st.session_state["transcription"] = text
                st.session_state["proc_time"] = elapsed
                st.session_state["audio_duration"] = duration
                st.rerun()
    if bundle is not None and audio is None:
        st.caption("Upload an audio file above to enable transcription.")

    # ---- Output --------------------------------------------------------------
    text = st.session_state.get("transcription")
    if text is not None and audio_ready:
        st.markdown('<div class="section-label">03 · Transcription</div>', unsafe_allow_html=True)
        if text:
            st.text_area(
                "Tamil transcription",
                value=text,
                height=180,
                label_visibility="collapsed",
            )
        else:
            st.warning("The model returned an empty transcription — the clip may contain no speech.")

        c1, c2 = st.columns(2)
        with c1:
            copy_button(text)
        with c2:
            st.download_button(
                "⬇️ Download as TXT",
                data=text.encode("utf-8"),
                file_name=f"{Path(st.session_state['audio_name']).stem}_transcription.txt",
                mime="text/plain",
                use_container_width=True,
            )

        elapsed = st.session_state.get("proc_time", 0.0)
        clip_len = st.session_state.get("audio_duration", 0.0)
        rtf = f"{clip_len / elapsed:.1f}× realtime" if elapsed > 0 and clip_len > 0 else ""
        s1, s2, s3 = st.columns(3)
        s1.markdown(metric_card("Words", f"{len(text.split()):,}"), unsafe_allow_html=True)
        s2.markdown(metric_card("Characters", f"{len(text):,}"), unsafe_allow_html=True)
        s3.markdown(metric_card("Processing Time", f"{elapsed:.2f}s", rtf), unsafe_allow_html=True)

    # ---- Performance metrics ---------------------------------------------------
    st.markdown('<div class="section-label">Model Performance Profile</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(metric_card("Model", "Whisper Small", MODEL_CARD["Base Model"]), unsafe_allow_html=True)
    m2.markdown(
        metric_card("Trainable Params", "1.77 M", "1,769,472 (0.73%)"), unsafe_allow_html=True
    )
    m3.markdown(
        metric_card("Total Params", "243.5 M", "243,504,384"), unsafe_allow_html=True
    )
    m4.markdown(metric_card("Method", "LoRA", "PEFT adapters"), unsafe_allow_html=True)
    m5.markdown(
        metric_card("Dataset", "Common Voice", "Tamil · 5,000 clips · 22 speakers"),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="margin-top:2.5rem;text-align:center;color:#5a6478;'
        'font-family:\'Spline Sans Mono\',monospace;font-size:0.72rem;letter-spacing:0.2em;">'
        'WHISPER SMALL · LORA · COMMON VOICE TAMIL · '
        + html.escape(BASE_MODEL_ID).upper()
        + "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
