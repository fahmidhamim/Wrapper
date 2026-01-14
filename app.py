import logging
import os
import shutil
import tempfile
from pathlib import Path
import hashlib
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from asr import transcribe_audio
from src.data_loader import load_all_documents
from src.document_rag import DocumentRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".mp4"}
ALLOWED_DOC_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".csv",
    ".xlsx",
    ".docx",
    ".json",
    ".pptx",
    ".ppt",
    ".html",
    ".htm",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tiff",
}
MIN_AUDIO_BYTES = 128

app = FastAPI(title="Voice Document RAG")
rag = DocumentRAG()
UPLOAD_DIR: str | None = None
FILE_HASHES: dict[str, str] = {}
DOC_COUNT = 0


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Voice Document RAG</title>
    <style>
      @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Instrument+Serif:opsz@8..96&display=swap");
      :root {
        --bg: #f7f2ec;
        --ink: #1f1d1b;
        --muted: #6b645f;
        --accent: #ea5a3b;
        --accent-soft: #ffd4c8;
        --panel: #ffffff;
        --shadow: 0 20px 60px rgba(20, 18, 16, 0.08);
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Space Grotesk", "Avenir Next", "Helvetica Neue", sans-serif;
        color: var(--ink);
        background: radial-gradient(circle at top, #fff4ea, transparent 60%),
          linear-gradient(135deg, #f7f2ec 0%, #f3e6da 45%, #f9f7f2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 32px;
      }
      .shell {
        width: min(980px, 100%);
        background: var(--panel);
        border-radius: 28px;
        box-shadow: var(--shadow);
        padding: 36px;
        position: relative;
        overflow: hidden;
        display: grid;
        gap: 22px;
      }
      .shell::before {
        content: "";
        position: absolute;
        top: -120px;
        right: -120px;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, var(--accent-soft), transparent 70%);
        opacity: 0.9;
        pointer-events: none;
      }
      header {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-bottom: 6px;
      }
      h1 {
        margin: 0;
        font-family: "Instrument Serif", "Georgia", serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 500;
      }
      p {
        margin: 0;
        color: var(--muted);
        font-size: 1rem;
      }
      .card {
        display: grid;
        gap: 16px;
        padding: 20px;
        border-radius: 18px;
        background: #fff7f1;
        border: 1px solid #f1e0d4;
      }
      .row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
      }
      input[type="file"] {
        flex: 1;
        min-width: 220px;
        padding: 12px;
        border-radius: 12px;
        border: 1px dashed #d8c8ba;
        background: #fff;
      }
      button {
        padding: 12px 20px;
        border: none;
        border-radius: 12px;
        background: var(--accent);
        color: #fff;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 10px 20px rgba(234, 90, 59, 0.25);
      }
      button.secondary {
        background: #f1e0d4;
        color: #4b3f37;
        box-shadow: none;
      }
      button:disabled {
        background: #c9b8ae;
        cursor: not-allowed;
        box-shadow: none;
      }
      button:hover:not(:disabled) {
        transform: translateY(-1px);
      }
      .status {
        font-size: 0.95rem;
        color: var(--muted);
      }
      textarea {
        width: 100%;
        min-height: 150px;
        padding: 14px;
        border-radius: 16px;
        border: 1px solid #ead9cc;
        font-size: 1rem;
        background: #fff;
        resize: vertical;
      }
      .chat-log {
        border-radius: 16px;
        border: 1px solid #ead9cc;
        background: #fff;
        padding: 12px;
        display: grid;
        gap: 12px;
        max-height: 320px;
        overflow-y: auto;
      }
      .chat-item {
        background: #fff7f1;
        border: 1px solid #f1e0d4;
        border-radius: 14px;
        padding: 12px;
        display: grid;
        gap: 6px;
      }
      .chat-actions {
        display: flex;
        gap: 8px;
        justify-content: flex-end;
      }
      .chat-label {
        font-size: 0.8rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }
      .chat-text {
        margin: 0;
        font-size: 1rem;
        color: var(--ink);
        white-space: pre-wrap;
      }
      .hint {
        font-size: 0.85rem;
        color: var(--muted);
      }
      @media (max-width: 640px) {
        .shell {
          padding: 24px;
        }
        .row {
          flex-direction: column;
          align-items: stretch;
        }
        button {
          width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <header>
        <h1>Voice Document RAG</h1>
        <p>Upload documents, then ask questions by voice. Answers are spoken back.</p>
      </header>
      <section class="card">
        <form id="docs-form">
          <div class="row">
            <input id="docs-files" type="file" multiple />
            <button id="docs-btn" type="submit">Upload Documents</button>
          </div>
          <div class="status" id="docs-status">
            Upload documents to build the knowledge base.
          </div>
        </form>
        <div class="hint">
          Supported: pdf, docx, txt, csv, xlsx, pptx, json, html, md, images
        </div>
      </section>
      <section class="card">
        <div class="row">
          <button id="record-btn" class="secondary" type="button">Record Question</button>
          <button id="stop-btn" type="button" disabled>Stop</button>
          <div class="status" id="record-status">Microphone ready.</div>
        </div>
        <form id="question-form">
          <div class="row">
            <input id="question-file" type="file" accept=".wav,.mp3,.m4a,.webm,.ogg,.mp4" />
            <button id="question-btn" type="submit">Ask from Audio File</button>
          </div>
        </form>
        <div class="status" id="qa-status">Waiting for a question.</div>
        <div class="status" id="topic-status">Topic: none</div>
        <div id="chat-log" class="chat-log" aria-live="polite"></div>
        <div class="row">
          <button id="reset-btn" class="secondary" type="button">Reset Conversation</button>
        </div>
        <div class="hint">
          Record a question or upload an audio file. Answers will be spoken aloud.
        </div>
      </section>
    </main>
    <script>
      const docsForm = document.getElementById("docs-form");
      const docsInput = document.getElementById("docs-files");
      const docsStatus = document.getElementById("docs-status");
      const docsBtn = document.getElementById("docs-btn");

      const recordBtn = document.getElementById("record-btn");
      const stopBtn = document.getElementById("stop-btn");
      const recordStatus = document.getElementById("record-status");
      const qaStatus = document.getElementById("qa-status");
      const resetBtn = document.getElementById("reset-btn");
      const topicStatus = document.getElementById("topic-status");
      const questionForm = document.getElementById("question-form");
      const questionFile = document.getElementById("question-file");
      const chatLog = document.getElementById("chat-log");

      let docsReady = false;
      let mediaRecorder = null;
      let recordedChunks = [];
      let micStream = null;
      let recordStream = null;
      let recorderMimeType = "";
      let audioCtx = null;
      let gainNode = null;

      function speak(text) {
        if (!("speechSynthesis" in window)) {
          return;
        }
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(utterance);
      }

      function pickMimeType() {
        const candidates = [
          "audio/webm;codecs=opus",
          "audio/webm",
          "audio/mp4",
          "audio/ogg;codecs=opus",
          "audio/ogg",
        ];
        return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
      }

      function extensionForMime(mimeType) {
        if (!mimeType) return "webm";
        if (mimeType.includes("mp4")) return "mp4";
        if (mimeType.includes("ogg")) return "ogg";
        if (mimeType.includes("webm")) return "webm";
        return "webm";
      }

      async function uploadDocuments(files) {
        if (!files || files.length === 0) {
          docsStatus.textContent = "Please choose at least one document.";
          return;
        }

        const formData = new FormData();
        for (const file of files) {
          formData.append("files", file);
        }

        docsStatus.textContent = "Uploading and indexing documents...";
        docsBtn.disabled = true;

        try {
          const response = await fetch("/upload", {
            method: "POST",
            body: formData,
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
          }
          docsReady = true;
          const baseText = `Ready. Loaded ${data.documents || 0} documents.`;
          if (data.duplicates && data.duplicates.length > 0) {
            docsStatus.textContent =
              baseText + ` Skipped ${data.duplicates.length} duplicate(s).`;
          } else if (data.files && data.files.length === 0) {
            docsStatus.textContent = baseText + " No new files added.";
          } else {
            docsStatus.textContent = baseText;
          }
        } catch (error) {
          docsReady = false;
          docsStatus.textContent = error.message || "Upload failed.";
        } finally {
          docsBtn.disabled = false;
        }
      }

      async function askWithAudio(file) {
        if (!docsReady) {
          qaStatus.textContent = "Upload documents first.";
          return;
        }
        if (!file) {
          qaStatus.textContent = "Provide an audio question.";
          return;
        }

        const formData = new FormData();
        formData.append("file", file);

        qaStatus.textContent = "Transcribing and answering...";
        recordBtn.disabled = true;
        stopBtn.disabled = true;

        try {
          const response = await fetch("/ask", {
            method: "POST",
            body: formData,
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "Answer failed.");
          }
          appendTurn(data.question || "", data.answer || "");
          qaStatus.textContent = "Answer ready.";
          updateTopic(data.topic || "");
          if (data.answer) {
            speak(data.answer);
          }
        } catch (error) {
          qaStatus.textContent = error.message || "Something went wrong.";
        } finally {
          recordBtn.disabled = false;
        }
      }

      docsForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        await uploadDocuments(docsInput.files);
      });

      questionForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        await askWithAudio(questionFile.files[0]);
      });

      recordBtn.addEventListener("click", async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          recordStatus.textContent = "Microphone not supported in this browser.";
          return;
        }
        if (!docsReady) {
          qaStatus.textContent = "Upload documents first.";
          return;
        }
        try {
          recordStatus.textContent = "Preparing microphone...";
          if (!recordStream) {
            micStream = await navigator.mediaDevices.getUserMedia({
              audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
              },
            });
            if (!audioCtx) {
              audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === "suspended") {
              await audioCtx.resume();
            }
            const source = audioCtx.createMediaStreamSource(micStream);
            gainNode = audioCtx.createGain();
            gainNode.gain.value = 1.6;
            const destination = audioCtx.createMediaStreamDestination();
            source.connect(gainNode);
            gainNode.connect(destination);
            recordStream = destination.stream;
          }
          recorderMimeType = pickMimeType();
          recordedChunks = [];

          const options = recorderMimeType
            ? { mimeType: recorderMimeType, audioBitsPerSecond: 128000 }
            : { audioBitsPerSecond: 128000 };
          mediaRecorder = new MediaRecorder(recordStream, options);

          mediaRecorder.addEventListener("dataavailable", (event) => {
            if (event.data && event.data.size > 0) {
              recordedChunks.push(event.data);
            }
          });

          mediaRecorder.addEventListener("stop", async () => {
            const mimeType = recorderMimeType || "audio/webm";
            const blob = new Blob(recordedChunks, { type: mimeType });
            if (!blob.size) {
              recordStatus.textContent =
                "No audio captured. Please try again and speak clearly.";
              recordBtn.disabled = false;
              stopBtn.disabled = true;
              return;
            }
            const extension = extensionForMime(mimeType);
            const file = new File([blob], `question.${extension}`, {
              type: mimeType,
            });

            recordStatus.textContent = "Uploading question...";
            await askWithAudio(file);
            recordStatus.textContent = "Microphone ready.";
            stopBtn.disabled = true;
          });

          mediaRecorder.addEventListener("start", () => {
            recordStatus.textContent = "Recording... speak now.";
          });
          await playBeepCountdown();
          mediaRecorder.start(100);
          recordBtn.disabled = true;
          stopBtn.disabled = false;
        } catch (error) {
          recordStatus.textContent =
            error && error.message
              ? error.message
              : "Could not access microphone.";
        }
      });

      stopBtn.addEventListener("click", () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
          mediaRecorder.stop();
        }
        recordBtn.disabled = false;
      });

      async function fetchStatus() {
        try {
          const response = await fetch("/status");
          const data = await response.json();
          docsReady = Boolean(data.documents);
          updateTopic(data.topic || "");
        } catch (error) {
          updateTopic("");
        }
      }

      function updateTopic(topic) {
        const label = topic && topic.trim() ? topic.trim() : "none";
        topicStatus.textContent = `Topic: ${label}`;
      }

      resetBtn.addEventListener("click", async () => {
        try {
          const response = await fetch("/reset", { method: "POST" });
          if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Reset failed.");
          }
          chatLog.innerHTML = "";
          updateTopic("");
          qaStatus.textContent = "Conversation reset.";
        } catch (error) {
          qaStatus.textContent = error.message || "Reset failed.";
        }
      });

      function appendTurn(question, answer) {
        const item = document.createElement("div");
        item.className = "chat-item";

        const qLabel = document.createElement("div");
        qLabel.className = "chat-label";
        qLabel.textContent = "Question";
        const qText = document.createElement("p");
        qText.className = "chat-text";
        qText.textContent = question || "(no transcription)";

        const aLabel = document.createElement("div");
        aLabel.className = "chat-label";
        aLabel.textContent = "Answer";
        const aText = document.createElement("p");
        aText.className = "chat-text";
        aText.textContent = answer || "Not found in document.";

        const actions = document.createElement("div");
        actions.className = "chat-actions";
        const repeat = document.createElement("button");
        repeat.type = "button";
        repeat.className = "secondary";
        repeat.textContent = "Repeat Answer";
        repeat.addEventListener("click", () => {
          if (!answer || !answer.trim()) {
            qaStatus.textContent = "No answer to repeat yet.";
            return;
          }
          speak(answer);
        });
        actions.appendChild(repeat);

        item.appendChild(qLabel);
        item.appendChild(qText);
        item.appendChild(aLabel);
        item.appendChild(aText);
        item.appendChild(actions);
        chatLog.appendChild(item);
        chatLog.scrollTop = chatLog.scrollHeight;
      }

      async function playTone(frequency, durationMs) {
        if (!audioCtx) {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === "suspended") {
          await audioCtx.resume();
        }
        return new Promise((resolve) => {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.frequency.value = frequency;
          osc.type = "sine";
          gain.gain.value = 0.12;
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start();
          setTimeout(() => {
            osc.stop();
            resolve();
          }, durationMs);
        });
      }

      async function playBeepCountdown() {
        recordStatus.textContent = "Get ready... 3";
        await playTone(880, 120);
        await new Promise((resolve) => setTimeout(resolve, 300));
        recordStatus.textContent = "Get ready... 2";
        await playTone(880, 120);
        await new Promise((resolve) => setTimeout(resolve, 300));
        recordStatus.textContent = "Get ready... 1";
        await playTone(880, 120);
      }

      fetchStatus();
    </script>
  </body>
</html>
"""


def _reset_upload_dir() -> None:
    global UPLOAD_DIR
    if UPLOAD_DIR and os.path.isdir(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR = None
    FILE_HASHES.clear()
    global DOC_COUNT
    DOC_COUNT = 0


def _ensure_upload_dir() -> str:
    global UPLOAD_DIR
    if not UPLOAD_DIR:
        UPLOAD_DIR = tempfile.mkdtemp(prefix="rag_upload_")
    return UPLOAD_DIR


def _unique_path(base_dir: str, filename: str) -> str:
    safe_name = Path(filename).name
    candidate = Path(base_dir) / safe_name
    if not candidate.exists():
        return str(candidate)

    stem = candidate.stem
    suffix = candidate.suffix
    for idx in range(1, 1000):
        attempt = Path(base_dir) / f"{stem}_{idx}{suffix}"
        if not attempt.exists():
            return str(attempt)

    raise HTTPException(status_code=500, detail="Too many files with same name.")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No documents provided.")

    upload_dir = _ensure_upload_dir()
    saved_files = []
    new_paths = []
    duplicates = []

    try:
        for file in files:
            if not file.filename:
                continue
            extension = Path(file.filename).suffix.lower()
            if extension not in ALLOWED_DOC_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported document type: {extension}. "
                        "Upload pdf, docx, txt, csv, xlsx, pptx, json, html, md, or images."
                    ),
                )

            content = await file.read()
            file_hash = _hash_bytes(content)
            if file_hash in FILE_HASHES:
                duplicates.append(Path(file.filename).name)
                continue

            dest_path = _unique_path(upload_dir, file.filename)
            with open(dest_path, "wb") as handle:
                handle.write(content)
            saved_files.append(Path(dest_path).name)
            new_paths.append(dest_path)
            FILE_HASHES[file_hash] = Path(dest_path).name

        if new_paths:
            documents = load_all_documents(upload_dir)
            if not documents:
                raise HTTPException(
                    status_code=400,
                    detail="No readable text found in uploaded documents.",
                )
            rag.set_documents(documents)
            global DOC_COUNT
            DOC_COUNT = len(documents)
        elif not rag.has_documents():
            raise HTTPException(
                status_code=400,
                detail="No readable text found in uploaded documents.",
            )

        return {
            "files": saved_files,
            "duplicates": duplicates,
            "documents": DOC_COUNT,
            "storage": Path(upload_dir).name,
        }
    except HTTPException:
        for path in new_paths:
            if os.path.exists(path):
                os.remove(path)
        raise
    except Exception:
        for path in new_paths:
            if os.path.exists(path):
                os.remove(path)
        logger.exception("Document upload failed.")
        raise HTTPException(status_code=500, detail="Document upload failed.")


@app.post("/ask")
async def ask(file: UploadFile = File(...)) -> dict:
    if not rag.has_documents():
        raise HTTPException(status_code=400, detail="No documents loaded yet.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio type. Use wav, mp3, m4a, webm, ogg, or mp4.",
        )

    temp_path = ""
    try:
        content = await file.read()
        if not content or len(content) < MIN_AUDIO_BYTES:
            raise HTTPException(
                status_code=400,
                detail="Audio too short or empty. Please record a longer question.",
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_path = temp_file.name
            temp_file.write(content)

        try:
            question = transcribe_audio(temp_path)
        except Exception:
            logger.exception("Failed to decode audio input.")
            raise HTTPException(
                status_code=400,
                detail="Could not decode audio. Try a longer recording or upload WAV/MP3.",
            )
        if not question.strip():
            raise HTTPException(
                status_code=400,
                detail="No speech detected in the question audio.",
            )

        answer = rag.answer(question, top_k=5)
        return {"question": question, "answer": answer, "topic": rag.current_topic()}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Question processing failed.")
        raise HTTPException(status_code=500, detail="Question processing failed.")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/status")
def status() -> dict:
    return {"documents": rag.has_documents(), "topic": rag.current_topic()}


@app.post("/reset")
def reset() -> dict:
    rag.reset_conversation()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False)
