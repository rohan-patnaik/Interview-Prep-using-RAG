import streamlit as st
import openai
import os
import time
import hashlib
import uuid
import wave
import pyaudio
import numpy as np
from dotenv import load_dotenv

# LangChain & Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

###############################################################################
# 0) Turn off file watching to avoid PyTorch or other conflicts
###############################################################################
# st.set_option("server.fileWatcherType", "none")

###############################################################################
# 1) Environment setup
###############################################################################
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "interview-transcripts-1"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(index_name)

###############################################################################
# 2) Streamlit session state (UI variables)
###############################################################################
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "candidate_text" not in st.session_state:
    st.session_state.candidate_text = ""
if "recorded_seconds" not in st.session_state:
    st.session_state.recorded_seconds = 0.0
if "doc_hashes" not in st.session_state:
    st.session_state["doc_hashes"] = {}

###############################################################################
# 3) Timed recording function (5 seconds)
###############################################################################
def record_5s_audio(filename="candidate_answer.wav", record_seconds=5):
    """
    Records from the default microphone for 'record_seconds' seconds,
    then saves the audio to a WAV file. Hardcodes 16-bit sample width.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    st.write(f"Recording for {record_seconds} second(s)...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    st.session_state.audio_file = filename
    st.session_state.recorded_seconds = record_seconds
    st.success(f"Recorded {record_seconds} seconds of audio.")
    st.info(f"Audio saved to {filename}")

###############################################################################
# 4) Embedding & Pinecone utilities
###############################################################################
@st.cache_data(show_spinner=False)
def embed_text(text):
    try:
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        return response["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error embedding text: {e}")
        return None

def upsert_document_chunks(chunks, batch_size=50):
    total = len(chunks)
    progress_bar = st.progress(0)
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        if emb is None:
            continue
        vectors.append((f"doc_chunk_{uuid.uuid4()}", emb, {"text": chunk}))
        progress_bar.progress((i + 1) / total)
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(batch)
    st.success("Documents have been embedded and upserted successfully!")

def retrieve_top_match(query_text, top_k=1):
    query_emb = embed_text(query_text)
    result = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    return result

def generate_missing_points_feedback(candidate_answer, ideal_answer, llm="openai"):
    instruction = (
        "You are an interview evaluator. Compare the candidate's answer to the ideal answer. "
        "Identify the key points that appear in the ideal answer but not in the candidate's answer. "
        "Only list those missing key points in a table with two columns: 'Key Point Missed' and 'Details/Explanation'."
    )
    prompt = f"Ideal Answer:\n{ideal_answer}\n\nCandidate's Answer:\n{candidate_answer}\n\n{instruction}"

    if llm == "openai":
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates interview answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error generating feedback: {e}")
            return None
    else:
        return "Local LLaMA: (Table of missing key points...)"

@st.cache_data
def get_content_hash(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def chunk_text(content, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(content)

###############################################################################
# 5) Streamlit UI
###############################################################################

st.title("RAG-based Interview Prep (5-Second PyAudio Edition)")

# 5.1) Upload Documents
st.header("1. Upload Documents")
uploaded_files = st.file_uploader(
    "Upload reference documents (TXT or PDF)",
    accept_multiple_files=True,
    type=["txt", "pdf"]
)

if uploaded_files:
    new_hashes = {}
    all_chunks = []
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            continue
        content_hash = get_content_hash(content)
        new_hashes[file.name] = content_hash
        chunks = chunk_text(content)
        all_chunks.extend(chunks)

    if new_hashes == st.session_state["doc_hashes"]:
        st.info("Documents unchanged. Using existing embeddings in Pinecone.")
    else:
        st.info("New or changed documents detected. Creating embeddings...")
        upsert_document_chunks(all_chunks)
        st.session_state["doc_hashes"] = new_hashes

# 5.2) Record Audio (5s)
st.header("2. Recording Audio (5 seconds)")

if st.button("Record 5s Audio"):
    record_5s_audio()

if st.session_state.recorded_seconds > 0:
    st.write(f"Last recorded audio length: {st.session_state.recorded_seconds} second(s).")

# 5.3) Transcription
import whisper

st.header("3. Transcription")
transcription_method = st.selectbox("Choose transcription method:", ["Local Whisper", "OpenAI Whisper API"])

if st.button("Transcribe Audio"):
    if not st.session_state.audio_file or not os.path.exists(st.session_state.audio_file):
        st.error("No recorded audio found. Please record first.")
    else:
        if transcription_method == "Local Whisper":
            with st.spinner("Transcribing locally..."):
                progress_bar = st.progress(0)
                for percent in range(0, 101, 20):
                    time.sleep(0.2)
                    progress_bar.progress(percent / 100)
                model = whisper.load_model("base")
                result = model.transcribe(st.session_state.audio_file)
                st.session_state.candidate_text = result["text"]
                st.success("Transcription complete!")
        else:
            with st.spinner("Transcribing via OpenAI Whisper API..."):
                with open(st.session_state.audio_file, "rb") as af:
                    transcript_response = openai.Audio.transcribe("whisper-1", af)
                st.session_state.candidate_text = transcript_response["text"]
                st.success("Transcription complete!")
        st.write("**Your Answer (Transcribed):**")
        st.write(st.session_state.candidate_text)

# 5.4) Retrieve Ideal Answer & Generate Feedback
st.header("4. Retrieve Ideal Answer & Generate Feedback")
llm_choice = st.selectbox("LLM for evaluation", ["openai", "llama_local"])

if st.button("Retrieve & Generate Feedback"):
    if not st.session_state.candidate_text:
        st.error("No candidate answer found. Please transcribe first.")
    else:
        with st.spinner("Retrieving the most relevant answer..."):
            query_result = retrieve_top_match(st.session_state.candidate_text, top_k=1)
            if query_result and "matches" in query_result and len(query_result["matches"]) > 0:
                best_chunk = query_result["matches"][0]["metadata"]["text"]
                st.subheader("Retrieved Ideal Answer")
                st.write(best_chunk)

                st.subheader("Feedback on Missing Points")
                feedback = generate_missing_points_feedback(
                    st.session_state.candidate_text,
                    best_chunk,
                    llm=llm_choice
                )
                if feedback:
                    st.write(feedback)
            else:
                st.warning("No matching chunk found in Pinecone. Please ensure you have uploaded documents.")
