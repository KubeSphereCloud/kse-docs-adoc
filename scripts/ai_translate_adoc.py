import os
import sys
import json
import time
from datetime import datetime, timezone
from openai import OpenAI

PROGRESS_FILE = "translation_progress.json"
CHUNK_SIZE = 10000  # characters per chunk (10KB)

def log(msg: str):
    """Print log message with RFC3339 timestamp."""
    ts = datetime.now(timezone.utc).isoformat()
    print(f"{ts} {msg}")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_progress(done_files):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done_files), f, ensure_ascii=False, indent=2)

def load_glossary(path: str) -> dict:
    """Load glossary dictionary from JSON file."""
    if not os.path.isfile(path):
        log(f"Warning: glossary file {path} not found, skipping glossary.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_glossary(text: str, glossary: dict) -> str:
    """Replace glossary terms in text before translation."""
    for zh, en in glossary.items():
        text = text.replace(zh, en)
    return text

def translate_chunk(client, model: str, text: str, glossary: dict) -> str:
    """Translate given text to English using chosen model, respecting glossary."""
    # Apply glossary replacements before sending to model
    # preprocessed = apply_glossary(text, glossary)
    preprocessed = text

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are a translator expert. Translate all text from ADOC(AsciiDoc) file to English but keep ADOC tags and format intact. "
                "Always respect the following glossary mappings:\n" +
                "\n".join([f"{k} -> {v}" for k,v in glossary.items()])
            )},
            {"role": "user", "content": preprocessed}
        ],
        temperature=0
    )
    translated = response.choices[0].message.content

    # Ensure glossary terms are enforced after translation
    for zh, en in glossary.items():
        translated = translated.replace(zh, en).replace(en, en)
    return translated

def chunk_text(text: str, size: int = CHUNK_SIZE):
    """Split text into chunks by size, preferring to cut at line breaks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # try to cut at nearest newline before end
        if end < len(text):
            newline_pos = text.rfind("\n\n", start, end)
            if newline_pos != -1 and newline_pos > start:
                end = newline_pos + 1
        chunks.append(text[start:end])
        start = end
    return chunks

def process_file(client, model: str, glossary: dict, path: str, done_files: set):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_text(content, CHUNK_SIZE)
        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            log(f"Sleep for 10 seconds to avoid rate limits")
            time.sleep(10)
            log(f"Translating chunk {i}/{len(chunks)} of {path}")
            translated_chunks.append(translate_chunk(client, model, chunk, glossary))

        translated = "\n".join(translated_chunks)

        with open(path, "w", encoding="utf-8") as f:
            f.write(translated)

        done_files.add(path)
        save_progress(done_files)
        log(f"✅ Successfully translated {path}")
    except Exception as e:
        log(f"❌ Error translating {path}: {e}")

def main():
    if len(sys.argv) < 4:
        log("Usage: python translate_adoc.py <SOURCE_DIR> <MODEL_NAME> <GLOSSARY_FILE>")
        log("Example: python translate_adoc.py ./docs DeepSeek-V3.2 glossary.json")
        sys.exit(1)

    source_dir = sys.argv[1]
    model = sys.argv[2]
    glossary_file = sys.argv[3]

    if not os.path.isdir(source_dir):
        log(f"Error: {source_dir} is not a valid directory")
        sys.exit(1)

    glossary = load_glossary(glossary_file)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url='https://openapi.coreshub.cn/v1')

    done_files = load_progress()

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".adoc"):
                src_path = os.path.join(root, file)
                if src_path in done_files:
                    log(f"⏩ Skipping already translated {src_path}")
                    continue
                log(f"Translating {src_path} with {model} (in place, glossary applied)")
                process_file(client, model, glossary, src_path, done_files)

if __name__ == "__main__":
    main()

