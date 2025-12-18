import os
import sys
import json
import time
from openai import OpenAI

PROGRESS_FILE = "translation_progress.json"

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
        print(f"Warning: glossary file {path} not found, skipping glossary.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_glossary(text: str, glossary: dict) -> str:
    """Replace glossary terms in text before translation."""
    for zh, en in glossary.items():
        text = text.replace(zh, en)
    return text

def translate_text(client, model: str, text: str, glossary: dict) -> str:
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

def process_file(client, model: str, glossary: dict, path: str, done_files: set):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        translated = translate_text(client, model, content, glossary)

        with open(path, "w", encoding="utf-8") as f:
            f.write(translated)

        done_files.add(path)
        save_progress(done_files)
        print(f"✅ Successfully translated {path}")
    except Exception as e:
        print(f"❌ Error translating {path}: {e}")
        # wait a bit before continuing, to avoid hammering API
        time.sleep(5)

def main():
    if len(sys.argv) < 4:
        print("Usage: python translate_adoc.py <SOURCE_DIR> <MODEL_NAME> <GLOSSARY_FILE>")
        print("Example: python translate_adoc.py ./docs DeepSeek-V3.2 glossary.json")
        sys.exit(1)

    source_dir = sys.argv[1]
    model = sys.argv[2]
    glossary_file = sys.argv[3]

    if not os.path.isdir(source_dir):
        print(f"Error: {source_dir} is not a valid directory")
        sys.exit(1)

    glossary = load_glossary(glossary_file)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url='https://openapi.coreshub.cn/v1')

    done_files = load_progress()

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".adoc"):
                src_path = os.path.join(root, file)
                if src_path in done_files:
                    print(f"⏩ Skipping already translated {src_path}")
                    continue
                print(f"Translating {src_path} with {model} (in place, glossary applied)")
                process_file(client, model, glossary, src_path, done_files)
                print(f"Sleep for 10 seconds to avoid rate limits")
                time.sleep(10)

if __name__ == "__main__":
    main()
