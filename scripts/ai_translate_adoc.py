import os
import sys
import json
import time
from openai import OpenAI

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
    preprocessed = apply_glossary(text, glossary)

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

def process_file(client, model: str, glossary: dict, path: str):
    """Read, translate, and overwrite HTML file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    translated = translate_text(client, model, content, glossary)

    with open(path, "w", encoding="utf-8") as f:
        f.write(translated)

def main():
    if len(sys.argv) < 4:
        print("Usage: python translate_adoc.py <SOURCE_DIR> <MODEL_NAME> <GLOSSARY_FILE>")
        print("Example: python translate_adoc.py ./docs qwen3-30b-a3b glossary.json")
        sys.exit(1)

    source_dir = sys.argv[1]
    model = sys.argv[2]
    glossary_file = sys.argv[3]

    if not os.path.isdir(source_dir):
        print(f"Error: {source_dir} is not a valid directory")
        sys.exit(1)

    glossary = load_glossary(glossary_file)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url='https://openapi.coreshub.cn/v1')

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".adoc"):
                src_path = os.path.join(root, file)
                print(f"Translating {src_path} with {model} (in place, glossary applied)")
                process_file(client, model, glossary, src_path)
                print(f"Sleep for 30 seconds to avoid rate limits")
                time.sleep(30)

if __name__ == "__main__":
    main()
