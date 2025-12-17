import os
import sys
import time
from openai import OpenAI

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url='https://openapi.coreshub.cn/v1')

def translate_text(text: str) -> str:
    """Translate given text to English using OpenAI API."""
    response = client.chat.completions.create(
        model="DeepSeek-V3.2",  # or another model you prefer
        messages=[
            {"role": "system", "content": "You are a translator expert. Translate all text from ADOC(AsciiDoc) file to English but keep ADOC tags and format intact."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def process_file(path: str):
    """Read, translate, and overwrite HTML file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    translated = translate_text(content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(translated)

def main():
    if len(sys.argv) < 2:
        print("Usage: python translate_adoc.py <SOURCE_DIR>")
        sys.exit(1)

    source_dir = sys.argv[1]

    if not os.path.isdir(source_dir):
        print(f"Error: {source_dir} is not a valid directory")
        sys.exit(1)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".adoc"):
                src_path = os.path.join(root, file)
                print(f"Translating {src_path} (in place)")
                process_file(src_path)
                print(f"Sleeping for 60 seconds to respect rate limits...")
                time.sleep(60)

if __name__ == "__main__":
    main()
