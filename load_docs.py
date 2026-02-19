from pathlib import Path

import whisper
from PyPDF2 import PdfReader

DOWNLOADS = Path.home() / "Downloads"

FILES = {
    "pdf": DOWNLOADS / "Living_Repository_AI_Literacy_Practices_Update_16042025_UqmogIt2HpLVokdcuzJL4mDvHk8_112203.pdf",
    "audio": [
        DOWNLOADS / "Red_Lines_and_Risks_in_the_AI_Act.m4a",
        DOWNLOADS / "The_Blueprint_For_Trustworthy_AI.m4a",
    ],
}


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def transcribe_audio(path: Path, model_name: str = "base") -> str:
    model = whisper.load_model(model_name)
    result = model.transcribe(str(path))
    return result["text"] # type: ignore


OUTPUT_DIR = Path(__file__).parent / "docs"


def save_as_markdown(doc: dict) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(doc["source"]).stem
    md_path = OUTPUT_DIR / f"{stem}.md"
    md_path.write_text(f"# {stem.replace('_', ' ')}\n\n{doc['text']}\n")
    return md_path


def load_all(whisper_model: str = "base") -> list[dict]:
    docs = []

    # PDF
    pdf_path = FILES["pdf"]
    print(f"Extracting text from {pdf_path.name} ...")
    docs.append({
        "source": pdf_path.name,
        "type": "pdf",
        "text": extract_pdf_text(pdf_path),
    })

    # Audio files
    for audio_path in FILES["audio"]:
        print(f"Transcribing {audio_path.name} ...")
        docs.append({
            "source": audio_path.name,
            "type": "audio",
            "text": transcribe_audio(audio_path, model_name=whisper_model),
        })

    return docs


if __name__ == "__main__":
    documents = load_all()
    for doc in documents:
        md_path = save_as_markdown(doc)
        print(f"Saved: {md_path}")
