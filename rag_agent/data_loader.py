import time
import pymupdf
from google import genai
from google.genai import errors, types
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    doc = pymupdf.open(path)
    chunks = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            chunks.extend(splitter.split_text(text))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    out = []
    batch = 10  # keeps each request well under the tokens-per-minute limit
    for i in range(0, len(texts), batch):
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=texts[i:i + batch],
                    config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
                )
                break
            except errors.ClientError as err:
                if err.code == 429 and attempt < 5:
                    time.sleep(60)  # wait out the per-minute limit, then retry this batch
                else:
                    raise
        out.extend(e.values for e in resp.embeddings)
        if i + batch < len(texts):
            time.sleep(25)  # pause only between batches, not after the last one
    return out