# Stock Research Assistant with SEC Filing Search

I built this to answer stock questions using two things at once: live market data from Yahoo Finance, and the text of SEC filings (10-Qs and 10-Ks) that I load into a vector database.

You type a question and an agent decides what it needs. Ask "How is NVDA trading, and what does its latest 10-Q say about revenue?" and it pulls the live price and also searches the filing. The page shows which tools it used and where the answer came from, so you can check it yourself.

<!-- Add a screenshot of the page here, e.g. ![screenshot](docs/screenshot.png) -->

## How it fits together

There are two small apps and a database:

```
Browser (index.html)
   |
   v
Stock agent (port 8001)  --- Yahoo Finance tools (price, history, news, fundamentals)
   |
   | when the question is about a filing
   v
RAG app (port 8000)  --- Gemini (turns text into vectors)
   |
   v
Qdrant (port 6333)  --- stores the filing chunks
```

**Loading a filing (done once per PDF):**
1. Read the text out of the PDF.
2. Cut it into chunks of about 1000 tokens, with 200 tokens of overlap so ideas on a boundary don't get split.
3. Turn each chunk into a vector with Gemini's embedding model.
4. Store the vectors in Qdrant, along with the chunk text and the file name.

Inngest runs these steps in the background, and its dashboard shows which step is running or failed.

**Asking a question:**
1. The question is turned into a vector with the same model.
2. Qdrant finds the stored chunks that are closest in meaning.
3. Those chunks go to Gemini with the question, and it writes the answer from them.

## What's in the folders

```
rag_agent/       the filing search service
  main.py        API endpoints and the Inngest functions
  data_loader.py PDF reading, chunking, embeddings
  vector_db.py   talks to Qdrant
  custom_types.py
  pyproject.toml

stock_agent/     the agent and the web page
  main.py        the agent and the /research endpoint
  tools.py       the tools the agent can call
  index.html     the front end (Bootstrap + plain JavaScript)
  robot.png
  requirements.txt
```

## What you need

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop
- Node.js (only for the Inngest dev server)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com)

## Running it

Use a separate terminal for each step.

**1. Start Qdrant**

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

Next time just run `docker start qdrant`. The volume keeps your data between restarts.

**2. Add your API key**

In both `rag_agent/` and `stock_agent/`, copy `.env.example` to `.env` and put your key in it:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

**3. Start the RAG app**

```bash
cd rag_agent
uv sync
uv run uvicorn main:app --reload --port 8000
```

**4. Start the Inngest dev server** (only needed when loading PDFs)

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
```

Its dashboard is at http://127.0.0.1:8288.

**5. Start the stock agent**

```bash
cd stock_agent
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

**6. Load a filing**

Get a 10-Q or 10-K from [SEC EDGAR](https://www.sec.gov/edgar/search/). EDGAR gives you HTML, so open the filing in Chrome, press Ctrl+P, and choose **Save as PDF**. Give the file a clear name like `NVDA_10Q.pdf`, because that name shows up as the source in answers.

Then go to http://127.0.0.1:8000/docs, open **POST /upload**, pick the PDF, and click Execute. It takes a couple of minutes (more on why below). You can watch it in the Inngest dashboard.

One thing that cost me some time: the PDF needs a real text layer. If you can highlight text in a viewer but loading finds nothing, save the PDF again using Chrome's "Save as PDF" option.

**7. Ask something**

Open `stock_agent/index.html` in your browser and try one of the example questions.

## The endpoints

RAG app, at http://127.0.0.1:8000/docs:

| Endpoint | What it does |
|---|---|
| `GET /search` | Returns the matching chunks and their sources. The agent uses this one. |
| `GET /query` | Does the search and also writes an answer with Gemini |
| `POST /upload` | Upload a PDF and start loading it |
| `POST /ingest` | Start loading a PDF that's already on your computer, by path |

Stock agent, at http://127.0.0.1:8001/docs:

| Endpoint | What it does |
|---|---|
| `POST /research` | Send `{"query": "..."}`, get back `topic`, `summary`, `sources` and `tools_used` |

## What the agent can use

| Tool | What it does |
|---|---|
| `stock_price_tool` | Latest price, previous close, day high and low |
| `stock_history_tool` | Price summary over a chosen period |
| `company_info_tool` | Sector, market cap, P/E ratios, dividend yield, 52-week range |
| `stock_news_tool` | Recent headlines |
| `wiki_tool` | A short Wikipedia lookup |
| `filings_rag_tool` | Searches the filings loaded into Qdrant |
| `save_text_to_file` | Saves the research output to a text file |

I capped it at 3 tool calls per question so replies stay quick and I don't burn through the API limits.

## Where this came from

The filing search started from a tutorial that used OpenAI. I rebuilt it on Gemini and changed a few things along the way:

- Swapped the PDF reader to PyMuPDF after the first reader returned nothing on my files.
- Added batching, pauses and retries, because Gemini's free tier limits how many tokens you can embed per minute.
- Added a `/search` endpoint so the stock agent can use the filing search as a tool, and `/upload` so I don't have to go through the Inngest dashboard.
- Wrote the stock agent, its tools and the web page myself.

## What it can't do yet

- **Search isn't filtered by company.** A question about one company can pull chunks from another company's filing. Adding a filter on ticker is the next fix.
- **Loading is slow and manual.** I add filings by hand, and the embedding calls are spaced out to stay under the free-tier limit. A paid plan would speed it up.
- **Tables lose their layout.** Text extraction is plain, so tables in filings come out as flat text.
- **It can be wrong.** The prompt tells the model to say so when the context doesn't contain the answer, but search can still miss the right passage. That's why sources are shown. Check important numbers against the filing.
- **It's a local dev setup.** CORS is wide open and there's no login, so don't put it on the public internet as it is.

## Ideas for later

- Pull new filings automatically from SEC EDGAR for a list of companies
- Save page numbers with each chunk so answers can point to a page
- Filter by company and filing type
- Write a set of test questions with known answers to measure accuracy

This is a learning project, not financial advice.
