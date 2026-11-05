import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from tools import (
    company_info_tool,
    filings_rag_tool,
    save_tool,
    stock_history_tool,
    stock_news_tool,
    stock_price_tool,
    wiki_tool,
)

load_dotenv()


class ResearchResponse(BaseModel):
  topic: str
  summary: str
  sources: list[str]
  tools_used: list[str]


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", convert_system_message_to_human=True, max_retries=5
)

system_prompt = """You are a research assistant that helps generate research reports, including stock/equity research.
For stock queries, use the stock tools to pull live price, historical performance, fundamentals, and news before writing the summary.
For questions about a company's SEC filings (revenue figures, net income, risk factors, management commentary), use filings_rag_tool.
Ticker FORMATTING :
-For US stocks,use the plain ticker (e.g. AAPL,TSLA)
- For Indian stocks, append the exchange suffix: use ".NS" for NSE (e.g. RELIANCE.NS, TCS.NS, INFY.NS) 
  unless the user specifically asks for BSE, in which case use ".BO".
- If the user gives a company name instead of a ticker (e.g. "Tata Motors"), infer the correct ticker 
  including the exchange suffix based on context (Indian company names should default to NSE with .NS).

CRITICAL INSTRUCTION:
Your final output MUST be a valid JSON string matching this exact format, with no backticks or markdown code blocks:
{
  "topic": "Topic Name",
  "summary": "Detailed summary text here...",
  "sources": ["source 1", "source 2"],
  "tools_used": ["tool_name_1", "tool_name_2"]
}
Efficiency Instruction:
Call a maximum of 3 tools per request. Do not make extra tool calls if you have sufficient info.
"""

tools = [
    wiki_tool,
    save_tool,
    stock_price_tool,
    stock_history_tool,
    company_info_tool,
    stock_news_tool,
    filings_rag_tool,
]

# Fixed parameter name: system_prompt=system_prompt
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #used this for development only so it is not safe the local host front end web address will be added
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
  query: str


@app.post("/research")
def research(request: QueryRequest):
  try:
    raw_response = agent.invoke({"messages": [("user", request.query)]})

    # Safely extract text content from the last message
    last_message_content = raw_response["messages"][-1].content
    if isinstance(last_message_content, list):
      final_content = "".join(
          block if isinstance(block, str) else block.get("text", "")
          for block in last_message_content
      ).strip()
    else:
      final_content = str(last_message_content).strip()

    # Clean JSON markdown backticks if present
    if final_content.startswith("```json"):
      final_content = final_content[7:-3].strip()
    elif final_content.startswith("```"):
      final_content = final_content[3:-3].strip()

    # Return parsed dictionary directly to FastAPI
    parsed_data = json.loads(final_content)
    return parsed_data

  except Exception as e:
    print("Error processing research query:", e)
    return {"error": str(e)}