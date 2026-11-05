from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from langchain_core.tools import tool
from datetime import datetime
import requests
import yfinance as yf

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
@tool
def wiki_tool(query: str)->str:
    """Search Wikipiedia for information on a topic."""
    try:
        return api_wrapper.run(query)
    except Exception as e:
        return f"Wikipidea search fail : {e}"


# ---------------- Stock tools (yfinance) ----------------
#we used ticker class here because the yfinance only know the ticker name of the company not the real name
#ticker.info or stock.info  has many features like company address,the thing they sell
@tool
def stock_price_tool(ticker: str) -> str:
    """Get the current/latest price and previous close for a stock ticker (e.g. AAPL, TSLA, RELIANCE.NS)."""
    try:
        stock = yf.Ticker(ticker)
        fi = stock.fast_info #fast_info provides commonly used stock information in a relatively lightweight way
        price = fi.get("lastPrice", "N/A") #dic get to get key values from fi
        prev_close = fi.get("previousClose", "N/A")
        currency = fi.get("currency", "")
        day_high = fi.get("dayHigh", "N/A")
        day_low = fi.get("dayLow", "N/A")
        return (
            f"{ticker.upper()} — Price: {price} {currency}, "
            f"Prev Close: {prev_close}, Day High: {day_high}, Day Low: {day_low}"
        )
    except Exception as e:
        return f"Stock price fetch failed for {ticker}: {e}"


@tool
def stock_history_tool(ticker: str, period: str = "1mo") -> str:
    """Get historical closing-price stats for a ticker. period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period) #? Use the value stored in the variable period as the period argument
        if hist.empty:
            return f"No historical data found for {ticker} over {period}"
        close = hist["Close"]
        return (
            f"{ticker.upper()} Close price summary ({period}): "
            f"start={close.iloc[0]:.2f}, end={close.iloc[-1]:.2f}, " #.2f means only go till 2 point place after value
            f"min={close.min():.2f}, max={close.max():.2f}, "
            f"change={((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100):.2f}%"
        )
    except Exception as e:
        return f"Stock history fetch failed for {ticker}: {e}"
  

@tool
def company_info_tool(ticker: str) -> str: # excepts a string and returns a string
    """Get company fundamentals for a ticker: sector, industry, market cap, P/E ratios, dividend yield, 52-week range."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fields = [
            "longName", "sector", "industry", "marketCap",
            "trailingPE", "forwardPE", "dividendYield",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]
        lines = [f"{f}: {info.get(f, 'N/A')}" for f in fields]
        return f"Company info for {ticker.upper()}:\n" + "\n".join(lines)
    except Exception as e:
        return f"Company info fetch failed for {ticker}: {e}"

#ticker.news fetches the news for that company

#stock.info
#stock.history()
#stock.dividends
# all are accessible by yf.Ticker object
@tool
def stock_news_tool(ticker: str) -> str:
    """Get recent news headlines for a stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]
        if not news:
            return f"No recent news found for {ticker}"
        headlines = [f"- {n.get('title', n.get('content', {}).get('title', 'No title'))}" for n in news]
        return f"Recent news for {ticker.upper()}:\n" + "\n".join(headlines)    # converts ticker value in upper case
    except Exception as e:
        return f"News fetch failed for {ticker}: {e}"


@tool
def filings_rag_tool(question: str) -> str:
    """Search company SEC filings (10-Q/10-K) for revenue, net income, risk factors and management commentary. Use for filing content, not live prices."""
    try:
        r = requests.get(
            "http://127.0.0.1:8000/search",
            params={"question": question, "top_k": 6},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if not data["contexts"]:
            return "No relevant filing content found."
        return "Sources: " + ", ".join(data["sources"]) + "\n\n" + "\n\n".join(data["contexts"])
    except Exception as e:
        return f"Filings search failed: {e}"
        #Features of python used in the tool.py

# List comprehension is an easier and more readable way to make a list

# example
#my _list=[]
# for n in nums:
#   my_list.append(n*n)
#print my_list

# my_list=[n*n for n in nums]-> list comprehension

# F-string : used to format the string it is used before the opening quotation mark
# Evaluate the expressions inside {} and insert their results into the string.

# letter= "hey my name is {} and I am from {}"
#country="india"
#name ="amogh"
#print (letter.format(country,name))
#print(f"Hey my name is {name} and I am from {country}")