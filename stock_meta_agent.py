import os
import sys
import socket
import time
import threading
import argparse
import json
import traceback

# A2A imports
from python_a2a import OpenAIA2AServer, run_server, A2AServer, AgentCard, AgentSkill
from python_a2a.langchain import to_langchain_agent, to_langchain_tool
from python_a2a.mcp import FastMCP

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Data & scraping imports
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin


def find_available_port(start_port: int = 5000, max_tries: int = 20) -> int:
    """
    Find an available TCP port on localhost.
    Tries ports from start_port up to start_port + max_tries - 1.
    If none are free, returns start_port + 1000 as a fallback.
    """
    for port in range(start_port, start_port + max_tries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("localhost", port))
            sock.close()
            return port
        except OSError:
            continue
    print(f"⚠️  Could not find an available port in range {start_port}-{start_port + max_tries - 1}")
    return start_port + 1000


def run_server_in_thread(
    server_func,
    server_obj,
    host: str = "0.0.0.0",
    port: int = None
) -> threading.Thread:
    """
    Run a server function (e.g., run_server or server.run) in a background thread.
    Sleeps briefly to give the server time to start.
    """
    def target():
        if port is not None:
            server_func(server_obj, host=host, port=port)
        else:
            server_func(server_obj)
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    time.sleep(2)
    return thread


def parse_arguments():
    """
    Parse command-line arguments:
      --a2a-port:  (optional) TCP port for the A2A server
      --mcp-port:  (optional) TCP port for the MCP server
      --model:     OpenAI model to use (default: gpt-4o)
      --temperature: sampling temperature for the LLM (default: 0.0)
    """
    parser = argparse.ArgumentParser(
        description="A2A + MCP + LangChain Stock Market Integration Example"
    )
    parser.add_argument(
        "--a2a-port",
        type=int,
        default=None,
        help="Port to run the A2A server on (default: auto-select)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=None,
        help="Port to run the MCP server on (default: auto-select)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0, deterministic)",
    )
    return parser.parse_args()


def check_api_key() -> bool:
    """
    Verify that the OPENAI_API_KEY environment variable is set.
    Returns False and exits if not found.
    """
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].strip():
        print("❌ Error: The environment variable OPENAI_API_KEY is not set.")
        print("   You must export your OpenAI API key before running this script.")
        print("   Example (Linux/macOS): export OPENAI_API_KEY=\"sk-...\"")
        return False
    return True


def main():
    """Main entry-point."""
    # 1) Check for OpenAI API key
    if not check_api_key():
        return 1

    # 2) Parse arguments
    args = parse_arguments()

    # 3) Determine ports (auto-select if not provided)
    a2a_port = args.a2a_port or find_available_port(5000, 20)
    mcp_port = args.mcp_port or find_available_port(7000, 20)

    print(f"🔍 A2A server port: {a2a_port}")
    print(f"🔍 MCP server port: {mcp_port}")

    # ----------------------------
    # Step 1: Create the OpenAI-powered A2A Server
    # ----------------------------
    print("\n📝 Step 1: Creating OpenAI-Powered A2A Server")

    # Define an AgentCard describing our “Stock Market Expert”
    agent_card = AgentCard(
        name="Stock Market Expert",
        description="An A2A agent specialized in stock market analysis and financial information.",
        url=f"http://localhost:{a2a_port}",
        version="1.0.0",
        skills=[
            AgentSkill(
                name="Market Analysis",
                description="Analyzing market trends, stock performance, and market indicators.",
                examples=[
                    "What's the current market sentiment?",
                    "How do interest rates affect tech stocks?",
                ],
            ),
            AgentSkill(
                name="Investment Strategies",
                description="Providing information about various investment approaches and strategies.",
                examples=[
                    "What is dollar cost averaging?",
                    "How should I diversify my portfolio?",
                ],
            ),
            AgentSkill(
                name="Company Analysis",
                description="Analyzing specific companies, their fundamentals, and performance metrics.",
                examples=[
                    "What are key metrics to evaluate a tech company?",
                    "How to interpret P/E ratios?",
                ],
            ),
        ],
    )

    # Instantiate an OpenAIA2AServer that fronts OpenAI’s API
    openai_server = OpenAIA2AServer(
        api_key=os.environ["OPENAI_API_KEY"],
        model=args.model,
        temperature=args.temperature,
        system_prompt=(
            "You are a stock market and financial analysis expert. "
            "Provide accurate, concise information about stocks, market trends, investment strategies, and financial metrics. "
            "Focus on educational content and avoid making specific investment recommendations or predictions."
        ),
    )

    # Wrap it in a standard A2AServer subclass so it can fit into the protocol
    class StockMarketExpert(A2AServer):
        def __init__(self, openai_server: OpenAIA2AServer, agent_card: AgentCard):
            super().__init__(agent_card=agent_card)
            self.openai_server = openai_server

        def handle_message(self, message: dict) -> dict:
            """
            Delegate incoming A2A messages to the OpenAI backend.
            Expects the A2A “message” dict; returns a response dict.
            """
            return self.openai_server.handle_message(message)

    # Instantiate the wrapped agent and start its server thread
    stock_agent = StockMarketExpert(openai_server, agent_card)

    a2a_server_url = f"http://localhost:{a2a_port}"
    print(f"\nStarting A2A server on {a2a_server_url}...")

    def run_a2a_server(server_obj, host="0.0.0.0", port=a2a_port):
        run_server(server_obj, host=host, port=port)

    a2a_thread = run_server_in_thread(run_a2a_server, stock_agent, port=a2a_port)

    # ----------------------------
    # Step 2: Create the MCP Server with “Finance Tools”
    # ----------------------------
    print("\n📝 Step 2: Creating MCP Server with Finance Tools")

    mcp_server = FastMCP(
        name="Finance Tools",
        description="Advanced tools for stock market analysis (stock_data + web_scraper).",
    )

    # ----- Tool #1: stock_data -----
    @mcp_server.tool(
        name="stock_data",
        description="Fetch stock data for one or more ticker symbols."
    )
    def stock_data(input_str=None, **kwargs):
        """
        Fetch stock data using yfinance with enhanced parsing.
        Accepts:
            • A single string (e.g., "AAPL, MSFT" or "apple")
            • Or keyword argument input="AAPL"
        Returns a JSON string containing metrics and summary.
        """
        try:
            # 1) Handle keyword vs positional input
            if input_str is None and "input" in kwargs:
                input_str = kwargs["input"]

            if input_str is None:
                return {"text": "Error: No ticker symbol provided."}

            input_str = str(input_str).strip()

            # 2) Extract tickers (supports comma-separated or plain text containing company names)
            tickers = []
            if "," in input_str:
                tickers = [t.strip().upper() for t in input_str.split(",") if t.strip()]
            else:
                # Find all letter-only tokens 1–5 chars long (e.g., "AAPL" or "MSFT")
                found = re.findall(r"\b[A-Za-z]{1,5}\b", input_str)
                tickers = [w.upper() for w in found] if found else []

            # 3) Map common company names → tickers if still empty
            if not tickers:
                common_stocks = {
                    "apple": "AAPL",
                    "microsoft": "MSFT",
                    "google": "GOOGL",
                    "amazon": "AMZN",
                    "tesla": "TSLA",
                    "nvidia": "NVDA",
                }
                for name, ticker_symbol in common_stocks.items():
                    if name.lower() in input_str.lower():
                        tickers.append(ticker_symbol)

            if not tickers:
                return {"text": "No valid ticker symbols found in input."}

            # 4) Determine period/interval based on keywords
            period = "1mo"
            interval = "1d"
            if "year" in input_str.lower() or "1y" in input_str.lower():
                period = "1y"
            elif "week" in input_str.lower() or "1w" in input_str.lower():
                period = "1wk"

            results = {}
            for ticker_symbol in tickers:
                try:
                    ticker_obj = yf.Ticker(ticker_symbol)
                    hist = ticker_obj.history(period=period, interval=interval)
                    if hist.empty:
                        results[ticker_symbol] = {"error": f"No data found for {ticker_symbol}"}
                        continue

                    info = ticker_obj.info
                    company_name = info.get("shortName", ticker_symbol)
                    sector = info.get("sector", "Unknown")
                    industry = info.get("industry", "Unknown")

                    latest = hist.iloc[-1]
                    earliest = hist.iloc[0]
                    price_change = float(latest["Close"]) - float(earliest["Close"])
                    percent_change = (price_change / float(earliest["Close"])) * 100

                    high_52week = info.get("fiftyTwoWeekHigh", "Unknown")
                    low_52week = info.get("fiftyTwoWeekLow", "Unknown")
                    avg_volume = info.get("averageVolume", "Unknown")
                    market_cap = info.get("marketCap", "Unknown")
                    pe_ratio = info.get("trailingPE", "Unknown")

                    summary = {
                        "ticker": ticker_symbol,
                        "company_name": company_name,
                        "sector": sector,
                        "industry": industry,
                        "latest_price": float(latest["Close"]),
                        "price_change": float(price_change),
                        "percent_change": float(percent_change),
                        "52_week_high": high_52week,
                        "52_week_low": low_52week,
                        "average_volume": avg_volume,
                        "market_cap": market_cap,
                        "pe_ratio": pe_ratio,
                        "period": period,
                        "interval": interval,
                        "data_points": len(hist),
                    }

                    results[ticker_symbol] = summary
                except Exception as e:
                    results[ticker_symbol] = {
                        "error": f"Error processing {ticker_symbol}: {str(e)}"
                    }

            return {"text": json.dumps(results)}

        except Exception as e:
            tb = traceback.format_exc()
            return {"text": f"Error: {str(e)}\nDetails: {tb}"}

    # ----- Tool #2: web_scraper -----
    @mcp_server.tool(
        name="web_scraper",
        description="Get financial news and company snapshot from Finviz (or simple URL handling)."
    )
    def web_scraper(input_str=None, **kwargs):
        """
        Scrape Finviz for the latest news headlines and company details.
        Accepts:
            • A ticker symbol (e.g., "AAPL")
            • A URL (in which case we return the URL)
            • Otherwise, recommend using a broader web search.
        Returns a JSON string with news items & snapshot fields.
        """
        try:
            if input_str is None and "input" in kwargs:
                input_str = kwargs["input"]

            if input_str is None:
                return {"text": "Error: No input provided."}

            input_str = str(input_str).strip()

            # If the input is exactly 1–5 letters: treat as ticker
            if re.match(r"^[A-Za-z]{1,5}$", input_str):
                ticker = input_str.upper()
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/91.0.4472.124 Safari/537.36"
                    )
                }
                url = f"https://finviz.com/quote.ashx?t={ticker.lower()}"

                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")

                # Scrape the news table (first 5 items)
                news_table = soup.find("table", {"id": "news-table"})
                news_items = []
                if news_table:
                    for row in news_table.find_all("tr")[:5]:
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            date_cell = cells[0]
                            title_cell = cells[1]
                            link = title_cell.find("a")
                            if link:
                                news_link = link["href"]
                                if not news_link.startswith("http"):
                                    news_link = urljoin(url, news_link)
                                news_items.append({
                                    "title": link.text.strip(),
                                    "link": news_link,
                                    "date": date_cell.text.strip(),
                                })

                # Scrape the “snapshot-table2” for company details
                company_details = {}
                snapshot_table = soup.find("table", {"class": "snapshot-table2"})
                if snapshot_table:
                    rows = snapshot_table.find_all("tr")
                    for row in rows:
                        cells = row.find_all("td")
                        for i in range(0, len(cells), 2):
                            if i + 1 < len(cells):
                                key = cells[i].text.strip()
                                value = cells[i + 1].text.strip()
                                if key and value:
                                    company_details[key] = value

                return {
                    "text": json.dumps({
                        "ticker": ticker,
                        "news_items": news_items,
                        "company_details": company_details,
                    })
                }

            # If the input starts with “http”, just echo back
            elif input_str.startswith("http"):
                return {
                    "text": json.dumps({
                        "url": input_str,
                        "message": "URL processing is simplified in this version.",
                    })
                }

            else:
                # Treat as a broadly defined topic
                topic = input_str.replace("topic:", "").strip()
                return {
                    "text": json.dumps({
                        "topic": topic,
                        "message": "Please use a general web search for detailed topic information.",
                    })
                }

        except Exception as e:
            tb = traceback.format_exc()
            return {"text": f"Error: {str(e)}\nDetails: {tb}"}

    # 3) Start the MCP server thread
    mcp_server_url = f"http://localhost:{mcp_port}"
    print(f"\nStarting MCP server on {mcp_server_url}...")

    def run_mcp_server(server_obj, host="0.0.0.0", port=mcp_port):
        server_obj.run(host=host, port=port)

    mcp_thread = run_server_in_thread(run_mcp_server, mcp_server, port=mcp_port)

    # 4) Wait briefly to ensure both servers are up
    print("\nWaiting for servers to initialize...")
    time.sleep(5)

    # 5) Verify MCP server /tools endpoint
    mcp_server_running = False
    try:
        resp = requests.get(f"{mcp_server_url}/tools", timeout=5)
        if resp.status_code == 200:
            mcp_server_running = True
    except:
        pass

    if not mcp_server_running:
        print(f"❌ MCP server failed to start on port {mcp_port}. Retrying on a new port...")
        mcp_port = find_available_port(8000, 20)
        mcp_server_url = f"http://localhost:{mcp_port}"
        print(f"🔍 New MCP server port: {mcp_port}")
        # Restart thread on new port
        mcp_thread = run_server_in_thread(run_mcp_server, mcp_server, port=mcp_port)
        time.sleep(5)

    # ----------------------------
    # Step 3: Convert A2A Agent to LangChain Tool
    # ----------------------------
    print("\n📝 Step 3: Converting A2A Agent to LangChain")
    try:
        langchain_agent = to_langchain_agent(a2a_server_url)
        print("✅ Successfully converted A2A agent to LangChain")
    except Exception as e:
        print(f"❌ Error converting A2A agent to LangChain: {e}")
        return 1

    # ----------------------------
    # Step 4: Convert MCP Tools to LangChain Tools
    # ----------------------------
    print("\n📝 Step 4: Converting MCP Tools to LangChain")
    try:
        stock_data_tool = to_langchain_tool(mcp_server_url, "stock_data")
        web_scraper_tool = to_langchain_tool(mcp_server_url, "web_scraper")
        print("✅ Successfully converted MCP tools to LangChain")
    except Exception as e:
        print(f"❌ Error converting MCP tools to LangChain: {e}")
        print("   Continuing with only the A2A agent available.")
        stock_data_tool = None
        web_scraper_tool = None

    # ----------------------------
    # Step 5: Test Individual Components
    # ----------------------------
    print("\n📝 Step 5: Testing Individual Components")

    # 5a) Test the A2A-based LangChain agent
    try:
        print("\nTesting A2A-based LangChain agent:")
        result = langchain_agent.invoke("What are some key metrics to evaluate a company's stock?")
        print("A2A Agent Response:")
        print(result.get("output", "<no output>"))
    except Exception as e:
        print(f"❌ Error using A2A-based LangChain agent: {e}")

    # 5b) Test the MCP-based LangChain tools (if available)
    if stock_data_tool and web_scraper_tool:
        try:
            print("\nTesting MCP-based LangChain tools:")

            print("\n1) Stock Data Tool (AAPL):")
            stock_res = stock_data_tool.invoke("AAPL")
            print(stock_res[:500] + "..." if isinstance(stock_res, str) and len(stock_res) > 500 else stock_res)

            print("\n2) Web Scraper Tool (AAPL):")
            web_res = web_scraper_tool.invoke("AAPL")
            print(web_res[:500] + "..." if isinstance(web_res, str) and len(web_res) > 500 else web_res)
        except Exception as e:
            print(f"❌ Error using MCP-based LangChain tools: {e}")
            traceback.print_exc()

    # ----------------------------
    # Step 6: Build the Meta-Agent
    # ----------------------------
    print("\n📝 Step 6: Creating Meta-Agent with Available Tools")
    try:
        # 6a) Instantiate an LLM for the meta-agent
        llm = ChatOpenAI(model=args.model, temperature=args.temperature)

        # 6b) Wrap each tool/function in a Python wrapper for safety
        def ask_stock_expert(query: str) -> str:
            """Ask the A2A agent (stock expert) a freeform question."""
            try:
                resp = langchain_agent.invoke(query)
                return resp.get("output", "No response from StockExpert.")
            except Exception as e:
                return f"Error querying StockExpert: {str(e)}"

        tools = [
            Tool(
                name="StockExpert",
                func=ask_stock_expert,
                description=(
                    "Ask the stock market expert questions about investing, market trends, financial concepts, etc."
                ),
            )
        ]

        # 6c) Add MCP tools if they were successfully converted
        if stock_data_tool:
            def get_stock_data(ticker_info) -> str:
                """Get stock data via the MCP 'stock_data' tool."""
                try:
                    if ticker_info is None:
                        return "Error: No ticker symbol provided."
                    if not isinstance(ticker_info, str):
                        ticker_info = str(ticker_info)
                    return stock_data_tool.invoke(ticker_info)
                except Exception as e:
                    return f"Error getting stock data: {str(e)}"

            tools.append(
                Tool(
                    name="StockData",
                    func=get_stock_data,
                    description=(
                        "Get historical stock data. "
                        "Input can be one or more ticker symbols (e.g., 'AAPL' or 'AAPL, MSFT')."
                    ),
                )
            )

        if web_scraper_tool:
            def get_financial_news(query: str) -> str:
                """Get financial news/company snapshot via the MCP 'web_scraper' tool."""
                try:
                    if query is None:
                        return "Error: No query provided."
                    if not isinstance(query, str):
                        query = str(query)
                    return web_scraper_tool.invoke(query)
                except Exception as e:
                    return f"Error getting financial news: {str(e)}"

            tools.append(
                Tool(
                    name="FinancialNews",
                    func=get_financial_news,
                    description=(
                        "Get financial news or company details. Input can be a ticker symbol, a URL, or a topic."
                    ),
                )
            )

        # 6d) Initialize a LangChain meta-agent that uses OpenAI Functions
        meta_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
        )

        # 6e) Test the final meta-agent
        print("\nTesting Meta-Agent with available tools:")
        test_query = "What are the current stock prices of Apple and Nvidia?"
        print(f"\nQuery: {test_query}")
        meta_result = meta_agent.invoke(test_query)
        print("\nMeta-Agent Response:")
        print(meta_result)
    except Exception as e:
        print(f"❌ Error creating or using meta-agent: {e}")
        traceback.print_exc()

    # ----------------------------
    # Keep the servers alive until Ctrl+C
    # ----------------------------
    print("\n✅ Integration successful! The A2A and MCP servers are still running in the background.")
    print("Press Ctrl+C to stop the servers and exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
