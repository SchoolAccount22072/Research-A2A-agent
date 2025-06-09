import os
import sys
import socket
import time
import threading
import argparse
import json
import traceback
from datetime import datetime
from dotenv import load_dotenv

# A2A imports
from python_a2a import OpenAIA2AServer, run_server, A2AServer, AgentCard, AgentSkill
from python_a2a.mcp import FastMCP
from python_a2a.langchain import to_langchain_agent, to_langchain_tool

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

# Web & parsing imports
import requests
from bs4 import BeautifulSoup
import re
import xml.etree.ElementTree as ET

# RSS parsing voor Medium
import feedparser
load_dotenv()

def find_available_port(start_port: int = 5000, max_tries: int = 20) -> int:
    """
    Vind een beschikbare TCP-poort op localhost.
    Probeert poorten van start_port tot start_port + max_tries - 1.
    Als er geen vrije poort wordt gevonden, retourneert start_port + 1000 als fallback.
    """
    for port in range(start_port, start_port + max_tries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("localhost", port))
            sock.close()
            return port
        except OSError:
            continue
    print(f"⚠️  Geen beschikbare poort gevonden in bereik {start_port}-{start_port + max_tries - 1}")
    return start_port + 1000


def run_server_in_thread(
    server_func,
    server_obj,
    host: str = "0.0.0.0",
    port: int = None
) -> threading.Thread:
    """
    Draai een serverfunctie (bijv. run_server of server.run) in een achtergrondthread.
    Wacht kort om de server de tijd te geven op te starten.
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
    Parseer command-line argumenten:
      --a2a-port:  (optioneel) TCP-poort voor de A2A-server
      --mcp-port:  (optioneel) TCP-poort voor de MCP-server
      --model:     OpenAI-model om te gebruiken (standaard: gpt-4o)
      --temperature: temperatuur voor de LLM (standaard: 0.2)
    """
    parser = argparse.ArgumentParser(
        description="A2A + MCP Lesplanner Integratie Voorbeeld (Functioneel Script)"
    )
    parser.add_argument(
        "--a2a-port",
        type=int,
        default=None,
        help="Poort om de A2A-server op te draaien (standaard: auto-select)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=None,
        help="Poort om de MCP-server op te draaien (standaard: auto-select)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI-model om te gebruiken (standaard: gpt-4o)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatuur voor generatie (standaard: 0.2)",
    )
    return parser.parse_args()


def check_api_key() -> bool:
    """
    Controleer of de OPENAI_API_KEY-omgeving variabele is gezet.
    Retourneert False en stopt als deze niet aanwezig is.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        print("❌ Fout: De omgeving variabele OPENAI_API_KEY is niet gezet.")
        print("   Voeg je sleutel toe in het bestand .env (in de root van je project).")
        return False
    return True


def main():
    """Hoofd entry-point."""
    # 1) Controleer OpenAI API-sleutel
    if not check_api_key():
        return 1

    # 2) Parseer argumenten
    args = parse_arguments()

    # 3) Bepaal poorten (auto-select als niet gespecificeerd)
    a2a_port = args.a2a_port or find_available_port(5000, 20)
    mcp_port = args.mcp_port or find_available_port(7000, 20)

    print(f"🔍 A2A-server poort: {a2a_port}")
    print(f"🔍 MCP-server poort: {mcp_port}")

    # ----------------------------
    # Stap 1: Maak de OpenAI-gestuurde A2A Server (Lesplanner Agent)
    # ----------------------------
    print("\n📝 Stap 1: Creëren van OpenAI-gestuurde A2A Server (Lesplanner)")

    # Definieer een AgentCard voor onze “Lesplanner”
    agent_card = AgentCard(
        name="Lesplanner",
        description="Een A2A-agent die gedetailleerde lesplannen genereert door info te verzamelen van Wikipedia, Medium, GitHub, YouTube, StackOverflow en arXiv.",
        url=f"http://localhost:{a2a_port}",
        version="1.3.0",
        skills=[
            AgentSkill(
                name="Inhoudsaggregatie",
                description=(
                    "Verzamel zowel theoretische als praktische informatie uit meerdere bronnen "
                    "(Wikipedia, Medium, GitHub, YouTube, StackOverflow, arXiv)."
                ),
                examples=[
                    "Haal kernconcepten van machinaal leren op uit Wikipedia.",
                    "Verkrijg top Medium-artikelen over diepe leermodellen.",
                    "Zoek relevante StackOverflow-vragen over Python-agenten.",
                    "Download recente arXiv-papers over Large Language Models.",
                ],
            ),
            AgentSkill(
                name="Les Synthese",
                description="Combineer alle verzamelde data tot een samenhangend, diepgaand lesplan met theorie, codevoorbeelden en oefeningen.",
                examples=[
                    "Maak een lesplan voor 'Inleiding tot K-Means clustering' met opgehaalde data.",
                    "Genereer een stapsgewijze Python-tutorial voor 'Verwerking van ongestructureerde data'.",
                ],
            ),
            AgentSkill(
                name="Bronaanbevelingen",
                description=(
                    "Doe aanbevelingen voor extra bronnen zoals YouTube-video's, GitHub-repository's, "
                    "artikelen, StackOverflow-discussies en wetenschappelijke papers."
                ),
                examples=[
                    "Noem 5 YouTube-tutorials over Python datastructuren.",
                    "Geef GitHub-repo's met voorbeeldprojecten voor natuurlijke taalverwerking.",
                    "Geef relevante StackOverflow-discussies over LLM-implementaties.",
                    "Geef 3 arXiv-papers over transformerarchitecturen.",
                ],
            ),
        ],
    )

    # Sterkere system_prompt die expliciet een iteratief “gesprek” tussen agents afdwingt in het Nederlands
    openai_server = OpenAIA2AServer(
        api_key=os.environ["OPENAI_API_KEY"],
        model = os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature = float(os.getenv("TEMPERATURE", "0.2")),
        system_prompt=(
            "Je bent een Lesplanner-agent. Je taak is om een **uitgebreid, diepgaand** "
            "lesplan in Markdown-formaat te genereren, volledig in het Nederlands. De gebruiker geeft een "
            "onderwerp (bijv. 'Week 4: Data Warehouse-optimalisatie & -onderhoud').\n\n"
            "Je werkt in meerdere fasen en laat je ‘sub-agents’ (de scrapers) elkaar context doorgeven:\n\n"
            "1. **Fase 1 – WikiScraper**:\n"
            "   • Roep WikiScraper aan om 10 sleutelzinnen (zowel theoretisch als praktisch) op te halen van NL Wikipedia.\n"
            "     • Als de pagina niet bestaat, zoek dynamisch op NL Wikipedia naar een gerelateerd artikel ('AI', 'LLM', 'Data Warehouse') en gebruik die.\n"
            "   • Geef de gevonden zinnen in één pakket door aan de volgende fase.\n\n"
            "2. **Fase 2 – MediumScraper**:\n"
            "   • Gebruik de output van Fase 1 (de Wiki-sleutelzinnen) als zoekcontext om de top 5 Medium-artikelen op te halen (titel + link).\n"
            "   • Voeg elke gevonden titel + link samen met beknopte samenvatting (indien beschikbaar) toe aan het contextpakket.\n\n"
            "3. **Fase 3 – GitHubScraper**:\n"
            "   • Gebruik de gecombineerde context (Wiki + Medium) om de top 5 GitHub-repository's te zoeken (naam, URL, omschrijving).\n"
            "   • Voeg de repository-gegevens toe aan het contextpakket.\n\n"
            "4. **Fase 4 – YouTubeScraper**:\n"
            "   • Met de context (Wiki + Medium + GitHub) zoek je de top 5 relevante YouTube-video's (URL, waar mogelijk titel).\n"
            "   • Voeg elk videolink-item toe met beknopte toelichting (of gebruik de video-omschrijving).\n\n"
            "5. **Fase 5 – StackOverflowScraper**:\n"
            "   • Gebruik de volledige context (Wiki + Medium + GitHub + YouTube) om de top 5 relevante StackOverflow-vragen op te halen (titel, URL, korte snippet van antwoord).\n"
            "   • Voeg deze toe aan het contextpakket.\n\n"
            "6. **Fase 6 – ArXivScraper**:\n"
            "   • Tot slot gebruik je de volledige context om de top 5 recente arXiv-papers te vinden (titel, URL, korte samenvatting).\n"
            "   • Voeg deze ook toe.\n\n"
            "7. **Synthetiseer** alle verzamelde data tot een **gestructureerd lesplan** in Markdown met de volgende secties (in het Nederlands), waarbij je telkens vermeldt wat de bron is (bijv. Wikipedia, Medium, GitHub, YouTube, StackOverflow, arXiv) en in elke sectie minstens twee alinea’s:\n"
            "   • **Week 4: Data Warehouse-optimalisatie & -onderhoud**\n"
            "     – Beschrijf hoe je dashboards en rapportages maakt in Power BI om inzichten te verkrijgen uit het Data Warehouse.\n"
            "     – Leg uit hoe je Power BI-rapporten publiceert, bewerkt en deelt binnen een organisatie.\n"
            "     – Bespreek data‐warehouse-synchronisatie: hoe je na de initiële laadtoestand wijzigingen (insert/update/delete) in operationele databases continu bijwerkt in het DWH.\n"
            "     – Leg technieken uit zoals Change Data Capture (CDC), ETL vs. ELT, en incrementele laadstrategieën.\n"
            "   • **Leerdoelen** (bullet points)\n"
            "   • **Inleiding** (context + belang)\n"
            "   • **Theoretische Basis** (definities, formules)\n"
            "   • **Kernconcepten & Algoritmen** (stapsgewijze uitleg)\n"
            "   • **Praktische Voorbeelden & Code Uitleg**:\n"
            "       – Minstens **twee onderscheidende Python-codevoorbeelden** in ```python```-blokken.\n"
            "       – Leg elke regel code uit in het Nederlands (minstens twee alinea’s toelichting per voorbeeld).\n"
            "       – Toon hoe je de code draait (`python voorbeeld.py`) en de verwachte uitvoer.\n"
            "   • **Casestudy of Praktijksituatie**\n"
            "   • **Oefeningen (met Uitgebreide Oplossingen)**:\n"
            "       – Minstens **vijf oefeningen** van oplopende moeilijkheid.\n"
            "       – Elke oefening bevat **meerdere deelvragen** (a, b, c) en stap-voor-stap oplossingen (minstens twee alinea’s toelichting per deelvraag).\n"
            "       – Zorg dat elke subkop in dit gedeelte minstens twee alinea’s tekst bevat.\n"
            "   • **Aanvullende Bronnen & Referenties** (5 per bron):\n"
            "       – Wikipedia (5 bronzinnen/titels met toelichting)\n"
            "       – Medium-artikelen (5 hyperlinks + toelichting)\n"
            "       – GitHub-repository’s (5 hyperlinks + toelichting)\n"
            "       – YouTube-video’s (5 hyperlinks + toelichting)\n"
            "       – StackOverflow-vragen (5 hyperlinks + toelichting)\n"
            "       – arXiv-papers (5 hyperlinks + toelichting)\n"
            "       – Suggesties voor verdere verdieping.\n"
            "   • **Samenvatting & Volgende Stappen** (minstens twee alinea’s)\n\n"
            "8. **Structureringsregels**:\n"
            "   • Elke subsectie moet minstens **twee alinea’s** bevatten.\n"
            "   • Gebruik duidelijke Nederlandse koppen (`##`, `###`).\n"
            "   • Vermeld onder elke paragraaf **welke bron** (Wikipedia, Medium, GitHub, YouTube, StackOverflow, arXiv) de informatie heeft bijgedragen.\n"
            "   • Alle tekst en voorbeeldcode in het Nederlands (code-commentaar in het Nederlands).\n"
            "   • Maak een **nieuw bestand** voor elke aanvraag; voeg nooit toe aan bestaande bestanden.\n"
            "   • Zorg dat elk stukje context (output van een eerdere fase) wordt meegenomen in latere fases, zodat de agent écht ‘kan doorpraten’. "
        ),
    )

    class LessonPlannerAgent(A2AServer):
        def __init__(self, openai_server: OpenAIA2AServer, agent_card: AgentCard):
            super().__init__(agent_card=agent_card)
            self.openai_server = openai_server

        def handle_message(self, message: dict) -> dict:
            """
            Stuur binnenkomende A2A-berichten naar de OpenAI-backend.
            Verwacht het A2A “message” dict; retourneert een response dict.
            """
            return self.openai_server.handle_message(message)

    # Instantiate en start de A2A-server in een achtergrondthread
    lesson_agent = LessonPlannerAgent(openai_server, agent_card)
    a2a_server_url = f"http://localhost:{a2a_port}"
    print(f"\nStarting A2A server op {a2a_server_url}...")

    def run_a2a_server(server_obj, host="0.0.0.0", port=a2a_port):
        run_server(server_obj, host=host, port=port)

    a2a_thread = run_server_in_thread(run_a2a_server, lesson_agent, port=a2a_port)

    # ----------------------------
    # Stap 2: Maak de MCP-server met “Les Aggregatie Tools”
    # ----------------------------
    print("\n📝 Stap 2: Creëren van MCP-server met Les Aggregatie Tools")

    mcp_server = FastMCP(
        name="Les Aggregatie Tools",
        description="Tools om lesinhoud te scrapen van Wikipedia NL, Medium, GitHub, YouTube, StackOverflow en arXiv.",
    )

    # ----- Tool #1: wiki_scraper -----
    @mcp_server.tool(
        name="wiki_scraper",
        description="Haal 10 sleutelzinnen (theoretisch + praktisch) op van de Nederlandse Wikipedia voor een onderwerp."
    )
    def wiki_scraper(input_str=None, **kwargs):
        """
        Scrape NL Wikipedia om tien relevante zinnen te krijgen over een opgegeven pagina titel.
        Als de directe pagina niet bestaat, voer dan een dynamische zoekopdracht uit en pak de eerste gerelateerde pagina.
        Retourneert een JSON-string met {'sentences': [ ... ]}.
        """
        try:
            if input_str is None and "input" in kwargs:
                input_str = kwargs["input"]
            if input_str is None:
                return {"text": "Error: Geen onderwerp opgegeven."}

            onderwerp_raw = str(input_str).strip()
            # Probeer eerst de exacte pagina op te halen
            def fetch_wiki_page(title):
                url = f"https://nl.wikipedia.org/wiki/{title.replace(' ', '_')}"
                headers = {"User-Agent": "Mozilla/5.0 (compatible; Lesplanner/1.0; +http://localhost/)"}
                resp = requests.get(url, headers=headers, timeout=10)
                return resp.status_code, resp.text

            # Stap 1: probeer directe pagina
            code, html = fetch_wiki_page(onderwerp_raw)
            if code == 404:
                # Stap 2: zoek via MediaWiki API
                search_url = (
                    "https://nl.wikipedia.org/w/api.php"
                    "?action=query&list=search&utf8=&format=json"
                    f"&srsearch={requests.utils.quote(onderwerp_raw)}"
                )
                search_resp = requests.get(search_url, timeout=10)
                if search_resp.status_code != 200:
                    return {"text": json.dumps({"error": "Wikipedia zoekactie mislukt."})}
                results = search_resp.json().get("query", {}).get("search", [])
                if not results:
                    return {"text": json.dumps({"error": f"Geen gerelateerde Wikipedia-pagina gevonden voor '{onderwerp_raw}'."})}
                # Pak de titel van het eerste zoekresultaat
                new_title = results[0].get("title", "")
                code, html = fetch_wiki_page(new_title)
                if code != 200:
                    return {"text": json.dumps({"error": f"Kan ook de gerelateerde pagina '{new_title}' niet ophalen."})}
            elif code != 200:
                return {"text": json.dumps({"error": f"Wikipedia-pagina voor '{onderwerp_raw}' niet beschikbaar (status {code})."})}

            # Parse de HTML
            soup = BeautifulSoup(html, "html.parser")
            paras = soup.select("div.mw-parser-output > p")
            sentences = []
            for p in paras:
                text = p.get_text().strip()
                # Split in zinnen
                for sent in re.split(r'(?<=[.!?])\s+', text):
                    clean = sent.strip()
                    if len(clean) >= 60:
                        sentences.append(clean)
                    if len(sentences) >= 10:
                        break
                if len(sentences) >= 10:
                    break

            if not sentences:
                return {"text": json.dumps({
                    "error": f"Geen geschikte inhoud gevonden voor '{onderwerp_raw}'."
                })}

            return {"text": json.dumps({"sentences": sentences})}

        except Exception as e:
            tb = traceback.format_exc()
            return {"text": f"Error: {str(e)}\nDetails: {tb}"}

    # ----- Tool #2: medium_scraper -----
    @mcp_server.tool(
        name="medium_scraper",
        description="Haal de top 5 Medium-artikels (titel + link) op voor een gegeven onderwerp."
    )
    def medium_scraper(input_str=None, **kwargs):
        """
        Haal top 5 Medium-artikelen op met behulp van de tag RSS-feed.
        Retourneert JSON-string met {'articles': [{title, link}, ...]}.
        """
        try:
            if input_str is None and "input" in kwargs:
                input_str = kwargs["input"]
            if input_str is None:
                return {"text": "Error: Geen onderwerp opgegeven."}

            tag = str(input_str).strip().lower().replace(" ", "-")
            rss_url = f"https://medium.com/feed/tag/{tag}"
            feed = feedparser.parse(rss_url)
            if not feed.entries:
                return {"text": json.dumps({
                    "error": f"Geen Medium-artikelen gevonden voor tag '{tag}'."
                })}

            articles = []
            for entry in feed.entries[:5]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                if title and link:
                    articles.append({"title": title, "link": link})

            return {"text": json.dumps({"articles": articles})}

        except Exception as e:
            tb = traceback.format_exc()
            return {"text": f"Error: {str(e)}\nDetails: {tb}"}

    # ----- Tool #3: github_scraper -----
    @mcp_server.tool(
        name="github_scraper",
        description="Haal de top 5 GitHub-repository's op voor een gegeven onderwerp."
    )
    def github_scraper(input_str=None, **kwargs):
        """
        Zoek via de GitHub API naar top repositories die overeenkomen met het onderwerp.
        Retourneert JSON-string met {'repos': [{name, url, description}, ...]}.
        """
        try:
            if input_str is None and "input" in kwargs:
                input_str = kwargs["input"]
            if input_str is None:
                return {"text": "Error: Geen onderwerp opgegeven."}

            query = str(input_str).strip()
            api_url = (
                f"https://api.github.com/search/repositories"
                f"?q={requests.utils.quote(query)}&sort=stars&order=desc&per_page=5"
            )
            headers = {"Accept": "application/vnd.github.v3+json"}
            resp = requests.get(api_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return {"text": json.dumps({
                    "error": f"GitHub API-aanvraag mislukt (status {resp.status_code})."
                })}

            data = resp.json()
            items = data.get("items", [])
            repos = []
            for item in items:
                name = item.get("full_name", "")
                url = item.get("html_url", "")
                desc = item.get("description", "") or ""
                repos.append({"name": name, "url": url, "description": desc})

            if not repos:
                return {"text": json.dumps({
                    "error": f"Geen GitHub-repo's gevonden voor '{query}'."
                })}

            return {"text": json.dumps({"repos": repos})}

        except Exception as e:
            tb = traceback.format_exc()
            return {"text": f"Error: {str(e)}\nDetails: {tb}"}

    # ----- Tool #4: youtube_scraper -----
    @mcp_server.tool(
        name="youtube_scraper",
        description="Haal de top 5 YouTube-video's op voor een gegeven onderwerp."
    )
    def youtube_scraper(input_str=None, **kwargs):
        """
        Scrape YouTube-zoekresultaten om de eerste 5 video-URL's te krijgen.
        Retourneert JSON-string met {'videos': [{title, url}, ...]}.
        """
        try:
            if input_str is None and "input" in kwargs:
                input_str = kwargs["input"]
            if input_str is None:
                return {"text": "Error: Geen onderwerp opgegeven."}

            query = str(input_str).strip()
            search_url = "https://www.youtube.com/results"
            params = {"search_query": query}
            headers = {"User-Agent": "Mozilla/5.0 (compatible; Lesplanner/1.0; +http://localhost/)"}
            resp = requests.get(search_url, params=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                return {"text": json.dumps({
                    "error": f"YouTube-zoekopdracht mislukt (status {resp.status_code})."
                })}

            html = resp.text
            # Zoek video-ID's (ongeveer)
            video_ids = re.findall(r"\/watch\?v=(.{11})", html)
            unieke_ids = []
            for vid in video_ids:
                if vid not in unieke_ids:
                    unieke_ids.append(vid)
                if len(unieke_ids) >= 5:
                    break

            videos = []
            for vid in unieke_ids:
                url = f"https://www.youtube.com/watch?v={vid}"
                videos.append({"title": "", "url": url})

            if not videos:
                return {"text": json.dumps({
                    "error": f"Geen YouTube-video's gevonden voor '{query}'."
                })}

            return {"text": json.dumps({"videos": videos})}

        except Exception as e:
            tb = traceback.format_exc()
            return {"text": f"Error: {str(e)}\nDetails: {tb}"}

    # ----- Tool #5: stackoverflow_scraper -----
    @mcp_server.tool(
        name="stackoverflow_scraper",
        description="Haal de top 5 relevante StackOverflow-vragen en -antwoorden op voor een gegeven onderwerp."
    )
    def stackoverflow_scraper(input_str=None, **kwargs):
        """
        Scrape de StackOverflow zoekresultaten voor een onderwerp.
        Retourneert JSON-string met {'questions': [{title, url, snippet}, ...]}.
        """
        try:
            if input_str is None and "input" in kwargs:
                input_str = kwargs["input"]
            if input_str is None:
                return {"text": "Error: Geen onderwerp opgegeven."}

            query = str(input_str).strip()
            search_url = f"https://stackoverflow.com/search?q={requests.utils.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; Lesplanner/1.0; +http://localhost/)"}
            resp = requests.get(search_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return {"text": json.dumps({
                    "error": f"StackOverflow-zoekopdracht mislukt (status {resp.status_code})."
                })}

            soup = BeautifulSoup(resp.text, "html.parser")
            question_summaries = soup.select(".question-summary")[:5]
            questions = []
            for summary in question_summaries:
                link_tag = summary.select_one(".question-hyperlink")
                if not link_tag:
                    continue
                title = link_tag.get_text().strip()
                url = f"https://stackoverflow.com{link_tag['href']}"
                snippet_tag = summary.select_one(".excerpt")
                snippet = snippet_tag.get_text().strip() if snippet_tag else ""
                questions.append({"title": title, "url": url, "snippet": snippet})
            if not questions:
                return {"text": json.dumps({
                    "error": f"Geen StackOverflow-vragen gevonden voor '{query}'."
                })}

            return {"text": json.dumps({"questions": questions})}

        except Exception as e:
            tb = traceback.format_exc()
            return {"text": f"Error: {str(e)}\nDetails: {tb}"}

    # ----- Tool #6: arxiv_scraper -----
    @mcp_server.tool(
        name="arxiv_scraper",
        description="Haal de top 5 recente arXiv-papers op voor een gegeven onderwerp."
    )
    def arxiv_scraper(input_str=None, **kwargs):
        """
        Query de arXiv API voor papers over een onderwerp.
        Retourneert JSON-string met {'papers': [{title, url, summary}, ...]}.
        """
        try:
            if input_str is None and "input" in kwargs:
                input_str = kwargs["input"]
            if input_str is None:
                return {"text": "Error: Geen onderwerp opgegeven."}

            query = str(input_str).strip()
            api_url = (
                "http://export.arxiv.org/api/query"
                f"?search_query=all:{requests.utils.quote(query)}&start=0&max_results=5"
            )
            resp = requests.get(api_url, timeout=10)
            if resp.status_code != 200:
                return {"text": json.dumps({
                    "error": f"arXiv API-aanvraag mislukt (status {resp.status_code})."
                })}

            root = ET.fromstring(resp.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            papers = []
            for entry in entries:
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                link_elem = entry.find("atom:link[@rel='alternate']", ns)
                title = title_elem.text.strip() if title_elem is not None else ""
                summary = summary_elem.text.strip() if summary_elem is not None else ""
                url = link_elem.attrib.get('href', '') if link_elem is not None else ""
                papers.append({"title": title, "url": url, "summary": summary})
            if not papers:
                return {"text": json.dumps({
                    "error": f"Geen arXiv-papers gevonden voor '{query}'."
                })}

            return {"text": json.dumps({"papers": papers})}

        except Exception as e:
            tb = traceback.format_exc()
            return {"text": f"Error: {str(e)}\nDetails: {tb}"}

    # 3) Start de MCP-server thread
    mcp_server_url = f"http://localhost:{mcp_port}"
    print(f"\nStarting MCP server op {mcp_server_url}...")

    def run_mcp_server(server_obj, host="0.0.0.0", port=mcp_port):
        server_obj.run(host=host, port=port)

    mcp_thread = run_server_in_thread(run_mcp_server, mcp_server, port=mcp_port)

    # 4) Wacht kort om te zorgen dat beide servers opstarten
    print("\nWachten tot servers initialized are...")
    time.sleep(5)

    # 5) Controleer MCP-server /tools endpoint
    mcp_server_running = False
    try:
        resp = requests.get(f"{mcp_server_url}/tools", timeout=5)
        if resp.status_code == 200:
            mcp_server_running = True
    except:
        pass

    if not mcp_server_running:
        print(f"❌ MCP-server kon niet starten op poort {mcp_port}. Probeer een nieuwe poort...")
        mcp_port = find_available_port(8000, 20)
        mcp_server_url = f"http://localhost:{mcp_port}"
        print(f"🔍 Nieuwe MCP-server poort: {mcp_port}")
        mcp_thread = run_server_in_thread(run_mcp_server, mcp_server, port=mcp_port)
        time.sleep(5)

    # ----------------------------
    # Stap 3: Converteer A2A Agent naar LangChain-tool
    # ----------------------------
    print("\n📝 Stap 3: Converteer A2A-agent naar LangChain")
    try:
        langchain_agent = to_langchain_agent(a2a_server_url)
        print("✅ Succesvol geconverteerd A2A-agent naar LangChain")
    except Exception as e:
        print(f"❌ Fout bij converteren van A2A-agent naar LangChain: {e}")
        return 1

    # ----------------------------
    # Stap 4: Converteer MCP-tools naar LangChain-tools
    # ----------------------------
    print("\n📝 Stap 4: Converteer MCP-tools naar LangChain-tools")
    try:
        wiki_tool             = to_langchain_tool(mcp_server_url, "wiki_scraper")
        medium_tool           = to_langchain_tool(mcp_server_url, "medium_scraper")
        github_tool           = to_langchain_tool(mcp_server_url, "github_scraper")
        youtube_tool          = to_langchain_tool(mcp_server_url, "youtube_scraper")
        stackoverflow_tool    = to_langchain_tool(mcp_server_url, "stackoverflow_scraper")
        arxiv_tool            = to_langchain_tool(mcp_server_url, "arxiv_scraper")
        print("✅ Succesvol geconverteerd MCP-tools naar LangChain")
    except Exception as e:
        print(f"❌ Fout bij converteren van MCP-tools naar LangChain: {e}")
        return 1

    # ----------------------------
    # Stap 5: Bouw de Meta-Agent
    # ----------------------------
    print("\n📝 Stap 5: Creëer Meta-Agent met Beschikbare Tools")
    try:
        # 5a) Maak een LLM voor de meta-agent
        llm = ChatOpenAI(model=args.model, temperature=args.temperature)

        # 5b) Wikkel elke MCP-tool in een Python-wrapper
        def get_wiki_info(topic: str) -> str:
            """Haal sleutelzinnen van Wikipedia op via MCP 'wiki_scraper'-tool."""
            try:
                if topic is None:
                    return "Error: Geen onderwerp opgegeven."
                return wiki_tool.invoke(topic)
            except Exception as e:
                return f"Error bij ophalen Wikipedia-info: {str(e)}"

        def get_medium_articles(topic: str) -> str:
            """Haal Medium-artikelen op via MCP 'medium_scraper'-tool."""
            try:
                if topic is None:
                    return "Error: Geen onderwerp opgegeven."
                return medium_tool.invoke(topic)
            except Exception as e:
                return f"Error bij ophalen Medium-artikelen: {str(e)}"

        def get_github_repos(topic: str) -> str:
            """Haal GitHub-repo's op via MCP 'github_scraper'-tool."""
            try:
                if topic is None:
                    return "Error: Geen onderwerp opgegeven."
                return github_tool.invoke(topic)
            except Exception as e:
                return f"Error bij ophalen GitHub-repo's: {str(e)}"

        def get_youtube_videos(topic: str) -> str:
            """Haal YouTube-video's op via MCP 'youtube_scraper'-tool."""
            try:
                if topic is None:
                    return "Error: Geen onderwerp opgegeven."
                return youtube_tool.invoke(topic)
            except Exception as e:
                return f"Error bij ophalen YouTube-video's: {str(e)}"

        def get_stackoverflow_qas(topic: str) -> str:
            """Haal StackOverflow Q&A op via MCP 'stackoverflow_scraper'-tool."""
            try:
                if topic is None:
                    return "Error: Geen onderwerp opgegeven."
                return stackoverflow_tool.invoke(topic)
            except Exception as e:
                return f"Error bij ophalen StackOverflow-vragen: {str(e)}"

        def get_arxiv_papers(topic: str) -> str:
            """Haal arXiv-papers op via MCP 'arxiv_scraper'-tool."""
            try:
                if topic is None:
                    return "Error: Geen onderwerp opgegeven."
                return arxiv_tool.invoke(topic)
            except Exception as e:
                return f"Error bij ophalen arXiv-papers: {str(e)}"

        tools = [
            Tool(
                name="WikiScraper",
                func=get_wiki_info,
                description="Haal tien relevante zinnen (theorie & praktijk) op van NL Wikipedia voor een onderwerp."
            ),
            Tool(
                name="MediumScraper",
                func=get_medium_articles,
                description="Haal titels en links van de top 5 Medium-artikelen voor een onderwerp."
            ),
            Tool(
                name="GitHubScraper",
                func=get_github_repos,
                description="Haal de top 5 GitHub-repository's op voor een onderwerp."
            ),
            Tool(
                name="YouTubeScraper",
                func=get_youtube_videos,
                description="Haal de URL's van de top 5 YouTube-video's op voor een onderwerp."
            ),
            Tool(
                name="StackOverflowScraper",
                func=get_stackoverflow_qas,
                description="Haal de top 5 relevante StackOverflow-vragen en -antwoorden op voor een onderwerp."
            ),
            Tool(
                name="ArXivScraper",
                func=get_arxiv_papers,
                description="Haal de top 5 recente arXiv-papers op voor een onderwerp."
            ),
        ]

        # 5c) Initialiseer een LangChain meta-agent die OpenAI-functies gebruikt
        meta_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
        )

    except Exception as e:
        print(f"❌ Fout bij creëren van meta-agent: {e}")
        traceback.print_exc()
        return 1

    # ----------------------------
    # Stap 6: Prompt de Gebruiker & Genereer Lesplan
    # ----------------------------
    print("\n🔍 Beschrijf in het Nederlands het onderwerp dat je wilt leren (bijv. 'Week 4: Data Warehouse-optimalisatie & -onderhoud').")
    raw_user_input = input("In enkele zinnen, beschrijf je lesonderwerp: ").strip()
    if not raw_user_input:
        print("❌ Fout: Geen invoer ontvangen. Afsluiten.")
        return 1

    # Wikkel de gebruikersinvoer in een expliciete meta-prompt in het Nederlands
    meta_prompt = f"""
Onderwerp Beschrijving (in het Nederlands):
\"\"\"
{raw_user_input}
\"\"\"

Instructies (in het Nederlands):
1. **Fase 1 – WikiScraper**:
   • Roep WikiScraper(onderwerp) aan om **10** sleutelzinnen (theorie & praktijk) te verzamelen van Wikipedia NL.
   • Als de directe pagina niet bestaat, zoek dynamisch naar een gerelateerde pagina ('Data Warehouse', 'Power BI', etc.) en gebruik die.
   • Stuur de output in één pakket door naar Fase 2.

2. **Fase 2 – MediumScraper**:
   • Gebruik de output van Fase 1 (Wiki-zinnen) als zoekcontext en roep MediumScraper(onderwerp) aan.
   • Haal **5** Medium-artikelen (titel + link) op. Voeg beknopte samenvatting toe indien beschikbaar.
   • Voeg deze items toe aan de context en stuur door naar Fase 3.

3. **Fase 3 – GitHubScraper**:
   • Gebruik de gecombineerde context (Wiki + Medium) en roep GitHubScraper(onderwerp) aan.
   • Haal **5** GitHub-repository's op (naam, URL, omschrijving).
   • Voeg deze toe aan de context en stuur door naar Fase 4.

4. **Fase 4 – YouTubeScraper**:
   • Met de context (Wiki + Medium + GitHub) roep je YouTubeScraper(onderwerp) aan.
   • Haal **5** YouTube-video’s op (URL, titel of korte toelichting).
   • Voeg deze toe aan de context en stuur door naar Fase 5.

5. **Fase 5 – StackOverflowScraper**:
   • Gebruik de volledige context (Wiki + Medium + GitHub + YouTube) en roep StackOverflowScraper(onderwerp) aan.
   • Haal **5** relevante StackOverflow-vragen en -antwoorden op (titel, URL, korte snippet).
   • Voeg deze toe aan de context en stuur door naar Fase 6.

6. **Fase 6 – ArXivScraper**:
   • Gebruik de volledige context en roep ArXivScraper(onderwerp) aan.
   • Haal **5** recente arXiv-papers op (titel, URL, samenvatting).
   • Voeg deze toe aan de context.

7. **Synthetiseer** alle verzamelde data tot één **uitgebreid Markdown-lesplan** in het Nederlands, met vermelding van de bron (Wikipedia, Medium, GitHub, YouTube, StackOverflow, arXiv) in elke paragraaf. Het document bevat ten minste de volgende secties, elke sectie minstens twee alinea’s:
   1. **Week 4: Data Warehouse-optimalisatie & -onderhoud**
      - Beschrijf hoe je dashboards en rapportages maakt in Power BI om inzichten te verkrijgen uit het Data Warehouse.
      - Leg uit hoe je Power BI-rapporten publiceert, bewerkt en deelt binnen een organisatie.
      - Bespreek data-warehouse-synchronisatie: hoe je na de initiële laadtoestand wijzigingen (insert/update/delete) in operationele databases continu bijwerkt in het DWH.
      - Leg technieken uit zoals Change Data Capture (CDC), ETL vs. ELT, en incrementele laadstrategieën.
   2. **Leerdoelen** (bullet points)
   3. **Inleiding** (context + belang)
   4. **Theoretische Basis** (definities, formules)
   5. **Kernconcepten & Algoritmen** (stapsgewijze uitleg)
   6. **Praktische Voorbeelden & Code Uitleg**:
        – Minstens **twee onderscheidende Python-codevoorbeelden** in ```python```-blokken.
        – Leg elke regel code uit in het Nederlands (minstens twee alinea’s toelichting per voorbeeld).
        – Toon hoe je de code draait (`python voorbeeld.py`) en de verwachte uitvoer.
   7. **Casestudy of Praktijksituatie**
   8. **Oefeningen (met Uitgebreide Oplossingen)**:
        – Minstens **vijf oefeningen** van oplopende moeilijkheid.
        – Elke oefening bevat **meerdere deelvragen** (a, b, c) en stap-voor-stap oplossingen (minstens twee alinea’s toelichting per deelvraag).
        – Zorg dat elke subkop in dit gedeelte minstens twee alinea’s tekst bevat.
   9. **Aanvullende Bronnen & Referenties** (5 per bron):
        – Wikipedia (5 bronzinnen/titels met toelichting)
        – Medium-artikelen (5 hyperlinks + toelichting)
        – GitHub-repository’s (5 hyperlinks + toelichting)
        – YouTube-video’s (5 hyperlinks + toelichting)
        – StackOverflow-vragen (5 hyperlinks + toelichting)
        – arXiv-papers (5 hyperlinks + toelichting)
        – Suggesties voor verdere verdieping.
  10. **Samenvatting & Volgende Stappen** (minstens twee alinea’s)

8. **Structureringsregels**:
   • Elke subsectie moet minstens **twee alinea’s** bevatten.
   • Gebruik duidelijke Nederlandse koppen (`##`, `###`).
   • Vermeld onder elke paragraaf **welke bron** (Wikipedia, Medium, GitHub, YouTube, StackOverflow, arXiv) de informatie heeft bijgedragen.
   • Alle tekst en voorbeeldcode in het Nederlands (code-commentaar in het Nederlands).
   • Maak een **nieuw bestand** voor elke aanvraag; voeg nooit toe aan bestaande bestanden.
   • Zorg dat elk stukje context (output van een eerdere fase) wordt meegenomen in latere fases, zodat de agent écht ‘kan doorpraten’.
"""

    print("\n💡 Lesplan aan het genereren. Dit kan even duren...\n")
    try:
        # Roep de meta-agent aan met onze expliciete meta_prompt
        meta_response = meta_agent.invoke(meta_prompt)
        # meta_response kan een dict met "output" zijn of een ruwe string, afhankelijk van LangChain-versie
        lesson_content = (
            meta_response
            if isinstance(meta_response, str)
            else meta_response.get("output", "")
        )

        if not lesson_content:
            print("❌ Fout: Lege respons van meta-agent.")
            return 1

        # Maak een uniek Markdown-bestandsnaam
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"files/lesplan_{timestamp}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(lesson_content)

        print(f"✅ Lesplan opgeslagen als '{filename}'.")
    except Exception as e:
        print(f"❌ Fout tijdens lesgeneratie: {e}")
        traceback.print_exc()
        return 1

    # ----------------------------
    # Houd servers kort draaiende om lopende verzoeken af te ronden
    # (Threads zijn daemonized en stoppen zodra dit script eindigt)
    # ----------------------------
    time.sleep(2)
    print("\n✋ Klaar. Afsluiten.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nProgramma onderbroken door gebruiker")
        sys.exit(0)
