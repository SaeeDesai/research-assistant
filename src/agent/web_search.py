"""
web_search.py

A web search tool for the agent. Used when a question needs
current information that isn't in the static paper corpus —
e.g. "what's the newest Llama model?"

Wraps the Tavily API and returns clean, formatted results
ready to be handed to the LLM as context.

"""

import os
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSearchTool:
    """
    Searches the web and returns formatted results.

    Why a class? 
    The Tavily client is created once and reused. Same pattern as the Embedder and RAGChain - 
    initialize the expensive thing once, reuse it for every search.
    """

    def __init__(self):
        api_key = os.getenv('TAVILY_API_KEY')
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not found in environment. "
                "Did you add it to your .env file?"
            )
        self.client = TavilyClient(api_key=api_key)
        logger.info("WebSearchTool initialized")

    def search(self, query: str, max_results: int = 5) -> str:
        """
        Search the web and return results as a formatted string.

        Why return a string, not raw results?
        The output of this tool becomes context for the LLM,
        exactly like retrieved chunks do in RAG. We format it
        into clean text the LLM can read and cite.

        Args:
            query:       The search query
            max_results: How many results to return

        Returns:
            Formatted string of search results
        """
        logger.info(f"Web searching: '{query}'")

        response = self.client.search(query, max_results=max_results)
        results = response.get('results', [])

        if not results:
            return "No web results found for this query."
        
        # Format each result into clean text for the LLM
        formatted_parts = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            content = result.get('content', '')
            url = result.get('url', '')
            formatted_parts.append(
                f"[Web Result {i} - {title}]\n{content}\nSource: {url}"
            )
        
        formatted = "\n\n".join(formatted_parts)
        logger.info(f"Found {len(results)} web results")
        return formatted
    

# --Test block--
if __name__ == '__main__':
    tool = WebSearchTool()

    print("=" * 60)
    print("WEB SEARCH TOOL TEST")
    print("=" * 60)

    query = "latest Llama model release 2026"
    print(f"\nQuery: {query}\n")

    result_text = tool.search(query, max_results=3)
    print(result_text[:1000])