from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from app.config.settings import settings


def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):

    if not query:
        return "No input provided"

    # Initialize LLM
    llm = ChatGroq(
        model=llm_id,
        api_key=settings.GROQ_API_KEY
    )

    # Prepare messages
    messages = [HumanMessage(content=msg) for msg in query]

    # Add system prompt
    if system_prompt:
        messages.insert(0, SystemMessage(content=system_prompt))

    # Tavily search (optional)
    if allow_search:
        try:
            search_tool = TavilySearch(max_results=2)
            search_results = search_tool.invoke(query[-1])

            messages.append(
                HumanMessage(content=f"Search Results:\n{search_results}")
            )

        except Exception as e:
            print("Search failed:", e)

    # Call LLM
    response = llm.invoke(messages)

    return response.content