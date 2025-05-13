import random
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from huggingface_hub import list_models

# 1. Web Search Tool - Sử dụng DuckDuckGo
search_tool = DuckDuckGoSearchRun()

# 2. Weather Information Tool
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the weather tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)

# 3. Hub Stats Tool
def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"
# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)

# # Example usage
# print(hub_stats_tool("facebook")) # Example: Get the most downloaded model by Facebook

# 4. News Tool - Mới
def get_latest_news(topic: str) -> str:
    """Fetches the latest news about a specific topic."""
    # Giả lập kết quả tin tức
    news_topics = {
        "ai": "Breaking: New GPT-5 model announced with unprecedented reasoning capabilities.",
        "technology": "Tech giants announce new AR glasses coming in 2024.",
        "politics": "World leaders gather for climate summit to discuss new emissions targets.",
        "sports": "National team wins championship in dramatic final match.",
        "entertainment": "Award-winning director announces new film project with A-list cast."
    }
    
    if topic.lower() in news_topics:
        return f"Latest news about {topic}: {news_topics[topic.lower()]}"
    else:
        return f"No recent news found about {topic}."

# Initialize the news tool
news_tool = Tool(
    name="get_latest_news",
    func=get_latest_news,
    description="Fetches the latest news about a specific topic."
)
