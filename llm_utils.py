import os
import requests
import json
# Comment out or remove load_dotenv() if you don't want to load environment variables
# from dotenv import load_dotenv

# If you don't want to load environment variables from a .env file, just set the values manually
# AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # Commented out or removed
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDMwMjdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.ZTRVKqXwtG_r6gQviyv27gQQ49mCvL-KUkqIRJnpfzI"  # Manually set the token (for testing)

if AIPROXY_TOKEN:
    print("AIPROXY_TOKEN:", AIPROXY_TOKEN[:5])
else:
    raise EnvironmentError("AIPROXY_TOKEN is not set in environment variables.")

# Set DEBUG_MODE manually if needed, or let it be "false" by default
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower()  # Default to false if not set

def chat_completion(prompt: str):
    """
    Make a chat completion API call.
    If DEBUG_MODE is True, return a mock response to save tokens.
    """
    if DEBUG_MODE.lower() == "true":  # Check if DEBUG_MODE is 'true'
        print("DEBUG_MODE enabled: Returning mock response for chat_completion.")
        # If the prompt asks about a credit card, return a mock credit card number.
        if "credit card" in prompt.lower():
            return {"choices": [{"message": {"content": "4026399336539356"}}]}
        else:
            # Otherwise, return the default mock response.
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "function": "count_days",
                            "params": {
                                "input_file": "/data/dates.txt",
                                "output_file": "/data/dates-wednesdays.txt",
                                "weekday_name": "Wednesday"
                            }
                        })
                    }
                }]
            }

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("API response:", response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        raise


response = chat_completion("How do I count Wednesdays in a file?")
print(response)

def generate_embeddings(texts):
    """
    Given a list of texts, generate embeddings using the AI Proxy service.
    If DEBUG_MODE is True, return a mock embedding.
    """
    if DEBUG_MODE.lower() == "true":  # Check if DEBUG_MODE is 'true'
        print("DEBUG_MODE enabled: Returning mock embeddings.")
        # Return a dummy embedding (e.g., a vector of zeros) for each text.
        embedding_vector = [0.0] * 768
        if isinstance(texts, list):
            return [{"embedding": embedding_vector} for _ in texts]
        else:
            return [{"embedding": embedding_vector}]
    
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error during API request: {str(e)}")
