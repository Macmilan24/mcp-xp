import requests

def fetch_galaxy_tool_source_code(tool_id,galaxy_url):
    """
    Fetches the XML source code of a Galaxy tool using its tool ID.

    Args:
        tool_id: The ID of the Galaxy tool to fetch the source code for.
        galaxy_url: The base URL of the Galaxy instance.
        api_key: The API key for authenticating with the Galaxy instance.

    Returns:
        A string containing the XML source code of the tool.
        Returns None if there was an error making the request.
    """
    url = f"{galaxy_url}api/tools/{tool_id}/raw_tool_source"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Galaxy tool source code: {e}")
        return None