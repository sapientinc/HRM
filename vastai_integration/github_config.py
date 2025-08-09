import xdg, os
from github import Github, Auth

DIRS = {
    'config': xdg.xdg_config_home(),
    'temp': xdg.xdg_cache_home()
}

GITHUB_KEY_FILE = os.path.join(DIRS['config'], "github", "github_api_key")

def load_github_token() -> str:
    """
    Loads the GitHub API key from the key file.

    Returns:
        str: The GitHub API key, or None if the file does not exist or is unreadable.
    """
    try:
        with open(GITHUB_KEY_FILE, 'r') as f:
            return f.read().strip()
    except (OSError, IOError):
        raise ValueError("Could not find github key")


def get_github_api() -> Github:
    auth = Auth.Token(load_github_token())
    return Github(auth=auth)
