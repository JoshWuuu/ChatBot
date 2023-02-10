import json

def data_loading(data_root):
    """
    sanity check for input string
    
    Input:
    - s: str, folder path
    Returns:
    - s: str, folder path
    """
    data_file = open(data_root + '/intents.json').read()
    data = json.loads(data_file)
    return data