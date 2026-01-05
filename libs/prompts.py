import re
from config import config
configs = config.load_config()

def prompt(prompt_name:str, prompt_param_dict: dict):
    prompt = configs['prompts'][prompt_name]
    
    for param in prompt['params']:
        output = prompt['template'].replace("{" + param + "}", prompt_param_dict[param])
    return output
