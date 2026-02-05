import re
from config import config
configs = config.load_config()

def get_prompt(prompt_name:str, prompt_param_dict: dict):
    prompt = configs['prompts'][prompt_name]
    output = prompt['template']
    
    for param in prompt['params']:
        val = str(prompt_param_dict.get(param, "")) 
        # initialize `output` outside the loop and update it inside the loop
        output = output.replace("{" + param + "}", val)
        
    return output
