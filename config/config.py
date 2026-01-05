import yaml
import os

def load_config():
    with open('./config/configs.yaml', 'r',encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            with open('./config/models.yaml', 'r',encoding='utf-8') as f:
                models = yaml.safe_load(f)
            with open('./config/data.yaml', 'r',encoding='utf-8') as f:
                data = yaml.safe_load(f)
            with open('./config/retrieval.yaml', 'r',encoding='utf-8') as f:
                retrieval = yaml.safe_load(f)
            with open('./config/prompts.yaml', 'r',encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            with open('./config/generate.yaml', 'r',encoding='utf-8') as f:
                generate = yaml.safe_load(f)
                
            config['models'] = models
            config['data'] = data
            config['retrieval'] = retrieval
            config['prompts'] = prompts
            config['generate'] = generate
        except yaml.YAMLError as e:
            print(e)
    # print(config)
    return config
