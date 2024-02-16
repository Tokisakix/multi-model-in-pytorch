import json

CONFIG_PATH = "config.json"

def load_config():
    CONFIG = json.load(open(CONFIG_PATH, "r"))
    return CONFIG




# ---TEST---

if __name__ == "__main__":
    CONFIG = load_config()
    print(CONFIG)