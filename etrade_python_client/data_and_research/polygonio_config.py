import configparser
import os

config = configparser.ConfigParser()
# Get absolute path to config.ini in the root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config.ini')

if os.path.exists(config_path):
    config.read(config_path)
    API_KEY = config.get('MASSIVE', 'API_KEY', fallback="")
else:
    API_KEY = ""