#!/usr/bin/env python3
from config.loader.ConfigLoader import ConfigLoader
import pprint

def main():
    cfg_path = 'config/files/settings.yml'
    loader = ConfigLoader(cfg_path)
    cfg = loader.get_config()
    print('Loaded config keys:', list(cfg.keys()))
    print('\nHosting config:')
    hosting = cfg.get('hosting')
    if hosting is None:
        print('No hosting config found')
        return
    pprint.pprint(hosting)

if __name__ == '__main__':
    main()

