import argparse

from overmind import Overmind

def main():
    parser = argparse.ArgumentParser(description='set params for overmind')
    parser.add_argument('--config', dest='config',
                        type=str, default=None, help='configuration file')
    parser.add_argument('--skip-init', dest='skip_init', action='store_true',
                        default=False, help='skip initializing device state table')
    parser.add_argument('--rt-mode', dest='rt_mode', action='store_true',
                        default=False, help='real time mode')
    
    parsed = parser.parse_args()

    ovm = Overmind()
    if parsed.config is not None:
        ovm.create_swarm(parsed.config, parsed.skip_init)
    else:
        raise ValueError("please specify config file")
    
    ovm.build_dep_graph(rt_mode=parsed.rt_mode)
    ovm.run_swarm(rt_mode=parsed.rt_mode)

if __name__ == '__main__':
    main()
