import cmd
import argparse
import json
import os
from pathlib import Path, PurePath

# from dist_swarm.dist_swarm_controller import DistSwarmController

class DistSwarmShell(cmd.Cmd):
    prompt = "[Swarm]$ "

    def preloop(self):
        # self.controller = DistSwarmController()

        logo = "█▀ █░█░█ ▄▀█ █▀█ █▀▄▀█ \n▄█ ▀▄▀▄▀ █▀█ █▀▄ █░▀░█ v. 0.01"
        hr = '-------------------------------------'
        print(hr)
        print(logo)
        print(hr)

    ### commands
    def do_create(self, inp):
        """
        create (config file for swarm, swarm)
        """
        args = inp.split(' ')
        if len(args) < 1:
            print('specify what to create (create config, create swarm)')
        if args[0] == 'config':
            self.create_config(args[1:])
        elif args[0] == 'swarm':
            self.create_swarm(args[1:])
        else:
            print('invalid command: {}'.format('create ' + args[0]))
        
    def do_edit(self, inp):
        """
        edit config file
        """
        if len(inp) < 1:
            print('specify what to edit (edit config)')
        if inp[0] == 'config':
            self.controller.edit_config(inp[1:])

        raise NotImplementedError()

    def do_remove_config(self, inp):
        """
        remove (config file for swarm, swarm)
        """
        if len(inp) < 1:
            print('specify what to create (remove config, remove swarm)')
        if inp[0] == 'config':
            self.controller.remove_config(inp[1:])
        elif inp[1] == 'swarm':
            self.controller.remove_swarm(inp[1:])
        else:
            print('invalid command: {}'.format('create ' + inp[0]))

    ### swarm commands    
    def create_swarm(self, inp):
        """
        create swarm from config file
        """
        parser = argparse.ArgumentParser(prog="config")
        parser.add_argument('--config', dest='config_file', 
                            default='configs/dist_swarm/controller_example.json')
        parser.add_argument('--name', dest='name', 
                            default='default_swarm')
        parsed = parser.parse_args(inp)
        print('created swarm {} with config file {}'.format(parsed.name, parsed.config_file))


    def do_config_swarm(self, inp):
        """
        configure a specificed swarm
        """
        raise NotImplementedError()

    def do_list_swarm(self, inp):
        """
        list all the swarms
        """
        raise NotImplementedError()

    def do_run_swarm(self, inp):
        """
        run swarm
        """
        raise NotImplementedError()

    def do_set_default_swarm(self, inp):
        """
        set default swarm
        """
        raise NotImplementedError()

    def do_show_swarm(self, inp):
        """
        show specifics of swarm
        """
        raise NotImplementedError()

    def do_remove_swarm(self, inp):
        """
        remove specified swarm
        """
        raise NotImplementedError()

    ### worker commands
    def do_show_worker(self, inp):
        """
        show descriptions of a worker
        """
        raise NotImplementedError()

    def do_exit(self, inp):
        return True

    ### control logics
    

def main():
    DistSwarmShell().cmdloop()

if __name__ == "__main__":
    main()