import cmd
import argparse

class DistSwarmShell(cmd.Cmd):
    prompt = "[Swarm]$ "

    def preloop(self):
        logo = "█▀ █░█░█ ▄▀█ █▀█ █▀▄▀█ \
                ▄█ ▀▄▀▄▀ █▀█ █▀▄ █░▀░█ v. 0.01"
        hr = '-------------------------------------'

    ### config commands
    def do_create_config(self):
        """
        create config file for swarm
        """
        raise NotImplementedError()

    def do_edit_config(self):
        """
        edit config file
        """
        raise NotImplementedError()

    def do_remove_config(self):
        """
        remove config file
        """
        raise NotImplementedError()

    ### swarm commands    
    def do_create_swarm(self):
        """
        create swarm from config file
        """
        parser = argparse.ArgumentParser(prog="config")
        parser.add_argument('-c')

    def do_config_swarm(self):
        """
        configure a specificed swarm
        """
        raise NotImplementedError()

    def do_list_swarm(self):
        """
        list all the swarms
        """
        raise NotImplementedError()

    def do_run_swarm(self):
        """
        run swarm
        """
        raise NotImplementedError()

    def do_show_swarm(self):
        """
        show specifics of swarm
        """
        raise NotImplementedError()

    def do_remove_swarm(self):
        """
        remove specified swarm
        """
        raise NotImplementedError()

    ### worker commands
    def do_show_worker(self):
        """
        show descriptions of a worker
        """
        raise NotImplementedError()