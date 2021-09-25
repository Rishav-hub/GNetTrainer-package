import yaml
def process_config(file_name):
    """FUnction to load a config file of type yaml

    Args:
        file_name (string): Name of the configuration file

    Returns:
        yaml: Load the configuration file and return a dictionary

    """
    with open(file_name, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            
            print(" loading Configuration of your experiment ..")
            return config
        except ValueError:
            print("INVALID yaml file format.. Please provide a good yaml file")
            exit(-1)



