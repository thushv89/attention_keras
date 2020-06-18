import logging
import logging.config
import yaml
import os


def get_logger(mod_name, log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    config_filepath = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'logger_config.yml')
    if os.path.exists(config_filepath):
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f.read())
            config["handlers"]["file"]["filename"] = os.path.join(log_dir, mod_name+'.log')
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(mod_name)
    logger.info("Started log {}".format(os.path.join(log_dir, mod_name)))
    return logger