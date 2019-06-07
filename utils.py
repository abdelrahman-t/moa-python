"""Utils."""
import logging
from typing import List

from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway, java_import

from dependencies import _MOA_JARS


def setup_logger(name: str, level) -> logging.Logger:
    """Create a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(f'{name}.log')
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        f'%(asctime)s [{name}] [%(levelname)-5.5s]  %(message)s')

    file_handler.setLevel(level)
    stream_handler.setLevel(level)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def setup_java_gateway(imports: List[str]):
    """
    Launch java gateway.

    :param imports: List of fully qualified class paths to import.
    """
    port = launch_gateway(classpath=_MOA_JARS, die_on_exit=True)

    params = GatewayParameters(
        port=port,
        auto_convert=True,
        auto_field=True,
        eager_load=True
    )

    gateway = JavaGateway(gateway_parameters=params)

    for import_ in imports:
        java_import(gateway.jvm, import_)

    return gateway
