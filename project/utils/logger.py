import logging

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,  # Detailed logging for debugging.
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
