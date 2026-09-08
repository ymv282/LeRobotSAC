import sys
import datetime
import os
import config

class TeeLogger:
    """Leitet stdout gleichzeitig an Konsole und Datei weiter."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # Zeilenweise flushen

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def log_output(name=config.logFileName, folder=config.logFolder):
    """Leitet stdout in eine logfile im Subfolder um (überschreibt existierende)."""
    os.makedirs(folder, exist_ok=True)
    log_filename = os.path.join(folder, name)
    sys.stdout = TeeLogger(log_filename)
    return log_filename
