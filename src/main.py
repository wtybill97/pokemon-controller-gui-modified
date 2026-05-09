import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import multiprocessing
from runtime_bootstrap import bootstrap_runtime
import os

bootstrap_runtime()

import ui


def main():
    ui.run()
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
