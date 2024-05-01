import os

from fcos.core.utils.checkpoint import DetectronCheckpointer as fcosDetectronCheckpointer

class DetectronCheckpointer(fcosDetectronCheckpointer):
    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return os.path.join(self.save_dir, last_saved)



