import os

from fcos.core.utils.checkpoint import DetectronCheckpointer as fcosDetectronCheckpointer

class DetectronCheckpointer(fcosDetectronCheckpointer):
    def __init__(self, cfg, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=None, logger=None):
        super().__init__(cfg, model, optimizer, scheduler, save_dir, save_to_disk, logger)
        self.file = None

    def load(self, f=None, force_file=False):
        
        if force_file and f is not None:
            self.file = f

        return super().load(f)

    def get_checkpoint_file(self):
        if self.file is not None:
            return self.file
        return super().get_checkpoint_file()



