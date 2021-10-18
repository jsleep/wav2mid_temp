from base.base_data_loader import BaseDataLoader
import numpy as np

# file-based dataloader
class Wav2MidDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)
        files = os.listdir(Path(self.config.preprocessor.data_dir) / 'preprocessed')

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
