from keras.datasets import mnist
import numpy as np
import os
from base.base_preprocessor import BasePreprocessor


class SimpleMnistPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super(SimpleMnistPreprocessor, self).__init__(config)

    def preprocess(self):
        data = mnist.load_data()
        np.save(os.path.join(
            self.config.preprocessor.data_dir,
            self.config.preprocessor.data_fmt.format(self.config)),
            data)
