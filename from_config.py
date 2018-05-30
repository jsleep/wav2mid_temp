from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys

def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    args = get_args()
    config = process_config(args.config)

    if hasattr(config,"comet_api_key"):
        from comet_ml import Experiment

    # create the experiments dirs
    create_dirs([
     config.callbacks.tensorboard_log_dir,
     config.callbacks.checkpoint_dir,
     config.preprocessor.data_dir])

    print('Creating the preprocessor.')
    preprocessor = factory.create("preprocessors."+config.preprocessor.name)(config)
    preprocessor.preprocess()

    print('Create the data generator.')
    data_loader = factory.create("data_loaders."+config.data_loader.name)(config)

    print('Create the model.')
    model = factory.create("models."+config.model.name)(config)

    print('Create the trainer')
    trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()

if __name__ == '__main__':
    main()
