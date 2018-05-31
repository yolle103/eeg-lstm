import run_model
import os
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.utils import to_categorical
def main():
    x, y = run_model.load_data()
    x, y = run_model.dataset_preprocess(x, y)
    y = to_categorical(y)
    model_path = './model.0019-0.61.hdf5'
    model = load_model(model_path)

    checkpoint = ModelCheckpoint(
            './model.{epoch:04d}-{val_loss:.2f}.hdf5',monitor='loss',
            verbose=1, save_best_only=True, mode='min')

    logger = CSVLogger(os.path.join(".", "training-20.log"))
    # fit the model

    callbacks_list = [checkpoint]
    model.fit(
            x, y, batch_size=32, 
            epochs=50, verbose=1, 
            validation_split=0.1, shuffle=True, 
            initial_epoch=20, callbacks=[checkpoint, logger])

if __name__ == '__main__':
    main()
