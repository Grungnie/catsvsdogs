import os
import sys
import json
from keras.models import model_from_json


# This is used to hide the printing of the flow from directory
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def save_model(model, model_dir, model_name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # compile details
    compile_details = {
        'loss': model.loss,
        'optimizer': str(model.optimizer).split(' ')[0].split('.')[-1].lower(),
        'metrics': model.metrics
    }
    with open("{}/{}-compile.json".format(model_dir, model_name), "w") as json_file:
        json.dump(compile_details, json_file)

    # serialize model to JSON
    model_json = model.to_json()
    with open("{}/{}.json".format(model_dir, model_name), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("{}/{}.h5".format(model_dir, model_name), overwrite=True)


def load_model(model_dir, epoch=None, best_acc=False, model_to_load=None):
    # Find the right model name
    best_score = 0
    model_to_load = model_to_load
    if model_to_load is None:
        for filename in os.listdir(model_dir):
            if filename.endswith(".h5"):

                model_name, file_epoch, train_acc, val_acc = filename[:-3].split('-')

                # if a specific epoch was requested return if found
                if epoch is not None and int(file_epoch) == epoch:
                    model_to_load = filename[:-3]
                    break
                elif not best_acc:
                    if int(file_epoch) > best_score:
                        model_to_load = filename[:-3]
                        best_score = int(file_epoch)
                elif best_acc:
                    if float(val_acc) > best_score:
                        model_to_load = filename[:-3]
                        best_score = float(val_acc)

                continue
            else:
                continue

    print('loading model: {}'.format(model_to_load))

    # load json and create model
    json_file = open('{}/{}.json'.format(model_dir, model_to_load), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("{}/{}.h5".format(model_dir, model_to_load))

    with open('{}/{}-compile.json'.format(model_dir, model_to_load)) as json_file:
        data = json.load(json_file)

    loaded_model.compile(loss=data['loss'],
                  optimizer=data['optimizer'],
                  metrics=data['metrics'])

    return loaded_model
