"""Implements the deep neural network and associated convenience functions."""
import json
import numpy as np
from learner import Learner
from keras.models import model_from_json
from keras.optimizers import Adam


class KerasLearner(Learner):
    """Keras implementation of base class Learner."""

    def __init__(self, model, field_names, field_dims, log_name='imlearn',
                 **options):
        """Initialize the Learner.

        Parameters
        ----------
        model : Keras model.
        field_names : list of strings indicating which fields to consume
                      from observations returned from the environment.
        field_dims : list of tuples indicating dimensions corresponding
                     to each of the fields above.
        log_name : string name under which to output log information.
        options : dict of optional keywords arguments
            loss
                loss type string for Keras model compilation
                (default: mean_absolute_error)
            optimizer
                Keras optimizer (default: Adam)
            optimizer_options
                dictionary of field names and values for optimizer
                (default: 'lr': 1.03e-3)
            metrics
                list of metrics to use for Keras model compilation
                (default: ['mean_squared_error'])
        """
        super(KerasLearner, self).__init__(field_names, field_dims, log_name)

        # Get option values or defaults
        if options is not None:
            loss = options.get('loss', 'mean_absolute_error')
            optimizer = options.get('optimizer', Adam)
            opt_options = options.get('optimizer_options', {'lr': 1.0e-3})
            metrics = options.get('metrics', ['mean_squared_error'])
        else:
            loss = 'mean_absolute_error'
            optimizer = Adam
            opt_options = {'lr': 1.0e-3}
            metrics = ['mean_squared_error']

        # Print model summary and compile
        self.logger.info(model.summary())

        # Compile model
        model.compile(loss=loss, optimizer=optimizer(**opt_options),
                      metrics=metrics)

        self.model = model

    @classmethod
    def load(cls, path, name, log_name='imlearn', **options):
        """Restore a Learner from file.

        Parameters
        ----------
        path - string containing full path to directory from
               which to load.
        name - string name of the KerasLearner to load.
        log_name - string name under which to produce logging information.
        options
            optional keywords arguments
            loss
                loss type string for Keras model compilation
                (default: mean_absolute_error)
            optimizer
                Keras optimizer (default: Adam)
            optimizer_options
                dictionary of field names and values for optimizer
                (default: 'lr': 1.03e-3)
            metrics
                list of metrics to use for Keras model compilation
                (default: ['mean_squared_error'])
        """
        model_structure_path, model_options_path, model_weights_path = \
            cls._get_path_names(path, name)

        with open(model_structure_path, 'r') as model_structure_file:
            model_structure = model_structure_file.read()
        model = model_from_json(model_structure)
        model.load_weights(model_weights_path)

        with open(model_options_path, 'r') as model_options_file:
            opt = json.load(model_options_file)
        field_names = opt['field_names']
        field_dims = opt['field_dims']

        # Return the loaded Learner
        return cls(model, field_names, field_dims, log_name, **options)

    def save(self, path, name):
        model_structure_path, model_options_path, model_weights_path = \
            self._get_path_names(path, name)

        options = {
            'field_names': self.fields,
            'field_dims': self.dims
        }

        model_structure = self.model.to_json()
        with open(model_structure_path, 'w') as model_structure_file:
            model_structure_file.write(model_structure)
        self.model.save_weights(model_weights_path)

        with open(model_options_path, 'w') as model_options_file:
            json.dump(options, model_options_file)

    @staticmethod
    def _get_path_names(path, name):
        if not path.endswith('/'):
            path += '/'
        structure_path = path + name + '.json'
        options_path = path + name + '_options.json'
        weights_path = path + name + '_weights.h5'
        return structure_path, options_path, weights_path

    def fit(self, observations, targets, **options):
        """Restore a Learner from file.

        Parameters
        ----------
        observations - A dict of arrays of observations collected during
                       algorithm executation.
        targets - Control targets corresponding to the observations.
        options
            keyword arguments to pass to the Keras model for fitting.
        """
        # NOTE: KeyError thrown if field is not in observations
        try:
            inputs = [np.array(observations[field]) for field in self.fields]
        except KeyError as e:
            self.logger.error(e, exc_info=True)

        samples = len(inputs[0])
        self.logger.info('Fit with {} samples'.format(samples))
        history = self.model.fit(inputs, np.array(targets), **options)
        self.logger.info(history.history)

    def get_policy(self):
        """Return the policy function for the KerasLearner."""
        def policy(observation):
            """Return an array of predictions given the observation.

            Parameters
            ----------
            observation
                dictionary-like object, each element of which is a numpy array
                containing a single sample of the input.
            """
            # NOTE: KeyError thrown if field is not in observation
            try:
                inputs = [np.array(observation[field]) for field in self.fields]
                return self.model.predict(inputs)
            except KeyError as e:
                self.logger.error(e, exc_info=True)

        return policy
