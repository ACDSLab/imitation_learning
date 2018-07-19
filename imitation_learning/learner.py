import logging
from abc import ABCMeta, abstractmethod


class Learner(object):
    """Base abstract class for a Learner."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, model, field_names, field_dims, log_name='imlearn'):
        """Initialize the Learner.

        Parameters
        ----------
        model
            Model structure of learner.
        field_names : list
            List of fields (from observations) to use for fit/predict
        field_dims : list
            List of tuples with dimensions of each field in field_names
        log_name : str
            Base name of logger (default: 'imlearn')
        """
        self.logger = logging.getLogger(log_name + '.learner')
        self.fields = field_names
        self.dims = field_dims

    @classmethod
    @abstractmethod
    def load(cls, path, name, log_name='imlearn', **options):
        """Restore a Learner from file.

        Parameters
        ----------
        path : str
            Path to directory containing the saved file
        name : str
            Mame of the Learner to load WITHOUT any file extensions
        log_name : str
            Base name of logger (default: 'imlearn')
        options
            Optional keywords arguments
        """
        return

    @abstractmethod
    def save(self, path, name):
        """Save the Learner to file.

        Parameters
        ----------
        path : str
            String path to directory to which to save the Learner
        name : str
            Name of the Learner to save WITHOUT any file extensions
        """
        return

    @abstractmethod
    def fit(self, observations, targets, **options):
        """Perform model fit.

        Parameters
        ----------
        observations
            Dictionary-like object, each element of which is a numpy array
            containing samples of the input.
        targets : numpy.ndarray
            Expert results, one array per sample of the input.
        options
            Optional keyword arguments
        """
        return

    @abstractmethod
    def get_policy(self):
        """Return the Learner's policy function."""
        return
