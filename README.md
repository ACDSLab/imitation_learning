# imlearn: A Python Framework for Imitation Learning

imlearn is a Python library for imitation learning.  At the moment, the only method implemented is the one described in:

**Agile Off-Road Autonomous Driving Using End-to-End Deep Imitation Learning.** Y. Pan, C. Cheng, K. Saigol, K. Lee,  X. Yan, E. Theodorou and B. Boots.  Robotics: Science and Systems (2018).

## About

imlearn is a generic framework for imitation learning in Python.  The framework itself does not depend on a particular choice of neural network library, for example.  See the `learners.py` file and the `KerasLearner` class to learn more about how to integrate your choice of neural network framework into imlearn.

## Using

Using imlearn requires implementing interfaces to your expert, environment, and learner.

### Experts
Experts provide an interface to an autonomous or human controller for the system of interest.  Your implementation should inherit from `Expert` and implement the appropriate interfaces.

### Environment
Environments provide an interface to the system to be controlled.  Your implementation should inherit from `Environment` and implement the appropriate interfaces.

### Learner
Learners are an interface to the learning model.  We provide a `KerasLearner` out of the box and you can easily extend the software to your choice of neural network framework using the `Learner` interface class.  For `KerasLearner`, you should just pass in an instance of your Keras neural network to a `KerasLearner` instance.

### Running an Experiment
Running an experiment using imlearn involves instantiating your expert, environment, and learner, and then passing them to the algorithm.  For a complete example, usable example using the [AutoRally](https://autorally.github.io/) platform's simulator, see our repository for [Imitation Learning with AutoRally](https://github.com/ACDSlab/imitation_learning_autorally).  Note that the [AutoRally](https://autorally.github.io/) software needs to be correctly installed, so please visit the AutoRally website.

## Software Authors
Keuntaek Lee, Kamil Saigol, Gabriel Nakajima An, Yunpeng Pan

## Citation
If you use imlearn in your work, please cite:

**Agile Off-Road Autonomous Driving Using End-to-End Deep Imitation Learning.** Y. Pan, C. Cheng, K. Saigol, K. Lee,  X. Yan, E. Theodorou and B. Boots.  Robotics: Science and Systems (2018).
