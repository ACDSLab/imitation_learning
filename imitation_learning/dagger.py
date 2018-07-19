"""Implement the Dagger algorithm."""
from collections import defaultdict
from datetime import datetime
import logging
from time import time
import numpy as np


def rollout(logger, system, policy, T, N, expert=None, mixing=0.0,
            mix_within_rollout=False):
    """Perform rollouts where each may be mixed between expert and learner."""
    data = {
        'observations': defaultdict(list),  # dict of np.arrays
        'targets': [],  # list of np.arrays
        'actions': [],  # list of np.arrays
        'rewards': []  # list of numbers
    }

    # TODO: Fix since initial observation is causing mismatch
    mix = expert is not None and not np.isclose(mixing, 0.0)
    for r in range(N):
        logger.info('= Waiting for Rollout {} ='.format(r))
        use_imlearn = False if mix and np.random.random < mixing else True
        autonomous = use_imlearn or expert.autonomous
        obs = system.wait_for_rollout(autonomous)
        if not mix_within_rollout or not mix:
            roller = 'imlearn' if use_imlearn else 'expert'
            logger.info('= Start Rollout {} using {} ='.format(r, roller))
        else:
            logger.info('= Start Mixed Rollout {} ='.format(r))
        for t in range(T):
            if mix_within_rollout:
                use_imlearn = \
                    False if mix and np.random.random < mixing else True
            action = policy(obs)
            target = action if use_imlearn else expert.action(obs)
            obs, reward, done = system.step(target)
            if not done:
                for field, value in obs:
                    data['observations'][field].append(value)
                data['targets'].append(target)
                data['actions'].append(action)
                data['rewards'].append(reward)
        if not mix_within_rollout or not mix:
            logger.info('= Completed Rollout {} using {} ='.format(r, roller))
        else:
            logger.info('= Completed Mixed Rollout {} ='.format(r))
    return data


def test_policy_sys(logger, system, policy, T, N):
    """Test the policy on the system using N rollouts of T timesteps each."""
    logger.info('=== Test with {} Rollouts of {} Timesteps ==='.format(N, T))
    data = rollout(logger, system, policy, T, N)
    r = [np.sum(data['rewards'][roll]) for roll in range(N)]
    mu = np.mean(r)
    sigma = np.var(r)
    logger.info('=== Reward Mean {} and Variance {} ==='.format(mu, sigma))
    return data


def save_data(logger, path, data, prefix=None, suffix=None):
    """Add the prefix and suffix to the filename and save the data."""
    if path is not None:
        logger.info('=== Saving Data ===')
        if not path.endswith('/'):
            path += '/'
        stamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')
        name = 'data_' + stamp
        if prefix is not None:
            name += '_' + prefix
        if suffix is not None:
            name += '_' + suffix
        np.savez(path + name, **data)
        logger.info('=== Saved Data {} to Path {} ==='.format(name, path))


def save_model(logger, path, learner):
    """Save the model."""
    if path is not None:
        logger.info('=== Saving Model ===')
        if not path.endswith('/'):
            path = path + '/'
        stamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')
        name = 'model_' + stamp
        learner.save(path, name)
        logger.info('=== Saved Model {} to Path {} ==='.format(name, path))


def dagger(system, expert, learner, timesteps, rollouts, iterations,
           mixing_rate=0.0, loaded_data=None, pretrain=False, test_expert=False,
           test_initial=False, test_policy=False, test_final=False,
           log_name='imlearn', **options):
    """Run the Dagger algorithm.

    system: Handle to the system (environment)
    expert: Handle to the expert
    learner: Handle to the learner
    timesteps: Maximum number of timesteps per rollout
    rollouts: Number of rollouts to perform per iteration
    iterations: Number of iterations -- set to 0 to skip Dagger iterations
    mixing_rate: Stochastic mixing rate in [0.0, 1.0] where higher numbers mean
                 more expert participation (default: 0.0)
    loaded_data: File name and path of data to load (default: None)
    pretrain: True if pretraining should be performed (default: False)
    test_expert: True if the expert policy should be tested before starting
                 (default: False)
    test_initial: True if the learned policy should be tested before starting
                  (default: False)
    test_policy: True if the learned policy should be tested after each
                 iteration (default: False)
    test_final: True if the learned policy should be tested after completing all
                iterations (default: False)
    log_name: Base name of the logger (default: 'imlearn')

    Reasonable defaults are provided for 'options' keyword arguments.
    field: description (default)

    data_save_path: string path to directory in which to save data (None)
    model_save_path: string path to directory in which to save model (None)
    T_test: integer timesteps for expert and policy tests (timesteps)
    n_roll_test: integer rollouts for expert and policy tests (rollouts)
    T_pretrain: integer timesteps for pretraining (timesteps)
    roll_pretrain: integer rollouts for pretraining (rollouts)
    options_pretrain: kwargs to pass to learner for pretraining (None)
    options_train: kwargs to pass to learner for training (None)
    mix_within_rollout: True if timesteps within a rollout should be mixed
                        according to mixing rate
                        False if entire rollouts should be performed by the
                        expert or the learner according to mixing rate (True)
    timesteps_initial: integer timesteps for initial test of the policy (timesteps)
    rollouts_initial: integer rollouts for initial test of the policy (rollouts)
    timesteps_final: integer timesteps for final test of the policy (timesteps)
    rollouts_final: integer rollouts for final test of the policy (rollouts)
    """
    logger = logging.getLogger(log_name + '.dagger')
    logger.info('=========== Start Dagger ===========')

    T = timesteps
    N = rollouts

    data_path = options.get('data_save_path', None)
    model_path = options.get('model_save_path', None)

    if test_expert or test_policy:
        T_test = options.get('T_test', T)
        N_test = options.get('n_roll_test', N)

    # Test the expert if required
    if test_expert:
        logger.info('====== Testing Expert ======')
        expert_test_data = test_policy_sys(logger, system, expert, T_test, N_test)
        save_data(logger, data_path, expert_test_data, prefix='expert_test')

    # Test the learned policy if required
    if test_initial:
        logger.info('====== Testing Initial Policy ======')
        T_initial = options.get('timesteps_initial', T)
        N_initial = options.get('rollouts_initial', N)
        policy = learner.get_policy()
        init_data = test_policy_sys(logger, system, policy, T_initial, N_initial)
        save_data(logger, data_path, init_data, prefix='initial_test')

    data = {
        'observations': defaultdict(list),
        'targets:': [],
        'actions': [],
        'rewards': []
    }

    # Load data if any given
    if loaded_data is not None:
        ldata = np.load(loaded_data)
        data['observations'] = ldata['observations'].item()
        data['targets'] = list(ldata['targets'])
        data['actions'] = list(ldata['actions'])
        data['rewards'] = list(ldata['rewards'])

    # Perform pretraining if required
    if pretrain:
        logger.info('====== Pretraining ======')
        T_pre = options.get('timesteps_pretrain', T)
        N_pre = options.get('rollouts_pretrain', N)
        tparams = options.get('options_pretrain', None)
        pretrain_data = rollout(logger, system, expert, T_pre, N_pre)
        for field, values in pretrain_data['observations']:
            # values is a list of np.arrays
            data['observations'][field].extend(values)
        data['targets'].extend(pretrain_data['targets'])
        data['actions'].extend(pretrain_data['actions'])
        data['rewards'].extend(pretrain_data['rewards'])
        save_data(logger, data_path, data, prefix='pretrain')
        if tparams is not None:
            learner.fit(data['observations'], data['targets'], **tparams)
        else:
            learner.fit(data['observations'], data['targets'])
        save_model(logger, model_path, learner)

    # Dagger iterations
    mixing = 1.0
    tparams = options.get('options_train', None)
    policy = learner.get_policy()
    # TODO: Question: Isn't it get_policy(obsearvation)?
    mix_within_rollout = options.get('mix_within_rollout', True)
    for iter in range(iterations):
        mixing *= mixing_rate
        data_new = rollout(logger, system, policy, expert, T, N, mixing,
                           mix_within_rollout)
        logger.info('====== Training Iteration {} ======'.format(iter))
        for field, values in data_new['observations']:
            # values is a list of np.arrays
            data['observations'][field].extend(values)
        data['targets'].extend(data_new['targets'])
        data['actions'].extend(data_new['actions'])
        data['rewards'].extend(data_new['rewards'])
        save_data(logger, data_path, data)
        if tparams is not None:
            learner.fit(data['observations'], data['targets'], **tparams)
        else:
            learner.fit(data['observations'], data['targets'])
        save_model(logger, model_path, learner)
        policy = learner.get_policy()
        if test_policy:
            logger.info('====== Testing Learned Policy {} ======'.format(iter))
            test_data = test_policy_sys(logger, system, policy, T_test, N_test)
            save_data(logger, data_path, test_data, prefix='policy_test',
                      suffix='iter' + str(iter))

    # Do a final test if required
    if test_final:
        logger.info('====== Testing Final Learned Policy ======')
        T_final = options.get('timesteps_final', T)
        N_final = options.get('rollouts_final', N)
        policy = learner.get_policy()
        final_test_data = test_policy_sys(logger, system, policy, T_final, N_final)
        save_data(logger, data_path, final_test_data, prefix='final_test')

    # Return the latest policy and the combined training data
    logger.info('============ End Dagger ============')
    return policy, data
