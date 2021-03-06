import os
import time

import numpy as np

import gym
import tensorflow as tf

from PIL import Image

DQN_configuration = {
    'Q_network': [
        ('input', {}),
        ('conv1', {'W_size': 8, 'stride': 4, 'in': 4, 'out': 16}),
        ('conv2', {'W_size': 4, 'stride': 2, 'in': 16, 'out': 32}),
        ('fc1', {'num_relus': 256}),
        ('output', {}),
    ],
    'Q_network_input_size': 84,
    'var_init_mean': 0.0,
    'var_init_stddev': 0.01,
    'minibatch_size': 32,
    'replay_memory_size': 10 ** 6,
    'agent_history_length': 4,
    'discount_factor': 0.95,
    'action_repeat': 4,
    'update_frequency': 4,
    'learning_rate': 0.00025,
    'rms_prop_decay': 0.95,
    'gradient_momentum': 0.0,
    'min_squared_gradient': 0.01,
    'initial_exploration': 1,
    'final_exploration': 0.1,
    'final_exploration_frame': 10 ** 6,
    'replay_start_size': 5 * (10 ** 4),
    'no-op_max': 30,
    'validation_size': 500,
    'evaluation_exploration': 0.05,
}


class ReplayMemory:
    def __init__(self, config, np_random):
        self.size = config['replay_memory_size']
        self.minibatch_size = m = config['minibatch_size']
        self.agent_history_length = h = config['agent_history_length']
        self.Q_network_input_size = s = config['Q_network_input_size']

        self.index = 0
        self.full = False
        self.states = np.zeros((self.size, s, s), dtype=np.uint8)
        self.actions = np.zeros((self.size), dtype=np.uint8)
        self.rewards = np.zeros((self.size), dtype=np.float32)
        self.terminals = np.zeros((self.size), dtype=np.uint8)
        self._np_random = np_random
        self._minibatch = {
            'states': np.zeros((m, s, s, h), dtype=np.float32),
            'actions': np.zeros(m, dtype=np.uint8),
            'rewards': np.zeros(m, dtype=np.float32),
            'next_states': np.zeros((m, s, s, h), dtype=np.float32),
            'terminals': np.zeros(m, dtype=np.uint8),
        }

    def store(self, state, action, reward, terminal):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = terminal
        self.index += 1
        if self.index == self.size:
            self.full = True
            self.index = 0

    def get_size(self):
        if self.full:
            current_size = self.size
        else:
            current_size = self.index
        return current_size

    def get_samples(self, sample_size):
        current_size = self.get_size()
        if current_size < sample_size:
            raise RuntimeError(
                'Number of states {} less than minibatch size {}'
                .format(current_size, sample_size)
            )
        samples = self._np_random.randint(
            low=(self.agent_history_length - 1),
            # XXX The following cannot sample self.states[self.size]
            # even when the next state is available as self.states[0].
            high=(current_size - 1),
            size=sample_size,
        )
        return samples

    def sample_minibatch(self):
        samples = self.get_samples(self.minibatch_size)

        self.fill_states(samples, self._minibatch['states'])
        self.fill_states(samples, self._minibatch['next_states'], offset=1)
        self._minibatch['actions'] = self.actions[samples]
        self._minibatch['rewards'] = self.rewards[samples]
        self._minibatch['terminals'] = self.terminals[samples]

        return self._minibatch

    def fill_states(self, samples, buf_states, offset=0):
        h = self.agent_history_length
        for i in range(len(samples)):
            j = samples[i]
            buf_states[i] = np.transpose(
                self.states[(j - h + 1 + offset):(j + 1 + offset),:,:],
                (1, 2, 0),
            )


class AtariDQNAgent:
    def __init__(
        self,
        game_name='Breakout-v0',
        random_seed=None,
    ):
        self.game_name = game_name

        log_dir = './log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        checkpoints_dir = './checkpoints'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        self._config = DQN_configuration
        self._env = gym.make(game_name)

        self._random_seed = random_seed
        # Seeding Numpy.
        self._np_random = np.random.RandomState(self._random_seed)
        if self._random_seed is not None:
            # Seeding Gym Atari environment.
            self._env.env._seed(self._random_seed)
            # Seeding Tensorflow.
            tf.set_random_seed(self._random_seed)

        self._replay_memory = ReplayMemory(
            config=self._config,
            np_random=self._np_random,
        )

        self._validation_states = None
        self._Q_network_layers = None
        self._train_op_input = {}
        self._loss_op = None
        self._validation_op_input = {}
        self._average_Q_op = None
        self._ave_reward = None
        self._tf_summary = {}
        self._tf_session = None
       
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()
            self._tf_session = tf.Session()
            self._tf_session.run(tf.global_variables_initializer())

    def _build_graph(self):
        self._Q_network_layers = []
        prev_layer = None
        for layer_name, layer_conf in self._config['Q_network']:
            with tf.variable_scope(layer_name):
                if 'input' in layer_name:
                    s = self._config['Q_network_input_size']
                    c = self._config['agent_history_length']

                    # The size of the minibatch can be either 32 when training
                    # or 1 when evaluating Q for a given state,
                    # therefore set to None.
                    new_layer = tf.placeholder(
                        tf.float32,
                        shape=(None, s, s, c),
                        name='input_layer',
                    )
                elif 'conv' in layer_name:
                    W_s = layer_conf['W_size']
                    s = layer_conf['stride']
                    i = layer_conf['in']
                    o = layer_conf['out']
                    W, b = self._get_filter_and_bias(
                        W_shape=(W_s, W_s, i, o),
                        b_shape=o,
                    )
                    a_tensor = tf.nn.conv2d(
                        prev_layer,
                        W,
                        strides=(1, s, s, 1),
                        padding='VALID',
                    )
                    a_tensor = tf.nn.bias_add(a_tensor, b)
                    new_layer = tf.nn.relu(a_tensor)
                elif 'fc' in layer_name:
                    n = layer_conf['num_relus']
                    _, h, w, c = prev_layer.shape.as_list()
                    in_dim = h * w * c
                    conv_out_flattened = tf.reshape(prev_layer, (-1, in_dim))
                    W, b = self._get_filter_and_bias(
                        W_shape=(in_dim, n),
                        b_shape=n,
                    )
                    new_layer = tf.nn.relu(
                        tf.nn.bias_add(
                            tf.matmul(conv_out_flattened, W), b
                        )
                    )
                elif 'output' in layer_name:
                    in_dim = prev_layer.shape.as_list()[-1]
                    num_actions = self.get_num_of_actions()
                    W, b = self._get_filter_and_bias(
                        W_shape=(in_dim, num_actions),
                        b_shape=(num_actions),
                    )
                    new_layer = tf.nn.bias_add(
                        tf.matmul(prev_layer, W), b
                    )

                self._Q_network_layers.append(new_layer)
                prev_layer = new_layer

        with tf.variable_scope('train_op'):

            ys = tf.placeholder(
                dtype=tf.float32,
                shape=(self._config['minibatch_size']),
                name='ys',
            )
            actions = tf.placeholder(
                dtype=tf.uint8,
                shape=(self._config['minibatch_size']),
                name='actions',
            )
            Q_input = self._Q_network_layers[0]

            self._train_op_input = {
                'states': Q_input,
                'actions': actions,
                'ys': ys,
            }

            Q_output = self._Q_network_layers[-1]
            one_hot_actions = tf.one_hot(
                actions,
                self.get_num_of_actions(),
            )
            Qs_of_action = tf.reduce_sum(
                tf.multiply(Q_output, one_hot_actions),
                axis=1,
            )
            self._loss_op = tf.reduce_mean(tf.square(ys - Qs_of_action))
            self._train_op = tf.train.RMSPropOptimizer(
                learning_rate=self._config['learning_rate'],
                decay=self._config['rms_prop_decay'],
                momentum=self._config['gradient_momentum'],
                epsilon=self._config['min_squared_gradient'],
                centered=True,
            ).minimize(self._loss_op)
            self._tf_summary['loss'] = tf.summary.scalar('loss', self._loss_op)

        with tf.variable_scope('validation_op'):
            self._validation_op_input = {
                'validation_states': self._Q_network_layers[0]
            }
            Q_output = self._Q_network_layers[-1]
            self._average_Q_op = tf.reduce_mean(
                tf.reduce_max(Q_output, axis=1)
            )
            self._tf_summary['Q'] = tf.summary.scalar('Q', self._average_Q_op)

            self._ave_reward = tf.Variable(
                0.0,
                name='ave_reward',
            )

            self._tf_summary['reward_per_episode'] = tf.summary.scalar(
                'reward_per_episode',
                self._ave_reward
            )

    def _get_filter_and_bias(self, W_shape, b_shape):
        W = tf.get_variable(
            'W',
            shape=W_shape,
            initializer=self._get_variable_initializer()
        )
        b = tf.get_variable(
            'b',
            shape=b_shape,
            initializer=self._get_variable_initializer()
        )
        return (W, b)

    def _get_variable_initializer(self):
        # TODO: Check the original initialization
        return tf.truncated_normal_initializer(
            mean=self._config['var_init_mean'],
            stddev=self._config['var_init_stddev'],
            seed=self._random_seed,
        )

    def get_num_of_actions(self):
        return self._env.action_space.n

    def preprocess_observation(
        self,
        observation,
        new_width=84,
        new_height=110,
        threshold=1,
    ):
        img = Image.fromarray(observation)
        img = img.convert('L')
        new_size = (new_width, new_height)
        img = img.resize(new_size)
        obs = np.array(img, dtype=np.uint8)
        obs = obs[-new_width:, :]
        return obs

    def play(
        self,
        max_num_of_steps=(10**6),
        max_num_of_episodes=None,
        train=True,
        save_path=None,
        step=None,
        log_name=None,
    ):

        s = self._config['Q_network_input_size']
        c = self._config['agent_history_length']
        rss = self._config['replay_start_size']
        n_steps = self._config['final_exploration_frame']
        min_epsilon = self._config['final_exploration']
        stats = {
            'average_loss': [],
            'average_reward': [],
            'average_Q': [],
        }

        with self._graph.as_default():
            saver = tf.train.Saver()
            if save_path is not None:
                saver.restore(self._tf_session, save_path)
                step = self._get_step_from_checkpoint(save_path)

            if step is None:
                step = 0

            if train:
                if log_name is None:
                    log_name = ('{:02}{:02}{:02}{:02}{:02}'
                                .format(*time.localtime()[1:6]))
                summary_writer = tf.summary.FileWriter(
                    logdir='log/{}'.format(log_name),
                    graph=self._graph
                )
            else:
                play_images = []
                actions = []
                rewards = []

            done = True
            episode = 0
            rewards_per_episode = []
            losses = []
            phi = np.empty((1, s, s, c), dtype=np.float32)

            while (
                step < max_num_of_steps
                or (max_num_of_episodes is not None
                    and episode < max_num_of_episodes)
            ):
                if done:
                    initial_observation = self._env.reset()
                    state = self.preprocess_observation(initial_observation)
                    done = False
                    phi.fill(0)
                    episode += 1
                    rewards_per_episode.append(0)
                    losses.append(0)
                else:
                    state = next_state

                if train:
                    if step < rss:
                        epsilon = 1
                    elif (step < n_steps):
                        epsilon = 1 - (1 - min_epsilon) / n_steps * step
                    else:
                        epsilon = min_epsilon
                else:
                    epsilon = self._config['evaluation_exploration'] 

                phi[0,:,:,:3] = phi[0,:,:,1:]
                phi[0,:,:,3] = np.array(state, dtype=np.float32)
                action = self._get_action(epsilon, phi)
                observation, reward, done, _ = self._env.step(action)
                step += 1
                rewards_per_episode[-1] += reward
                next_state = self.preprocess_observation(observation)

                if train:
                    self._replay_memory.store(state, action, reward, int(done))

                    if (self._replay_memory.get_size() < rss):
                        continue
                    elif (self._validation_states is None):
                        vs = self._config['validation_size']
                        self._validation_states = np.zeros(
                            (vs, s, s, c), dtype=np.float32,
                        )
                        samples = self._replay_memory.get_samples(vs)
                        self._replay_memory.fill_states(
                            samples,
                            self._validation_states,
                        )

                    loss, loss_summary_str = self._optimize_Q()
                    summary_writer.add_summary(loss_summary_str, step)
                    losses[-1] += loss

                    if (step % 1000 == 0):
                        ave_loss = np.mean(loss)

                        ave_Q, Q_summary_str = self._tf_session.run(
                            [self._average_Q_op, self._tf_summary['Q']],
                            feed_dict={
                                self._Q_network_layers[0]:
                                    self._validation_states
                            },
                        )
                        summary_writer.add_summary(Q_summary_str, step)

                        ave_reward = np.mean(rewards_per_episode)

                        _, reward_summary_str = self._tf_session.run(
                            [tf.assign(self._ave_reward, ave_reward),
                             self._tf_summary['reward_per_episode']]
                        )
                        summary_writer.add_summary(reward_summary_str, step)

                        print(
                            'step: {}, ave. loss: {:g}, '
                            'ave. reward: {:g}, ave. Q: {:g}'
                            .format(
                                step,
                                ave_loss,
                                ave_reward,
                                ave_Q,
                            )
                        )
                        stats['average_loss'].append(ave_loss)
                        stats['average_reward'].append(ave_reward)
                        stats['average_Q'].append(ave_Q)

                    if (step % 10000 == 0 or step == max_num_of_steps):
                        save_path = saver.save(
                            self._tf_session,
                            'checkpoints/{}'.format(self.game_name),
                            global_step=step,
                        )
                        print('checkpoint saved at {}'.format(save_path))

                    if (step % 50000 == 0):
                        rewards_per_episode = rewards_per_episode[-1:]
                        losses = losses[-1:]
                else:
                    actions.append(action)
                    rewards.append(reward)
                    play_images.append(Image.fromarray(observation))


            if train:
                if (step >= max_num_of_steps):
                    print(
                        'Finished: reached the maximum number of steps {}.'
                        .format(max_num_of_steps)
                    )
                summary_writer.close()
                return stats
            else:
                play_images[0].save(
                    'play.gif',
                    save_all=True,
                    append_images=play_images[1:],
                )
                return (actions, rewards)

    def _get_action(self, epsilon, phi):
        if (self._np_random.rand() < epsilon):
            return self._env.action_space.sample()
        else:
            return self._get_action_from_Q(phi)

    def _get_action_from_Q(self, phi):
        Qs = self._get_Q_values(phi)
        # TODO: Tiebreaking
        return np.argmax(Qs)

    def _get_Q_values(self, states):
        assert(states.ndim == 4)
        assert(states.dtype == np.float32)
        return self._tf_session.run(
            self._Q_network_layers[-1],
            feed_dict={self._Q_network_layers[0]: states},
        )

    def _optimize_Q(self):
        gamma = self._config['discount_factor']
        minibatch = self._replay_memory.sample_minibatch()
        mask = (1 - minibatch['terminals'])
        Qs_next = self._get_Q_values(minibatch['next_states'])
        ys = minibatch['rewards'] + gamma * (
            np.amax(
                Qs_next * mask[:, np.newaxis],
                axis=1,
            )
        )

        _, loss, loss_summary_str = self._tf_session.run(
            [self._train_op, self._loss_op, self._tf_summary['loss']],
            feed_dict={
               self._train_op_input['states']: minibatch['states'],
               self._train_op_input['actions']: minibatch['actions'],
               self._train_op_input['ys']: ys,
            }
        )

        return (loss, loss_summary_str)

    def _get_step_from_checkpoint(self, save_path):
        return int(save_path.split('-')[-1])