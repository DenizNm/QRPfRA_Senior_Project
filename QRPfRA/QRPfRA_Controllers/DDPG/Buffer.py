import numpy as np
import tensorflow as tf

class Buffer:
    def __init__(self, critic_model, actor_model, critic_optimizer, actor_optimizer,
                 buffer_capacity=200000, batch_size=2, state_dim=23, action_dim=12, sample_count=100):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.sample_count = sample_count

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim))

        self.critic_model = critic_model
        self.actor_model = actor_model
        #self.target_critic = target_critic
        #self.target_actor = target_actor
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.gamma = 0.99

        self.num_actions = action_dim
        self.num_states = state_dim


    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1


    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with (tf.GradientTape() as tape):
            y = reward_batch
            critic_value = self.critic_model([state_batch, action_batch],
                                             training=True)  # Add tf.expand_dims
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value)) - 100

            tf.print("Critic loss: ", critic_loss)

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([next_state_batch, action_batch], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value) - 1e-6 * tf.math.reduce_mean(tf.math.square(actions))
            tf.print("Actor loss: ", actor_loss)
            
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    @tf.function
    def learn(self):
        # Initialize batch lists
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        # Get the current buffer size
        buffer_size = min(self.buffer_counter, self.buffer_capacity)

        for _ in range(self.batch_size):  # for each batch
            # Randomly select a starting index
            if buffer_size > self.sample_count:
                start_index = np.random.randint(0, buffer_size - self.sample_count)
                #np.random.randint(0, buffer_size - self.sample_count)
                #buffer_size - self.sample_count #
            else:
                start_index = 0

            end_index = start_index + self.sample_count

            # Slice self.sample_count sequential samples from each buffer
            state_batch.append(self.state_buffer[start_index:end_index])

            action_batch.append(self.action_buffer[end_index])

            reward_batch.append(self.reward_buffer[start_index:end_index])
            next_state_batch.append(self.next_state_buffer[start_index:end_index])

        # Convert lists to numpy arrays and reshape
        state_batch = np.array(state_batch, dtype=np.float32).reshape(-1, self.batch_size, self.sample_count, self.state_buffer.shape[
            1])  # Add dtype=np.float32


        action_batch = np.array(action_batch, dtype=np.float32).reshape(-1, self.batch_size, self.num_actions)  # Add dtype=np.float32
        print("Action batch: ", action_batch)
        print("Action batch shape: ", action_batch.shape)

        reward_batch = np.array(reward_batch, dtype=np.float32).reshape(-1, self.batch_size, self.sample_count, 1)
        next_state_batch = np.array(next_state_batch, dtype=np.float32).reshape(-1, self.batch_size, self.sample_count,
                                                                                self.next_state_buffer.shape[
                                                                                    1])  # Add dtype=np.float32

        # Convert to tensors
        state_batch = tf.convert_to_tensor(state_batch)
        action_batch = tf.convert_to_tensor(action_batch)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch)

        state_batch = tf.reshape(state_batch, (-1, 1, self.batch_size, self.sample_count, self.state_buffer.shape[1]))
        #action_batch = tf.reshape(action_batch, (-1, 1, self.batch_size, self.sample_count, self.action_buffer.shape[1]))
        reward_batch = tf.reshape(reward_batch, (-1, 1, self.batch_size, self.sample_count, 1))
        next_state_batch = tf.reshape(next_state_batch, (-1, 1, self.batch_size, self.sample_count, self.next_state_buffer.shape[1]))

        self.update(state_batch, action_batch, reward_batch, next_state_batch)