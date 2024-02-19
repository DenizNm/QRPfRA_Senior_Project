import sys
sys.path.append("..")
from gymnasium.utils.env_checker import check_env
import QRPfRA
from OU_Noise import OUActionNoise as ou_noise
import numpy as np
from Buffer import Buffer
from actor_network import get_actor
from critic_network import get_critic
import tensorflow as tf
import matplotlib.pyplot as plt


env = QRPfRA.QRPfRA_v3()
check_env(env)
obs = env.reset()
env.render_mode = "human"

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


std_dev = 0.25
ou_noise = ou_noise(mean=np.zeros(num_actions), std_deviation=float(std_dev) * np.ones(num_actions), dt=0.02)


batch_size = 1
sample_count = 80

input_shape_for_states = (1, batch_size, sample_count, num_states)
input_shape_for_actions = (1, num_actions)#(1, batch_size, sample_count, num_actions)

actor_model = get_actor(input_shape_for_states, num_actions)
critic_model = get_critic(input_shape_for_states, input_shape_for_actions)

"""target_actor = get_actor(input_shape_for_states, num_actions)
target_critic = get_critic(input_shape_for_states, input_shape_for_actions)

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())"""


def policy(buffer_state, noise_object):
    # Reshape the state to match the expected input shape of the actor model
    targeted_sample_shape = (-1, 1, batch_size, sample_count, num_states)
    buffer_state = tf.reshape(buffer_state, targeted_sample_shape)

    sampled_actions = tf.squeeze(actor_model(buffer_state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    # We make sure action is within bounds

    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    legal_action = tf.math.multiply(legal_action, 100)
    legal_action = tf.math.round(legal_action + 1e-3)
    legal_action = tf.math.divide(legal_action, 100)

    return np.squeeze(legal_action)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.


# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.legacy.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.legacy.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
#gamma = 0.99


# Used to update target networks
# tau = 0.005

buffer = Buffer(critic_model, actor_model,
                critic_optimizer, actor_optimizer,
                batch_size=batch_size, sample_count=sample_count,
                state_dim=num_states, action_dim=num_actions)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Initialize the state buffer with the first state
state_buffer = np.zeros((batch_size, sample_count, num_states))

# Takes about 4 min to train
for ep in range(total_episodes):

    prev_state = env.reset()
    if len(prev_state) == 2:
        prev_state, _ = prev_state
    else:
        prev_state = prev_state
    episodic_reward = 0

    # Fill the state buffer with the initial state
    state_buffer[:] = prev_state

    while True:
        #tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        if len(prev_state) == 2:
            prev_state, _ = prev_state
        else:
            prev_state = prev_state

        # Update the state buffer with the new state
        state_buffer = np.roll(state_buffer, -1, axis=1)
        state_buffer[0, 0, :] = prev_state

        # Reshape the state buffer to match the expected input shape of the actor model
        tf_prev_state = tf.reshape(state_buffer, (1, batch_size, sample_count, num_states))
        #tf.print("State buffer: ", state_buffer)
        #tf.print("Prev state: ", tf_prev_state)

        action = np.array(policy(tf_prev_state, ou_noise)).flatten()

        # Recieve state and reward from environment.
        state, reward, done, _, info = env.step(action)
        print("Reward: ", reward)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        #update_target(target_actor.variables, actor_model.variables, tau)
        #update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-2:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
plt.savefig("DDPG.png")