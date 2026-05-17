import random
import numpy as np
import os
import pickle

class SeqReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        
        self.flat_buffer = [] 
        self.position = 0
        
        self.trajectories = []


        self._n_step_indices_cache = {}

    def push(self, state, action, reward, next_state, done):
        current_traj = {
            'states': [], 'actions': [], 'rewards': [], 'dones': []
        }
        
        num_transitions = state.shape[0]

        for i in range(num_transitions):
            self.flat_buffer.append((
                state[i], action[i], reward[i], next_state[i], done[i]
            ))

            if not current_traj['states']:
                current_traj['states'].append(state[i])
            
            current_traj['actions'].append(action[i])
            current_traj['rewards'].append(reward[i])
            current_traj['dones'].append(done[i])
            current_traj['states'].append(next_state[i]) # add s_{t+1}

            if done[i][0]:
                # trajectory end
                self.trajectories.append({
                    'states': np.array(current_traj['states']),     #  (T+1, D_s)
                    'actions': np.array(current_traj['actions']),   #  (T, D_a)
                    'rewards': np.array(current_traj['rewards']),   #  (T, 1)
                    'dones': np.array(current_traj['dones'])        #  (T, 1)
                })
                
                # reset
                current_traj = {
                    'states': [], 'actions': [], 'rewards': [], 'dones': []
                }
        
        # Handle the remaining incomplete data.
        if current_traj['states']:
            self.trajectories.append({
                'states': np.array(current_traj['states']),
                'actions': np.array(current_traj['actions']),
                'rewards': np.array(current_traj['rewards']),
                'dones': np.array(current_traj['dones'])
            })

        self.position = len(self.flat_buffer) % self.capacity
        print(f"SeqReplayMemory: Processed {num_transitions} transitions into {len(self.trajectories)} trajectories.")


    def sample(self, batch_size):
        batch = random.sample(self.flat_buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def _get_n_step_indices(self, n_step):
        """
        Precompute all valid (trajectory index, step start index) pairs.
        """
        # Check buffer
        if n_step in self._n_step_indices_cache:
            return self._n_step_indices_cache[n_step]

        indices = []
        for traj_idx, traj in enumerate(self.trajectories):

            traj_len_transitions = len(traj['actions'])
            max_start_idx = traj_len_transitions - n_step
            
            for i in range(max_start_idx + 1):
                indices.append((traj_idx, i))
        
        valid_indices = np.array(indices)
        self._n_step_indices_cache[n_step] = valid_indices
        return valid_indices

    def sample_sequential_transitions(self, batch_size, n_step):

        valid_indices = self._get_n_step_indices(n_step)
        

        sample_indices = valid_indices[
            np.random.randint(0, len(valid_indices), size=batch_size)
        ]
        

        batch_s_h = []
        batch_a_h = []
        batch_rewards_n = [] # (Batch, n_step, 1)
        batch_dones_n = []   # (Batch, n_step, 1)
        batch_s_h_plus_n = []
        
        for traj_idx, start_idx in sample_indices:
            traj = self.trajectories[traj_idx]
            
            # s_h (start state)
            batch_s_h.append(traj['states'][start_idx])
            
            # a_h (start action)
            batch_a_h.append(traj['actions'][start_idx])
            
            # r_h to r_{h+n-1}
            batch_rewards_n.append(
                traj['rewards'][start_idx : start_idx + n_step]
            )
            
            # d_h to d_{h+n-1}
            batch_dones_n.append(
                traj['dones'][start_idx : start_idx + n_step]
            )
            
            # s_{h+n}
            batch_s_h_plus_n.append(
                traj['states'][start_idx + n_step]
            )
            
        return (
            np.stack(batch_s_h),
            np.stack(batch_a_h),
            np.stack(batch_rewards_n),
            np.stack(batch_dones_n),
            np.stack(batch_s_h_plus_n)
        )

    def __len__(self):
        return len(self.flat_buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if save_path is None:
            save_path = "checkpoints/replay_memory_{}_{}".format(env_name, suffix)
        print('Saving replay memory to {}'.format(save_path))

        data_to_save = {
            'flat_buffer': self.flat_buffer,
            'trajectories': self.trajectories,
            'position': self.position
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_buffer(self, save_path):
        print('Loading replay memory from {}'.format(save_path))
        with open(save_path, "rb") as f:
            data_to_load = pickle.load(f)
            if isinstance(data_to_load, dict):
                self.flat_buffer = data_to_load['flat_buffer']
                self.trajectories = data_to_load['trajectories']
                self.position = data_to_load['position']
            else:
               
                print("Warning: Loading old buffer format. Re-processing trajectories...")
                self.flat_buffer = data_to_load
                self.position = len(self.flat_buffer) % self.capacity
                
                states, actions, rewards, next_states, dones = map(np.array, zip(*self.flat_buffer))
                
                self.flat_buffer = [] 
                self.push(states, actions, rewards, next_states, dones)

        self._n_step_indices_cache = {}
        print(f"Loaded {len(self.flat_buffer)} transitions in {len(self.trajectories)} trajectories.")