import math
import numpy as np
import time
from typing import NamedTuple, List, Tuple
import pickle

ALPHA = 0.6
UNIFORM_SAMPLE_PROBABILITY = 1e-3
IMPORTANCE_SAMPLING_START = 0.4
IMPORTANCE_SAMPLING_END = 1.0
IMPORTANCE_SAMPLING_DECAY_START = 200_000
IMPORTANCE_SAMPLING_DECAY_STEPS = 5_000_000

class Experience(NamedTuple):
    state: np.ndarray       # what was the state @ time T
    action: int             # action that was taken in S @ time T
    reward: float           # immediate reward
    next_state: np.ndarray  # next state
    done: bool              # has the episode finished

class PrioritizeExperience:
    def __init__(self, state : np.ndarray, action : int, reward: float, next_state: np.ndarray, done: bool, td_error: float):
        self.state = state          # what was the state @ time T
        self.action = action        # action that was taken in S @ time T
        self.reward = reward        # immediate reward
        self.next_state = next_state  # next state
        self.done = done            # has the episode finished
        self.td_error = td_error    # this is required for replay buffer's weighted computation

    @property
    def priority(self):
        # PrioritizedReplay research paper says "the transition with the absolute largest TD error.." so 
        # we take the absolute value here
        assert math.isinf(self.td_error) == False

        p_abs = abs(self.td_error)
        if p_abs == 0:
            return 0    # an experience with 0 priority is not required to be sampled so we dont do a p_abs ** 0 on it
        else:
            return p_abs ** ALPHA


class AnnealingRate:
    """
    Takes a starting value for a rate and decays it to an ending value. Decay starts after a given number of 
    steps have passed and reaches the ending_value in a given number of decay steps
    """
    def __init__(self, start_value : float, end_value : float, after_steps : int, decay_steps: int):
        """
        once after_steps timesteps have crossed, it will start to decay the rate till it reaches end_value
        """
        self.start_value = start_value
        self.end_value = end_value
        self._decay_after_steps = after_steps
        self._delta = (self.end_value - self.start_value) / decay_steps
        self._total_steps = self._decay_after_steps + decay_steps
        self.reset()
        
        assert decay_steps >= 1, 'Decay steps have to be >= 1'

    def __len__(self) -> int:
        return self._steps

    def reset(self):
        self._steps = 0
        self._current_value = self.start_value
        self.get_current = self.ret_start

    def ret_start(self):
        self._steps += 1
        if self._steps > self._decay_after_steps:
            self.get_current = self.ret_decay
            return self.get_current()
        else:
            return self.start_value

    def ret_decay(self):
        self._steps += 1
        if self._steps > self._total_steps:
            self.get_current = self.ret_end
            return self.get_current()
        else:
            self._current_value += self._delta
            return self._current_value

    def ret_end(self):
        return self.end_value
    
    @property 
    def current(self):
        return self.get_current()

class FindResult(NamedTuple):
    index : int
    experience : PrioritizeExperience

class SumTree:
    """
    Picks an experience randomly, but the chance of an experience to be picked up is based on how high is the
    priority of that experience. For priority, the td_error is used.
    """
    def __init__(self, leaf_nodes : List[PrioritizeExperience], max_size = None):
        """
        leaf_nodes: this is the list where experience is being stored. SumTree uses
            the same memory location for accessing the leaf_nodes' td_error
        """
        size = max_size if max_size != None else len(leaf_nodes)
        self.leaf_nodes = leaf_nodes
        
        # the last layer of a binary tree has 2^h + 1 nodes, where h is the height of the
        # tree. We use the inverse function to find out the number of nodes
        self.nonleaf_count = 2 ** math.ceil(np.log2(size))
        self.nodes = np.zeros(shape = self.nonleaf_count)
        
        # although we don't keep the leaf nodes with us but to find out the parent
        # we need to know where the non-leaf nodes would have been kept in the tree
        self.leaf_start = self.nonleaf_count
        #print(f'non-leaf size: {self.nonleaf_count}, self.leaf_start {self.leaf_start}')

    @property
    def root(self) -> float:
        return self.nodes[1]
    
    def __len__(self):
        return len(self.leaf_nodes)
    
    def set_priority(self, index : int):
        #self.nodes[index] = item_priority
        #self.leaf_index = (self.leaf_index + 1) % self.size

        # had we kept the leaf_nodes here, the index would have been
        # after self.leaf_start_index
        #print(f'Index added: {index}')
        
        # Since we don't have the leaf nodes with us, we need to figureout which 
        # two nodes make the left and right. So 0,1 make a pair, 2, 3 make another
        # So whatever index the new leaf was stored at, the left will be even and 
        # right would be the odd number
        left_index = index & 0xFFFFFFFE
        right_index = index | 1
        
        left_p = self.leaf_nodes[left_index].priority if self.leaf_nodes[left_index] != None else 0
        right_p = self.leaf_nodes[right_index].priority if self.leaf_nodes[right_index] != None else 0
        
        leaf_sum = left_p + right_p
        #print(f'index: {index}, left: {left_index}, right:{right_index}, leaf_sum: {leaf_sum}')
              
        # figure out where the leaf node would have been kept
        leaf_index = index + self.leaf_start
        parent = leaf_index // 2
        self.nodes[parent] = leaf_sum
              
        #print(f'First parent: {parent} for index: {index}')

        level = 1
        max_level = 1 + math.floor(math.log2(self.nonleaf_count * 2))

        parent //= 2
        while parent > 0:
            self.nodes[parent] = self.nodes[parent * 2] + self.nodes[parent * 2 + 1]
            #print(f'Next parent: {parent} = {self.nodes[parent]}')
            parent //= 2

            level += 1
            assert level < max_level

    def update_priorities(self, indices : List[int]):
        for index in indices:
            self.set_priority(index)
    
    def find(self, random_priority : float) -> FindResult:
        assert random_priority <= self.root, f'Random priority: {random_priority} exceds root {self.root}'
        
        # root is at 1st element of the array not the 0th
        index = 1
        p = random_priority

        # start from the root and go till the start of leaf nodes, then choose
        # the index of the leaf node that covers the priority given
        while index * 2 < self.leaf_start:
            left_index = index * 2
            right_index = index * 2 + 1
            
            left_value = self.nodes[left_index]
            if p < left_value:
                index = left_index
            else:
                p -= left_value
                index = right_index

        assert left_index & 1 == 0      # left index has to be even numbered
        assert right_index & 1 == 1     # right index has to be odd numbered
         
        # The leaf node's priority has to be considered as well in deciding
        # which of the two children covers the given p
        left_index = (index * 2) - self.leaf_start
        right_index = left_index + 1
        
        #print(f'left: {left_index}, right: {right_index}, priority: {priority}')
              
        left_value = self.leaf_nodes[left_index].priority
        if p < left_value or self.leaf_nodes[left_index] == None:
            index = left_index
        else:
            index = right_index
    
        return FindResult(index, self.leaf_nodes[index])
    
    def pick_random(self) -> FindResult:
        return self.find(np.random.uniform(0, self.root))

    def reset(self):
        self.nodes = np.zeros(shape = self.nonleaf_count)

    def recompute(self):
        self.reset()
        # In each go both left and right nodes will be summed, so the loop below only
        # has to traverse the left nodes of the leaf memory
        for i in range(0, len(self.leaf_nodes), 2):
            self.set_priority(i)

class ReplayBuffer:
    SERIALIZE_VERSION = 1

    class SerializeException(Exception):
        def __init__(self, message):
            super().__init__(message)

    def __init__(self, maxlen : int, batch_size: int, seed: int = None):
        self.maxlen = maxlen
        # we are not using a deque here so that in future we can remove the 
        # least rewarding states with new states rather than using a cyclic method
        self.memory = [None] * self.maxlen
        if seed != None:
            np.random.seed(seed)
        self.batch_size = batch_size
        self.clear()
        
    def __len__(self):
        return min(self.count, self.maxlen)

    @property
    def capacity(self):
        return self.maxlen

    @property 
    def is_full(self):
        return self.count >= self.maxlen
    
    def clear(self):
        self.count = 0
        self.index = 0

    def add(self, experience: Experience):
        self.memory[self.index] = experience
        self.count += 1
        index = self.index
        self.index = (self.index + 1) % self.maxlen
        return index

    def _sample(self):
        return np.random.choice(len(self), self.batch_size)                        

    def separate_columns(self, experiences)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for e in experiences:
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            dones.append(e.done)

        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones, dtype='float'))

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = self._sample()
        experiences = [self.memory[i] for i in indices]
        return self.separate_columns(experiences), indices

    def serialize(self):
        return self.memory

    def deserialize(self, data):
        self.memory = data
        self.count = len(self.memory)
        self.maxlen = len(self.memory)
        self.index = 0

    def dump(self, filename):
        with open(filename, 'wb') as f:
            data = {
                'version': ReplayBuffer.SERIALIZE_VERSION,
                'serialized': self.serialize()
            }
            pickle.dump(data, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            if data['version'] != ReplayBuffer.SERIALIZE_VERSION:
                raise ReplayBuffer.SerializeException('Data file is not from this version')

        serialized = data['serialized']
        self.deserialize(serialized)


# This basic idea could be explained as "train more on data that surprises you"
class ReplayBufferWithPriority(ReplayBuffer):
    def __init__(self, maxlen : int, batch_size: int, is_rate: AnnealingRate = None, seed: float = None):
        super().__init__(maxlen, batch_size, seed)
        self.sumtree = SumTree(self.memory)
        self.max_priority = 1
        if is_rate != None:
            self.is_rate = is_rate
        else:
            self.is_rate = AnnealingRate(IMPORTANCE_SAMPLING_START, IMPORTANCE_SAMPLING_END, 
                                IMPORTANCE_SAMPLING_DECAY_START, IMPORTANCE_SAMPLING_DECAY_STEPS)
        
    def add(self, experience : PrioritizeExperience):
        index = super().add(experience)
        self.sumtree.set_priority(index)

    def add_with_max(self, state, action, reward, next_state, done):
        index = super().add(PrioritizeExperience(state, action, reward, next_state, done, self.max_priority))
        self.sumtree.set_priority(index)

    # PaperCheck: why did their code have this?
    def _find_probabilities(self, indices) -> np.ndarray:
        uniform_prob = 1. / len(self)
        p_sum = self.sumtree.root

        if p_sum > 0:
            priorities_prob = np.array([self.memory[i].priority / p_sum for i in indices])
            return (1 - UNIFORM_SAMPLE_PROBABILITY) * priorities_prob + UNIFORM_SAMPLE_PROBABILITY * uniform_prob
        else:
            return np.array([UNIFORM_SAMPLE_PROBABILITY * uniform_prob] * len(indices))

    def _replace_few_uniformly(self, indices) -> List[int]:
        """
        To gaurantee that few nodes are still chosen randomly
        """
        return np.where(np.random.uniform(size = self.batch_size) < UNIFORM_SAMPLE_PROBABILITY,
            np.random.choice(len(self), size = self.batch_size),
            indices)

    def _importance_sampling(self, indices):
        probabilities = self._find_probabilities(indices)
        N = len(self)
        weights = (1. / (N * probabilities)) ** self.is_rate.get_current()
        weight_scaled = weights / np.max(weights)
        assert np.isfinite(weight_scaled).all(), 'Some of the weights are not finite any more'
        return weight_scaled

    def _sample(self) -> Tuple[List[int], np.ndarray]:
        """
        Returns a uniformly choosen experience based on the td_errors

        Shortcoming: what if the sumtree has some nodes that do not have priorities set?
        """
        if self.sumtree.root > 0:
            # divide the total priority in equal number of K ranges
            bucket_size = self.sumtree.root / self.batch_size

            # a generator that returns a pair of (low, high) priorities to pick from sumtree
            bucket_priorities = (np.random.uniform(bucket_size * i, bucket_size * (i + 1)) for i in range(self.batch_size))

            indices = [self.sumtree.find(p).index for p in bucket_priorities]
        else:
            # select randomly as the sumtree root is 0
            indices = np.random.choice(len(self), self.batch_size)

        indices = self._replace_few_uniformly(indices)        
        weights = self._importance_sampling(indices)
        return indices, weights

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices, weights = self._sample()
        experiences = [self.memory[i] for i in indices]
        separated = super().separate_columns(experiences)
        return (indices, *separated, weights)

    def update_priorities(self, indices : List[int], td_error : np.ndarray):
        for i, error in zip(indices, td_error):
            self.memory[i].td_error = abs(error)

            # update max_priority so that any new experience added to the replay buffer
            # will have a chance to be picked
            p = self.memory[i].priority
            if p > self.max_priority:
                self.max_priority = p

        self.sumtree.update_priorities(indices)

    def deserialize(self, filename):
        super().deserialize(filename)

        # Update priorities in the sum tree. Clear the current one and use a new tree
        self.sumtree = SumTree(self.memory)
        self.sumtree.recompute()
