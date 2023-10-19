'''
Modified from https://github.dev/keirp/stochastic_offline_envs
'''

from collections import namedtuple

# Each policy step
PolicyStep = namedtuple("PolicyStep", ["action", "info"])

class BasePolicy:
	'''
    Basic policy type of agent
    '''
	def reset(self):
		pass

	def sample(self, obs, reward, t):
		'''
        Sample action based on obs, reward and timestep
        '''
		raise NotImplementedError

	def update(self, total_interactions):
		"""Use this to update epsilon"""
		pass

	@property
	def name(self):
		return self._name
	
