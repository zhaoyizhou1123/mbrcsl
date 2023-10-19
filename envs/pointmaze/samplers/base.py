'''
Modified from https://github.dev/keirp/stochastic_offline_envs
'''

class BaseSampler:

	def collect_trajectories(self, *args, **kwargs):
		"""Sample at most n_interactions data and return"""
		raise NotImplementedError