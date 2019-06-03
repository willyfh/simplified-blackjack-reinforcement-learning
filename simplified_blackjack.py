"""
__author__  = Willy Fitra Hendria
"""
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

numbers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
colors = ['red', 'black']
# actions = ['hit', 'stick']
# states = [dealer_card, player_card]
N0 = 10
gamma = 1.0
LEARN_N_EPISODES = 1000
EVAL_N_EPISODES = 100



def advance(s, a):
	"""  takes as input a state s (dealer’s first card 3–12 and the player’s sum 1–21),
	and an action a (hit or stick), and returns a sample of the next state s′
	(which may be terminal if the game is finished)
	and reward r ∈ {1, 0, -1} for winning, draw, and loosing.
	"""
	dealer_sum = s[0]
	player_sum = s[1]
	r = 0
	is_done = False
	if (a == 0): # hit
		drawed_card = draw_card()
		if drawed_card[1] == 'black':
			player_sum = player_sum + drawed_card[0]
		else: #red
			player_sum = player_sum - drawed_card[0]
		
		if player_sum < 1 or player_sum > 21: #lose
			r = -1
			is_done = True
	else: #stick
		# dealer starts taking turns
		dealer_sum = s[0]
		while dealer_sum < 15 and dealer_sum > 1:
			drawed_card = draw_card()
			if drawed_card[1] == 'black':
				dealer_sum = dealer_sum + drawed_card[0]
			else: #red
				dealer_sum = dealer_sum - drawed_card[0]
		if dealer_sum > 21 or dealer_sum < 1: # player is win
			r = 1
		else:
			if (dealer_sum > player_sum):
				r = -1
			elif (dealer_sum < player_sum):
				r = 1
			else: #draw
				r = 0
		is_done = True		
	next_state = (dealer_sum, player_sum)		
	return next_state, r, is_done

def draw_first_card():
	""" At the start of the game, both the player and the dealer draw one black card
	"""
	i = np.random.choice(10)
	drawed_number = numbers[i]
	dealer_card = (drawed_number, 'black')
	
	i = np.random.choice(10)
	drawed_number = numbers[i]
	player_card = (drawed_number, 'black')
	
	return dealer_card, player_card
	
def first_state():
	""" return the first state
	"""
	dealer_first_card, player_first_card = draw_first_card()
	s = (dealer_first_card[0], player_first_card[0])
	return s
	
def draw_card():
	""" Each draw from the deck results in a value between 3 and 12 (uniformly distributed)
	with a color of red (probability .3) or black (probability .7). 
	"""
	i = np.random.choice(10)
	drawed_number = numbers[i]
	j = np.random.choice(2, p=[0.3, 0.7])
	drawed_color = colors[j]
	drawed_card = (drawed_number, drawed_color)
	return drawed_card


	
def sarsa(n_episodes, landa):
	""" sarsa lambda with n ε-greedy exploration strategy
	"""
	Q = defaultdict(lambda: np.zeros(2)) # state action value
	Ns = defaultdict(lambda:0) # number of times state s has been visited
	Nsa = defaultdict(lambda: np.zeros(2)) # number of times action a has been selected from state s
	for ep in range(n_episodes):
		e = defaultdict(lambda: np.zeros(2)) # eligibility trace
		current_state = first_state()
		epsilon = epsilon_fn(Ns[current_state])
		current_action = epsilon_greedy_action(Q[current_state], epsilon)
		is_done = False
		while not is_done:
			Ns[current_state] += 1
			Nsa[current_state][current_action] += 1
			
			# observe next state and reward
			next_state, reward, is_done = advance(current_state, current_action)
			
			#choose next action using epsilon greedy
			epsilon = epsilon_fn(Ns[next_state])
			next_action = epsilon_greedy_action(Q[next_state], epsilon)
			
			# compute state action value
			delta = reward + gamma * Q[next_state][next_action] - Q[current_state][current_action]
			e[current_state][current_action] += 1
			alpha = alpha_fn(Nsa[current_state][current_action])
			for s in Q:
				for a in [0,1]:
					Q[s][a] += (alpha * delta * e[s][a])
					e[s][a] = gamma * landa * e[s][a]
			
			current_state = next_state
			current_action = next_action
	return Q		

def linear_function_approximation(n_episodes, landa):
	""" Using a simple coarse coding value function approximator that is based on a state feature 
	"""
	theta = np.random.random(36)
	epsilon = 0.1
	alpha = 0.05
	for ep in range(n_episodes):
		e = np.zeros(36) # eligibility trace
		current_state = first_state()
		current_state_features = get_state_features(current_state)
		current_Q, current_action = epsilon_greedy_action_lfa(current_state_features, epsilon, theta)
		current_Q_features = get_state_action_features(current_state_features, current_action)
		is_done = False
		while not is_done:
			
			# observe next state and reward
			next_state, reward, is_done = advance(current_state, current_action)
			
			# get next state feature
			next_state_features = get_state_features(next_state)
			
			# choose next action using epsilon greedy
			next_Q, next_action = epsilon_greedy_action_lfa(next_state_features, epsilon, theta)
			
			# get next state action features
			next_Q_features = get_state_action_features(next_state_features, next_action)
			
			# compute theta
			delta = reward + gamma * next_Q - current_Q
			e = np.add((gamma * landa * e), current_Q_features)
			theta = np.add(theta, alpha * delta * e)
			
			current_Q = next_Q
			current_Q_features = next_Q_features
			current_action = next_action
		
		# compute state action value
		Q = defaultdict(lambda: np.zeros(2))
		for player in range(21):
			for dealer in range(10):
				for a in [0,1]:
					s = (dealer+3, player+1)
					phi = get_state_action_features(get_state_features(s), a)
					Q[s][a] = phi.dot(theta)
	return Q
	
def alpha_fn(count):
	return 1/count
	
def epsilon_fn(count):
	return N0/(N0+count)

def epsilon_greedy_action(Qs, epsilon):
	p = np.random.uniform()
	if p < epsilon: # exploration
		i = np.random.choice(2)
	else: #exploitation
		i = np.argmax(Qs)
	return i
	
def epsilon_greedy_action_lfa(state_features, epsilon, theta):
	p = np.random.uniform()
	if p < epsilon: # exploration
		a = np.random.choice(2)
		Q = get_state_action_features(state_features, a).dot(theta)
		return Q, a
	else: # exploitation
		Q_hit = get_state_action_features(state_features, 0 ).dot(theta)
		Q_stick = get_state_action_features(state_features, 1).dot(theta)
		if Q_hit > Q_stick:
			return Q_hit, 0
		elif Q_hit < Q_stick:
			return Q_stick, 1
		else:
			a = np.random.choice(2)
			Q = get_state_action_features(state_features, a).dot(theta)
			return Q, a
		
def	get_state_features(state):
	""" convert state into state features in binary vector
	"""
	dealer_card_sum = state[0]
	player_card_sum = state[1]
	
	dealer_intervals = [(3,6),(6,9),(9,12)]
	player_intervals = [(1,6),(4,9),(7,12),(10,15),(13,18),(16,21)]
	
	features = []
	for i in range(len(dealer_intervals)):
		for j in range(len(player_intervals)):
			dealer_interval = dealer_intervals[i]
			player_interval = player_intervals[j]
			if (dealer_interval[0] <= dealer_card_sum <= dealer_interval[1]) and (player_interval[0] <= player_card_sum <= player_interval[0]):
				features.append(1)
			else:
				features.append(0)
	return features
	
		
		
def get_state_action_features(state_features, action):
	""" create a state action feature vector size 36 based on the given action
	"""
	if action == 0: #hit
		return np.append(state_features, np.zeros(18))
	else: # stick
		return np.append(np.zeros(18), state_features)



def accumulate_reward(Q, n_episodes):
	""" accumulate reward based on the given policy
	"""
	sum_reward = 0
	for i in range(n_episodes):
		current_state = first_state()
		is_done = False
		while not is_done:
			current_action = np.argmax(Q[current_state])
			next_state, reward, is_done = advance(current_state, current_action)
			current_state = next_state	
		sum_reward += reward
	return sum_reward

def drawRewardPlot(sum_rewards, lambdas):
	""" plot the accumulated reward for the next 100 episodes against λ
	"""
	plt.plot(lambdas, sum_rewards)
	plt.xlabel('lambda')
	plt.ylabel('accumulative rewards over 100 episodes')
	plt.show()

###
### main program
###
print()
print("1. Test 'advance' function")
print("2. Sarsa Lambda")
print("3. Linear value function approximation")
print()
i = input("input (1 -  3):")
print()

if i == '1':
	print("---------------------------------------")
	print("This is for testing 'advance' function")
	print("---------------------------------------")
	is_done = False
	s = first_state()
	print("first state:",s)
	while not is_done:
		action = input("input (0 for hit, or 1 for stick): ")
		if action == '0' or action == '1':
			next_state, reward, is_done = advance(s, int(action))
			print("next state:",next_state,", reward:", reward,", is_done:",is_done)
			s = next_state
		else:
			print("action is not defined")
elif i == '2':
	print("---------------------------------------")
	print("Sarsa lambda")
	print("---------------------------------------")
	Qlambda = []
	lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	for l in lambdas:
		Q = sarsa(LEARN_N_EPISODES, l)
		Qlambda.append(Q)

	sum_rewards = []
	for f in Qlambda:
		sum_reward = accumulate_reward(f, EVAL_N_EPISODES)
		sum_rewards.append(sum_reward)
		
	print(sum_rewards)
	drawRewardPlot(sum_rewards, lambdas)
elif i == '3':
	print("---------------------------------------")
	print("Coarse coding value function approximator")
	print("---------------------------------------")
	Qlambda = []
	lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	for l in lambdas:
		Q = linear_function_approximation(LEARN_N_EPISODES, l)
		Qlambda.append(Q)
		
	sum_rewards = []
	for f in Qlambda:
		sum_reward = accumulate_reward(f, EVAL_N_EPISODES)
		sum_rewards.append(sum_reward)
		
	print(sum_rewards)
	drawRewardPlot(sum_rewards, lambdas)
else:
	print("not defined")



