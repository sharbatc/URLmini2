"""
U&RL: Miniproject2 - Mountain Car problem "solved" with cont. state SARSA(lambda) algo
Authors: Andras Ecker, Sharbatanu Chatterjee
"""

import numpy as np
import sys
import pylab as plb
import mountaincar
from figures import plot_escape_time, plot_mult_escape_times


class Agent():
    """ A very clever agent for the mountain-car task """

    def __init__(self, x_size, xd_size, w_ini, mc=None, parameter1=3.0, weights=None):
        
        if mc is None:
            self.mc = mountaincar.MountainCar()
        else:
            self.mc = mc

        self.parameter1 = parameter1
        
        self.centers_x, self.sigmax = np.linspace(-150, 30, num=x_size, endpoint=True, retstep=True)  # -150,30 is the size of the map
        self.centers_xd, self.sigmay = np.linspace(-15, 15, num=xd_size, endpoint=True, retstep=True)  # -15,15 is the size of the map
        
        # broadcast for the matrix multiplications
        self.centers_x = np.atleast_2d(self.centers_x)
        self.centers_xd = np.atleast_2d(self.centers_xd).T
            
        if weights is None:
            if w_ini == 0.:
                self.weights = np.zeros((3, x_size, xd_size))  # tensor instead of dictionary with 3 keys
            elif w_ini == 1.:
                self.weights = np.ones((3, x_size, xd_size))  # tensor instead of dictionary with 3 keys
            else:
                print("w_ini should be 0 or 1")
        else:
            self.weights = weights
        
    def calculate_r(self):
        """ Calculates firing rate of the input layer neurons, based on state of the agent
        return: firing rate array of the neurons
        """

        return np.exp(-(self.mc.x-self.centers_x)**2/(self.sigmax**2) - (self.mc.x_d-self.centers_xd)**2/(self.sigmay**2))
        
    def calculate_Q(self):
        """ Calculates Q(s,a) for every possible action
        return: (rates - just to reuse)  + Q(s,a) array [sum(weights_a*rates), ..., ...]
        """
        
        r = self.calculate_r()
        # weights('a' x 'x' x 'xd') . r('x' x 'xd') = (Q_actions)
        Qs = np.tensordot(self.weights, r, 2) # i.e. (3x10x10) . (10x10) -> (3,)
        return r, Qs
        
    def normalize(self, x):
        """normalize input array with (x-mean)/std"""
        
        if np.std(x) != 0:
            return (x - np.mean(x))/np.std(x)
        else:
            return x - np.mean(x)
        
    def softmax_policy(self, tau):
        """ Calculates probabilities based on Boltzmann distribution
        :param tau: temperature parameter of Boltzmann distribution (exploration vs. exploration)
        return: (rates, Q values - just to reuse) +  list of probabilities (for the 3 possible actions)
        """

        r, Q = self.calculate_Q()  # calc Q values
        Q_n = self.normalize(Q / tau)  # normalize before np.exp() !!! -> more stable !!
        exp_Q_n = np.exp(Q_n)
        p_n = exp_Q_n / exp_Q_n.sum()
        return r, Q, p_n
        
    def learn(self, n_steps=200, lambda_=0.95, tau=0.05, eta=0.01, visualize=True):
        """ SARSA(lambda) implementation (with softmax policy)
        :param n_step: max steps before termination (or it breaks if it reaches the reward)
        :param lambda_: decaying rate for eligibility trace
        :param tau: param of softmax_policy
        :param eta: learning rate
        :param visualize: set to True for step-by-step visualization of the learning process
        """

        gamma = 0.95  # based on the description of the project
        
        if visualize:       
            # prepare for the visualization
            plb.ion()
            mv = mountaincar.MountainCarViewer(self.mc)
            mv.create_figure(n_steps, n_steps)
            plb.draw()
            plb.show()
            plb.pause(1e-7)

        # ============================ core SARSA(lambda) algo ============================

        # init eligibility trace
        e = np.zeros_like(self.weights)
        # init car
        self.mc.reset()
        r, Qs, p = self.softmax_policy(tau)
        #a_old = np.random.randint(0,3,1)[0]  # gives 0 or 1 or 2 -> random action at the 1st step
        a_old = np.random.choice([0,1,2],p=p)
        Q_old = Qs[a_old]

        # run untill it gets the reward
        for i in range(n_steps):
            self.mc.apply_force(a_old-1)  # scale [0,2] to [-1,1]
            self.mc.simulate_timesteps(100, 0.01)  # take action a and go to an other state, observe reward
            
            #choose new action according to softmax policy
            r, Qs, p = self.softmax_policy(tau)
            a_selected = np.random.choice([0,1,2],p=p)

            # calculate delta
            Q = Qs[a_selected]
            #print("Q:", Q, "Q_old:", Q_old)
            delta = self.mc.R - Q_old + gamma*Q
                
            # update eligibility trace and weights    
            e *= gamma * lambda_  # decay eligibility trace
            e[a_selected] += r  # reinforce e-trace based on the actions taken
            self.weights += eta * delta * e  # update weights

            a_old = a_selected
            Q_old = Q

        # ============================ core SARSA(lambda) algo ============================
            
            if visualize:          
                # update the visualization
                mv.update_figure()
                plb.draw()
                plb.show()
                plb.pause(1e-7)
            
            # check for rewards
            if self.mc.R > 0.0:
                print ("Reward obtained at t = %s"%self.mc.t)
                break
                
        if visualize:       
            plb.close('all')
            
        return self.mc.t

    def episodes(self, max_episodes=250,
                 n_steps=5000, lambda_=0.95, tau=0.05, eta=0.01, visualize=True):
        """ Runs multiple episodes with 1 agent and plots escape time
        :param max_episodes: ...
        :param n_step, lambda_, tau, eta, x_size, xd_size, visualize: params of learn()
        """

        esc_ts = []  # list to store escape times
        for n in range(max_episodes):
            print("episode:%s"%n)           
            t = self.learn(n_steps, lambda_, tau, eta, visualize)  # updates self.weights based on SARSA rule
            esc_ts.append(t)
            
        return esc_ts


if __name__ == "__main__":
    eta = 0.01
    x_sizes = [20.]#[10., 20.]
    xd_sizes = [20.]#[10., 20.]
    w_inis = [1.]#[0., 1.]
    taus = [0.05]#[1e-5, 1., 100.]
    lambdas = [0.95]#[0, 0.95]
    
    """
    for x_size in x_sizes:
        for xd_size in xd_sizes:
            for w_ini in w_inis:
                for tau in taus:
                    for lambda_ in lambdas:
                        print("x_size:%s, xd_size:%s, w_ini:%s, tau:%.5f, lambda:%.2f"%(x_size, xd_size, w_ini, tau, lambda_))
                        
                        a = Agent(x_size=x_size, xd_size=xd_size, w_ini=w_ini)
                        escape_times = a.episodes(max_episodes=200, n_steps=5000,
                                                lambda_=lambda_, tau=tau, eta=eta,
                                                visualize=False)
    
                        print("Escape times:", escape_times)
                        plot_escape_time(escape_times, tau, lambda_, x_size, xd_size, w_ini)
    """
    # run 10 agents and average the runs:
    mult_esc_ts = []  # to store multiple escape times
    for i in range(0, 10):
        a = Agent(x_size=20, xd_size=20, w_ini=1)
        escape_times = a.episodes(max_episodes=250, n_steps=5000,
                                  lambda_=0.95, tau=0.05, eta=0.01,
                                  visualize=False)
        mult_esc_ts.append(escape_times)
                                  
    plot_mult_escape_times(mult_esc_ts, tau=0.05, lambda_=0.95, x_size=20, xd_size=20, w_ini=1)
                                   
    print("##### Terminated #####")
    
    
               
    
    
    
    
    
    
    
    
    
