"""
U&RL: Miniproject2 - Mountain Car problem "solved" with cont. state SARSA(lambda) algo
Authors: Andras Ecker, Sharbatanu Chatterjee
"""

import numpy as np
import sys
import pylab as plb
import mountaincar
from figures import plot_escape_time


class Agent():
    """ A very clever agent for the mountain-car task """

    def __init__(self, x_size, xd_size, w_ini, mc=None, parameter1=3.0, weights=None): #unchanged from given
        
        if mc is None:
            self.mc = mountaincar.MountainCar()
        else:
            self.mc = mc

        self.parameter1 = parameter1
        
        self.x_centers, self.sigmax = np.linspace(-150, 30, num=x_size, endpoint=True, retstep=True)
        self.xd_centers, self.sigmay = np.linspace(-15, 15, num=xd_size, endpoint=True, retstep=True)
        
        if weights is None:
            a_size = 3
            # weight dictionary matrix (dim:x_size*xd_size*a_size)
            self.weights = {}
            for a in np.arange(0, a_size):
                for x in self.x_centers:
                    for x_d in self.xd_centers:
                        self.weights[a, x, x_d] = w_ini
        else:
            self.weights = weights
        
    def calculate_r(self, xj, xdj):
        """ Calculates firing rate of the input layer neurons, based on state of the agent
        :param xj,xdj: ids/place of the current neuron
        return: firing rate of the neuron
        """
        return np.exp(-((xj-self.mc.x)**2/self.sigmax**2) - ((xdj-self.mc.x_d)**2/self.sigmay**2))
        
    def calculate_Q(self, a):
        """ Calculates Q(s,a)
        :param a: action
        return: Q(s,a) as sum(weights*rates)
        """
        sum = 0
        for key, weight in self.weights.items():
            if key[0] == a:
                sum += weight*self.calculate_r(key[1], key[2])  # key = (a, xj, xdj)

        return sum
        
    def softmax_policy(self, tau):
        """ Calculates probabilities based on Boltzmann distribution
        :param tau: temperature parameter of Boltzmann distribution (exploration vs. exploration)
        return: dict of Q values, list of probabilities (for the 3 possible actions)
        """
        Qs = {}  # dict just to store calculated Q-values and speed up the code
        Qs[0] = self.calculate_Q(0)
        Qs[1] = self.calculate_Q(1)
        Qs[2] = self.calculate_Q(2)
        Pa0 = np.exp(Qs[0] / tau)
        Pa1 = np.exp(Qs[1] / tau)
        Pa2 = np.exp(Qs[2] / tau)
        P_tot = Pa0 + Pa1 + Pa2
        
        return Qs, [Pa0/P_tot, Pa1/P_tot, Pa2/P_tot]
        
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
            
        a_size = 3
        # e dictionary matrix (dim:x_size*xd_size*a_size)
        e = {}
        for a in np.arange(0, a_size):
            for x in self.x_centers:
                for x_d in self.xd_centers:
                    e[a, x, x_d] = 0.    

        # ============================ core SARSA(lambda) algo ============================

        # init
        self.mc.reset()
        a_old = np.random.randint(0,3,1)[0]  # gives 0 or 1 or 2 -> random action at the 1st step
        Q_old = self.calculate_Q(a_old)

        # run untill it gets the reward
        for i in range(n_steps):
            self.mc.apply_force(a_old-1)  # scale [0,2] to [-1,1]
            self.mc.simulate_timesteps(100, 0.01)  # take action a and go to an other state, observe reward
            
            #choose new action according to softmax policy
            Qs, p = self.softmax_policy(tau)
            a_selected = np.random.choice([0,1,2],1,p)[0]

            # calculate delta
            Q = Qs[a_selected]
            delta = self.mc.R - Q_old + gamma*Q
            
            # update e and w
            for key, _ in e.items():                         
                e[key] *= gamma * lambda_
                if key[0] == a_selected:
                    r = self.calculate_r(key[1], key[2])  # key = (a, xj, xdj)
                    e[key] += r
                # finally update weights !
                self.weights[key] += eta * delta * e[key]

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
        
        """
        # kind of debigging tool ...
        print("some random e-values:", e[2,-50.,-15.], e[2,-90.,5.], e[0,-110.,5.], e[1,-110.,5.], e[2,-110.,5.],
              e[1,-130.,5.], e[0,-50.,-15.], e[0,-70.,5.], e[1,-70.,15.], e[2,-70.,15.])
        print("corresponding weights:", self.weights[2,-50.,-15.], self.weights[2,-90.,5.], self.weights[0,-110.,5.],
                                        self.weights[1,-110.,5.], self.weights[2,-110.,5.], self.weights[1,-130.,5.],
                                        self.weights[0,-50.,-15.], self.weights[0,-70.,5.], self.weights[1,-70.,15.], self.weights[2,-70.,15.])
        """
            
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
    x_sizes = [10.]#[10., 20.]
    xd_sizes = [10.]#[10., 20.]
    w_inis = [0., 1.]#[0., 1.]
    taus = [0.001]#[1e-5, 1., 100.]
    lambdas = [0.95]#[0, 0.95]
    
    for x_size in x_sizes:
        for xd_size in xd_sizes:
            for w_ini in w_inis:
                for tau in taus:
                    for lambda_ in lambdas:
                        print("x_size:%s, xd_size:%s, w_ini:%s, tau:%.5f, lambda:%.2f"%(x_size, xd_size, w_ini, tau, lambda_))
                        
                        a = Agent(x_size=x_size, xd_size=xd_size, w_ini=w_ini)
                        escape_times = a.episodes(max_episodes=250, n_steps=5000,
                                                lambda_=lambda_, tau=tau, eta=eta,
                                                visualize=False)
    
                        print("Escape times:", escape_times)
                        plot_escape_time(escape_times, tau, lambda_, x_size, xd_size, w_ini)
                                     
    print("##### Terminated #####")
               
    
    
    
    
    
    
    
    
    
