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

    def __init__(self, mc=None, parameter1=3.0): #unchanged from given
        
        if mc is None:
            self.mc = mountaincar.MountainCar()
        else:
            self.mc = mc

        self.parameter1 = parameter1
        
    def calculate_r(self, xj, xdj, a, x, xd, sigmax, sigmay):
        """ Calculates firing rate of the input layer neurons, based on state of the agent
        :param xj,xdj: ids of the current neuron
        :param x, xd: current state of the car
        :param a: action taken
        :param sigmax, sigmay: params of the (Gaussian) basis function, based on the grid size
        return: firing rate of the neuron
        """
        return np.exp(-((xj-x)/sigmax)**2 - ((xdj-xd)/sigmay)**2)
        
    def calculate_Q(self, x, xd, a, w, sigmax, sigmay):
        """ Calculates Q(s,a)
        :param x, xd, a: state + action
        :param w: synaptic weights (stored in a dictionary... maybe a matrix would be faster)
        :param sigmax, sigmay: params of calculate_r
        return: Q(s,a) as sum(w*rate)
        """
        sum = 0
        for keys,weight in w.items():
            if keys[0]==a:
                sum += weight*self.calculate_r(keys[1], keys[2], keys[0], x, xd, sigmax, sigmay)  # keys = (a, xj, xdj, a) but can't pass as tuple

        return sum
        
    def softmax_policy(self, w, tau, sigmax, sigmay):
        """ Calculates probabilities based on Boltzmann distribution
        :param w: synaptic weights (stored in a dictionary... maybe a matrix would be faster)
        :param tau: temperature parameter of Boltzmann distribution (exploration vs. exploration)
        :param sigmax, sigmay: params of calculate_r
        return: list of probabilities (for the 3 possible actions)
        """
        Pa1 = np.exp(self.calculate_Q(self.mc.x, self.mc.x_d, 0, w, sigmax, sigmay) / tau)
        Pa2 = np.exp(self.calculate_Q(self.mc.x, self.mc.x_d, 1, w, sigmax, sigmay) / tau)
        Pa3 = np.exp(self.calculate_Q(self.mc.x, self.mc.x_d, 2, w, sigmax, sigmay) / tau)
        P_tot = Pa1 + Pa2 + Pa3
        
        return [Pa1/P_tot, Pa2/P_tot, Pa3/P_tot]
        
    def learn(self, w, n_steps=200, lambda_=0.95, tau=0.05, eta=0.01, x_size=10, xdot_size=10, visualize=True):
        """ SARSA(lambda) implementation (with softmax policy)
        :param w: weights (dictionary) at the beginning of the episode (first time initialized in episodes())
        :param n_step: max steps before termination (or it breaks if it reaches the reward)
        :param lambda_: decaying rate for eligibility trace
        :param tau: param of softmax_policy
        :param eta: learning rate
        :param x_size, xdot_size: discretization of the state place
        :param visualize: set to True for step-by-step visualization of the learning process
        """
        
        sigmax = 180/x_size
        sigmay = 30/xdot_size
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

        # init
        self.mc.reset()
        x_old = self.mc.x
        x_d_old = self.mc.x_d
        a_old = np.random.randint(0,3,1)[0]
        e_old = 0

        # run untill it gets the reward
        for i in range(n_steps):
            self.mc.apply_force(a_old-1)
            self.mc.simulate_timesteps(100, 0.01) #take action a, observe

            #choose new action according to softmax policy
            p = self.softmax_policy(w, tau, sigmax, sigmay)
            a_selected = np.random.choice([0,1,2],1,p)[0]

            delta_t = self.mc.R + gamma*self.calculate_Q(self.mc.x, self.mc.x_d, a, w, sigmax, sigmay) - self.calculate_Q(x_old, x_d_old, a, w, sigmax, sigmay)
            
            for keys,_ in w.items():                               
                e = gamma*lambda_*delta_t*e_old
                if keys[0]==a:
                    r_j = self.calculate_r(keys[1], keys[2], keys[0], self.mc.x, self.mc.x_d, sigmax, sigmay)  # keys = (a, xj, xdj, a) but can't pass as tuple
                    e += e + r_j
                # finally update weights !
                w[keys] += eta*delta_t*e

            x_old = self.mc.x
            x_d_old = self.mc.x_d
            a_old = a_selected
            e_old = e
            
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
            plb.draw()
            plb.show()
            #plb.waitforbuttonpress(timeout=3)
            plb.close('all')
            
        return w, self.mc.t

    def episodes(self, max_episodes=100,
                 n_steps=3000, lambda_=0.95, tau=0.05, eta=0.01, x_size=10, xdot_size=10, visualize=True,
                 w_ini=0):
        """ Runs multiple episodes with 1 agent and plots escape time
        :param max_episodes: ...
        :param n_step, lambda_, tau, eta, x_size, xdot_size, visualize: params of learn()
        :param w_ini: initial weight "distribution"
        """
        
        a_size = 3
        # weight dictionary matrix (dim:x_size*xdot_size*a_size)
        w = {}
        for a in np.arange(0, a_size):
            for x in np.arange(0, x_size):
                for x_d in np.arange(0, xdot_size):
                    w[a, x, x_d] = w_ini

        esc_ts = []  # list to store escape times
        for n in range(max_episodes):
            print('\repisode =', n)           
            w_new, t = self.learn(w, n_steps, lambda_, tau, eta, x_size, xdot_size, visualize)
            w = w_new  # just to make sure ...
            esc_ts.append(t)
            
        return esc_ts


if __name__ == "__main__":
    eta = 0.01
    x_sizes = [10]#[10, 20]
    xdot_sizes = [10]#[10, 20]
    w_inis = [0]#[0, 1]
    taus = [0.05]#[1e-10, 1, 1e10]
    lambdas = [0.95]#[0, 0.95]
    
    for x_size in x_sizes:
        for xdot_size in xdot_sizes:
            for w_ini in w_inis:
                for tau in taus:
                    for lambda_ in lambdas:
                        print("x_size:%s, xdot_size:%s, w_ini:%s, tau:%.3f, lambda:%.3f"%(x_size, xdot_size, w_ini, tau, lambda_))
                        a = Agent()
                        escape_times = a.episodes(max_episodes=10, n_steps=3000,
                                                lambda_=lambda_, tau=tau, eta=eta,
                                                x_size=x_size, xdot_size=xdot_size,
                                                visualize=False,
                                                w_ini=w_ini)
    
                        print("Escape times:", escape_times)
                        plot_escape_time(escape_times, tau, lambda_, x_size, xdot_size, w_ini)
                                     
    print("##### Terminated #####")
               
    
    
    
    
    
    
    
    
    
