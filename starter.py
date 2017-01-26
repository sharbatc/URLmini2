import sys

import pylab as plb
import numpy as np
import mountaincar

class DummyAgent(): #name to be changed later
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mc=None, parameter1=3.0): #unchanged from given
        
        if mc is None:
            self.mc = mountaincar.MountainCar()
        else:
            self.mc = mc

        self.parameter1 = parameter1
        
    def calculate_r(self, xj, xdj, a, x, xd, sigmax, sigmay):
        return np.exp(-((xj-x)/sigmax)**2 - ((xdj-xd)/sigmay)**2)
        
    def calculate_Q(self, x, xd, a, w, sigmax, sigmay):
        sum = 0
        for keys,weight in w.items():
            if keys[2]==a:
                sum += weight*self.calculate_r(keys[0], keys[1], keys[2], x, xd, sigmax, sigmay)  # keys = (xj, xdj, a) but can't pass as tuple

        return sum
        
    def softmax_policy(self, w, tau, sigmax, sigmay):
        Pa1 = np.exp(self.calculate_Q(self.mc.x, self.mc.x_d, 0, w, sigmax, sigmay) / tau)
        Pa2 = np.exp(self.calculate_Q(self.mc.x, self.mc.x_d, 1, w, sigmax, sigmay) / tau)
        Pa3 = np.exp(self.calculate_Q(self.mc.x, self.mc.x_d, 2, w, sigmax, sigmay) / tau)
        P_tot = Pa1 + Pa2 + Pa3
        
        return [Pa1/P_tot, Pa2/P_tot, Pa3/P_tot]
        
    def learn(self, n_steps=200, lambda_=0.95, tau=0.05, eta=0.01, visualize=True):
        """We will look to implement the Sarsa(lambda) algorithm pseudocode Fig 7.11"""
        
        if visualize:       
            # prepare for the visualization
            plb.ion()
            mv = mountaincar.MountainCarViewer(self.mc)
            mv.create_figure(n_steps, n_steps)
            plb.draw()
            plb.show()
            plb.pause(1e-7)

        x_size = 10  # this should be a parameter
        xdot_size = 10  # this should be a parameter
        sigmax = 180/x_size
        sigmay = 30/xdot_size
        a_size = 3  # 3 possible actions
        gamma = 0.95  # based on the description of the project

        # weight dictionary matrix
        w = {}
        w_ini = 0  # this should be a parameter
        for a in np.arange(0,a_size,1):
            for x in np.arange(0,x_size,1):
                for x_d in np.arange(0,xdot_size,1):
                    w[a,x,x_d] = w_ini
        """
        # dictionary for storing Q-values
        Q = {} 
        for x in np.arange(0,x_size,1):
            for x_d in np.arange(0,xdot_size,1):
                for a in np.arange(0,a_size,1):
                    Q[x,x_d,a] = 0
        """            
         

        # dictionary for storing e-values
        e = {} 
        for x in np.arange(0,x_size,1):
            for x_d in np.arange(0,xdot_size,1):
                for a in np.arange(0,a_size,1):
                    e[x,x_d,a] = 0

        # ============================ core SARSA(lambda) algo ============================

        # init
        self.mc.reset()
        x_old = self.mc.x
        x_d_old = self.mc.x_d
        a_old = np.random.randint(0,3,1)[0]

        # run untill it gets the reward
        for i in range(n_steps):
            self.mc.apply_force(a_old-1)
            self.mc.simulate_timesteps(100,0.01) #take action a, observe

            #choose new action according to softmax policy
            p = self.softmax_policy(w, tau, sigmax, sigmay)
            a_selected = np.random.choice([0,1,2],1,p)[0]

            delta = self.mc.R + gamma*self.calculate_Q(self.mc.x, self.mc.x_d, a, w, sigmax, sigmay) - self.calculate_Q(x_old, x_d_old, a, w, sigmax, sigmay)            

            for keys,_ in w.items():
                w[keys] += eta*delta*e[keys]
                e[keys] = gamma*delta*e[keys]

            e[(x_old,x_d_old,a_old)] += 1
            x_old = mc.x
            x_d_old = mc.x_d
            a_old = a_selected
            
        # ============================ core SARSA(lambda) algo ============================
            
            if visualize:          
                # update the visualization
                mv.update_figure()
                plb.draw()
                plb.show()
                plb.pause(1e-7)
            
            # check for rewards
            if self.mc.R > 0.0:
                print ("\rreward obtained at t = ", self.mc.t)
                break
                
        if visualize:       
            plb.draw()
            plb.show()
            plb.waitforbuttonpress(timeout=3)

    def episodes(self, max_episodes=100, n_steps=200, lambda_=0.95, tau=0.05, eta=0.01):
        """run multiple episodes with 1 agent"""

        for n in range(max_episodes):
            print('\repisode =', n)           
            self.learn(n_steps, lambda_, tau, eta)


if __name__ == "__main__":
    d = DummyAgent()
    d.episodes()
    
    
    
