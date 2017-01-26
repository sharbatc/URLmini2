import sys

import pylab as plb
import numpy as np
import mountaincar

class DummyAgent(): #name to be changed later
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, parameter1 = 3.0): #unchanged from given
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

    def visualize_trial(self, n_steps=200,lambda=0.95,tau=0.05,eta=0.01):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
        plb.show()
        plb.pause(1e-7)
            
        max_episodes = 100
        # make sure the mountain-car is reset - 1 agent (have to do 10 times)
        self.mountain_car.reset()

        # below - episodes for 1 agent
        for n in range(max_episodes):
            #print('\rt =', self.mountain_car.t)
            print('\repisode =', n)
            sys.stdout.flush()
            
            ## we learn action
            learn(n_steps,lambda,tau,eta)

            # update the visualization
            mv.update_figure()
            plb.draw()
            plb.show()
            plb.pause(1e-7)
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print ("\rreward obtained at t = ", self.mountain_car.t)
                break
                
        plb.draw()
        plb.show()
        plb.waitforbuttonpress(timeout=3)

    def softmax_policy(self,tau):
        Pa1 = np.exp(calculate_Q((mc.x,mc.x_d),0)/tau)
        Pa2 = np.exp(calculate_Q((mc.x,mc.x_d),1)/tau)
        Pa3 = np.exp(calculate_Q((mc.x,mc.x_d),2)/tau)
        P_tot = Pa1 + Pa2 + Pa3
        return [Pa1/P_tot,Pa2/P_tot,Pa3/P_tot]

    def calculate_Q((x,xd),a):
        sum = 0
        for keys,weight in w.items():
            if keys[2]==a:
                sum+=weight*calculate_r(keys,(x,xd),sigmax,sigmay)

        return sum

    def calculate_r((xj,xdj,a),(x,xd),sigmax,sigmay):
        return np.exp(-((xj-x)/sigmax)**2-((xdj-xd)/sigmay)**2)



    def learn(self, n_steps=200,lambda=0.95,tau=0.05,eta=0.01):
        #We will look to implement the Sarsa(lambda) algorithm pseudocode Fig 7.11

        #input layer (place fields)
        x_size = 10
        xdot_size = 10
        sigmax = 180/x_size
        sigmay = 30/xdot_size
        # input_layer = np.zeros((x_size,xdot_size))

        #output layer
        a_size = 3
        output_layer = np.zeros(a_size)

        #weight dictionary matrix
        w = {}
        w_ini = 0
        for a in np.arange(0,a_size,1):
            for x in np.arange(0,x_size,1):
                for x_d in np.arange(0,xdot_size,1):
                    w[(a,x,x_d)] = w_ini

        #dictionary for storing Q-values
        Q = {} 
        for x in np.arange(0,x_size,1):
            for x_d in np.arange(0,xdot_size,1):
                for a in np.arange(0,a_size,1):
                    Q[(x,x_d,a)] = 0

        #dictionary for storing e-values
        e = {} 
        for x in np.arange(0,x_size,1):
            for x_d in np.arange(0,xdot_size,1):
                for a in np.arange(0,a_size,1):
                    e[(x,x_d,a)] = 0



        mc = self.mountain_car.reset
        x_old = mc.x
        x_d_old = mc.x_d
        a_old = np.random.randint(0,3,1)[0]

        for i in range(n_steps):
            mc.apply_force(a_old-1)
            mc.simulate_timesteps(100,0.01) #take action a, observe

            #choose new action according to softmax policy
            p = softmax_policy(tau)
            a_selected = np.random.choice([0,1,2],1,p)[0]

            delta = mc.R + gamma*Q[(mc.x,mc.x_d,a_selected)] - Q[(x_old,x_d_old,a)]
            

            for keys,_ in Q.items():
                w[keys] += eta*delta*e[keys]
                e[keys] = gamma*delta*e[keys]

            e[(x_old,x_d_old,a_old)] += 1
            x_old = mc.x
            x_d_old = mc.x_d
            a_old = a_selected

if __name__ == "__main__":
    d = DummyAgent()
    d.visualize_trial()
    plb.show()
