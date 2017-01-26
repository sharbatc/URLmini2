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

    def visualize_trial(self, n_steps=50):
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
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()


        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            sys.stdout.flush()
            
            ## we learnaction
            learn()


            # simulate the timestep
            #self.mountain_car.simulate_timesteps(100, 0.01)

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

    def softmax_policy(self,Q, temperature):
        Pa1 = np.exp(calculate_Q((mc.x,mc.x_d),0)/temperature)
        Pa2 = np.exp(calculate_Q((mc.x,mc.x_d),1)/temperature)
        Pa3 = np.exp(calculate_Q((mc.x,mc.x_d),2)/temperature)
        P_tot = Pa1 + Pa2 + Pa3
        return [Pa1/P_tot,Pa2/P_tot,Pa3/P_tot]

    def calculate_Q((x,xd),a):
        sum = 0
        for keys,weight in w.items():
            if keys[2]==a:
                sum+=weight*calculate_r(keys,(x,xd))
        return sum

    def calculate_r((xj,xdj,a),(x,xd),sigmax,sigmay):
        return np.exp(-((xj-x)/sigmax)**2-((xdj-xd)/sigmay)**2)



    def learn(self):
        # This is your job! 
        #We will look to implement the Sarsa(lambda) algorithm pseudocode Fig 7.11

        #input layer (place fields)
        x_size = 20
        xdot_size = 20
        sigmax = 180/x_size
        sigmay = 30/xdot_size
        # input_layer = np.zeros((x_size,xdot_size))

        #output layer
        a_size = 3
        output_layer = np.zeros(a_size)

        #weight matrix
        w = {}
        w_ini = 0
        for a in np.arange(0,a_size,1):
            for x in np.arange(0,x_size,1):
                for x_d in np.arange(0,xdot_size,1):
                    w[(a,x,x_d)] = w_ini

        #dictionary for storing Q-values
        Q = {} 
        for i in np.arange(0,x_size,1):
            for j in np.arange(0,xdot_size,1):
                for k in np.arange(0,a_size,1):
                    Q[(i,j,k)] = 0
                    
        #dictionary for storing e-values
        e = {} 
        for i in np.arange(0,x_size,1):
            for j in np.arange(0,xdot_size,1):
                for k in np.arange(0,a_size,1):
                    e[(i,j,k)] = 0



        mc = self.mountain_car.reset
        x_old = mc.x
        x_d_old = mc.x_d
        a_old = np.random.randint(0,3,1)[0]

        for i in range(n_steps):
            mc.apply_force(a_old-1)
            mc.simulate_timesteps(100,0.01) #take action a, observe

            #choose new action according to softmax policy
            p = softmax_policy()
            a_selected = np.random.choice([0,1,2],1,p)[0]

            delta = mc.R + gamma*Q[(mc.x,mc.x_d,a_selected)] - Q[(x_old,x_d_old,a)]
            e[(x_old,x_d_old,a)] += 1

            for keys,_ in Q.items():
                w[keys] += eta*delta*e[keys]
                e[keys] = gamma*delta*e[keys]
            x_old = mc.x
            x_d_old = mc.x_d
            a_old = a_selected










if __name__ == "__main__":
    d = DummyAgent()
    d.visualize_trial()
    plb.show()
