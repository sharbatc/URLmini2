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
            
            ## choose a random action, here we change to learned action
            #self.mountain_car.apply_force(np.random.randint(3) - 1)
            self.mountain_car = 



            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

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
        Pa1 = np.exp(Q[(mc.x,mc.x_d),a]/temperature)
        Pa2 = np.exp(Q)



    def learn(self):
        # This is your job! 
        #We will look to implement the Sarsa(lambda) algorithm pseudocode Fig 7.11

        #input layer (place fields)
        x_size = 20
        xdot_size = 20
        input_layer = np.zeros((x_size,xdot_size))        

        #output layer
        a_size = 3
        output_layer = np.zeros(a_size)

        #weight matrix
        w = {}
        w_ini = 0
        for a in np.arange(0,a_size,1):
            for i in np.arange(0,x_size,1):
                for j in np.arange(0,xdot_size,1):
                    w[(a,i,j)] = w_ini

        #dictionary for storing Q-values
        Q = {} 
        for i in np.arange(0,x_size,1):
            for j in np.arange(0,xdot_size,1):
                for k in np.arange(0,a_size,1):
                    Q[(i,j,k)] = 0



        for i in range(n_episode):
            mc = self.mountain_car
            mountain_car.x, mountain_car.x_d



if __name__ == "__main__":
    d = DummyAgent()
    d.visualize_trial()
    plb.show()
