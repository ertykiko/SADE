from mesa import Agent, Model
from mesa.time import RandomActivation

#https://mesa.readthedocs.io/en/master/tutorials/intro_tutorial.html
#https://towardsdatascience.com/introduction-to-mesa-agent-based-modeling-in-python-bcb0596e1c9a


class MachineAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.age = 0

    def step(self):
        # The agent's step will go here.
        # Increase it's age 
        age += 1

        print("Hi, I am machine  " + str(self.unique_id) + ".")


class MachineModel(Model):
    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = MachineAgent(i, self)
            self.schedule.add(a)
    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()

#--------


emptyModel = MachineModel(100)
emptyModel.step()
