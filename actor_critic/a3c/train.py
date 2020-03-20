import torch
import torch.multiprocessing as mp
from actor_critic.actor_critic import ActorCritic
from actor_critic.actions import get_action_space
from actor_critic.a3c.worker import Worker
#NEW
import pickle

class A3CTrainer:
    def __init__(self, params, model_path):
        self.params = params
        self.model_path = model_path
        self.num_of_processes = mp.cpu_count()
        self.global_model = ActorCritic(self.params.stack_size,
                                        get_action_space())
        self.global_model.share_memory()

    def run(self):
        processes = []
        for process_num in range(self.num_of_processes):
            worker = Worker(process_num, self.global_model, self.params, autosave = True)# NEW
            
            processes.append(worker)
            worker.start()

        torch.save(self.global_model.state_dict(), self.model_path)
        # CHANGE
        for i, process in enumerate(processes):
            
            #NEW
            pickle.dump(process.log_loss, open('models/log_loss{}.pkl'.format(i), 'wb'))
            #NEW
            pickle.dump(process.log_reward, open('models/log_reward{}.pkl'.format(i), 'wb'))

            process.join()


        
