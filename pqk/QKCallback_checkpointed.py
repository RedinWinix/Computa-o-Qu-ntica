import datetime
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import json



#calback function
class QKCallback:
    
    num_iteration = 0
   

    def __init__(self, checkpoint_path: str = None, checkpoint_every: int = 1) -> None:
          self._data = [[] for i in range(5)]
          # checkpoint_path: full path to a JSON file that gets overwritten every
          # `checkpoint_every` iterations with the latest parameters. If None,
          # no live checkpointing happens (original behaviour).
          self.checkpoint_path = checkpoint_path
          self.checkpoint_every = max(1, checkpoint_every)

    #callback function  
    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
            
            """
            Args:
                x0: number of function evaluations
                x1: the parameters
                x2: the function value
                x3: the stepsize
                x4: whether the step was accepted
            """

            self.num_iteration +=1

            print(f'**********************')
            print(f'Print callback. Iteration {self.num_iteration}')
            print(f'Number of function evaluations: {x0}')
            print(f'The paramenters: {x1}')
            print(f'The function value: {x2}')
            print(f'The stepsize: {x3}')
            print(f'Whether the step was accepted: {x4}')
            print(f'**********************')

            self._data[0].append(x0)
            self._data[1].append(x1.tolist())
            self._data[2].append(x2)
            self._data[3].append(x3)
            self._data[4].append(x4)

            #write a live checkpoint to disk so a crash/interrupt doesn't lose
            #hours of training - overwrites the same file each time (cheap: a
            #handful of floats), and is written atomically (tmp file + rename)
            #so a checkpoint is never left half-written.
            if self.checkpoint_path is not None and self.num_iteration % self.checkpoint_every == 0:
                self._write_checkpoint(x0, x1, x2, x3, x4)

            #return True if you want stop the training
            stop_training = False
            return stop_training

    def _write_checkpoint(self, x0, x1, x2, x3, x4):
        checkpoint = {
            'iteration': self.num_iteration,
            'num_function_evaluations': int(x0),
            'parameters': x1.tolist(),
            'function_value': float(x2),
            'stepsize': float(x3) if x3 is not None else None,
            'accepted': bool(x4) if x4 is not None else None,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        tmp_path = self.checkpoint_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(checkpoint, f, indent=3)
        os.replace(tmp_path, self.checkpoint_path)  # atomic on POSIX and Windows

    @staticmethod
    def load_checkpoint(checkpoint_path: str):
        '''
        Load a checkpoint written by _write_checkpoint. Returns None if the
        file doesn't exist (e.g. first run, nothing to resume).
        '''
        if not os.path.exists(checkpoint_path):
            return None
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    
    def plot_data(self):
        QKCallback._plot_data(self._data)

    #plot data from list of data
    def _plot_data(data):
        plt.rcParams["font.size"] = 20          
        plt.plot([i + 1 for i in range(len(data[0]))], np.array(data[2]), c="k", marker="o")
        plt.show()


    #save my feature map
    def save(self, prefix = ''):
        #create a csv file with feature maps
        current_timestamp = time.time()
        datetime_object = datetime.datetime.fromtimestamp(current_timestamp)
        formatted_datetime = datetime_object.strftime("%Y%m%d%H%M%S")
        csv_file = '../qfm/callback/' + prefix + str(formatted_datetime) + '.json'

        json_str = json.dumps(self._data, indent= 3)        

        main_path = os.path.dirname(__file__)
        file_path = os.path.join(main_path, csv_file)

        #store the features map
        with open(file_path, 'w') as f:
            f.write(json_str)

    def plot_data_file(file = ''):
        data_to_plot = None
        
        # open file and read the content in a list
        with open(file, 'r') as _file:
            content = _file.read()
            data_to_plot = json.loads(content)
                   
        QKCallback._plot_data(data_to_plot)



if __name__ == '__main__':
    ##check the reader    
    QKCallback.plot_data_file(file='qfm/callback/callback_20240704230145.json')
     
     




        
                   
                   

            
            
    

