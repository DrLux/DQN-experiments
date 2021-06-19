from pathlib import Path
import tracemalloc

def make_dir(dirpath):
  Path(dirpath).mkdir(parents=True,exist_ok=True)

  
class Chronometer():
  def __init__(self):
    self.result = dict()  

  def set_checkpoint(self, cname):
    self.result[cname] = time.time()  

  def take_time(self,cname):
    self.result[cname] = time.time() - self.result[cname] 

  def show_result(self):
    for k,v in self.result.items():
      print('Elapsed time for {} = {:1.5f}'.format(k,v))

  
class Profiler():
  def __init__(self):
    self.result = dict()  

  def trace(self,cname):
    memory_used, _ = tracemalloc.get_traced_memory()
    memory_used = memory_used / 10 ** 6
    self.result[cname] = memory_used  

  def show_result(self):
    for k,v in self.result.items():
      print('Memory used for {} = {:1.5f} MB'.format(k,v))
