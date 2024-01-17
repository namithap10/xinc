from builtins import breakpoint
import torch 
from collections import OrderedDict

"""
    Class to do direct add and subtract operations on state dicts.

"""

class StateDictOperator():
    def __init__(self,state_dict):

        if (type(state_dict) is OrderedDict) or (type(state_dict) is dict):
            self.state_dict = state_dict
        else:
            raise TypeError("StateDictOperator only accepts dicts as input")
        
        self.frame_number = None #optional param that attributes a state dict to a particular frame.
        self.quant_bit = None #optional 

    def to_device(self,device):
        for key in self.state_dict.keys():
            self.state_dict[key] = self.state_dict[key].to(device)

    def equal(self,compare_state):

        if type(compare_state) is StateDictOperator:
            compare = compare_state.state_dict
        else:
            compare = compare_state

        for idx,(key, value) in enumerate(self.state_dict.items()):
            if torch.all(self.state_dict[key]!=compare[key]):
                return False
        return True


    def multiply_constant(self,constant):
        current_state = {}
        for idx,(key, value) in enumerate(self.state_dict.items()):
            current_state[key] = value * constant
        return current_state

    def add(self,compare_state):

        if type(compare_state) is StateDictOperator:
            compare = compare_state.state_dict
        else:
            compare = compare_state
        
        current_state = {}
        for idx,(key, value) in enumerate(self.state_dict.items()):
            current_state[key] = value + compare[key]
        
        if type(compare_state) is StateDictOperator:
            return StateDictOperator(current_state)

        return current_state

    def subtract(self,compare_state,strict=True):
        if type(compare_state) is StateDictOperator:
            compare = compare_state.state_dict
        else:
            compare = compare_state

        current_state = {}
        for idx,(key, value) in enumerate(self.state_dict.items()):
            try:
                current_state[key] = value - compare[key]
            except:
                if not strict:
                    """
                        Retaining current state.
                    """
                    current_state[key] = value
        
        if type(compare_state) is StateDictOperator:
            return StateDictOperator(current_state)

        return current_state

    def multiply(self,compare_state):

        if type(compare_state) is float or type(compare_state) is int:
            current_state = self.multiply_constant(compare_state)
            return StateDictOperator(current_state)

        if type(compare_state) is StateDictOperator:
            compare = compare_state.state_dict
        else:
            compare = compare_state

        current_state = {}
        for idx,(key, value) in enumerate(self.state_dict.items()):
            current_state[key] = value * compare[key]
        
        if type(compare_state) is StateDictOperator:
            return StateDictOperator(current_state)

        return current_state

    def divide(self,compare_state):

        if type(compare_state) is StateDictOperator:
            compare = compare_state.state_dict
        else:
            compare = compare_state

        current_state = {}
        for idx,(key, value) in enumerate(self.state_dict.items()):
            current_state[key] = value / compare[key]
        
        if type(compare_state) is StateDictOperator:
            return StateDictOperator(current_state)

        return current_state

    def filter(self,value):

        """
            Make all abs(entries) less than value zero.
        """

        current_state = {}
        for idx,(key, v) in enumerate(self.state_dict.items()):
            #current_state[key] = value * (value>value)
            current_state[key] = torch.where(torch.abs(v)<=value,torch.zeros_like(v),v)
        
        self.state_dict = current_state
        return StateDictOperator(current_state)

    
    def __eq__(self,compare_state) -> bool:
        return self.equal(compare_state)

    def __call__(self,compare_state):
        return self.add(compare_state)
    
    def __add__(self,compare_state):
        return self.add(compare_state)
    
    def __sub__(self,compare_state):
        return self.subtract(compare_state)
    
    def __mul__(self,compare_state):
        return self.multiply(compare_state)
    
    def __div__(self,compare_state):
        return self.divide(compare_state)
    
    def __radd__(self,compare_state):
        return self.add(compare_state)
    
    def __rsub__(self,compare_state):
        return self.subtract(compare_state)
    
    def __rmul__(self,compare_state):
        return self.multiply(compare_state)
    
    def __rdiv__(self,compare_state):
        return self.divide(compare_state)
    

