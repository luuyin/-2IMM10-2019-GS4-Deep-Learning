import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

class Dataiterator():

    def __init__(self, img_in, cap_in, cap_out, vocab_size=1651, seq_length=34, decoder_dim=300, batch_size=32):
        
        self.img_in = img_in
        self.cap_in = cap_in 
        self.cap_out = cap_out
        self.states = np.zeros((len(img_in), decoder_dim)) 
        self.num_data = len(img_in) 
        self.vocab_size = vocab_size
        self.batch_size = batch_size 
        self.seq_length = seq_length
        self.reset() # initial: shuffling examples and set index to 0
        
    
    def onehotencoding(self, data):
                 
        return to_categorical(data, num_classes=self.vocab_size+1, dtype='int32')
    
    def __iter__(self): # iterates data
        
        return self


    def reset(self): # initials
        
        self.idx = 0
        self.order = np.random.permutation(self.num_data) # shuffling examples by providing randomized ids 
        
    def __next__(self): # return model inputs - outputs per batch
        
        X_ids = [] # hold ids per batch 

        while len(X_ids) < self.batch_size:

            X_id = self.order[self.idx] # copy random id from initial shuffling
            X_ids.append(X_id)

            self.idx += 1 # 
            if self.idx >= self.num_data: # exception if all examples of data have been seen (iterated)
                self.reset()
                raise StopIteration()
    
        batch_img_in = self.img_in[np.array(X_ids)] # X values (encoder input) per batch
        batch_cap_in = self.cap_in[np.array(X_ids)] # y_in values (decoder input) per batch
        batch_cap_out = self.cap_out[np.array(X_ids)]
        batch_states = self.states[np.array(X_ids)] # state values (decoder state input) per batch
        batch_y = self.onehotencoding(batch_cap_out)
        
     
        
        return batch_img_in, batch_cap_in, batch_states, list(batch_y.swapaxes(0,1))

    # return all data examples 
    def all(self):
      
        y = self.onehotencoding(self.cap_out)
        
        return self.img_in, self.cap_in, self.states, list(y.swapaxes(0,1))

