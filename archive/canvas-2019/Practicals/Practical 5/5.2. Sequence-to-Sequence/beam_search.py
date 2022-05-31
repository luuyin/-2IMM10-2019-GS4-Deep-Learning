"""This file contains code to run beam search decoding"""

import tensorflow as tf
import numpy as np
import data


class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, log_probs):
    """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
    """
    self.tokens = tokens
    self.log_probs = log_probs
    

  def extend(self, token, log_prob):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob])

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)

  
def decode_one_step(cap_inputs, img_feats, decoder_model, rnn_dim, len_hyps, beam_size, words_indices, indices_words):
  
  topk_probs = []
  topk_ids = []
  
  for i in range(len_hyps):
    
    cap_input = cap_inputs[i]
    img_feat = img_feats[i]
    
 
    pred_prob_t = decoder_model.predict([cap_input] + [img_feat] )
   
    topk_prob = np.sort(pred_prob_t[0][0], axis=-1)[-(beam_size):][::-1]
    
    topk_logprob = -np.log(topk_prob)
        
    topk_id = np.argsort(pred_prob_t[0][0], axis=-1)[-(beam_size):][::-1]
    
    topk_probs.append(topk_logprob)
    topk_ids.append(topk_id)
    
  return topk_probs, topk_ids
  
def run_beam_search(encoder_model, decoder_model, words_indices, indices_words, enc_in_seq, \
                    min_dec_steps, max_dec_steps, rnn_dim,  num_hyps, beam_size, filepath):
  
  #Performs beam search decoding on the given example.
  
  
  encoder_output = encoder_model.predict(enc_in_seq)
 

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[words_indices.get('startseq')],
                     log_probs=[0.0]                      
                     # zero vector of length attention_length
                     ) for _ in range(num_hyps)]
  
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)

  steps = 0
  while steps < max_dec_steps and len(results) < beam_size:
    
    latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
    
    len_hyps = len(latest_tokens)
    
    img_feats = np.array([encoder_output] * len_hyps)
    
    
    dec_inputs = latest_tokens
    dec_inputs = np.array(dec_inputs)
    dec_inputs = np.reshape(dec_inputs, (dec_inputs.shape[0], 1))
    
    
    # Run one step of the decoder to get the new info
    topk_log_probs, topk_ids = decode_one_step(
                        dec_inputs, img_feats, decoder_model, rnn_dim, len_hyps, beam_size, words_indices, indices_words)

    topk_log_probs = np.array(topk_log_probs)
    topk_ids = np.array(topk_ids)
    
    
    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    for i in range(num_orig_hyps):
      h = hyps[i]  # take the ith hypothesis and new decoder state info
      for j in range(beam_size):  # for each of the top 2*beam_size hyps:
        # Extend the ith hypothesis with the jth option
        
        logprobs= np.array(topk_log_probs[i,j])
        new_hyp = h.extend(token=topk_ids[i,j],
                           log_prob=logprobs)
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
    for h in sort_hyps(all_hyps): # in order of most likely h
      if h.latest_token == words_indices.get('endseq'): # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= min_dec_steps:
          results.append(h)
      else: # hasn't reached stop token, so continue to extend this hypothesis
        hyps.append(h)
      if len(hyps) == beam_size or len(results) == beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1

  # At this point, either we've got beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results)

  # Return the hypothesis with highest average log prob
  return hyps_sorted

def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=False)
