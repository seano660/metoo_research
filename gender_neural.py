import json
import os
import re
import sys

import numpy as np

from demographer.utils import tf
from demographer.utils import download_pretrained_models
from demographer.demographer import Demographer
from demographer.tflm import SymbolTable
from demographer.naacl_twitter import NaaclTwitter
from demographer.gender import load_name_dictionary


class NeuralGenderDemographer(Demographer):
  name_key = 'gender_neural'

  def __init__(self, model_dir=None, dtype='n', use_name_dictionary=True):
    """Initalizes a class for an Gender neural classifier

    Constructor needs to load a classifier from tensorflow

    Args:
        model_dir: where does the model live?

    """
    self.dtype = dtype
    if not model_dir:
      dir = os.path.dirname(sys.modules['demographer'].__file__)
      model_dir = os.path.join(dir, 'models', 'mw_neural')
      if not os.path.exists(model_dir):
          assert download_pretrained_models(os.path.join(dir, 'models'), 'mw_neural')

    self.name_dictionary = None

    self.graph = tf.Graph()
    with self.graph.as_default():
      saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_dir, 'model.meta'))
      self.sess = tf.compat.v1.Session()
      saver.restore(self.sess, os.path.join(model_dir, 'model'))

      self.inputs = self.graph.get_tensor_by_name('inputs:0')
      self.logits = self.graph.get_tensor_by_name('logits:0')

    if use_name_dictionary:
      self.asciiex = re.compile(r'[a-zA-Z]+')
      self.name_dictionary = load_name_dictionary(None)

    with open(os.path.join(model_dir, "vocab.json"), "r") as inf:
      obj = json.loads(inf.readline())
      self._vocab = SymbolTable.from_json(obj)
      self._pad = NaaclTwitter.PAD
      self._split = NaaclTwitter.SPLIT
      self._max_name_length = 32  # this was set dynamically by data length

    with open(os.path.join(model_dir, "labels.json"), "r") as inf:
      obj = json.loads(inf.readline())
      self._labels = SymbolTable.from_json(obj)

  def encode(self, name, screen=None):
    tokens = self._encode(name)
    if screen is not None:
      tokens.append(self._vocab.idx(self._split))
      tokens += self._encode(screen)
    tokens += [self._vocab.idx(self._pad)
               for i in range(self._max_name_length - len(tokens))]
    return tokens[:self._max_name_length]

  def _encode(self, name):
    return [self._vocab.idx(char) for char in name]

  def process_tweet(self, tweet):
    if 'user' in tweet:
      user_info = tweet.get('user')
    else:
      user_info = tweet

    name_string = user_info.get('name')
    if self.dtype == 'ns':
      screen_string = user_info.get('screen')
    else:
      screen_string = None

    if self.name_dictionary:
      match = self.asciiex.search(name_string.split(' ')[0])
      if match and match.group(0).lower() in self.name_dictionary:
        firstname = match.group(0).lower()
        w_ct = self.name_dictionary[firstname]['F']
        m_ct = self.name_dictionary[firstname]['M']
        tot = (w_ct + m_ct) * 1.0
        prob_w = w_ct / tot
        prob_m = m_ct / tot
        if prob_w > prob_m:
          value = "woman"
        else:
          value = "man"

        result = {
            "value": value, "name": "gender",
            "annotator": "Gender Namelist",
            "scores": {"woman": prob_w, "man": prob_m}}
        return {self.name_key: result}

    # get logits from saved model
    inputs = self.encode(name_string, screen_string)
    inputs = np.expand_dims(inputs, 0)
    logits = self.sess.run(self.logits, {self.inputs: inputs})
    logits = np.squeeze(logits)

    # find probabilities over labels
    label_probs = {}
    logits_sum = np.sum(logits)
    for label_i in range(len(self._labels)):
      label = self._labels._i2v[str(label_i)]
      label_probs[label] = logits[label_i] / logits_sum
    prediction = self._labels._i2v[str(np.argmax(logits))]

    result = {"value": prediction,
              "name": "gender",
              "annotator": "Neural Gender Classifier",
              "scores": label_probs}

    return {self.name_key: result}