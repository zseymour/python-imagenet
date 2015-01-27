# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:57:23 2015

@author: zach
"""
from nltk.corpus import wordnet as wn
import pandas as pd

import itertools
import os
import pickle
import random

SYNSET_FORMAT = "n{:0>8}"

class ImageHierarchy(object):
    
    def __init__(self, image_dir, caffe_root, dataset="val"):
        self.IMAGE_DIR = image_dir
        self.CAFFE_ROOT = caffe_root
        self.dataset = dataset
        bet_file = pickle.load(open(os.path.join(self.CAFFE_ROOT, "data/ilsvrc12/imagenet.bet.pickle"), "rb"))
        self.words = bet_file['words']
                
        with open(os.path.join(self.CAFFE_ROOT, "data/ilsvrc12/synset_words.txt")) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')
        
        with open(os.path.join(self.CAFFE_ROOT,"data/ilsvrc12",dataset+".txt")) as f:
            image_df = pd.DataFrame([
                {
                    'file_name': l.strip().split(' ')[0],
                    'label': int(' '.join(l.strip().split(' ')[1:]).split(',')[0])
                }
                for l in f.readlines()
            ])
        self.images = image_df.sort('file_name').join(self.labels, on="label")
        
    def is_visual(self, word):
       return len(self.get_synsets(word)) > 0
       
    def get_synsets(self,word):
        return wn.synsets(word,pos=wn.NOUN)
        
    def get_synset_id(self, wn_synset):
        return SYNSET_FORMAT.format(wn_synset.offset())
    
    def get_leaf_synsets(self, synset):
        synsets_to_check = [synset]
        
        while synsets_to_check:
            synset = synsets_to_check.pop()
            children = synset.hyponyms()
            if children:
               synsets_to_check.extend(children)
            else:
                yield synset   
        
    def fetch_images(self, word, number_of_images=100):
        synset = self.get_synsets(word)[0]
        
        all_images = list(itertools.chain(*[self.fetch_images_for_synset(s) for s in self.get_leaf_synsets(synset)]))
        all_images = map(self.make_image_absolute, all_images)
        if len(all_images) < number_of_images:
            return all_images
        else:
            return random.sample(all_images, number_of_images)
        
    def fetch_images_for_synset(self, synset):
        synset_id = self.get_synset_id(synset)
        return list(imagenet.images.loc[imagenet.images["synset_id"] == synset_id].file_name)
        
    def make_image_absolute(self, file_name):
        return os.path.join(self.IMAGE_DIR, file_name)
        
if __name__ == "__main__":
    imagenet = ImageHierarchy('/home/zach/Downloads/imagenet/val','/home/zach/caffe')

