"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import pandas as pd

class AnalogyStats:

    def __init__(self):

        self.start = []
        self.start_label = []
        self.end = []
        self.end_label = []
        self.start_pair = []
        self.start_pair_label = []
        self.end_pair = []
        self.end_pair_label = []
        self.depth = []
        self.nature = []
        self.target = []
        self.confidence = []
        self.final_decision = []
    
    def append_stats_in_generation(self, start, start_label, end, end_label, start_pair, start_pair_label, end_pair, end_pair_label, depth_diff, nature, number):
        for _ in range(number):
            self.start.append(start)
            self.start_label.append(start_label)
            self.end_label.append(end_label)
            self.start_pair.append(start_pair)
            self.start_pair_label.append(start_pair_label)
            self.end_pair.append(end_pair)
            self.end_pair_label.append(end_pair_label)
            self.end.append(end)
            self.depth.append(depth_diff)
            self.nature.append(nature)

    def append_stats_in_voting(self, target, final_decision, number):
        for _ in range(number):
            self.target.append(target)
            self.final_decision.append(final_decision)

    def save_stats(self, path):

        analogies_df = pd.DataFrame({'start': self.start, 
                                     "start label": self.start_label, 
                                     'end': self.end, 
                                     "end label": self.end_label, 
                                     "target": self.target,
                                     'depth difference': self.depth,
                                     'start pair': self.start_pair,
                                     'start pair label': self.start_pair_label,
                                     'end pair': self.end_pair,
                                     'end pair label': self.end_pair_label,
                                     "nature": self.nature,
                                     'confidence': self.confidence,
                                     'analogy decision': self.final_decision})
        
        analogies_df.to_csv(path, header=True, index=False)

