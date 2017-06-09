# -*- mode: Python; coding: utf-8 -*-
'''
Hannah Provenza
CS 134A Statistical Methods in NLP
Programming Assignment
Naïve Bayes Classifier
'''

from classifier import Classifier
from numpy import product
from corpus import Document, BlogsCorpus, NamesCorpus
import sys
import math

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""
    def __init__(self, model={}):
        try:
            self.labels = model["labels"]
            self.feature_counts = model["feature_counts"]
            self.features = set(model["features"])
        except:
            self.labels = {}
            self.feature_counts = {}
            self.features = set()

    def labels(self):
        return self.labels.keys()

    def features(self):
        return self.model["features"]

    def model(self):
        return {"labels": self.labels, "feature_counts": self.feature_counts, "features": self.features}

    def train(self, instances):
        # get counts
        print "Number of instances is: " + str(len(instances))
        for instance in instances:
            try:
                self.labels[instance.label] += 1
            except KeyError:
                self.labels[instance.label] = 1
                self.feature_counts[instance.label] = {}
            for feature in set(instance.features()):
                try:
                    self.feature_counts[instance.label][feature] += 1
                except KeyError:
                    self.feature_counts[instance.label][feature] = 1

        features = set([item for label in self.labels.keys() for item in self.feature_counts[label].keys()])
        self.features = features

        # add +1 smoothing to all label-feature combinations
        for label in self.labels:
            for feature in features:
                try:
                    self.feature_counts[label][feature] += 1
                except:
                    self.feature_counts[label][feature] = 1

        # calculate log probabilities of features
        total_feature_count = sum([self.feature_counts[label][feature] for feature in self.features])
        for label in self.feature_counts.keys():
            for feature in self.feature_counts[label].keys():
                self.feature_counts[label][feature] = math.log(float(self.feature_counts[label][feature]) / total_feature_count)


        # calculate log probabilities of labels
        for label in self.labels:
            self.labels[label] = math.log(float(self.labels[label]) / total_feature_count)
        # save model
        # self.model = {"labels": self.labels, "feature_counts": self.feature_counts, "features": self.features}
        # self.save("half_smoothing_NB.model")
        print("done training")

    def calc_likelihood(self, instance, label):
        probs = [self.feature_counts[label][feature] for feature in set(instance.features()) if feature in self.features]
        return sum(probs)

    def calc_prior(self, label):
        return self.labels[label]

    def calc_posterior(self, label, instance):
        return self.calc_likelihood(instance, label) + self.calc_prior(label)

    def classify(self, instance):
        return max(self.labels, key=lambda x: self.calc_posterior(x, instance))