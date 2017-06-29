from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy.stats
import math
import sys


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    gaussianTh = 100  # if attr has more possible values than th, use gaussian distribution
    useBagOfWord = False  # currently not used
    likelihood_matrix_default_ = dict()
    likelihood_matrix_ = dict()
    likelihood_paras_ = dict()

    def __init__(self, gaussianTh=100, useBagOfWord=False):
        self.gaussianTh = gaussianTh
        self.useBagOfWord = useBagOfWord

    def fit(self, X, y=None):
        # generate p(class)
        classNames, classCounts = np.unique(y, return_counts=True)
        self.p_class_ = zip(classNames, classCounts / (len(y) + 0.0))

        # transverse features, determine use gaussian(1) or not(0)
        self.use_gaussian_ = np.zeros(len(X))
        X = np.transpose(X)
        y = np.transpose(y)
        for i in range(len(X)):
            # generate (class, all feature values list)
            classFeaturePair = dict()
            for (xn, yn) in zip(X[i], y):
                if yn in classFeaturePair:
                    classFeaturePair[yn] = np.append(classFeaturePair[yn], xn)
                else:
                    classFeaturePair[yn] = np.array([xn])
            # generate likelihood matrix for discrete features,
            # gaussian parameters for continuous features
            _, featureCounts = np.unique(X[i], return_counts=True)
            if len(featureCounts) > self.gaussianTh:
                self.use_gaussian_[i] = 1
                for className in self.p_class_:
                    featureValues = classFeaturePair[className[0]]
                    mean = np.mean(featureValues)
                    std = np.std(featureValues)
                    self.likelihood_paras_[(i, className[0])] = (mean, std)
            else:
                self.use_gaussian_[i] = 0
                for className in self.p_class_:
                    featureValues = classFeaturePair[className[0]]
                    classFeatureNames, classFeatureCounts = np.unique(featureValues, return_counts=True)
                    # use laplace smoothing
                    self.likelihood_matrix_default_[(i, className[0])] = 1.0 / (len(featureValues) + len(classFeatureNames) + 1.0)
                    for (classFeatureName, likelihood) in zip(classFeatureNames, (classFeatureCounts + 1.0) / (len(featureValues)  + len(classFeatureNames) + 0.0)):
                        self.likelihood_matrix_[(i, className[0], classFeatureName)] = likelihood
        return self

    def _meaning(self, x):
        # returns class name according to fitted classifier
        # notice underscore on the beginning
        maxLikelihood = -sys.maxint - 1
        res = self.p_class_[0][0]
        for classK in self.p_class_:
            totalLikelihood = 0
            for i in range(len(x)):
                if 1 == self.use_gaussian_[i]:
                    paras = self.likelihood_paras_[(i, classK[0])]
                    confRange = np.array([-1.96, 1.96]) * paras[1] + paras[0]
                    if x[i] < confRange[0] or x[i] > confRange[1]:
                        likelihood = self.gaussian(paras[0], paras[1], confRange[0])
                        # likelihood = scipy.stats.norm(paras[0], paras[1]).pdf(confRange[0])
                    else:
                        likelihood = self.gaussian(paras[0], paras[1], x[i])
                        # likelihood = scipy.stats.norm(paras[0], paras[1]).pdf(x[i])
                else:
                    if (i, classK[0], x[i]) in self.likelihood_matrix_:
                        likelihood = self.likelihood_matrix_[(i, classK[0], x[i])]
                    else:
                        likelihood = self.likelihood_matrix_default_[(i, classK[0])]
                totalLikelihood += math.log(likelihood)
            totalLikelihood += math.log(classK[1])
            if totalLikelihood > maxLikelihood:
                res = classK[0]
                maxLikelihood = totalLikelihood
        return res

    def gaussian(self, miu, delta, x):
        x = (x - miu) / delta
        return (math.exp(-x*x/2) / (math.sqrt(2 * math.pi))) / delta

    def predict(self, X, y=None):
        try:
            getattr(self, "likelihood_matrix_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return([self._meaning(x) for x in X])

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X)))