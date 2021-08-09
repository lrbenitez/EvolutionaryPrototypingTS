# EvolutionaryPrototypingTS
An Evolutionary Approach for Efficient Prototyping of Large Time Series Datasets

BRIEF EXPLANATION

## Main Features ##

Enumerating keywords

## Installation ##

pip ...

## Dependencies ##

* numpy
* pandas
* deap
* scikit-learn
* matplotlib

# Getting Started

Bellow there is a Quickstart Guide to EvolutionaryPrototypingTS.

## Basic example ##
 
Next piece of code shows a basic usage of the library. In it is shown how to import a file and x

.. code-block:: python

	>>> from ga_segments.ga import GA_segments
	>>> import pandas as pd
	
	>>> series = pd.read_csv('./data/50words_TRAIN', header=None).values[:, 1:]
	
	>>> ga = GA_segments()
	>>> centroid, best_fitness, log = ga.calculate_centroids(series)
	
	
Example of use of Nearest Centroid algorithm with GA-Segments:

.. code-block:: python

	>>> from ga_segments.nc import NC
	>>> import pandas as pd
	>>> from sklearn.model_selection import train_test_split
	
	>>> series = pd.read_csv('./data/50words_TRAIN', header=None)
	>>> x, y = series[:, 1:], series[:, 0]
	>>> x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
	
	>>> nc = NC()
	>>> nc.fit(x_train, y_train)
	>>> nc.predict(x_val)
	>>> nc.labels
  
  
 ## Citing ## 
 
 If eeglib has been useful in your research, please, consider citing the next article.

## Documents related ##

This library was initialy a Final Degree Project and you can find the documentation of the development in the next link:

* Final Degree Project Documentation (Spanish)

Later it was extented as part of a Master's thesis that can be found in the next link:



### Scientific papers ###

There are also some papers related to this library that can be seen bellow:

Characterisation of mobile-device tasks by their associated cognitive load through EEG data processing
eeglib: computational analysis of cognitive performance during the use of video games
