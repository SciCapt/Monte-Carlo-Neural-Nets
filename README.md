# Monte-Carlo-Neural-Nets

## Overview / Lore

A package made to see how well a basic architecture of neural nets could be. The nets can be created with custom input and output heights, as well as full customization of the hidden layers' sizes (height and count of layers) and activation functions that are applied at each space between the layers.

The basic operation of these nets is that they can be trained to some data set (or some score in an unsupervised case) by randomly 'tweaking' the different parameter weight values within the net. These weight values are clipped to be restrained to the range [-1, 1].

If training a net in a supervised fashion, the training function, GenTrain(), iteratively creates 'generations' or 'batches' of nets from some parent net, and uses the one with the most improvement as the next parent net. This effectively creates a training for the nets via gradient decent.

This is on PyPI, view the latest release at:
https://pypi.org/project/mcnets/


## Quickstart
### Example Code (Fit to f(x) = ln(x))
Since V0.2.0, the activation functions can be fully customized per layer in the created nets. This allows for very easy curve fitting applications such as the example below. Note that the section "Curve Fitting Examples" uses two outdated methods relative the this code example, but are kept to show progress that has been made.
```
from mcnets import *
from math import log
import numpy as np

# Make a neural net 
## Layer heights of 1, 10, 10 and 1
## 3 Spaces/Activation Functions: first ELU, then ATAN, then NONE (linear)
net = AdvNet(1, [10, 10], 1, ["ELU", "ATAN", "NONE"])

# Show net characteristics
print(net)

# Make data
xTrain = [*range(30)]
yTrain = [log(x + 1) for x in xTrain]
inData = np.array(xTrain)
valData = np.array(yTrain)

# Thin data for training data (10 points)
xThin, yThin, xIndicies = thinData(inData, valData, 10)

# Train the net to the data and display the results
net, R2 = genTrain(net, xThin, yThin, R2Goal = 1)
predictions = Forecast(net, inData, valData)
```
Using a straight forward method like this (essentially using the net as some functions f(x)) didn't work well before V0.2.1. However presently, being able to customize the activation function used at each layer space allows for this to now be possible and work very well (R^2 > 0.999 typically for the training data points in this code example).


## Old Curve Fitting Examples
Whole the quickstart above shows the current ability of nets to converge onto a small set of data points, up to 1000 points have been done in under 10 seconds. There were a few versions before this, however. Below are some examples of the limited training possible before, shown purely for comparison to what is now possible.

### First (choppy) Method
The original method I used these nets for curve fitting is by taking advantage of the non-linear behavior of the RELU calculation. At the time, I also had an ELU option implemented, but it didn't give me any useful fit back. This method worked, and did allow the net to have some "intelligence" to fit to the given data to, but it was very choppy at best, and required massive nets to do an okay-ish job. Below is a "short" training session example of what it gives.

Training Details:
- Net Size = 1176 parameters
- Training Time = 213.4 s

Result:

![](Examples/ghFit1b.png)

### Second Method (fitting slopes using ELU)
The new method still makes use of the AdvNet, but instead of directly using the output as some sort of "y" value, the output is used a a sort of rough derivative estimator (ie. a slope) that gets added to the previous point. By finding the points this way, using the ELU hidden function, much nicer results can be generated. Also importantly, note the much smaller net and training time as compared to above.

Training Details:
- Net Size = 312 params.
- Training Time = 18.1 s

Result:

![](Examples/ghFit2b.png)


## Speed Tests
### Convention
A speed test was completed on what I like to refer as 'rectangular' neural nets. This is, excluding the input and output layer sizes, each hidden layer has some size/height H and there are W layers (the net has a hidden 'width' of W). So, again excluding in/out sizes as they will be the same, the nets can be referenced as their hidden size of HxW. For example, a net with sizes 1x12x12x12x1 (in and out size of 1) as used in the above quickstart example can be thought of as a 12x3 net.

### Results
Using this convention, a test was conducted on 50^2 nets, that is, with H and W ranging from 1 to 50. The test simply consisted of performing the same numerical calculation (using the .Calculate() method) 100 times with the same net. The time recorded was then the total time divided by the 100 repeats to give the average calculation time. This test was done to all of the 2500 different nets which took a few minutes as the median calculation time was about 1.2 ms. Here is the pretty heat map that sums up the results better than the entirety of this paragraph if I kept rambling:

![](Examples/ghSpeedTest1a.png)

Note that the cooler colors are faster, and the warmer colors are the slower times.

The most important take-away is that 'tall' and 'wide' nets of equal parameters take significantly different times. That is, having only a few hidden layers that are tall (ie. like a 12x3 net) is faster than the reverse with a similar number of parameters (ie. like a 5x13 net). It is quite worse, in fact:

Net 1:
- Shape = 12x3
- Param.s = 312
- Avg. calculation time = 49.2 microseconds

Net 2:
- Shape = 5x13
- Param.s = 310
- Avg. calculation time = 124 microseconds
