# Monte-Carlo-Neural-Nets

## Overview / Lore

A package made to see how well a basic architecture of neural nets could be. The nets can be created with custom input and output heights, as well as full customization of the hidden layers' sizes (height and count of layers).

The basic operation of these nets is that they can be trained to some data set (or not, currently working on a 'self-training' chess AI example) by randomly 'tweaking' the different parameter weight values within the net. These weight values are clipped to be restrained to the range [-1, 1].

By default, a RELU-type function is applied at every layer calculation to give non-linearites such that more advanced calculations are actually possible.

This is on PyPI, view the latest release at:
https://pypi.org/project/mcnets/


## Quickstart
### Example Code (Fit to f(x) = ln(x))
Using the old method (which is no longer the best, see Curve Fitting Examples) a net can be created and trained in the following fashion:
```
from mcnets import *
from math import log
import numpy as np

# Make a neural net (layer heights of 1, 12, 12, 12, 1)
net = AdvNet(1, [12]*3, 1)

# Show net characteristics
print(net)

# Make training data
xTrain = [*range(30)]
yTrain = [log(x + 1) for x in xTrain]
inData = np.array(xTrain).reshape(len(xTrain), 1)
valData = np.array(yTrain).reshape(len(yTrain), 1)

# Train the net to the data and display the results
net, avgError = CycleTrain(net, inData, valData, maxCycles=5, 
                           plotResults=True, hiddenFnc="RELU")
```
This will yield some similar plot to the one of the old method shown below. For a small number of data points this is quick and works rather well. However, with a larger number of data points, the fit is worse and becomes noticably more choppy (again, as shown below).

## Curve Fitting Examples
### Old Method
The original method I used these nets for curve fitting is by taking advantage of the non-linear behavior of the RELU calculation. At the time, I also had an ELU option implemented, but it didn't give me any useful fit back. This method worked, and did allow the net to have some "intelligence" to fit to the given data to, but it was very choppy at best, and required massive nets to do an okay-ish job. Below is a "short" training session example of what it gives.

Training Details:
- Net Size = 1176 parameters
- Training Time = 213.4 s

Result:

![](Examples/ghFit1b.png)

### New Method (fitting slopes using ELU)
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
