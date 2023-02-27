# Monte-Carlo-Neural-Nets

## Overview

A package made to see how well a basic architecture of neural nets could be. The nets can be created with custom input and output heights, as well as full customization of the hidden layers' sizes (height and count of layers).

The basic operation of these nets is that they can be trained to some data set (or not, currently working on a 'self-training' chess AI example) by randomly 'tweaking' the different parameter weight values within the net. These weight values are clipped to be restrained to the range [-1, 1].

By default, a RELU-type function is applied at every layer calculation to give non-linearites such that more advanced calculations are actually possible.

This is on PyPI, view the latest release at:
https://pypi.org/project/mcnets/

## Curve Fitting Examples
### Old Method
The original method I used these nets for curve fitting is by taking advantage of the non-linear behavior of the RELU calculation. At the time, I also had an ELU option implemented, but it didn't give me any useful fit back. This method worked, and did allow the net to have some "intelligence" to fit to the given data to, but it was very choppy at best, and required massive nets to do an okay-ish job. Below is a "short" training session example of what it gives.

Training Details:
- Net Size = 1176 parameters
- Training Time = 213.4 s

Result:

![](Examples/ghFit1b.png)

### New Method (fitting slopes using ELU)
The new method still makes use of the same net, but instead of directly using the output as some sort of "y" value (ie y = f(x)), the output is used a a sort of rough derivative estimator (ie., a slope) that gets added to the previous point. Also, I found that using the ELU calculations withint the layers was optimal in this case. Overall, this method gives fantastic results compared to the old method as shown below. Also note the smaller net size and the much shorter training session used!

Training Details:
- Net Size = 312 params.
- Training Time = 18.1 s

Result:

![](Examples/ghFit2b.png)
