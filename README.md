
Requires:
 tensorflow >= 1.0.0
 sklearn


TODO:

- check parameters are all needed
- handle unknown users and items
- binary targets with log-loss ??
- side data ??
- momentum ??
- check gradients are all on correct variables
- max iters and stop early using validation set
- other cost functions
- other regularization scemes (like the one I was trying at first)
- maintain speed (Fit-time. 100 full-batch iters in 32 seconds.)
- auto-select learning rate
- add random seed support
- Double check docstrings
- Write docs and publish on readthedocs
- package for python 2 & 3
- publish on PyPI




References:

- http://katbailey.github.io/post/matrix-factorization-with-tensorflow/
- https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
- http://scikit-learn.org/stable/developers/
- http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
