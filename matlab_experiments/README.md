Lifting Layers: Matlab Experiments
================

Requirements
-------------------

1. You need to have Matlab2018a or later
2. You need the Matlab Neural Networks Toolbox and likely also the Image Processing Toolbox

Usage - Synthetic Examples
-------------------

- Simply run synthetic1d.m or synthetic2d.m to reproduce the corresponding experiments from the paper

Usage - Image Classification
-------------------

- Run cifar10_test.m or cifar100_test.m to reproduce the corresponding experiments from the paper. The corresponding datasets will be downloaded automatically. Calling plotResultsCifarX_test.m after running one of the above .m files will generate the plots.
- Run cifar10_test_differentL.m to test the effect of different lifting dimensions on the classification result - an experiments illustrated in the supplementary material of our paper.