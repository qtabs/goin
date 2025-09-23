# Inference comparison between implementations and against leaky integrator

Folder containing the script to compare the COIN inference implementation between Python and Matlab, and assess it against the baseline leaky integrator (goin/opt_coin/test_opt_coin.py).

To run goin/opt_coin/test_opt_coin.py:

- Download / clone the repos:

https://github.com/clelf/goin/ (contains the script  and Alejandro's generative model)

https://github.com/clelf/COIN_Python (COIN Python implementation with adaptation about sqrt values)

https://github.com/jamesheald/COIN/

The script currently runs with the repos being stored like this:

workspace/goin/

workspace/COIN_Python/

workspace/COIN/

- Adapt the variables coin_base, thirdparty_base and goin_base according to the repos' paths in goin/addCoinPaths.m

- Create a conda environment from goin/opt_coin/environment_opt_coin.yml