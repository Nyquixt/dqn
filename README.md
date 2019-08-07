# Deep Q-Network (Re)Implementation

This implementation is based on an implementation of Chapter 6 in the book Deep Reinforcement Learning Hands-On by Maxim Lapan.

## Why the Reimplementation?

Although in the book, the author mentions that there is a library called ptan that handles everything in the Deep RL area for Pytorch, 
I prefer pure implementation to help people understand the algorithm.

I made some tweaks in the environment wrappers and some observation handlers to minimize the usage of RAM during training, as for more
complicated environments, the Replay Buffer could get to 1 million in capacity.
