# Ising-Simulations
A set of c++ codes which simulate the 1D, 2D, and ANNN Ising Models

# 1D Ising
Files: 1DIsingFinal.cpp


# 2D Ising
Files: 2DIsingFinal.cpp


# ANNNI Ising + ML Model
Files: ANNNImodelfinal.cpp & RMBfinal.cpp
ANNNImodel.cpp runs 1000 simulations (int N) of length (int L) and saves each chain's initial and final state to InitialConfigurations.txt and FinalConfigurations.txt, and also computes the correlation C(x) between spins, outputting IsingCorrelation.txt
RBMfinal.cpp trains a restricted boltzmann machine on the final chain states from ANNNImodel.cpp. Once the weights of the model are determined, N initial chains are put through the model to be turned in to probable final chain states. Correlation C(x) is computed for the RBM and output under RBMCorrelation.txt for comparison to IsingCorrelation.txt

