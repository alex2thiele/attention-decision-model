# attention-decision-model
# this model is based on the original 2 stage decision network model by Wimmer/Compte (doi:10.1038/ncomms7177, https://www.ncbi.nlm.nih.gov/pubmed/25649611). 
# It has been modified to reduce its winner take all characteristics, and the assumed input fluctuations into the sensory network are radically reduced.
# Thus, it can be interpreted as a stationary stimulus, where fluctuations arise simply through noisy inputs from earlier processing stages. 
# The integration circuit (2 overall populations), where 3*10 cells in each of these has their NMDA current drive reduced, to mimick NMDAR blockade to different degrees.
# The attentional modulation is implemented by adding unbalanced feedback to the sensory circuit (it could alternatively be done by adding exitatory drive to one of the two 
# integration circuit populations (e.g. akin to "A source for feature based attention in the prefrontal cortex", (2015), Bichot et al. http://www.ncbi.nlm.nih.gov/pubmed/26526392). 
# On each run it calculates the attentional modulation (AUROC) in the integration circuit between populations of 10 cells equally affected by NMDAR blockade, and it calculates the 
# drug modulation against the unmodulated population (AUROC) and writes these to a file. 
# The exact outcome varies from instantiation to instantiation, due to randomness in connectivity on each run and tiny variations in input 'noisiness'.
# requires brian (https://briansimulator.org/getting-started/), does not work with brian2
# runs under python2.X
# simulations were performed under Spyder (Python2.7).
