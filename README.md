ieee_cec_2014_nmmso
===================

Codebase for IEEE Congress on Evolutionary Computation 2014 paper, containing the Niching Migratory Multi-Swarm Optimiser, and two other multi-modal optimisers from the literature

Implementation provided corresponds to that used in:
"Running Up Those Hills: Multi-Modal Search with the Niching Migratory
Multi-Swarm Optimiser"
by Jonathan E. Fieldsend
published in Proceedings of the IEEE Congress on Evolutionary Computation,
pages 2593-2600, 2014

Dependencies: you will need the Netlab toobox by Ian Nabney (as the NMMSO uses it's dist2 function for calculating the sqaured distance between matrices).

Please use "help function_name" at the commandline in Matlab to get a discription of the function usage -- note that the "_iterative" versions run the algorithm for a single generation (and can take state of the previous generation end as input). This allows a step-through of the algorithm for behaviour analysis. The non iterative versions essentially wrap the iterative versions. I've put in code comments and subroutines are (hopefully!) not too opaque in naming. Any queries, bug fixes, etc., please email me. 
