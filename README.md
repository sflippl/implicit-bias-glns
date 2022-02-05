## The Implicit Bias of Gradient Descent on Generalized Gated Linear Networks

This folder contains the code to reproduce the data in "The Implicit Bias of Gradient Descent on Generalized Gated Linear Networks."

To reproduce the 'data' folder:

- Activate the environment provided as the 'environment.yml' file.
- Run 'shallow_gln.sh', 'shallow_gln_cp.sh', 'shallow_gln_cp_2.sh', 'deep_gln.sh', 'deep_gln_cp.sh', 'relu_net.sh', and 'relu_net_cp.sh'.
  The convex optimization scripts (end in cp) must be run after their gradient descent counterparts (do not end in cp).

We then created an R package to package the cleaned datasets ('glnsanalysis', available in the folder 'implicit-bias-glns-analysis'). To access the datasets, build the packages. The notebooks in the folder 'vignettes' reproduce the figures.

