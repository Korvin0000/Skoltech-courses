# ml_stoch_diff_eq_2023
Stochastic Differential Equations for Generative Modeling.

One can find code for replication and research tasks in the folders [replication-task](/replication-task/) and [research-task](/research-task/).

To run the evaluation for research-task just run [main.py](/research-task/main.py). You will need 20 GB of GPU memory for it. The code will download pretrained Generator and Discriminator from [here](https://github.com/csinva/gan-vae-pretrained-pytorch.git) and run Langevin dynamics for it with further FID calculation for CIFAR-10 via [torch_fidelity](https://torch-fidelity.readthedocs.io/en/latest/) framework.

For replication task we used this [repo](https://github.com/yang-song/score_sde_pytorch.git). The experiments for replication-task were made via running [/replication-task/skoltech_main.py](/replication-task/skoltech_main.py). One can find bash files for results evaluation. The list of bash files for VE and VP is as follows: 

 - [/replication-task/exp_c_1000.sh](/replication-task/exp_c_1000.sh) --- corrector only 1000 steps
 - [/replication-task/exp_c_2000.sh](/replication-task/exp_c_2000.sh) --- corrector only 2000 steps
 - [/replication-task/exp_p_1000.sh](/replication-task/exp_p_1000.sh) --- predictor with reversed diffusion 1000 steps
 - [/replication-task/exp_p_2000.sh](/replication-task/exp_p_2000.sh) --- predictor with reversed diffusion 2000 steps
 - [/replication-task/exp_p_as_1000.sh](/replication-task/exp_p_as_1000.sh) --- predictor with ancestral sampling 1000 steps 
 - [/replication-task/exp_p_as_2000.sh](/replication-task/exp_p_as_2000.sh) --- predictor with ancestral sampling 2000 steps 
 - [/replication-task/exp_pc_1000.sh](/replication-task/exp_pc_1000.sh) --- predictor-corrector with reversed diffusion 1000 steps 
 - [/replication-task/exp_pc_as_1000.sh](/replication-task/exp_pc_as_1000.sh) --- predictor-correcor with ancestral sampling 1000 steps
  
 - [/replication-task/exp_c_1000_vp.sh](/replication-task/exp_c_1000.sh) --- corrector only 1000 steps (for VP)
 - [/replication-task/exp_c_2000_vp.sh](/replication-task/exp_c_2000.sh) --- corrector only 2000 steps (for VP)
 - [/replication-task/exp_p_1000_vp.sh](/replication-task/exp_p_1000.sh) --- predictor with reversed diffusion 1000 steps (for VP)
 - [/replication-task/exp_p_2000_vp.sh](/replication-task/exp_p_2000.sh) --- predictor with reversed diffusion 2000 steps (for VP)
 - [/replication-task/exp_p_as_1000_vp.sh](/replication-task/exp_p_as_1000.sh) --- predictor with ancestral sampling 1000 steps (for VP)
 - [/replication-task/exp_p_as_2000_vp.sh](/replication-task/exp_p_as_2000.sh) --- predictor with ancestral sampling 2000 steps (for VP)
 - [/replication-task/exp_pc_1000_vp.sh](/replication-task/exp_pc_1000.sh) --- predictor-corrector with reversed diffusion 1000 steps (for VP)
 - [/replication-task/exp_pc_as_1000_vp.sh](/replication-task/exp_pc_as_1000.sh) --- predictor-correcor with ancestral sampling 1000 steps (for VP) 


Fids were derived via [torch_fidelity](https://torch-fidelity.readthedocs.io/en/latest/) framework (see [here](/replication-task/calculate_fids.py) for the details). One can find final results in the [folder](/replication-task/fids/). 
