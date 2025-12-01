This is the code for the article Real-Time Approximation of Hydraulic Erosion: A Parameter-Conditioned Deep Learning Approach
To replicate it, you can use the Map_Generator.py file to generate 10.000 prior samples, then, use the Erosion_simulation.py file to erode 
all these maps with Numba-JIT compilation. This is the pipeline for the procedural generation of the ground-truth pairs. 
The GAN_Model.py details the architecture and training strategy for the parameter-conditioned cGAN, you can start this file to train the network yourself. 
Training time estimate: (8 Hours using 8GB of VRAM, if your Graphics card is CUDA available and you have more than 8 GB of VRAM,
you can increase the number of workers in the training section of the Gan_Model file, every worker uses approximately 1.96GB of VRAM), or you can use the weights of the model I trained for the article, 
which can be found here https://drive.google.com/drive/folders/1r90NX1SYCDUVXGxiKFwSwcNfzQFxr--Z?usp=drive_link
by deleting the training loop in the file, running the GAN_Model.py file and then replacing the weights of the Generator and Discriminator for the ones
in Google Drive. 

Once you have a working model, all experiment and test files can be run, as they operate with the internal methods defined in the Map Generator, Fluid Simulation and Gan Model files. 

Preliminary results can be seen here

![alt text]([https://github.com/[Stivizea]/[Real-Time-Approximation-of-Hydraulic-Erosion-with-CGANS]/blob/[main]/Figure_Recursive_Error_Analysis.png?raw=true](https://github.com/Stivizea/Real-Time-Approximation-of-Hydraulic-Erosion-with-CGANS/blob/main/Figure_Recursive_Error_Analysis.png?raw=true))
