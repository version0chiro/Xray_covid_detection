These are the set of commands used in all three devices:

•	Windows

o	Jupyter notebook and python 3.6 was used to train models from scratch using keras framework with tensorflow backend (CUDA enabled)

o	The model generated can be found in the model folder of this repository

o	SSH was done to the linux machine running ubuntu to execute git related commands

•	Linux(Ubuntu)

o	The vitis-ai-tutorials was cloned from the following git repository to convert keras model into elf models. https://github.com/Xilinx/Vitis-AI-Tutorials.git

o	Commands run inside the repository:

	./docker_run.sh xilinx/vitis-ai:latest (to start the docker)

	source ./0_setenv.sh (initializing many of the parameters for the specific model)

	source ./2_keras2tf.sh (starting the keras to tensorflow conversion)

	source ./4_quant.sh (quantization of the images placed inside the calib_images folder)

•	A custom datagen.py script was written to convert the calib_images into proper format for the model input shape 

	source ./6_compile.sh ( compiling the model with path to the dcf and arch json file inside the DOUCZDX8G path)

o	The above command must have generated an elf file if executed properly.

•	Ultra96V2 (Petalinux)

o	The DPU runner was used to utilise the built elf model. Threads were used for the static version and no threads were utilised for the live feed version

o	The target board being Ultra96V2 obtained the code via git function. The original git repo with all the commits can be find here : 
https://github.com/version0chiro/DPU_Covid_19_detection_target

o	The repository has a run.py script that uses the dpu to run the elf model with xrays provided. 

o	We have tested two of our own data that was not feed into the model, both of them were covid-19 positive and the model returned accurate results.
