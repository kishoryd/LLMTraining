This folder contains the training of a large language model (LLM). 


***Distributed data parallelism***


   DDP is implemented for the GPT2 model to improve the training process, and the Python script of the DDP implementation is given in **train_ddp_gpt_2.py**. 

   To run train_ddp_gpt_2.py in a cluster with a **single node and multiple GPU**, use  **torchrun --nproc_per_node=2 train_ddp_gpt_2.py**.

   To run the train_ddp_gpt_2.py in a cluster with **multi-node multi-GPU**, then use 
   
   
   for 1st node 
   
   **torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=node_name --master_port=4422 train_ddp_gpt_2.py** 
   
   for 2nd node
   
   **torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=node_name --master_port=4422 train_ddp_gpt_2.py**.


   ***Tensor parallelism***

   TP is implemented in the python script with name **train_tp_gpt_2.py**

   To run the train_tp_gpt_2.py with **single node multi-GPU** use the command as **torchrun --nproc_per_node=2 train_tp_gpt_2.py**

   To run the train_tp_gpt_2.py in a cluster with **multi-node multi-GPU**, then use 
   
   
   for 1st node 
   
   **torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=node_name --master_port=4422 train_tp_gpt_2.py** 
   
   for 2nd node
   
   **torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=node_name --master_port=4422 train_tp_gpt_2.py**

   ***Pipeline parallelism***

   TP is implemented in the python script with name **train_pp_gpt_2.py**

   To run the train_pp_gpt_2.py with **single node multi-GPU** use the command as **torchrun --nproc_per_node=2 train_pp_gpt_2.py**

   To run the train_pp_gpt_2.py in a cluster with **multi-node multi-GPU**, then use 
   
   
   for 1st node 
   
   **torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=node_name --master_port=4422 train_pp_gpt_2.py** 
   
   for 2nd node
   
   **torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=node_name --master_port=4422 train_pp_gpt_2.py**

   To check how many GPUs are being used during the implementation of python script use command as **watch -n 0.1 nvidia-smi** after loging into that specified node.
