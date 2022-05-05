# Neural-Architecture-Search-
CV project on NAS 

Our end goal is to create a Neural Architecture Search system which can learn convolution operations from scratch 
and assemble the CV related operations to perform better than a traditional CNN. Below is the list of operations 
that our model focuses on optimizing for the network. 

After going through the studies on Neural Architecture Search we considered the most relatable for our references.  
The first one is the Fast and Practical Neural Architecture which uses Directely Acyclic graph as a block of 
operations. The vertex represents operations such as element-wise addition, concatination and spit operation.
The edges represents arithmetic operation, such as convolution or pooling, and identity mapping. The system 
shows its great generalization ability on ImageNet and ADE20K datasets for classification and semantic segmentation.

The second one is the Surrogate-Assisted Neural Architecture search which built for generating task specific models. 
The surrogate helps at two levels: 1 at architecture level to improve sample efficiency and 2: at weights level
to improve gradiesnt descent efficiency. On standard datasets (CIFAR-10, and ImageNet), the system matches the 
state-of-the-art with a search cost of one day.
