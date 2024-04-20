<h1 align="centre">Meta-Heuristic CNNs</h1>

**Team Name** - Pixel Swarm <br>
**Project Title** - Meta-Heuristics vs Backpropagation: A Fresh Look at CNN Model Parameter Optimization for Image Classification.

<h2>1. Introduction</h2>

<h3>1.1. Abstract</h3>
Most of the Neural Network models are trained using gradient based backpropagation 
technique to optimize model parameters which are prone to be stuck at local optimal value 
rather than reaching the globally optimal parameters. There are various techniques to 
improve the simple Stochastic Gradient Descent (SGD) like Learning Rate Scheduling and 
Momentum, but these techniques does not resolve the aforementioned limitation. In this 
project, we aim to compare Backpropagation with Meta-Heuristic Optimization Algorithms 
by analyzing their performance on training CNN models for Image Classification tasks. 
There are a few population-based meta-heuristic algorithms like Particle Swarm 
Optimization (PSO) and Grey Wolf Optimization (GWO) which can achieve globally optimal 
parameters and so, we aim to check the feasibility of such optimization techniques in CNN 
model architectures.

<h3>1.2. Problem Statement</h3>
The aim of this project is to analyze the performances of meta-heuristic optimization 
algorithms in contrast to gradient-based backpropagation techniques for Convolutional 
Neural Network model training. 

<h3>1.3. Method Details</h3>

<ol type="a">
    <li>**Meta-Heuristic algorithms** -  Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Grey Wolf Optimization (GWO), and Ant Colony Optimization (ACO). </li>
    <li>**CNN model architectures** - AlexNet, VGG-16, and ResNet-50. As all the above 
    three models have been trained on Image Classification task of the ImageNet 
    challenge (ILSCRC), we’ll be using the classification task of ILSVRC 2017 as our 
    dataset for model training and model inferencing.</li>
    <li>**Dataset** - ILSVRC 2017 (Image Classification)</li>
</ol>

