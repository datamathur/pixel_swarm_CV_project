<h1 align="center">Meta-Heuristic CNNs</h1>

<b>Team Name</b> - Pixel Swarm <br>
<b>Project Title</b> - Meta-Heuristics vs Backpropagation: A Fresh Look at CNN Model Parameter Optimization for Image Classification.<br>
<h4><b>Project Members</b></h4> 

<ol type="1">
<li>Utkarsh Mathur (<a href="https://github.com/datamathur">datamathur</a>, <a href="mailto:umathur@buffalo.edu">umathur@buffalo.edu</a>)</li>
<li>Mahammad Iqbal Shaik (<a href="mailto:mahammad@buffalo.edu">mahammad@buffalo.edu</a>)</li>
</ol>

<h2>1. Introduction</h2>

<h3>1.1. Abstract</h3>
<p>Most of the Neural Network models are trained using gradient based backpropagation 
technique to optimize model parameters which are prone to be stuck at local optimal value 
rather than reaching the globally optimal parameters. There are various techniques to 
improve the simple Stochastic Gradient Descent (SGD) like Learning Rate Scheduling and 
Momentum, but these techniques does not resolve the aforementioned limitation. In this 
project, we aim to compare Backpropagation with Meta-Heuristic Optimization Algorithms 
by analyzing their performance on training CNN models for Image Classification tasks. 
There are a few population-based meta-heuristic algorithms like Particle Swarm 
Optimization (PSO) and Grey Wolf Optimization (GWO) which can achieve globally optimal 
parameters and so, we aim to check the feasibility of such optimization techniques in CNN 
model architectures.</p>

<h3>1.2. Problem Statement</h3>

<p>The aim of this project is to analyze the performances of meta-heuristic optimization 
algorithms in contrast to gradient-based backpropagation techniques for Convolutional 
Neural Network model training.</p>

<h3>1.3. Method Details</h3>

<ol type="a">
    <li><b>Meta-Heuristic algorithms</b> -  Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Grey Wolf Optimization (GWO), and Ant Colony Optimization (ACO).</li>
    <li><b>CNN model architectures</b> - LeNet, AlexNet, VGG-16, and ResNet-50. As all the above 
    three models have been trained on Image Classification task of the ImageNet 
    challenge (ILSCRC), we’ll be using the classification task of ILSVRC 2017 as our 
    dataset for model training and model inferencing.</li>
    <li><b>Dataset</b> - MNIST, CIFAR10, ILSVRC 2017 (Image Classification)</li>
</ol>

<h3>1.4. Project Analysis</h3>

<p>Since we are trying to change the model optimization techniques we intend to perform analysis on the following two grounds:</p>

<ol type="a">
    <li><b>Quality of model training</b> – Based on the model inferencing performed on models 
    trained with different optimization techniques, we will be comparing the qualities of 
    trained model.</li>
    <li><b>Computational Cost</b> – Since most of the deep-CNN models are computationally 
    expensive we will be analyzing the computational resources required by the novel 
    optimization techniques in contrast to gradient-based backpropagation techniques.</li>
</ol>


<h2>2. Background</h2>

<h3>2.1. Meta-Heuristic Optimization</h3>

<h3>2.2. Genetic Algorithm</h3>

<h3>2.3. Particle Swarm Optimization</h3>

<h3>2.4. Ant Colony Optimization</h3>

<h3>2.5. Grey Wolf Optimization</h3>

<h3>2.6. Convolutional Neural Networks</h3>

<h3>2.7. Image Classification</h3>

<h2>3. Methodology</h2>

<h3>3.1. Image Classification CNNs</h3>

<h3>3.2. Model Training</h3>

<h3>3.3. Model Inference</h3>

<h2>4. Experiments</h2>

<h3>4.1. MNIST LeNet<h3>

<h3>4.2. CIFAR-10 LeNet</h3>

<h3>4.3. ImageNet AlexNet</h3>

<h3>4.4. ImageNet VGG-16</h3>

<h3>4.5. ImageNet ResNet-50</h3>

<h2>5. Results & Analysis</h2>

<h3>5.1. Model Performances</h3>

<h3>5.2. Computation Analysis</h3>

<h2>6. References</h2>

<ol type="1">
    <li></li>
<ol>