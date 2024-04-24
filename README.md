<h1 align="center">Meta-Heuristic CNNs</h1>
<p align="center"><i>(Utkarsh Mathur)</i></p>

<b>Team Name</b> - Pixel Swarm <br>
<b>Course</b> - CSE 573: Computer Vision and Image Processing (University at Buffalo, State University of New York)<br>
<b>Instructor</b> - Dr. Sreyasee Das Bhattacharjee (<b>TA</b> - Bhushan Mahajan)<br>
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
<p>N/A</p>

<h3>2.1. Meta-Heuristic Optimization</h3>
<p>N/A</p>

<h3>2.2. Genetic Algorithm</h3>
<p>N/A</p>

<h3>2.3. Particle Swarm Optimization</h3>
<p>N/A</p>

<h3>2.4. Ant Colony Optimization</h3>
<p>N/A</p>

<h3>2.5. Grey Wolf Optimization</h3>
<p>N/A</p>

<h3>2.6. Convolutional Neural Networks</h3>
<p>N/A</p>

<h3>2.7. Image Classification</h3>
<p>N/A</p>

<h2>3. Methodology</h2>
<p>Most of the evolutionary optimization algorithms can handle vector and tensors as their population. Model parameters for Deep Learning models usually consists of collections of 3D or 4D tensors of varying dimensionality. To address this we treat the collections parameters atomically (i.e. one convolution layer at a time).</p>

<h3>3.1. Image Classification CNNs</h3>
<p>N/A</p>

<h3>3.2. Model Training</h3>
<p>N/A</p>

<h3>3.3. Model Inference</h3>
<p>N/A</p>

<h2>4. Experiments</h2>
<p>N/A</p>

<h3>4.1. MNIST LeNet<h3>
<p>N/A</p>

<h3>4.2. CIFAR-10 LeNet</h3>
<p>N/A</p>

<h3>4.3. ImageNet AlexNet</h3>
<p>N/A</p>

<h3>4.4. ImageNet VGG-16</h3>
<p>N/A</p>

<h3>4.5. ImageNet ResNet-50</h3>
<p>N/A</p>

<h2>5. Results & Analysis</h2>
<p>N/A</p>

<h3>5.1. Model Performances</h3>
<p>N/A</p>

<h3>5.2. Computation Analysis</h3>
<p>N/A</p>

<h2>6. References</h2>

<ol type="1">
    <li>Z. Michalewicz, Genetic Algorithms + Data Structures = Evolution Programs. Berlin, Heidelberg: Springer Berlin Heidelberg, 1996. doi: https://doi.org/10.1007/978-3-662-03315-9.‌</li>
    <li>S. Katoch, S. S. Chauhan, and V. Kumar, “A review on genetic algorithm: past, present, and future,” Multimedia Tools and Applications, vol. 80, no. 5, Oct. 2020, doi: https://doi.org/10.1007/s11042-020-10139-6.</li>
    <li>J. Kennedy and R. Eberhart, “Particle swarm optimization,” Proceedings of ICNN’95 - International Conference on Neural Networks, vol. 4, pp. 1942–1948, 1995, doi: https://doi.org/10.1109/icnn.1995.488968.</li>
    <li>T. Alam, S. Qamar, A. Dixit, and M. Benaida, “Genetic Algorithm: Reviews, Implementations, and Applications,” Jun. 2020, doi: https://doi.org/10.48550/arxiv.2007.12673.</li>
    <li>A. G. Gad, “Particle Swarm Optimization Algorithm and Its Applications: A Systematic Review,” Archives of Computational Methods in Engineering, vol. 29, no. 5, pp. 2531–2561, Apr. 2022, doi: https://doi.org/10.1007/s11831-021-09694-4.</li>
    <li>M. Dorigo, M. Birattari, and T. Stutzle, “Ant colony optimization,” IEEE Computational Intelligence Magazine, vol. 1, no. 4, pp. 28–39, Nov. 2006, doi: https://doi.org/10.1109/MCI.2006.329691.</li>
    <li>S. Mirjalili, S. M. Mirjalili, and A. Lewis, “Grey Wolf Optimizer,” Advances in Engineering Software, vol. 69, pp. 46–61, Mar. 2014, doi: https://doi.org/10.1016/j.advengsoft.2013.12.007.</li>
    <li>J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” 2009 IEEE Conference on Computer Vision and Pattern Recognition, Jun. 2009, doi: https://doi.org/10.1109/cvpr.2009.5206848</li>
    <li>Li Deng, “The MNIST Database of Handwritten Digit Images for Machine Learning Research [Best of the Web],” IEEE Signal Processing Magazine, vol. 29, no. 6, pp. 141–142, Nov. 2012, doi: https://doi.org/10.1109/msp.2012.2211477.</li>
    <li>A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” undefined, 2009, Accessed: May 11, 2022. [Online]. Available: https://www.semanticscholar.org/paper/Learning-Multiple-Layers-of-Features-from-Tiny-Krizhevsky/5d90f06bb70a0a3dced62413346235c02b1aa086</li>
    <li>Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998, doi: https://doi.org/10.1109/5.726791.</li>
    <li>A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” Communications of the ACM, vol. 60, no. 6, pp. 84–90, May 2012, doi: https://doi.org/10.1145/3065386.</li>
    <li>K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” arXiv.org, Apr. 10, 2015. https://arxiv.org/abs/1409.1556</li>
    <li>K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” arXiv.org, Dec. 10, 2015. https://arxiv.org/abs/1512.03385</li>
    <li>R. Mohapatra, “rohanmohapatra/torchswarm,” GitHub, Apr. 20, 2024. https://github.com/rohanmohapatra/torchswarm (accessed Apr. 20, 2024).</li>
    <li>A. P. Sansom, “Torch PSO,” GitHub, Aug. 01, 2022. https://github.com/qthequartermasterman/torch_pso (accessed Apr. 20, 2024).‌</li>
    <li>H. Faris, “7ossam81/EvoloPy,” GitHub, Apr. 20, 2024. https://github.com/7ossam81/EvoloPy/tree/master (accessed Apr. 20, 2024).‌</li>
</ol>