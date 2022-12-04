# MS-Independent-Study-Fall-2022

This is the project reposetory for Independent Study Fall 2022 at University of Coloradoo Boulder.

## Independent Study Goal -

- Understand how self driving cars work, read papers, go through courses
- Train a self-driving car using behavioral cloning in a simulated environment

## Bi- Weekly Plan -

- 8/22/2022	: Data simulation and generation and Data loader
- 9/5/2022	: Start implementing the NVIDIA architecture
- 9/19/2022	: Explore different prediction metrics apart from the ones mentioned in the paper
- 10/3/2022	: Explore on different architectures suitable in comparison to the current architecture
- 10/17/2022 : Run experiments, Evaluate on other simulated environments(if time permits)
- 10/31/2022 : Explore different data augmentation techniques
- 11/14/2022 : Work on optimization, and hyper-parameter tuning, etc.
- 11/28/2022 : Write a report( preferably IEEE format same as last IS report)
- 12/12/2022 : Get a review/feedback

## Resources -

1. Udacity's self-driving car simulator - https://github.com/udacity/self-driving-car-sim
2. NVIDIA's paper- End to End Learning for Self-Driving Cars
3. Reference for data augmentation- Improving Behavioral Cloning with Human-Driven Dynamic Dataset Augmentation

## Project Status- 

### Week 1: Data Generation -

- For data generation I am using Udacity's open source car simulator
- The tool was created by Udacity using Unity for there nanodegree students, later they open sourced it for boarder research community
- To generate the dataset, one needs to play the game in the simulator in driving/training mode. This was particularly challenging for me

### Week 3 : Training the model -

- The idea is to start with simple model
- Gradually I am building design the atchitecture based on the NVIDIA paper. I am not replicating there entire atchitecture but just take the basic intuition since I do not have the same computation power as them. This would be a gradual and experimental process over next couple of weeks. 
- Here is Nvidia's architecture-

<img src="images\NVIDIA-architecture.png" width="500" height="350">

### Week 5: Decide on Evaluation Metrics -

- The model training based on the images captured from driving the car
- The output that model trains on is the steering angle
- Here, I am keeping the evaluation metrics simple to accuracy and F1 score

### Week 7: Improve The Network -

- Next couple of weeks' goal is improve the basic model and work on the architecture
- Tune the hyper parameter which might take longer than i originally planned for

### Week 9: Run experiments, Evaluate on other simulated environments -

- Plan is to note parameters and setting for every experiment and compare outputs

### Week 11: Explore different data augmentation techniques -
- I used basic data augmentation techniques, such as cropping, flipping etc.

<img src="output\track1.png"> 
 <img src="output\track1_cropped.png" width="200" height="200"> <img src="output\track1_flipped.png" width="200" height="200"> 

### Week 12-14: Work on optimization, and hyper-parameter tuning, etc. -

- I spent this week on trying different neural network architechture and improving the model performance
#### Final parameters:

| Hyper-Parameters  | Values  |
|---|---|
|  Epochs |  5  |
| Optimizer  | Adam   |
| Learning Rate  |  0.001  |
| Train-Test Split |  80-20%  |
| Batch Size  |  64  |
|  Loss Function |  MSE  |
| Activation Function  |  RELU  |
| Dropout  |  35%  |


### Week 15: Write a report( preferably IEEE format same as last IS report)

