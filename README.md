# NonMaxSuppression-with-Reinforcement-Learning
A Reinforcement Learning Agent Doing Non Maximum Suppression For Object detection.


# Setup
Clone the repository:
```shell
$ git clone https://github.com/blurry-mood/NonMaxSuppression-with-Reinforcement-Learning
```
Install Requirements (or first create a virual environment):
```shell
$ cd NonMaxSuppression-with-Reinforcement-Learning
$ pip3 install -r requirements
```

# Dataset
The used dataset is **WIDER FACE**. It can be found [here](http://shuoyang1213.me/WIDERFACE/).  
Please download the training, validation, testing images and the face annotations file, and place them inside `dataset/`.  
Finally execute this command:
```shell
$ cd NonMaxSuppression-with-Reinforcement-Learning/dataset
$ sh setup.sh
```

# Train
Training the agent is done by executing this command:
```shell
$ cd NonMaxSuppression-with-Reinforcement-Learning/src
$ python3 train_agent.py
```
After every episode, the agent will be saved the model to `artifacts/dqn.pth`.
A UI will be shown to visualize the training process. 
At every step in the episode, chosen bounding-boxes will be visualized.

# Evaluate


# Citation
```
@inproceedings{yang2016wider,
                Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
                Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                Title = {WIDER FACE: A Face Detection Benchmark},
                Year = {2016}}
```
```
@inproceedings{deng2019retinaface,
                title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
                author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
                booktitle={arxiv},
                year={2019}
```
