# The Basics of Reinforcement Learning

This is the code I wrote while learning RL, it implements 4 models of increasing complexity. 
Using OpenAI Gymnasium for the environments, and its able to save your trained models and load and use them for inference


# Layout

"grid.py" is the first project, it is a simple grid based environment and it showcases the foundations of Reinforcement Learning and Markov Decision Process (MDP).
It showcases 4 methods of learning ("Monte Carlo", "SARSA", "Expected SARSA", "Q-learning") which can even be run in parallel to watch them compete in real time.

Each of the 4 models has a train and a model file, the model holds the type of neural network that is to be used with the learning algorithm 
and the train file is the file you run to create/train the model according to the specific learning algorithm.
REMEMBER to set all file names and hyper-parameters at the top of the file to your liking before running the train file!

- MODEL 1: Policy Gradient/REINFORCE ("pg_train.py", "pg_model.py")
- MODEL 2: Deep Q Network ("dqn_train.py", "dqn_model.py")
- MODEL 3: Proximal Policy Optimization ("ppo_train", "ppo_model.py")
- MODEL 4: Soft Actor Critic ("sac_train.py", "sac_model.py")

"eval.py" Is for running inference/evaluating model performance. It can use a model trained with any of the 4 train files. 
It doesn't random sample for actions like the training loops do and rather just uses an argmax to choose the next action.
Can show a dashboard with the gameplay, or can be used with no dashboard to speed through episodes and get a quick average reward in the terminal.

"data_visualizer.py" Uses matplotlib to display any saved data you'd like that was saved in a csv using the train files. Remember to set the params at the top to display the right stuff.

"helpers.py" Holds all the functions that are used in multiple files across the codebase.


# Requires

pytorch
gymnasium
matplotlib