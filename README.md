# Pytorch_Policy_Search_Optimizer
A tool for optimizing RL policy modules based on random search (https://arxiv.org/abs/1803.07055)

It wrapps any Pytorch's module for a Reinforcement Learning policy into an optimizer that updates its parameter in order to maximize a reward.

## Usage
### create a pso wrapping a policy:
```
policy = nn.Linear(num_inputs, num_outputs, bias=True)
policy.weight.data.fill_(0)
policy.bias.data.fill_(0)
pso = PSO(policy, lr=0.05, std=0.02, b=8, n_directions=8)
```
### sample perturbations (in both positive and negative direction):
```
pso.sample()
```
### make decision
```
action = pso.evaluate(state, direction=0, side="left") 
```
(direction = index of explored direction, side = positive or negative perturbation)

### update
```
pso.reward(reward, direction=0, side="left")
```

## Example
See example.py for a suggestion of augmented random search implementation.

### mujoko's halfcheetah runs after 100 episodes:
![HalfCheetah_GIF](img/HalfCheetah.gif)
