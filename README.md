# INF581Project

INF581 course project

Using A3C & DQN for car racing environment


## References
1. Asynchronous Methods for Deep Reinforcement Learning: [article](https://arxiv.org/pdf/1602.01783.pdf)
2. Reinforcement Car racing with A3C: [link](https://sites.google.com/view/jesikmin/course-projects/reinforcement-car-racing-with-a3c)
3. Reinforcement Learning for a Simple Racing Game: [link](https://web.stanford.edu/class/aa228/reports/2018/final150.pdf)
4. A3C tutorial: [Simple Reinforcement Learning with Tensorflow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
5. Self-driving toy car using the Asynchronous Advantage Actor-Critic algorithm: [tutor](https://www.endpoint.com/blog/2018/08/29/self-driving-toy-car-using-the-a3c-algorithm)


## Run the code

```bash
python run.py -m <model> --<action>
```
Possible values for parameter `action` are: `train` and `evaluate`.

Possible values for parameter `model` are: `dqn` and `a3c`.

