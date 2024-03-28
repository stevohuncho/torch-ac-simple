from torch_ac_simple import Agent
from torch_ac_simple.plot import plot_eval, PlotEvalData
import matplotlib.pyplot as plt

def test_a2c_agent_creation() -> Agent:
    return Agent("MiniGrid-Empty-6x6-v0", "a2c_test")

def test_a2c_agent_train(a: Agent):
    a.train(1000, "a2c", log_interval=10, save_interval=10)

def test_a2c_agent_eval(a: Agent):
    a.eval(100)

def test_ppo_agent_creation() -> Agent:
    return Agent("MiniGrid-Empty-6x6-v0", "ppo_test")

def test_ppo_agent_train(a: Agent):
    a.train(1000, "ppo", log_interval=10, save_interval=10)

def test_ppo_agent_eval(a: Agent):
    a.eval(100)

def test_plot_eval():
    plot_eval([PlotEvalData("storage/a2c_test/eval.csv", "A2C"), PlotEvalData("storage/ppo_test/eval.csv", "PPO", 'red')])
    plt.savefig("storage/a2c_test/eval.png")

if __name__ == "__main__":
    a = test_a2c_agent_creation()
    print("A2C AGENT CREATION passed!")
    test_a2c_agent_train(a)
    print("A2C AGENT TRAINING passed!")
    test_a2c_agent_eval(a)
    print("A2C AGENT EVALUATION passed!")

    a = test_ppo_agent_creation()
    print("PPO AGENT CREATION passed!")
    test_ppo_agent_train(a)
    print("PPO AGENT TRAINING passed!")
    test_ppo_agent_eval(a)
    print("PPO AGENT EVALUATION passed!")

    test_plot_eval()

