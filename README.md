# AI Pendulum Balancer

An intelligent pendulum control system that uses **neuroevolution** to learn how to balance an inverted pendulum in real-time. Watch as neural networks evolve from random movements to expert balancing through artificial evolution.


https://github.com/user-attachments/assets/d57f049e-8251-40b6-8ebf-cb922ab75b48


## 🧠 What It Does

This project demonstrates **machine learning in action** using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The AI starts with no knowledge of physics and learns through generations of evolution and trial and error to master pendulum control.

Watch the training in action:


https://github.com/user-attachments/assets/7ebe6dea-4667-4cb8-8512-e06de83961d1


## 🚀 Quick Start

```bash
# Install dependencies
pip install pygame pymunk neat-python

# Launch the application
python main_menu.py
```

## 🎮 Control Modes

### 🤖 AI Control
- Load a pre-trained neural network
- Watch the AI automatically balance the pendulum

### 👨‍💻 Manual Control  
- Use arrow keys to control the pendulum

### 🏋️ Training Mode
- Run the NEAT algorithm to evolve new AI controllers
- Networks compete to minimize balancing time
- Best performer saved as `best_network.pkl`

## 🧩 Neural Network Architecture

- **Inputs**: Pendulum angle, angular velocity, pivot position
- **Output**: Movement command for the pivot
- **Evolution**: Both weights and topology evolve over generations

## 📁 Project Structure

```
├── 🎯 main_menu.py              # Application launcher
├── pendulum_simulation/
│   ├── 🤖 AI_control.py         # AI demonstration
│   ├── 🏋️ train.py              # Neuroevolution training
│   ├── 👨‍💻 manual_control.py     # Human control
│   ├── 🔧 commons.py            # Pendulum physics
│   └── ⚙️ neat_config.txt       # AI training parameters
```

## 💡 Acknowledgments

**Inspiration:**
- [How to train simple AIs](https://www.youtube.com/watch?v=EvV5Qtp_fYg) - An excellent video

**Resources:**
- [NEAT-Python](https://neat-python.readthedocs.io/) - Implementation of NEAT, a genetic algorithm for generating evolving artificial neural networks
- [Pygame](https://www.pygame.org/) - Graphics library
- [Pymunk](https://www.pymunk.org/) - Physics simulation
- [Walking with NEAT](https://github.com/monokim/PyHuman) - Similar project

This project was developed with assistance from large language models serving as a collaborative tool for refinement and explanation.

---

