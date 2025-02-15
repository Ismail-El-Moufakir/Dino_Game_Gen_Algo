# Genetic Algorithm Dino Game

This project implements a simple genetic algorithm to train agents (Dino) in a game-like environment using PyTorch and Pygame. The agents learn to avoid obstacles by jumping and walking, evolving over multiple generations to improve their performance.


## Installation

To run this project, you need to have Python installed along with the necessary libraries. Follow these steps to set up the environment:

1. Clone this repository:
    ```bash
    git clone https://github.com/Ismail-El-Moufakir/Dino_Game_Gen_Algo
    cd Dino_Game_Gen_Algo
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    torch
    numpy
    matplotlib
    ```

4. Ensure you have the necessary assets (images for Dino's actions). Place the images in the `assets` directory:
    ```
    assets/
        Walk_Left.png
        Walk_Right.png
        Jump.png
    ```

## Usage

To run the genetic algorithm training process, simply execute the main script:

```bash
python test.py
