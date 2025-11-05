# FoilML - Airfoil Design and Optimization

This repository contains tools for airfoil design and optimization using neural networks.

## FoilGen - Web Interface

**FoilGen** is the website folder containing the interactive web application for designing custom airfoils.

🌐 **Live Website**: [https://foilgen.fly.dev](https://foilgen.fly.dev)

For detailed documentation on FoilGen, see the [FoilGen README](FoilGen/README.md).

## Model Architecture

FoilGen uses a two-stage neural network pipeline:

1. **Aero → AF512**: Converts aerodynamic parameters (Reynolds number, Mach number, Cl, Cd, Cl/Cd curves, thickness) into an intermediate `AF512` format
   - Uses LSTM-based sequence encoder for processing aerodynamic curves
   - Outputs 1024-dimensional AF512 representation

2. **AF512 → XY**: Converts the `AF512` representation into actual X-Y coordinates for the airfoil
   - Takes 1024-dimensional AF512 input
   - Outputs 2048 values (1024 X-coordinates + 1024 Y-coordinates)

## Acknowledgments

- **Mike Quayle** and **bigfoil.com** - Provided the airfoil data used for training the neural networks
- **Peter D. Sharpe** - Creator of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) and [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox), which are core components of this project
