# FoilGen Quick Start Guide

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open in browser:**
   Navigate to `http://localhost:8090`

## What's Included

- **app.py** - Flask web server
- **interactive_airfoil_design.py** - Core model definitions and functions
- **Model files:**
  - `aero_to_af512_model.pth` - Neural network for aerodynamic to airfoil features
  - `af512_to_xy_model.pth` - Neural network for airfoil features to coordinates
  - `feature_normalization.npy` - Feature normalization data
- **Frontend:**
  - `templates/index.html` - Main web interface
  - `static/style.css` - Styling
  - `static/script.js` - Interactive functionality
- **Output folder** - Generated airfoils will be saved here

## Usage Tips

1. Start by entering flight conditions (chord length and speed)
2. Set your angle of attack range
3. Add control points for Cl and Cl/Cd curves either by:
   - Typing "alpha,value" in the input field
   - Clicking directly on the charts
4. Click "Update Interpolation" to see the spline curve
5. Enter thickness values and generate your airfoil!

The application is self-contained - all required files are in this directory.
