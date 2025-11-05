# Interactive Airfoil Design Web Application

A beautiful, user-friendly web interface for designing custom airfoils using neural networks.

## Features

- 🎨 Modern, responsive web interface
- 📊 Interactive charts using Plotly
- ✈️ Real-time airfoil generation
- 📈 Visualize aerodynamic curves
- 💾 Download generated airfoil .dat files
- 🎯 Spline interpolation for smooth curves

## Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Files Are Present**

   The required model files should already be in the FoilGen directory:
   - `aero_to_af512_model.pth`
   - `af512_to_xy_model.pth`
   - `feature_normalization.npy`

3. **Navigate to FoilGen Directory**

   ```bash
   cd FoilGen
   ```

4. **Run the Web Application**

   ```bash
   python app.py
   ```

5. **Open in Browser**

   Navigate to `http://localhost:8090` in your web browser.

## Usage

1. **Enter Flight Conditions**: Input chord length (ft) and speed (mph)
2. **Set Angle Range**: Define the angle of attack range
3. **Define Cl Curve**: Add control points for lift coefficient (format: "alpha,value")
4. **Define Cl/Cd Curve**: Add control points for lift-to-drag ratio (format: "alpha,value")
5. **Set Thickness**: Enter minimum and maximum thickness values
6. **Generate**: Click "Generate Airfoil" to create your custom airfoil

The application will automatically interpolate between your control points using splines and generate the airfoil shape.

## File Structure

```
FoilGen/
├── app.py                      # Flask web application
├── interactive_airfoil_design.py  # Core functions and models
├── aero_to_af512_model.pth     # Neural network model (Aero → AF512)
├── af512_to_xy_model.pth       # Neural network model (AF512 → XY)
├── feature_normalization.npy   # Feature normalization data
├── templates/
│   └── index.html              # Frontend HTML
├── static/
│   ├── style.css               # Styling
│   └── script.js               # Frontend JavaScript
├── requirements.txt            # Python dependencies
└── output/                     # Generated files (created automatically)
```

## API Endpoints

- `GET /` - Main page
- `POST /api/compute_reynolds` - Compute Reynolds and Mach numbers
- `POST /api/generate_alpha` - Generate alpha vector
- `POST /api/interpolate_points` - Interpolate control points
- `POST /api/generate_airfoil` - Generate airfoil from inputs
- `GET /api/download_dat` - Download generated .dat file

## Notes

- The application loads models on startup, so the first request may take a moment
- Make sure you have sufficient memory for the neural network models
- Charts update in real-time as you add control points
