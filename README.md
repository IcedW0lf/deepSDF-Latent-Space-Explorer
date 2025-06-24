# DeepSDF Auto-Decoder Latent Space Explorer

## [Try the Demo](https://icedw0lf.github.io/deepSDF-Latent-Space-Explorer/)

<p align="center">
  <img alt="DeepSDF Auto-Decoder Latent Space Explorer Demo" src="asserts/autodecoder.gif" width="75%">
</p>

### About
This application is an interactive visualization that allows you to explore the latent space of 2D geometric shapes generated using an auto-decoder approach inspired by the [DeepSDF paper](https://arxiv.org/pdf/1901.05103) by Park et al. Unlike traditional VAEs, auto-decoders directly optimize latent codes for each shape along with the decoder network, providing a more direct and interpretable latent representation.

The visualization generates four basic geometric shapes (circle, triangle, square, hexagon) as signed distance fields (SDFs) and learns a 2D latent space representation that can be explored interactively in the browser.


### Implementation Details

The auto-decoder model was implemented using TensorFlow/Keras and the training code is located in the `scripts/AutoDecoder.ipynb` Jupyter notebook. The model consists of:

1. **Learnable latent codes**: Each shape has its own optimizable latent vector (2D for visualization)
2. **Decoder network**: Maps from latent space to 28x28 SDF images
3. **Joint optimization**: Both latent codes and decoder weights are optimized simultaneously

Key features:
- **SDF Generation**: Uses mathematical SDF functions to generate perfect geometric shapes
- **Auto-Decoder Training**: Direct optimization of latent codes without an encoder
- **Interactive Exploration**: Real-time generation of shapes by hovering over the latent space
- **TensorFlow.js Integration**: Model runs entirely in the browser using WebGL acceleration

### Why Latent Space Interpolation?

This project demonstrates a key advantage of latent space interpolation over traditional linear interpolation for geometric shapes:

**Semantically Meaningful Interpolation for Manifold-Constrained Shapes**:

 When interpolating directly between shape parameters or pixel values, linear interpolation often creates meaningless "islands" or artifacts that don't represent valid geometric shapes. In contrast, latent space interpolation ensures that all intermediate points lie on the learned manifold of valid shapes.


This approach is particularly valuable for:
- **Shape morphing applications** in computer graphics, architecture, and robotics  that needs to maintain geometric validity
- **Design exploration** where all variations must be meaningful
- **Generative modeling** that respects the underlying structure of geometric data


### Technical Stack

- **Backend Training**: TensorFlow/Keras, NumPy, Matplotlib
- **Frontend**: React.js for UI, HTML Canvas for image rendering
- **Visualization**: D3.js for interactive scatter plot
- **Model Deployment**: TensorFlow.js for in-browser inference
- **Shape Generation**: Custom SDF (Signed Distance Field) functions

### Project Structure

```
├── scripts/
│   ├── AutoDecoder.ipynb     # Main training notebook
│   └── sdf_generator.py      # SDF shape generation utilities
├── src/
│   ├── components/           # React components
│   ├── containers/           # Main app container
│   └── encoded.json         # Pre-computed latent space grid
└── public/
    └── models/              # Trained TensorFlow.js model
```

### Getting Started

1. **Install dependencies**: `npm install`
2. **Train the model**: Run the Jupyter notebook in `scripts/AutoDecoder.ipynb`
3. **Start the app**: `npm start`
4. **Explore**: Hover over the scatter plot to generate shapes in real-time!

### References

This implementation is inspired by and builds upon:

**DeepSDF: Learning Continuous Signed Distance Representations for Shape Completion**  
Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S. (2019)  
*IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*  
[arXiv:1901.05103](https://arxiv.org/abs/1901.05103)

**Original Interactive Visualization Framework**  
VAE Latent Space Explorer by Taylor Denouden (April 2018)  
[GitHub Repository](https://github.com/tayden/VAE-Latent-Space-Explorer)  

_Auto-Decoder adaptation and SDF integration for geometric shapes interpolation from latent space by Wo Lin (June 2025)_
