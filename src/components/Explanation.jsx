import React from "react";

import "./Explanation.css";

const Explanation = () => (  <div align="left">
    <div className="textbox">
      <div className="header">
        <h2>
          A brief explanation of the auto-decoder approach for geometric shapes
        </h2>
      </div>
      <div className="content">
        <p>
          The above visualization was created using a 2-dimensional latent space learned by an 
          auto-decoder model inspired by the DeepSDF paper. Unlike traditional variational autoencoders (VAEs), 
          auto-decoders directly optimize latent codes for each shape along with the decoder network parameters.
        </p>
        <p>
          Four geometric shapes (circle, triangle, square, hexagon) were generated as Signed Distance Fields (SDFs) 
          and used to train the auto-decoder. Each shape has its own learnable latent code that gets optimized 
          during training. The latent codes displayed in the scatterplot represent the learned 2D embeddings 
          for these shapes. Moving your mouse around this plot samples latent vectors and passes them to the 
          decoder network, which generates new geometric shapes displayed via an HTML Canvas.
        </p>
        <p>
          All code runs in the browser using TensorFlow.js. The model was originally implemented in 
          TensorFlow/Keras and exported for browser use. React and D3 are responsible for handling 
          the page updates and drawing the chart visualization. The shapes are generated using 
          mathematical SDF functions that provide perfect geometric representations.
        </p>
      </div>
    </div>

    <div className="textbox">
      <div className="header">
        <h2>
          What are auto-decoders and how do they differ from VAEs?
        </h2>
      </div>
      <div className="content">
        <p>
          Auto-decoders, introduced in the DeepSDF paper, take a different approach to learning latent 
          representations compared to traditional VAEs:
        </p>
        <ul>
          <li><strong>No encoder network:</strong> Instead of learning an encoder to map inputs to latent codes, 
          auto-decoders directly optimize a latent code for each training example.</li>
          <li><strong>Joint optimization:</strong> Both the decoder network weights and the latent codes 
          are optimized simultaneously during training.</li>
          <li><strong>More direct control:</strong> This approach often provides more direct and interpretable 
          control over the latent space since each training example has its own optimized representation.</li>
          <li><strong>Better for geometric data:</strong> Auto-decoders work particularly well with geometric 
          data like SDFs, where precise reconstruction is important.</li>
        </ul>
        <p>
          For more details on auto-decoders, check out the{" "}
          <a href="https://arxiv.org/abs/1901.05103" target="_blank" rel="noopener noreferrer">
            DeepSDF paper
          </a>{" "}
          by Park et al.
        </p>
      </div>
    </div>

    <div className="textbox">
      <div className="header">
        <h2>
          What are Signed Distance Fields (SDFs)?
        </h2>
      </div>
      <div className="content">
        <p>
          Signed Distance Fields are mathematical functions that describe the distance from any point 
          in space to the nearest surface of a shape. The "signed" part means:
        </p>
        <ul>
          <li><strong>Negative values:</strong> Points inside the shape</li>
          <li><strong>Zero:</strong> Points exactly on the surface</li>
          <li><strong>Positive values:</strong> Points outside the shape</li>
        </ul>
        <p>
          SDFs are particularly useful for machine learning because they provide a continuous, 
          differentiable representation of shapes that captures both interior and exterior structure. 
          This makes them ideal for generative models that need to learn and synthesize geometric forms.
        </p>
      </div>
    </div>

    {/* <div className="textbox">
      <div className="header">
        <h2>Implementation details</h2>
      </div>
      <div className="content">
        <h4>Encoder</h4>
        <code>
          2D convolution (32 filters, 3x3 kernel, stride 2, padding 'same')<br />
          batch normalization<br />
          leaky relu activation<br />
          <br />
          2D convolution (64 filters, 3x3 kernel, stride 2, padding 'same')<br />
          batch normalization<br />
          leaky relu activation<br />
          <br />
          2D convolution (128 filters, 3x3 kernel, stride 2, padding 'same')<br />
          batch normalization<br />
          leaky relu activation<br />
          <br />
          flatten to one dimensional vector<br />
          <br />
          dense layer, output size 100<br />
          relu activation<br />
          <br />
          two parallel dense layers, output size 2 each
        </code>
        <h4>Variational layer</h4>
        <p>
          The final two output layers of size 2 would now typically be used to
          decode back to their original as best possible. These distilled layers
          try to capture the essence of the differences between different
          images. That is, these latent factors represented a distinct image
          using just 4 values.
          <br />
          In VAEs, we constrain these latent factors so they roughly follow a
          unit Gaussian distribution. This restriction allows us to generate new
          images later by simply sampling values from the unit Gaussian, and
          decoding that latent vector back to a normal image. This operation is
          as follows:
        </p>
        <code>
          mu := [2 values from one of the final dense layers]<br />
          sigma := [2 values from the other final dense layer]<br />
          <br />
          # sample 2 values from unit Gaussian<br />
          epsilon = [e1, e2] ~ N(0, 1)<br />
          <br />
          z = mu + e^(sigma / 2) * epsilon
        </code>
        <h4>Decoder</h4>
        <p>The Decoder inverts the Encoder layers</p>
        <code>
          dense layer of size 100<br />
          dense layer of size 2048<br />
          relu activation <br />
          reshape size 2048 vector to (128, 4, 4)
          <br />
          <br />
          2d convolution transpose (64 filters, 3x3 kernel)<br />
          batch normalization<br />
          leaky relu activation<br />
          <br />
          2d convolution transpose (32 filters, 3x3 kernel)<br />
          batch normalization<br />
          {/* leaky relu activation<br />
          <br />
          2d convolution transpose (1 filter, 3x3 kernel)<br />
          sigmoid activation
        </code>
      </div>
    </div> */}
  </div>
);

export default Explanation;
