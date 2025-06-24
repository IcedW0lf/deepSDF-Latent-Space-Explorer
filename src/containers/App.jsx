import React, { Component } from "react";
import * as tf from "@tensorflow/tfjs";

import ImageCanvas from "../components/ImageCanvas";
import XYPlot from "../components/XYPlot";
import Explanation from "../components/Explanation";
import { rounder } from "../utils";

import "./App.css";

import encodedData from "../encoded.json";

const MODEL_PATH = process.env.PUBLIC_URL + "/models/generatorjs/model.json";

class App extends Component {
  constructor(props) {
    super(props);
    this.getImage = this.getImage.bind(this);
    this.previousTensor = null; // Keep track of previous tensor

    this.state = {
      model: null,
      digitImg: tf.zeros([28, 28]),
      latentX: -2.5, // Initialize with corner coordinate
      latentY: -2.5  // Initialize with corner coordinate
    };
  }  async componentDidMount() {
    try {
      console.log("Loading model from:", MODEL_PATH);
      
      // First check if the model file is accessible
      const response = await fetch(MODEL_PATH);
      if (!response.ok) {
        throw new Error(`Model file not found: ${response.status} ${response.statusText}`);
      }
      
      const modelData = await response.text();
      console.log("Model JSON first 100 chars:", modelData.substring(0, 100));
      
      const model = await tf.loadLayersModel(MODEL_PATH);
      this.setState({ model }, async () => {
        // Generate initial image after model is loaded and state is updated
        const digitImg = await this.getImage();
        this.setState({ digitImg });
      });
    } catch (error) {
      console.error("Error loading model:", error);
      console.error("Model path:", MODEL_PATH);
    }
  }
  async getImage() {
    const { model, latentX, latentY } = this.state;
    if (!model) return tf.zeros([28, 28]);
    
    const zSample = tf.tensor2d([[latentX, latentY]]);
    const prediction = model.predict(zSample);
    const result = prediction
      .mul(tf.scalar(255.0))
      .reshape([28, 28]);
    // Clean up intermediate tensors but keep the result
    zSample.dispose();
    prediction.dispose();
    
    return result;
  }

  componentWillUnmount() {
    // Clean up tensors when component unmounts
    if (this.state.digitImg) {
      this.state.digitImg.dispose();
    }
    if (this.previousTensor) {
      this.previousTensor.dispose();
    }
    if (this.state.model) {
      this.state.model.dispose();
    }
  }
  render() {
    return this.state.model === null ? (
      <div>Loading, please wait</div>
    ) : (
      <div className="App">
        <h1>DeepSDF Auto-Decoder Latent Space Explorer</h1>
        <div className="ImageDisplay">
          <ImageCanvas
            width={500}
            height={500}
            imageData={this.state.digitImg}
          />
        </div>

        <div className="ChartDisplay">
          <XYPlot
            data={encodedData}
            width={500 - 10 - 10}
            height={500 - 20 - 10}
            xAccessor={d => d[0]}
            yAccessor={d => d[1]}
            colorAccessor={d => d[2]}
            margin={{ top: 20, bottom: 10, left: 10, right: 10 }}
            onHover={async ({ x, y }) => {
              this.setState({ latentY: y, latentX: x });
              
              // Store reference to current tensor for later disposal
              this.previousTensor = this.state.digitImg;
              
              const digitImg = await this.getImage();
              
              this.setState({ digitImg }, () => {
                // Use setTimeout to dispose after React has fully updated
                setTimeout(() => {
                  if (this.previousTensor && this.previousTensor !== this.state.digitImg) {
                    this.previousTensor.dispose();
                    this.previousTensor = null;
                  }
                }, 0);
              });
            }}
          />        </div>
        <p>Latent X: {rounder(this.state.latentX, 3)}</p>
        <p>Latent Y: {rounder(this.state.latentY, 3)}</p>
        <div className="Explanation">
          <Explanation />
        </div>

        <h5>Auto-Decoder implementation by Wo Lin(June 2025) • Original VAE by Taylor Denouden (April 2018)</h5>
      </div>
    );
  }
}

export default App;
