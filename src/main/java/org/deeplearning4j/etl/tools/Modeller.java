package org.deeplearning4j.etl.tools;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author raver119@gmail.com
 */
public class Modeller {
    public static Model getConvolutionalModel() {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(119)
                .iterations(1) // Training iterations as above
                .regularization(true).l2(0.0005)
                .learningRate(.01)//.biasLearningRate(0.02)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .graphBuilder()
                .addInputs("input")
                .addLayer("cnn1", new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(20)
                        .dropOut(0.95)
                        .activation(Activation.IDENTITY)
                        .build(),"input")
                .addLayer("pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build(),"cnn1")
                .addLayer("cnn2", new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build(),"pool1")
                .addLayer("pool2", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build(),"cnn2")
                .addLayer("dense", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build(),"pool2")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build(),"dense")
                .setOutputs("output")
                .setInputTypes(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false).build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }
}
