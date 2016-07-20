using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Protocols;

namespace NeuralNetwork
{
    internal class SigmoidNeuronFunction : INeuronFunction
    {

        public Vector<double> NeuronFunctionValue(Vector<double> input)
        {
            return 1/(1 + input.PointwiseExp());
        }

        public double NeuronFunctionDerivativeValue(Vector<double> input)
        {
            var functionValue = NeuronFunctionValue(input);
            return functionValue * (1- functionValue);
        }
    }
}
