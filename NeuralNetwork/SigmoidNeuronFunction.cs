using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using NeuralNetwork.Protocols;

namespace NeuralNetwork
{
    internal class SigmoidNeuronFunction : INeuronFunction
    {

        public Vector<double> NeuronFunctionValue(Vector<double> input)
        {
            var i = -input;
            return 1/(1 + i.PointwiseExp());
        }

        public Vector<double> NeuronFunctionDerivativeValue(Vector<double> input)
        {
            var functionValue = NeuronFunctionValue(input);
            return  functionValue.PointwiseMultiply(1 - functionValue);
        }
    }
}
