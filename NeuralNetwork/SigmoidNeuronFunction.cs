using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using NeuralNetwork.Protocols;

namespace NeuralNetwork
{
    internal class SigmoidNeuronFunction : INeuronFunction
    {

        public Matrix<double> NeuronFunctionValue(Matrix<double> input)
        {
            var i = -input;
            return 1/(1 + i.PointwiseExp());
        }

        public Matrix<double> NeuronFunctionDerivativeValue(Matrix<double> input)
        {
            var functionValue = NeuronFunctionValue(input);
            return  functionValue.PointwiseMultiply(1 - functionValue);
        }
    }
}
