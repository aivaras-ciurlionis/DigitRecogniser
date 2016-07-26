using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Protocols
{
    public interface INeuronFunction
    {
        Matrix<double> NeuronFunctionValue(Matrix<double> input);
        Matrix<double> NeuronFunctionDerivativeValue(Matrix<double> input);
    }
}
