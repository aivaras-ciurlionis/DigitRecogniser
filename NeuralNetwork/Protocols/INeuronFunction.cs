using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Protocols
{
    public interface INeuronFunction
    {
        Vector<double> NeuronFunctionValue(Vector<double> input);
        Vector<double> NeuronFunctionDerivativeValue(Vector<double> input);
    }
}
