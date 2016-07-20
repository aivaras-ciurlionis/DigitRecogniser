using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Protocols
{
    public interface INeuronFunction
    {
        Vector<double> NeuronFunctionValue(Vector<double> input);
        double NeuronFunctionDerivativeValue(Vector<double> input);
    }
}
