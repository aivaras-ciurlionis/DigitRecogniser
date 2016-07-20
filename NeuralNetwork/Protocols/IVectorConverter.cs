using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Protocols
{
    public interface IVectorConverter
    {
        Vector<double> EnumerableToVector(IEnumerable<double> array);
        IEnumerable<Vector<double>> EnumerablesToVectors(IEnumerable<IEnumerable<double>> arrays);
        Vector<double> ExpectedOutputToVector(int expectedOutput);
    }
}