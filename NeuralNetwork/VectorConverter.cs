using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Protocols;

namespace NeuralNetwork
{
    public class VectorConverter : IVectorConverter
    {
        public Vector<double> EnumerableToVector(IEnumerable<double> array)
        {
            return new DenseVector(array.ToArray());
        }

        public IEnumerable<Vector<double>> EnumerablesToVectors(IEnumerable<IEnumerable<double>> arrays)
        {
            return arrays.Select(EnumerableToVector).ToList();
        }

        public Vector<double> ExpectedOutputToVector(int expectedOutput)
        {
            var vector = Vector<double>.Build.Sparse(10);
            vector[expectedOutput] = 1.0;
            return vector;
        }
    }
}
