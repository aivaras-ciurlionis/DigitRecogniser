using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    internal class PairOfMatrices
    {
        public Matrix<double> M1 { get; set; }
        public Matrix<double> M2 { get; set; }
    }
}
