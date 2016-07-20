using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    internal class NablaTupple
    {
        public IList<Matrix<double>> NablaB { get; set; }
        public IList<Matrix<double>> NablaW { get; set; }
    }
}
