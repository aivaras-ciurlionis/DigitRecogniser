using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NetworkInputFormat
    {
        public IEnumerable<double> Input { get; set; }
        public int ExpectedOutput { get; set; }
    }
}
