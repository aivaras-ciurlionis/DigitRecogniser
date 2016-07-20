using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Protocols;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly IList<int> _sizes;
        private readonly int _layersCount;
        private readonly IList<Vector<double>> _biases;
        private readonly IList<Matrix<double>> _weights;
        private readonly IContinuousDistribution _seedDistribution;
        private readonly INeuronFunction _neuronFunction;

        private int _batchSize;
        private double _learningRate;

        private void InitialiseBiases()
        {
            foreach (var size in _sizes.Skip(1))
            {
                _biases.Add(Vector<double>.Build.Random(size, _seedDistribution));
            }
        }

        private void InitialiseWeights()
        {
            for (var i = 0; i < _sizes.Count - 1; i++)
            {
                _weights.Add(Matrix<double>.Build.Random(_sizes[i + 1], _sizes[i], _seedDistribution));
            }
        }

        public NeuralNetwork(IEnumerable<int> sizes)
        {
            _neuronFunction = new SigmoidNeuronFunction();
            _seedDistribution = new Normal(0, 0.5, new Random());
            _biases = new List<Vector<double>>();
            _weights = new List<Matrix<double>>();
            var enumerable = sizes as int[] ?? sizes.ToArray();
            _sizes = enumerable;
            _layersCount = _sizes.Count;
            InitialiseBiases();
            InitialiseWeights();
        }

        public Tuple<IList<Matrix<double>>, IList<Matrix<double>>> BackPropagation(Vector<double> x, Vector<double> y)
        {
            return null;
        }

        public Vector<double> FeedForward(Vector<double> input)
        {
            var zipped = _biases.Zip(_weights, (b, w) => new { Bias = b, Weight = w });
            return zipped.Aggregate(input, (current, z) => _neuronFunction.NeuronFunctionValue(z.Weight * current + z.Bias));
        }

        public void UpdateMiniBatch(ref IEnumerable<NetworkInputFormat> miniBatch)
        {
            var converter = new VectorConverter();
            // empty gradient matrixes
            var nablaB = _biases
                .Select(b => Matrix<double>.Build.Dense(b.Count, 1))
                .ToList();
            var nablaW = _weights
                .Select(w => Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount))
                .ToList();

            foreach (var trainingBatch in miniBatch)
            {
                var x = converter.EnumerableToVector(trainingBatch.Input);
                var y = converter.ExpectedOutputToVector(trainingBatch.ExpectedOutput);
            }
        }

        public void ExecuteEpoch(
             NetworkInputFormat[] trainData,
             NetworkInputFormat[] testData
            )
        {
            var rnd = new Random();
            // Shuffle train data
            var trainShuffled = trainData.OrderBy(d => rnd.Next());
            for (var i = 0; i < trainData.Length / _batchSize; i++)
            {
                var miniBatch = trainShuffled.Skip(i * _batchSize).Take(_batchSize);
                UpdateMiniBatch(ref miniBatch);
            }
        }

        public void LearnAndTest(
            IEnumerable<NetworkInputFormat> trainData,
            int epochCount,
            int batchSize,
            double learningRate,
             IEnumerable<NetworkInputFormat> testData
            )
        {
            _batchSize = batchSize;
            _learningRate = learningRate;
            var networkInputFormats = trainData as NetworkInputFormat[] ?? trainData.ToArray();
            var inputFormats = testData as NetworkInputFormat[] ?? testData.ToArray();
            for (var i = 0; i < epochCount; i++)
            {
                ExecuteEpoch(networkInputFormats, inputFormats);
            }
        }
    }
}
