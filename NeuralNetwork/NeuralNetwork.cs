using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Permissions;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Protocols;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly IList<int> _sizes;
        private readonly int _layersCount;
        private readonly IList<Matrix<double>> _biases;
        private readonly IList<Matrix<double>> _weights;
        private readonly IContinuousDistribution _seedDistribution;
        public readonly INeuronFunction NeuronFunction;

        private int _batchSize;
        private double _learningRate;

        private void InitialiseBiases()
        {
            foreach (var size in _sizes.Skip(1))
            {
                _biases.Add(Matrix<double>.Build.Random(size, 1, _seedDistribution));
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
            NeuronFunction = new SigmoidNeuronFunction();
            _seedDistribution = new Normal(0, 0.25, new Random());
            _biases = new List<Matrix<double>>();
            _weights = new List<Matrix<double>>();
            var enumerable = sizes as int[] ?? sizes.ToArray();
            _sizes = enumerable;
            _layersCount = _sizes.Count;
            InitialiseBiases();
            InitialiseWeights();
        }

        public int Evaluate(IEnumerable<NetworkInputFormat> testData)
        {
            var converter = new VectorConverter();
            var zipped = testData.Select(t => new { t.Input, t.ExpectedOutput });
            var sum = 0;
            foreach (var zip in zipped)
            {
                var testResults = FeedForward(converter.EnumerableToVector(zip.Input));
                if (testResults.MaximumIndex() == zip.ExpectedOutput)
                {
                    sum++;
                }
            }
            return sum;
        }

        public Tuple<IList<Matrix<double>>, IList<Matrix<double>>> BackPropagation(Vector<double> x, Vector<double> y)
        {
            var nablaB = _biases
             .Select(b => Matrix<double>.Build.Dense(b.RowCount, 1))
             .ToList();

            var nablaW = _weights
                .Select(w => Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount))
                .ToList();

            var activation = x;
            var activations = new List<Vector<double>> { activation };
            var zs = new List<Vector<double>>();
            var zipped = _biases.Zip(_weights, (b, w) => new { Bias = b, Weight = w });

            foreach (var zip in zipped)
            {
                var z = (zip.Weight * activation + zip.Bias.Column(0));
                zs.Add(z);
                activation = NeuronFunction.NeuronFunctionValue(z);
                activations.Add(activation);
            }

            var cd = CostDerivative(activations.Last(), y);
            var delta = cd.PointwiseMultiply(NeuronFunction.NeuronFunctionDerivativeValue(zs.Last()));
            nablaB[nablaB.Count - 1] = delta.ToColumnMatrix();
            nablaW[nablaB.Count - 1] = delta.OuterProduct(activations[activations.Count - 2]);
            for (var i = 2; i < _layersCount; i++)
            {
                var z = zs[zs.Count - i];
                var sp = NeuronFunction.NeuronFunctionDerivativeValue(z);
                delta = (_weights[_weights.Count - i + 1].Transpose() * delta).PointwiseMultiply(sp);
                nablaB[nablaB.Count - i] = delta.ToColumnMatrix();
                nablaW[nablaB.Count - i] = delta.OuterProduct(activations[activations.Count - i - 1]);
            }
            return new Tuple<IList<Matrix<double>>, IList<Matrix<double>>>(nablaB, nablaW);
        }

        private Matrix<double> MatrixFromVector(IList<double> input)
        {
            return Matrix<double>.Build.Dense(input.Count, 1, (i, j) => input[i]);
        }

        private Vector<double> CostDerivative(Vector<double> a, Vector<double> y)
        {
            return a - y;
        }

        public Vector<double> FeedForward(Vector<double> input)
        {
            var zipped = _biases.Zip(_weights, (b, w) => new { Bias = b, Weight = w });
            return zipped.Aggregate(input, (current, z) => NeuronFunction.NeuronFunctionValue(z.Weight * current + z.Bias.Column(0)));
        }

        public void UpdateMiniBatch(ref List<NetworkInputFormat> miniBatch)
        {
            var converter = new VectorConverter();
            // empty gradient matrixes
           var nablaB = _biases
                .Select(b => Matrix<double>.Build.Dense(b.RowCount, 1)).ToList();
            var nablaW = _weights
                .Select(w => Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount)).ToList();

            foreach (var trainingBatch in miniBatch)
            {
                var x = converter.EnumerableToVector(trainingBatch.Input);
                var y = converter.ExpectedOutputToVector(trainingBatch.ExpectedOutput);
                // do some magic
                var backpropResult = BackPropagation(x, y);
                for (var i = 0; i < _layersCount - 1; i++)
                {
                    nablaB[i] += backpropResult.Item1[i];
                    nablaW[i] += backpropResult.Item2[i];
                }
            }

            var learnRateLength = _learningRate / miniBatch.Count;

            for (var i = 0; i < _layersCount - 1; i++)
            {
                _weights[i] -= learnRateLength * nablaW[i];
                _biases[i] -= learnRateLength * nablaB[i];
            }
        }


        public Tuple<int, int> ExecuteEpoch(
             NetworkInputFormat[] trainData,
             NetworkInputFormat[] testData
            )
        {
            var rnd = new Random();
            // Shuffle train data
            var trainShuffled = trainData.OrderBy(d => rnd.Next()).ToList();
            var batchCount = trainData.Length/_batchSize;
            for (var i = 0; i < batchCount; i++)
            {
                var miniBatch = trainShuffled.Skip(i*_batchSize).Take(_batchSize).ToList();
                UpdateMiniBatch(ref miniBatch);
                if (i % (batchCount / 100) == 0)
                {
                    Console.Write("*");
                }
            }
            Console.WriteLine();
            Console.WriteLine("Evaluating test");
            var test =  Evaluate(testData);
          //  Console.WriteLine("Evaluating train");
          //  var train = Evaluate(trainData);
            Console.WriteLine();
            return new Tuple<int, int>(test, 0);
        }

        public void SetMagicParameters(int batchSize, double learningRate)
        {
            _batchSize = batchSize;
            _learningRate = learningRate;
        }

        public void LearnAndTest(
            IEnumerable<NetworkInputFormat> trainData,
            int epochCount,
            int batchSize,
            double learningRate,
             IEnumerable<NetworkInputFormat> testData
            )
        {
            SetMagicParameters(batchSize, learningRate);
            var networkInputFormats = trainData as NetworkInputFormat[] ?? trainData.ToArray();
            var inputFormats = testData as NetworkInputFormat[] ?? testData.ToArray();
            for (var i = 0; i < epochCount; i++)
            {
                ExecuteEpoch(networkInputFormats, inputFormats);
            }
        }
    }
}
