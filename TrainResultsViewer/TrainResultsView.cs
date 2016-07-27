using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace TrainResultsViewer
{
    class TrainResultsView
    {
        private const int TestCount = 10000;
        private const int TrainCount = 50000;

        private IEnumerable<IEnumerable<double>> _testImages;
        private IEnumerable<IEnumerable<double>> _trainImages;

        private IEnumerable<int> _testLabels;
        private IEnumerable<int> _trainLabels;

        private List<NeuralNetwork.NetworkInputFormat> _prepTestData;
        private List<NeuralNetwork.NetworkInputFormat> _prepTrainData;

        private NeuralNetwork.NeuralNetwork _network;

        private void CreateNetwork()
        {
            _network = new NeuralNetwork.NeuralNetwork(new List<int> { 28 * 28, 30, 10 });
            _network.SetMagicParameters(10, 3);
        }

        private void LoadAndPrepareData()
        {
            var loader = new MNISTDataLoader.MnistDataLoader(AppDomain.CurrentDomain.BaseDirectory + "/Data");
            

            _testImages = loader.GetTestImages(TestCount);
            Console.WriteLine("Loaded test images");
            _testLabels = loader.GetTestLabels(TestCount);
            Console.WriteLine("Loaded test labels");

            _trainImages = loader.GetTrainImages(TrainCount);
            Console.WriteLine("Loaded train images");
            _trainLabels = loader.GetTrainLabels(TrainCount);
            Console.WriteLine("Loaded train labels");

            var ti = _testImages.ToList();
            var tl = _testLabels.ToList();

            var tri = _trainImages.ToList();
            var trl = _trainLabels.ToList();


            _prepTestData = ti.Select((t, i) => new NeuralNetwork.NetworkInputFormat
            {
                Input = new List<double>(t),
                ExpectedOutput = tl[i]
            }).ToList();

            _prepTrainData = tri.Select((t, i) => new NeuralNetwork.NetworkInputFormat
            {
                Input = new List<double>(t),
                ExpectedOutput = trl[i]
            }).ToList();
            Console.WriteLine("Data prepared.");
            Console.WriteLine();
        }

        public void ExecuteEpoch(int epochNumber)
        {
            var result = _network.ExecuteEpoch(_prepTrainData.ToArray(), _prepTestData.ToArray());
            Console.WriteLine($"Epoch {epochNumber}: TestDataAcuracy: {result.Item1}/{TestCount} TrainDataAcuracy:{result.Item2}/{TrainCount}");
            Console.WriteLine();
        }


        private void ExecuteTraining()
        {
          
            for (var i = 0; i < 200; i++)
            {
                var stopwatch = new Stopwatch();
                stopwatch.Start();
                var delim = new string('-', 100);
                Console.WriteLine(delim);
                ExecuteEpoch(i + 1);
                stopwatch.Stop();
                Console.WriteLine($"Time elapsed: {stopwatch.Elapsed}");
            }
        }



        public static void Main(string[] args)
        {
            var v = new TrainResultsView();
            v.LoadAndPrepareData();
            v.CreateNetwork();
            Console.WriteLine("Executing....");
            Console.WriteLine();
            // v.Test();
            v.ExecuteTraining();
           // Console.ReadKey();

        }
    }
}
