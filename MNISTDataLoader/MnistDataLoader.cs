using System.Collections.Generic;

namespace MNISTDataLoader
{
    public class MnistDataLoader
    {
        private const string TestLabelsFileName = "t10k-labels.idx1-ubyte";
        private const string TrainLabelsFileName = "train-labels.idx1-ubyte";

        private const string TestImagesFileName = "t10k-images.idx3-ubyte";
        private const string TrainImagesFileName = "train-images.idx3-ubyte";

        private readonly string _dataLocation;
        public MnistDataLoader(string dataLocation)
        {
            _dataLocation = dataLocation;
        }

        public IEnumerable<IEnumerable<double>> GetTestImages(int? count)
        {
            var testImageLoader = new ImagesLoader(_dataLocation + "/" + TestImagesFileName);
            return testImageLoader.GetImages(count);
        }

        public IEnumerable<int> GetTestLabels(int? count)
        {
            var testLabelLoader = new LabelLoader(_dataLocation + "/" + TestLabelsFileName);
            return testLabelLoader.GetLabels(count);
        }

        public IEnumerable<IEnumerable<double>> GetTrainImages(int? count)
        {
            var trainingImageLoader = new ImagesLoader(_dataLocation + "/" + TrainImagesFileName);
            return trainingImageLoader.GetImages(count);
        }

        public IEnumerable<int> GetTrainLabels(int? count)
        {
            var trainingLabelLoader = new LabelLoader(_dataLocation + "/" + TrainLabelsFileName);
            return trainingLabelLoader.GetLabels(count);
        }

    }
}
