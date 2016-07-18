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

        public IEnumerable<IEnumerable<int>> GetTestImages()
        {
            var testImageLoader = new ImagesLoader(_dataLocation + "/" + TestImagesFileName);
            return testImageLoader.GetImages();
        }

        public IEnumerable<int> GetTestLabels()
        {
            var testLabelLoader = new LabelLoader(_dataLocation + "/" + TestLabelsFileName);
            return testLabelLoader.GetLabels();
        }

        public void GetData()
        {
         
            var trainingLabelLoader = new LabelLoader(_dataLocation + "/" + TrainLabelsFileName);
            var trainingImageLoader = new ImagesLoader(_dataLocation + "/" + TrainImagesFileName);
            var trainLabels = trainingLabelLoader.GetLabels();
        }

    }
}
