using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
namespace MNistImageViewer
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private int _counter = 0;

        private IEnumerable<IEnumerable<double>> _testImages;
        private IEnumerable<IEnumerable<double>> _trainImages;

        private IEnumerable<int> _testLabels;
        private IEnumerable<int> _trainLabels;

        private List<NeuralNetwork.NetworkInputFormat> _prepTestData;
        private List<NeuralNetwork.NetworkInputFormat> _prepTrainData;

        private NeuralNetwork.NeuralNetwork _network;

        private void PaintImage(int pixelSize, int offset, int rows, int cols, IEnumerable<int> values)
        {
            var g = CreateGraphics();
            var enumerable = values as int[] ?? values.ToArray();
            var i = 0;
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++)
                {
                    var grayColor = byte.MaxValue - enumerable.ElementAt(i);
                    var b = new SolidBrush(Color.FromArgb(grayColor,
                        grayColor,
                        grayColor));

                    g.FillRectangle(b, offset + c * pixelSize, offset + r * pixelSize, pixelSize, pixelSize);
                    i++;
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            var loader = new MNISTDataLoader.MnistDataLoader(AppDomain.CurrentDomain.BaseDirectory + "/Data");



            _testImages = loader.GetTestImages(100);
            _testLabels = loader.GetTestLabels(100);

            _trainImages = loader.GetTrainImages(10000);
            _trainLabels = loader.GetTrainLabels(10000);

            var ti = _testImages.ToList();
            var tl = _testLabels.ToList();

            var tri = _trainImages.ToList();
            var trl = _trainLabels.ToList();


            _prepTestData = ti.Select((t, i) => new NeuralNetwork.NetworkInputFormat
            {
                Input = new List<double>(t), ExpectedOutput = tl[i]
            }).ToList();

            _prepTrainData = tri.Select((t, i) => new NeuralNetwork.NetworkInputFormat
            {
                Input = new List<double>(t), ExpectedOutput = trl[i]
            }).ToList();


            _network = new NeuralNetwork.NeuralNetwork(new List<int> { 28 * 28, 10 });
            _network.SetMagicParameters(10, 100);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            _counter++;

            var cnt = _network.ExecuteEpoch(_prepTrainData.ToArray(), _prepTestData.Take(100).ToArray());

            
            label1.Text = _counter.ToString();
            label2.Text = cnt.Item1 + " " + cnt.Item2;
        }

    }
}
