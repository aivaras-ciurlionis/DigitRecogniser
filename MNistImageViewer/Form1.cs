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

        private IEnumerable<IEnumerable<int>> _testImages;
        private IEnumerable<int> _testLabels;

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
            _testImages = loader.GetTestImages();
            _testLabels = loader.GetTestLabels();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            PaintImage(8, 50, 28, 28, _testImages.ElementAt(_counter));
            label1.Text = _testLabels.ElementAt(_counter).ToString();
            _counter++;
        }

    }
}
