using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MNISTDataLoader
{
    public class LabelLoader : IDisposable
    {
        private readonly BinaryReader _binaryReader;

        public LabelLoader(string fileName)
        {
            var labelFile = File.Open(fileName, FileMode.Open);
            _binaryReader = new BinaryReader(labelFile);
        }

        public IEnumerable<int> GetLabels(int? count)
        {
            var labels = new List<int>();
            _binaryReader.ReadInt32(); // Skip magic number
            var setSize = HeaderInt32Converter.Convert(_binaryReader.ReadBytes(4));
            for (var i = 0; i < setSize; i++)
            {
                labels.Add(_binaryReader.ReadByte());
            }
            return labels.Take(count??setSize);
        }

        public void Dispose()
        {
            _binaryReader?.Dispose();
        }

    }
}
