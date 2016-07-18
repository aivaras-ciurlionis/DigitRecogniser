using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MNISTDataLoader
{
    public class ImagesLoader : IDisposable
    {
        private readonly BinaryReader _binaryReader;

        public ImagesLoader(string fileName)
        {
            var imagesFile = File.Open(fileName, FileMode.Open);
            _binaryReader = new BinaryReader(imagesFile);
        }

        private static List<int> GetImagePixels(IEnumerable<byte> bytes)
        {
            return bytes.Select(singleByte => (int) singleByte).ToList();
        }

        public IEnumerable<IEnumerable<int>> GetImages()
        {
            var images = new List<List<int>>();
            _binaryReader.ReadInt32(); // Skip magic number
            var imagesCount = HeaderInt32Converter.Convert(_binaryReader.ReadBytes(4));
            var rowsCount = HeaderInt32Converter.Convert(_binaryReader.ReadBytes(4));
            var columnCount = HeaderInt32Converter.Convert(_binaryReader.ReadBytes(4));
            for (var i = 0; i < imagesCount; i++)
            {
                var singleImageBytes = _binaryReader.ReadBytes(rowsCount * columnCount);
                images.Add(GetImagePixels(singleImageBytes));
            }
            return images;
        }

        public void Dispose()
        {
            _binaryReader?.Dispose();
        }

    }
}
