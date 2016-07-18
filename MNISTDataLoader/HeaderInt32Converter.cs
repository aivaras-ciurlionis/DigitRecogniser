using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNISTDataLoader
{
    public static class HeaderInt32Converter
    {
        public static int Convert(byte[] bytes)
        {
            if (bytes.Length != 4)
            {
                throw new ArgumentException();
            }
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
