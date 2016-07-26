using System;

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
