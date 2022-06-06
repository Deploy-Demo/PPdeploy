using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;


namespace ModelInferUI
{
    public class ImageHelper
    {
        // mat转bitmap
        public static Bitmap MatToBitmap(Mat image)
        {
            return OpenCvSharp.Extensions.BitmapConverter.ToBitmap(image);
        }

        // mat转BitmapImage
        public static BitmapImage MatToBitmapImage(Mat image)
        {
            Bitmap bitmap = MatToBitmap(image);
            using (MemoryStream stream = new MemoryStream())
            {
                bitmap.Save(stream, System.Drawing.Imaging.ImageFormat.Png); // 坑点：格式选Bmp时，不带透明度

                stream.Position = 0;
                BitmapImage result = new BitmapImage();
                result.BeginInit();
                // According to MSDN, "The default OnDemand cache option retains access to the stream until the image is needed."
                // Force the bitmap to load right now so we can dispose the stream.
                result.CacheOption = BitmapCacheOption.OnLoad;
                result.StreamSource = stream;
                result.EndInit();
                result.Freeze();
                return result;
            }
        }

        // RGB value collection of pseudo color map (color_map) - from paddleX
        public static byte[] get_color_map_list(int num_classes = 256)
        {
            num_classes += 1;
            byte[] color_map = new byte[num_classes * 3];
            for (int i = 0; i < num_classes; i++)
            {
                int j = 0;
                int lab = i;
                while (lab != 0)
                {
                    color_map[i * 3] |= (byte)(((lab >> 0) & 1) << (7 - j));
                    color_map[i * 3 + 1] |= (byte)(((lab >> 1) & 1) << (7 - j));
                    color_map[i * 3 + 2] |= (byte)(((lab >> 2) & 1) << (7 - j));

                    j += 1;
                    lab >>= 3;
                }
            }
            color_map = color_map.Skip(3).ToArray();
            return color_map;
        }


        // 灰度map转换为伪彩map
        public static void gray_image_to_pseudo_image(Mat grayImage, out Mat pseudoImage)
        {
            byte[] color_map = get_color_map_list(256);
            pseudoImage = Mat.Zeros(grayImage.Size(), MatType.CV_8UC3);
            for(int i=0;i<pseudoImage.Cols;i++)
            {
                for(int j=0;j<pseudoImage.Rows;j++)
                {
                    byte label = grayImage.At<byte>(j, i);
                    // 黑色认为是背景标注，不进行涂色
                    if (label != 0)
                    {
                        pseudoImage.At<Vec3b>(j, i) = new Vec3b(color_map[label * 3 + 0], color_map[label * 3 + 1], color_map[label * 3 + 2]);
                    }
                }
            }
        }

        // 保持宽高比的缩放: 可用于获取固定尺寸大小图片的同时保持图片不变形
        // 参考：mmcv/image/geometric/imrescale
        // 返回的scale_factor为实际缩放的比例，scale_factor = dst/src = min(目标H/源H, 目标W/源W)
        public static void rescale(Mat src, ref Mat dst, int target_w, int target_h, ref double scale_factor)
        {
            int src_w = src.Cols;
            int src_h = src.Rows;
            // 计算实际比例因子
            double divH = target_h * 1.0 / src_h;
            double divW = target_w * 1.0 / src_w;
            scale_factor = divH > divW ? divW : divH;
            Mat _src = new Mat();
            int new_w = (int)(src.Cols * scale_factor + 0.5);
            int new_h = (int)(src.Rows * scale_factor + 0.5);
            Cv2.Resize(src, _src, new OpenCvSharp.Size(new_w, new_h));
            // roi移动
            // 创建目标图像
            if (src.Channels() == 1) dst = Mat.Zeros(target_h, target_w, MatType.CV_8UC1);
            else dst = Mat.Zeros(target_h, target_w, MatType.CV_8UC3);
            Rect roi = new Rect(0, 0, _src.Cols, _src.Rows);
            Mat _dst = new Mat(dst, roi);
            _src.CopyTo(_dst);
        }
    }
}
