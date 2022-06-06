using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Forms;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using OpenCvSharp;

namespace ModelInferUI
{
    

    /// <summary>
    /// VisualizeWindow.xaml 的交互逻辑
    /// </summary>
    public partial class VisualizeWindow : System.Windows.Window
    {
        public MetaInformation metaInfos;

        //public string resultDir;
        public List<string> srcPaths;
        public List<string> mapPaths;
        public List<string> xmlPaths;
        public string resultDir { set; get; }
        public byte[] colormap;

        public ObservableCollection<ShowData> showDatas = null;

        public VisualizeWindow(MetaInformation metaInfos)
        {
            InitializeComponent();
            this.metaInfos = metaInfos;

            // 数据绑定
            showDatas = new ObservableCollection<ShowData>();
            lstImglist.Items.Clear();
            lstImglist.ItemsSource = showDatas;

            this.txtResultDir.DataContext = this;

            // 生成颜色表： nclass*3
            colormap = ImageHelper.get_color_map_list(256);

            // 默认参数
            if (metaInfos.SaveResult)
            {
                resultDir = metaInfos.SaveDir;
                if (Directory.Exists(resultDir))
                {
                    GenerateShowData(resultDir);
                }
                else
                {
                    System.Windows.Forms.MessageBox.Show("Error: Can not find any image files！");
                }
            }
        }

        public void GenerateShowData(string resultDir)
        {
            List<ShowData> showDatas_ = new List<ShowData>();
            srcPaths = Directory.GetFiles(resultDir, "src*.bmp").ToList<string>();
            mapPaths = Directory.GetFiles(resultDir, "map*.bmp").ToList<string>();
            xmlPaths = Directory.GetFiles(resultDir, "res*.xml").ToList<string>();
            int num = srcPaths.Count;
            if (mapPaths.Count == 0) mapPaths = Enumerable.Repeat("*", num).ToList();
            if (xmlPaths.Count == 0) xmlPaths = Enumerable.Repeat("*", num).ToList();
            if (num != mapPaths.Count || num != xmlPaths.Count)
            {
                System.Windows.Forms.MessageBox.Show("Error: result files not integrate.");
            }
            for(int i=0; i<srcPaths.Count; i++)
            {
                string[] tmp = srcPaths[i].Split('\\');
                int len = tmp.Length;
                int id = Convert.ToInt32(tmp[len-1].Split('.')[0].Substring(3));
                showDatas_.Add(new ShowData(srcPaths[i], xmlPaths[i], mapPaths[i], id));
            }
            // 对结构体数组排序，基于id
            showDatas_.Sort((x, y) => { return x.Id.CompareTo(y.Id); });
            // 放入带notify的数组，并建立绑定
            showDatas = new ObservableCollection<ShowData>(showDatas_);
            lstImglist.ItemsSource = showDatas;
        }

        private void btnResultDir_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
                if (folderBrowserDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    resultDir = folderBrowserDialog.SelectedPath.Trim();

                    if (Directory.Exists(resultDir))
                    {
                        GenerateShowData(resultDir);
                    }
                    else
                    {
                        System.Windows.Forms.MessageBox.Show("Error: Can not find any image files！");
                    }
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Choose Image Dir Error: {ex.Message}");
            }
        }

        // 针对ListViewItem的通用鼠标双击事件
        // 参考：https://docs.microsoft.com/zh-cn/dotnet/desktop/wpf/controls/how-to-handle-the-mousedoubleclick-event-for-each-item-in-a-listview?view=netframeworkdesktop-4.8
        private void ListViewItem_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            ShowData shown = ((System.Windows.Controls.ListViewItem)sender).Content as ShowData;
            if (shown == null) return;
            // det, clas, mask
            if(shown.ResPath != null)
            {
                object res = XmlSerializerHelper.ReadXML(shown.ResPath, typeof(ResultData));
                string types = ((ResultData)res).type;
                if(types == "seg")
                {
                    Mat img_mat = Cv2.ImRead(shown.SrcPath);                   
                    Mat map_mat = Cv2.ImRead(shown.MapPath, ImreadModes.Grayscale);
                    Mat img_merge = Mat.Zeros(img_mat.Size(), MatType.CV_8UC3);
                    Mat map_pseudo;
                    ImageHelper.gray_image_to_pseudo_image(map_mat, out map_pseudo);
                    Cv2.AddWeighted(map_pseudo, 0.65, img_mat, 0.35, 1, img_merge);
                    this.imgShow.Source = ImageHelper.MatToBitmapImage(img_merge);
                }
                else if(types == "det")
                {
                    Mat img_mat = Cv2.ImRead(shown.SrcPath);
                    
                    int numRegion = ((ResultData)res).numRegions;
                    float threshold = 0.5F;
                    for(int i=0; i<numRegion; i++)
                    {
                        Region region = ((ResultData)res).regions[i];
                        int label = region.labelId;
                        string labeldesc = region.labelDesc;
                        float score = region.score;
                        float left = region.xx[0];
                        float top = region.yy[0];
                        float right = region.xx[1];
                        float down = region.yy[1];
                        if(score > threshold)
                        {
                            int[] color_ = { (int)(colormap[label%256*3 + 0]),
                                             (int)(colormap[label%256*3 + 1]),
                                             (int)(colormap[label%256*3 + 2]) };

                            // 获取文本区域的大小
                            var text_size = Cv2.GetTextSize($"{labeldesc}-{label}-{score:f2}",HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                            // 获取文本区域的左下顶点 -- 右上角
                            int left_down_x = (int)left + 10;
                            int left_down_y = (int)top + 10;

                            // 绘制矩形，书写类别
                            Cv2.Rectangle(img_mat, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(img_mat, $"{labeldesc}-{label}-{score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 0.5, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 1, LineTypes.Link4);
                        }
                    }

                    this.imgShow.Source = ImageHelper.MatToBitmapImage(img_mat);
                }
                else if(types == "clas")
                {
                    Mat img_mat = Cv2.ImRead(shown.SrcPath);
                    //
                    Region region = ((ResultData)res).regions[0];
                    int label = region.labelId;
                    string labeldesc = region.labelDesc;
                    float score = region.score;
                    int[] color_ = { (int)(colormap[label % 256 * 3 + 0]),(int)(colormap[label % 256 * 3 + 1]),(int)(colormap[label % 256 * 3 + 2]) };
                    // 获取文本区域的大小
                    int left_down_x = 0 + 10;
                    int left_down_y = 0 + 10;
                    var text_size = Cv2.GetTextSize($"{labeldesc}-{label}-{score:f2}",HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                    Cv2.PutText(img_mat, $"{labeldesc}-{label}-{score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 0.5, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 1, LineTypes.Link4);
                    this.imgShow.Source = ImageHelper.MatToBitmapImage(img_mat);
                }
                else if(types == "mask")
                {
                    Mat img_mat = Cv2.ImRead(shown.SrcPath);                   
                    Mat map_mat = Cv2.ImRead(shown.MapPath, ImreadModes.Grayscale);
                    Mat img_merge = Mat.Zeros(img_mat.Size(), MatType.CV_8UC3);
                    Mat map_pseudo;
                    ImageHelper.gray_image_to_pseudo_image(map_mat, out map_pseudo);
                    Cv2.AddWeighted(map_pseudo, 0.65, img_mat, 0.35, 1, img_merge);

                    int numRegion = ((ResultData)res).numRegions;
                    float threshold = 0.5F;
                    for (int i = 0; i < numRegion; i++)
                    {
                        Region region = ((ResultData)res).regions[i];
                        int label = region.labelId;
                        string labeldesc = region.labelDesc;
                        float score = region.score;
                        float left = region.xx[0];
                        float top = region.yy[0];
                        float right = region.xx[1];
                        float down = region.yy[1];
                        if (score > threshold)
                        {
                            int[] color_ = { (int)(colormap[label % 256 * 3 + 0]),
                                             (int)(colormap[label % 256 * 3 + 1]),
                                             (int)(colormap[label % 256 * 3 + 2]) };

                            // 获取文本区域的大小
                            var text_size = Cv2.GetTextSize($"{labeldesc}-{label}-{score:f2}",
                                             HersheyFonts.HersheySimplex, 1, 2, out int baseline);  // 1倍大小的HersheySimplex，高度为22
                            // 获取文本区域的左下顶点 -- 右上角
                            int left_down_x = (int)left + 10;
                            int left_down_y = (int)top + 10;

                            // 绘制矩形，书写类别
                            Cv2.Rectangle(img_merge, new OpenCvSharp.Rect((int)left, (int)top, (int)right, (int)down), new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 2, LineTypes.AntiAlias);//LineTypes.AntiAlias:反锯齿效果
                            Cv2.PutText(img_merge, $"{labeldesc}-{label}-{score:f2}", new OpenCvSharp.Point(left_down_x, left_down_y), HersheyFonts.HersheySimplex, 0.5, new OpenCvSharp.Scalar(color_[0], color_[1], color_[2]), 1, LineTypes.Link4);
                        }
                    }
                    this.imgShow.Source = ImageHelper.MatToBitmapImage(img_merge);
                }
            }
            else
            {
                throw new Exception($"Can not find xml result file");
            }
        }

        public void showPredictResult()
        {

        }

    }


}
