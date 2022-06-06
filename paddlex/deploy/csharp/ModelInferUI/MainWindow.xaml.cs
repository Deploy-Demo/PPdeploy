using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Forms;     // FolderBrowserDialog, 需要手动引用System.Windows.Forms
using System.IO;                // Directory
using OpenCvSharp;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Collections.Concurrent;
using System.Threading;

namespace ModelInferUI
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        public static MetaInformation metaInfos = new MetaInformation();
        public List<string> imagePaths = null;
        public List<string> modelFiles = null;
        public List<IntPtr> models = new List<IntPtr>();
        public List<Task> tasks = new List<Task>();
        List<string> precisionModes = new List<string> { "fp32", "fp16", "int8" };
        byte[] paddlex_model_type = new byte[10];
        public int MaxQueueLength = 100;

        public static int MaxBoxNum = 100;
        public static int MaxStrLen = 50;
        private bool warmuped = false;
        private int cycleIndex = 0;
        private Stopwatch timer = new Stopwatch();
        private static ConcurrentQueue<PredictResult> resultQueue = new ConcurrentQueue<PredictResult>();
        private CancellationTokenSource tokenSource = new CancellationTokenSource();

        public MainWindow()
        {
            InitializeComponent();
            this.gridMain.DataContext = metaInfos;
            this.Closing += MainWindow_Closing;
            Task saveTask = Task.Factory.StartNew(Callback_SaveResult, TaskCreationOptions.LongRunning);

        }

        private void ClearAllModels()
        {
            int t = models.Count;
            if (t > 0)
            {
                for (int i = 0; i < models.Count; i++)
                {
                    InferModel.ModelObjDestruct(models[i]);
                }
                metaInfos.Log = metaInfos.Log + $"[Info] Finished destruct {t} models.\n";
            }
            models.Clear();
            
        }

        private void MainWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            ClearAllModels();
        }

        private void btnImgDir_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
                if (folderBrowserDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    metaInfos.ImgDir = folderBrowserDialog.SelectedPath.Trim();
                    metaInfos.Log = metaInfos.Log + "[Info] Test Image Dir: " + this.txtImgDir.Text + "\n";

                    if (Directory.Exists(metaInfos.ImgDir))
                    {
                        imagePaths = Directory.GetFiles(metaInfos.ImgDir).ToList<string>();
                        metaInfos.Log = metaInfos.Log + $"[Info] Found {imagePaths.Count()} image files.\n";
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

        private void btnModelDir_Click(object sender, RoutedEventArgs e)
        {
            metaInfos.ModelRootDir = "";
            metaInfos.ModelDirs.Clear();
            string inputDir = "";
            try
            {
                FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
                if (folderBrowserDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    inputDir = folderBrowserDialog.SelectedPath.Trim();
                    metaInfos.ModelRootDir = inputDir;
                }

                if (Directory.Exists(inputDir))
                {
                    // 如果输入路径包含文件，则为单模型
                    if (Directory.GetFiles(inputDir).Length > 0)
                    {
                        metaInfos.Log = metaInfos.Log + "[Info] Chosen 1 Model Dir\n";
                        modelFiles = Directory.GetFiles(inputDir).ToList<string>();

                        metaInfos.ModelDirs.Add(inputDir);
                        metaInfos.ModelFiles.Add(modelFiles);

                        metaInfos.Log = metaInfos.Log + "    Dir: " + inputDir + "\n";
                        for (int i = 0; i < modelFiles.Count(); i++)
                        {
                            metaInfos.Log = metaInfos.Log + $"        {modelFiles[i]}\n";
                        }
                    }
                    // 如果输入路径包含文件夹，则为多模型
                    else if (Directory.GetDirectories(inputDir).Length > 0)
                    {
                        // 多模型模式下，不需要手动更新模型，直接可以开始逐个模型推理(包含模型创建和推理)
                        metaInfos.NeedUpdate = false;
                        metaInfos.IsReady = true;
                        // 获取每个模型的文件夹列表
                        string[] dirs = Directory.GetDirectories(inputDir);
                        metaInfos.Log = metaInfos.Log + $"[Info] Chosen {dirs.Length} Model Dirs\n";
                        foreach (string _dir in dirs)
                        {
                            List<string> _files = Directory.GetFiles(_dir).ToList<string>();
                            if (_files.Count > 0)
                            {
                                metaInfos.ModelDirs.Add(_dir);
                                metaInfos.ModelFiles.Add(_files);
                            }
                            else throw new Exception("[ERROR] Some of model folder did not contain model file.");
                            metaInfos.Log = metaInfos.Log + $"    Dir: {_dir}\n";
                            for (int i = 0; i < _files.Count(); i++)
                            {
                                metaInfos.Log = metaInfos.Log + $"        {_files[i]}\n";
                            }
                        }
                    }
                }
                else
                {
                    System.Windows.Forms.MessageBox.Show("Error: Can not find any model files！");
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Choose Model Dir Error: {ex.Message}");
            }
        }

        private void cmbThreads_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (this.cmbThreads.SelectedValue != null)
            {
                if (this.cmbThreads.SelectedIndex >= 0)  // index从0开始
                {
                    metaInfos.NumModels = this.cmbThreads.SelectedIndex + 1;
                    metaInfos.Log = metaInfos.Log + $"[Info] Choose {metaInfos.NumModels} thread for {metaInfos.NumModels} models.\n";
                }
            }
        }

        private void cmbModelType_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (this.cmbModelType.SelectedValue != null)
            {
                if (this.cmbModelType.SelectedIndex >= 0)
                {
                    metaInfos.ModelType = this.cmbModelType.SelectedValue.ToString().Split(':')[1].ToString().Trim();
                    metaInfos.Log = metaInfos.Log + $"[Info] Model Type: {metaInfos.ModelType}.\n";
                }
            }
        }

        private void btnUpdate_Click(object sender, RoutedEventArgs e)
        {
            if (metaInfos.ModelType == null) metaInfos.Log += $"[Error] model type is Null.\n";
            else if (metaInfos.NumModels <= 0) metaInfos.Log += $"[Error] model threads is <=0.\n";
            else if (metaInfos.ModelDirs.Count == 0) metaInfos.Log += $"[Error] model dir is Null.\n";
            else if (metaInfos.GpuId < 0 || metaInfos.GpuId > 7) metaInfos.Log += $"[Error] GPU id is not valid.\n";
            else
            {
                Task t = Start_Update(metaInfos.ModelDirs[0]);
            }
        }

        private async void btnStart_Click(object sender, RoutedEventArgs e)
        {
            // 因为开始按钮可能单独按下，所以更新按钮的参数检查部分需要重新检查
            if (metaInfos.ImgDir == null) metaInfos.Log += $"[Error] image dir is Null.\n";
            else if (metaInfos.ModelType == null) metaInfos.Log += $"[Error] model type is Null.\n";
            else if (metaInfos.NumModels <= 0) metaInfos.Log += $"[Error] model threads is <=0.\n";
            else if (metaInfos.ModelDirs[0] == null) metaInfos.Log += $"[Error] model dir is Null.\n";
            else if (metaInfos.GpuId < 0 || metaInfos.GpuId > 7) metaInfos.Log += $"[Error] GPU id is not valid.\n";
            else
            {
                metaInfos.IsReady = false;
                // 单模型
                if (metaInfos.ModelDirs.Count == 1)
                {
                    await Task.WhenAll(Start_Predict());
                }
                // 多模型
                else
                {
                    if (models.Count() > 0) ClearAllModels();
                    for (int i = 0; i < metaInfos.ModelDirs.Count(); i++)
                    {
                        // await确保顺序执行更新、预测
                        await Task.WhenAll(Start_Update(metaInfos.ModelDirs[i]));
                        await Task.WhenAll(Start_Predict());
                    }
                }
            }
        }

        private async Task Start_Update(string model_dir)
        {
            // 模型创建之前的参数检查放在按钮触发函数中
            await Task.Run(() =>
            {
                // 按钮不可用
                metaInfos.IsReady = false;
                metaInfos.NeedUpdate = false;
                warmuped = false;

                ClearAllModels();

                // 新增fine tune参数
                // TODO：是否把该参数放界面
                metaInfos.ShapeRangeInfoPath = "./shape_range_info.pbtxt";

                // 创建新模型
                for (int i = 0; i < metaInfos.NumModels; i++)
                {
                    // 模型初始化
                    IntPtr minInputSize = IntPtr.Zero;
                    IntPtr maxInputSize = IntPtr.Zero;
                    IntPtr optInputSize = IntPtr.Zero;
                    IntPtr paddlex_model_type_ = Marshal.AllocHGlobal(10);
                    try
                    {
                        if (!metaInfos.UseTrt)
                        {
                            IntPtr model_ = InferModel.ModelObjInit(metaInfos.ModelType, model_dir, metaInfos.GpuId, metaInfos.UseTrt, paddlex_model_type_);
                            models.Add(model_);
                            metaInfos.Log = metaInfos.Log + $"[Info] Finished create [No.{i + 1}] {metaInfos.ModelType} model from {model_dir}\n";
                        }
                        else
                        {
                            metaInfos.Log = metaInfos.Log + $"[Info] Creating models with TensorRT acceleration will take a long long time, please wait with patient...\n";
                            //Thread.Sleep(500);
                            int[] minInputSize_ = new int[4] { 1, 3, metaInfos.MinH, metaInfos.MinW };
                            int[] maxInputSize_ = new int[4] { 1, 3, metaInfos.MaxH, metaInfos.MaxW };
                            int[] optInputSize_ = new int[4] { 1, 3, metaInfos.OptH, metaInfos.OptW };
                            minInputSize = Marshal.AllocHGlobal(4 * 4);
                            maxInputSize = Marshal.AllocHGlobal(4 * 4);
                            optInputSize = Marshal.AllocHGlobal(4 * 4);
                            Marshal.Copy(minInputSize_, 0, minInputSize, 4);
                            Marshal.Copy(maxInputSize_, 0, maxInputSize, 4);
                            Marshal.Copy(optInputSize_, 0, optInputSize, 4);

                            //IntPtr model_ = InferModel.ModelObjInit(metaInfos.ModelType, model_dir, metaInfos.GpuId, metaInfos.UseTrt, paddlex_model_type_,
                            //    minInputSize, maxInputSize, optInputSize, metaInfos.Precision, metaInfos.MinSubgraphSize);
                            metaInfos.Log = metaInfos.Log + $"[Info] Collecting trt model shape range info and save to {metaInfos.ShapeRangeInfoPath}...\n";
                            IntPtr model_ = InferModel.ModelObjInit(metaInfos.ModelType, model_dir, metaInfos.GpuId, metaInfos.UseTrt, paddlex_model_type_,
                                minInputSize, maxInputSize, optInputSize, metaInfos.Precision, metaInfos.MinSubgraphSize,
                                metaInfos.TargetWidth, metaInfos.TargetHeight, metaInfos.ShapeRangeInfoPath);
                            
                            models.Add(model_);
                            metaInfos.Log = metaInfos.Log + $"[Info] Finished create [No.{i + 1}] {metaInfos.ModelType} with TensorRT acceleration model!\n";
                        }
                        // paddlex获取模型实际类型并更新ModelType
                        if (metaInfos.ModelType == "paddlex")
                        {
                            Marshal.Copy(paddlex_model_type_, paddlex_model_type, 0, 10);
                            string tmp = System.Text.Encoding.ASCII.GetString(paddlex_model_type);
                            metaInfos.ModelType = tmp.Split(new char[] { '\0' })[0];
                        }
                    }
                    catch (Exception ex)
                    {
                        throw ex;
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(paddlex_model_type_);
                        if (metaInfos.UseTrt)
                        {
                            Marshal.FreeHGlobal(minInputSize);
                            Marshal.FreeHGlobal(maxInputSize);
                            Marshal.FreeHGlobal(optInputSize);
                        }
                    }
                }
                metaInfos.IsReady = true;
            });
        }

        private async Task Start_Predict()
        {
            // 预热
            await Task.Run(() =>
            {
                if (metaInfos.WarmUp && warmuped == false)
                {
                    // warmup次数改为100次
                    int warmup_cycles = 100;
                    Mat src = Cv2.ImRead(imagePaths[0], ImreadModes.Unchanged);
                    ResumeParam param = new ResumeParam();
                    int h = src.Rows;
                    int w = src.Cols;
                    int c = src.Channels();

                    metaInfos.Log = metaInfos.Log + $"[Info] Start run warmup with {models.Count} models with {warmup_cycles} cyles.\n";
                    timer.Reset();
                    timer.Start();
                    for(int j = 0; j < warmup_cycles; j++)
                    {
                        for (int i = 0; i < models.Count; i++)
                        {
                            tasks.Add(CreateTask(metaInfos.ModelType, models[i], src.Data, i, w, h, c, param));
                        }
                        for (int i = 0; i < tasks.Count; i++) tasks[i].Start();
                        try
                        {
                            Task.WaitAll(tasks.ToArray());
                        }
                        catch (Exception ex)
                        {
                            throw ex;
                        }
                        tasks.Clear();
                    }
                    timer.Stop();
                    metaInfos.ElapsedTime = metaInfos.ElapsedTime + timer.ElapsedMilliseconds;
                    metaInfos.Log = metaInfos.Log + $"[Info] Finished {models.Count} models warm up with {warmup_cycles} cycles.Warm up Total elapse: {metaInfos.ElapsedTime} ms, Average elapse: {metaInfos.ElapsedTime / models.Count / warmup_cycles} ms.\n";
                    metaInfos.ElapsedTime = 0;
                }
            });
            // 推理
            await Task.Run(() =>
            {
                while (cycleIndex < metaInfos.CycleTime)
                {
                    metaInfos.Log = metaInfos.Log + $"[Info]===== Cycle{cycleIndex + 1} Start =====\n";
                    for (int i = 0; i < imagePaths.Count(); i++)
                    {
                        Mat src = Cv2.ImRead(imagePaths[i], ImreadModes.Unchanged);
                        // 调整图片尺寸(实际部署可省略)
                        int srcW = src.Cols;
                        int srcH = src.Rows;
                        double scale_factor = 0.0;
                        ResumeParam param = new ResumeParam();
                        if(metaInfos.TargetHeight !=0 && metaInfos.TargetWidth != 0)
                        {
                            ImageHelper.rescale(src, ref src, metaInfos.TargetWidth, metaInfos.TargetHeight, ref scale_factor);
                            param.scale_factor = scale_factor;
                            param.src_height = srcH;
                            param.src_width = srcW;
                        }
                            
                        //Cv2.ImWrite("D:\\resized.jpg", src);

                        int h = src.Rows;
                        int w = src.Cols;
                        int c = src.Channels();


                        int idx = i % metaInfos.NumModels;
                        tasks.Add(CreateTask(metaInfos.ModelType, models[idx], src.Data, i, w, h, c, param));
                        // 如果模型全部匹配或者到达最后一张图
                        if ((idx == metaInfos.NumModels - 1) || i == imagePaths.Count - 1)
                        {
                            timer.Reset();
                            timer.Start();
                            for (int j = 0; j < tasks.Count; j++)
                            {
                                tasks[j].Start();
                            }
                            try
                            {
                                Task.WaitAll(tasks.ToArray());
                                metaInfos.Log = metaInfos.Log + $"[Info] predicted {tasks.Count} images.\n";
                                
                            }
                            catch (Exception ex)
                            {
                                throw ex;
                            }
                            tasks.Clear();
                            timer.Stop();
                            metaInfos.ElapsedTime = metaInfos.ElapsedTime + timer.ElapsedMilliseconds;
                        }
                    }
                    cycleIndex++;
                }
            });

            metaInfos.Log = metaInfos.Log + $"[Info] Finished all {metaInfos.CycleTime} cycles.\n";
            metaInfos.Log = metaInfos.Log + $"[Info] Total images: {imagePaths.Count * metaInfos.CycleTime}, Total elapse: {metaInfos.ElapsedTime} ms, Average elapse: [{metaInfos.ElapsedTime / (imagePaths.Count * metaInfos.CycleTime)}] ms.\n";
            metaInfos.Log = metaInfos.Log + "---------------------------------------------------------------------------\n";
            // 恢复初始值
            warmuped = true;
            cycleIndex = 0;
            metaInfos.ElapsedTime = 0;
            metaInfos.IsReady = true;
            timer.Reset();
        }


        // 分割推理  unsigned char* resultMap
        public void Callback_Predict_Seg(IntPtr modelObj, IntPtr imageData, int id, int width, int height, int channels, ResumeParam param)
        {
            IntPtr resultMap = IntPtr.Zero;
            try
            {
                int size1 = width * height;
                resultMap = Marshal.AllocHGlobal(size1);
                InferModel.ModelObjPredict_Seg(modelObj, imageData, width, height, channels, resultMap);
                Mat resultMat = new Mat(height, width, MatType.CV_8UC1, resultMap);

                if (metaInfos.SaveResult)
                {
                    EnqueueResult(new PredictResult(id, width, height, channels, metaInfos.SaveDir, param, (long)imageData, (long)resultMap));
                }
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        // 检测推理  float* output, int* nBoxesNum, char* LabelList
        public void Callback_Predict_Det(IntPtr modelObj, IntPtr imageData, int id, int width, int height, int channels, ResumeParam param)
        {
            IntPtr output = IntPtr.Zero;
            IntPtr nBoxesNum = IntPtr.Zero;
            IntPtr LabelList = IntPtr.Zero;

            try
            {
                output = Marshal.AllocHGlobal(MaxBoxNum * 6 * 4); // 最多100个box, 每个box=24个字节，100*24=2400
                nBoxesNum = Marshal.AllocHGlobal(4);
                LabelList = Marshal.AllocHGlobal(MaxBoxNum * MaxStrLen); // 最多100个box，每个box=49个char描述+1个空格，100*50=5000

                //Stopwatch timer1 = new Stopwatch();
                //timer1.Reset();
                //timer1.Start();
                InferModel.ModelObjPredict_Det(modelObj, imageData, width, height, channels, output, nBoxesNum, LabelList);
                //timer1.Stop();
                //metaInfos.Log = metaInfos.Log + $"[Info] seperate time test: {timer1.ElapsedMilliseconds} ms.\n";

                if (metaInfos.SaveResult)
                {
                    EnqueueResult(new PredictResult(id, width, height, channels, metaInfos.SaveDir, param, (long)imageData, 0, (long)output, (long)nBoxesNum, (long)LabelList));
                }
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        // 分类推理 IntPtr score, IntPtr category, IntPtr category_id
        public void Callback_Predict_Cls(IntPtr modelObj, IntPtr imageData, int id, int width, int height, int channels, ResumeParam param)
        {
            IntPtr score = IntPtr.Zero;
            IntPtr category = IntPtr.Zero;
            IntPtr category_id = IntPtr.Zero;
            try
            {
                score = Marshal.AllocHGlobal(4);  // float
                category = Marshal.AllocHGlobal(MaxStrLen); // 50个char
                category_id = Marshal.AllocHGlobal(4); // int
                InferModel.ModelObjPredict_Cls(modelObj, imageData, width, height, channels, score, category, category_id);

                if (metaInfos.SaveResult)
                {
                    EnqueueResult(new PredictResult(id, width, height, channels, metaInfos.SaveDir, param, (long)imageData, 0, 0, 0, 0, (long)score, (long)category, (long)category_id));
                }
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        // mask推理 float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList
        public void Callback_Predict_Mask(IntPtr modelObj, IntPtr imageData, int id, int width, int height, int channels, ResumeParam param)
        {
            IntPtr box_output = IntPtr.Zero;
            IntPtr mask_output = IntPtr.Zero;
            IntPtr nBoxesNum = IntPtr.Zero;
            IntPtr LabelList = IntPtr.Zero;
            try
            {
                box_output = Marshal.AllocHGlobal(MaxBoxNum * 6 * 4);
                mask_output = Marshal.AllocHGlobal(width * height);
                nBoxesNum = Marshal.AllocHGlobal(4);
                LabelList = Marshal.AllocHGlobal(MaxBoxNum * MaxStrLen);
                InferModel.ModelObjPredict_Mask(modelObj, imageData, width, height, channels, box_output, mask_output, nBoxesNum, LabelList);

                if (metaInfos.SaveResult)
                {
                    EnqueueResult(new PredictResult(id, width, height, channels, metaInfos.SaveDir, param, (long)imageData, 0, 0, 0, 0, 0, 0, 0, (long)box_output, (long)mask_output, (long)nBoxesNum, (long)LabelList));
                }
            }
            catch (Exception ex)
            {
                throw ex;
            }
        }

        public Task CreateTask(string taskType, IntPtr modelObj, IntPtr imageData, int id, int width, int height, int channels, ResumeParam param)
        {
            if (taskType == "seg")
                return new Task(() => Callback_Predict_Seg(modelObj, imageData, id, width, height, channels, param));
            else if (taskType == "det")
                return new Task(() => Callback_Predict_Det(modelObj, imageData, id, width, height, channels, param));
            else if (taskType == "clas")
                return new Task(() => Callback_Predict_Cls(modelObj, imageData, id, width, height, channels, param));
            else if (taskType == "mask")
                return new Task(() => Callback_Predict_Mask(modelObj, imageData, id, width, height, channels, param));
            else
                throw new System.Exception($"[Error] taskType can only be: clas/seg/det/mask, but got {taskType}");
        }

        private void ckbSave_Checked(object sender, RoutedEventArgs e)
        {
            // 如果选择保存，则指定目标文件夹
            try
            {
                FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
                var res = folderBrowserDialog.ShowDialog();
                if (res == System.Windows.Forms.DialogResult.OK)
                {
                    metaInfos.SaveDir = folderBrowserDialog.SelectedPath.Trim();
                    metaInfos.Log = metaInfos.Log + "[Info] Save Result Dir: " + metaInfos.SaveDir + "\n";

                    if (!Directory.Exists(metaInfos.SaveDir))
                    {
                        System.Windows.Forms.MessageBox.Show("Error: Save Dir does not Exist！");
                    }
                }
                else if(res == System.Windows.Forms.DialogResult.Cancel)
                {
                    metaInfos.SaveResult = false;
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Choose Save Dir Error: {ex.Message}");
            }
        }

        private void txtInfo_TextChanged(object sender, TextChangedEventArgs e)
        {
            txtInfo.ScrollToEnd();
        }

        private void EnqueueResult(PredictResult results)
        {
            if(resultQueue.Count < MaxQueueLength)
            {
                resultQueue.Enqueue(results);
            }
            else
            {
                metaInfos.Log = metaInfos.Log + $"[WARN] result queue length exceed {MaxQueueLength}, Space not enough or save speed is too slow.\n";
            }
        }

        // 对结果队列进行保存
        private void Callback_SaveResult()
        {
            while(!this.tokenSource.Token.IsCancellationRequested)
            {
                try
                {
                    if(resultQueue.Count>0)
                    {
                        PredictResult resultData = null;
                        if(resultQueue.TryDequeue(out resultData))
                        {
                            SaveResult(resultData);
                        }
                    }
                }
                catch(Exception ex)
                {
                    throw ex;
                }
                Thread.Sleep(50);
            }
        }


        // 保存预测结果：注意如果在界面指定了目标图片尺寸，则这里保存和显示的就是目标图片尺寸的预测结果，不会进行恢复操作
        // 如果希望恢复成原始图像，可借助predictResult中的ResumeParam对象预存的数据进行恢复
        public void SaveResult(PredictResult pred)
        {
            // save seg result
            if (pred.seg_resultMap != IntPtr.Zero)
            {
                // map转轮廓点集
                // 点集存入xml
                // TODO: 当前seg直接保存src和map，没有统一成xml，便于直接用mat做addweight，无需引入提取轮廓和点集的操作
                MatType srcType = MatType.CV_8UC1;
                if (pred.channel == 3) srcType = MatType.CV_8UC3;
                Mat src = new Mat(pred.height, pred.width, srcType, pred.src);
                Mat map = new Mat(pred.height, pred.width, MatType.CV_8UC1, pred.seg_resultMap);
                Cv2.ImWrite($"{pred.saveDir}\\src{pred.id}.bmp", src);
                Cv2.ImWrite($"{pred.saveDir}\\map{pred.id}.bmp", map);

                ResultData resultData = new ResultData();
                resultData.type = "seg";
                resultData.id = pred.id;
                resultData.regions = new Region[1];
                Region region = new Region();
                resultData.regions[0] = region;
                resultData.numRegions = 1;
                XmlSerializerHelper.WriteXML(resultData, $"{pred.saveDir}\\res{pred.id}.xml", typeof(ResultData));
                pred.Dispose();
            }
            // TODO: save det result
            else if (pred.det_nBoxesNum != IntPtr.Zero)
            {
                int[] numBoxes = new int[1];
                float[] output = new float[MaxBoxNum * 6];
                byte[] labelList = new byte[MaxBoxNum * MaxStrLen];
                Marshal.Copy(pred.det_nBoxesNum, numBoxes, 0, 1);
                Marshal.Copy(pred.det_output, output, 0, MaxBoxNum * 6);
                Marshal.Copy(pred.det_LabelList, labelList, 0, MaxBoxNum * MaxStrLen);
                /* save action here */
                MatType srcType = MatType.CV_8UC1;
                if (pred.channel == 3) srcType = MatType.CV_8UC3;
                Mat src = new Mat(pred.height, pred.width, srcType, pred.src);
                Cv2.ImWrite($"{pred.saveDir}\\src{pred.id}.bmp", src);

                ResultData resultData = new ResultData();
                resultData.type = "det";
                resultData.id = pred.id;
                resultData.regions = new Region[numBoxes[0]];
                string[] labelList_ = System.Text.Encoding.ASCII.GetString(labelList).Trim().Split(' ');
                int idx = 0;
                for (int i = 0; i < numBoxes[0]; i++)
                {
                    if(output[i * 6 + 1] > 0.5) // score > 0.5
                    {
                        Region region = new Region();
                        region.labelId = (int)output[i * 6 + 0];
                        region.score = output[i * 6 + 1];
                        region.numPoints = 2;
                        region.labelDesc = labelList_[i];
                        region.xx = new float[region.numPoints];
                        region.yy = new float[region.numPoints];
                        for (int j = 0; j < 2; j++)
                        {
                            region.xx[j] = output[i * 6 + 2 + 2 * j];
                            region.yy[j] = output[i * 6 + 3 + 2 * j];
                        }
                        resultData.regions[idx] = region;
                        idx++;
                    }
                }
                if (idx > 0)
                {
                    Array.Resize<Region>(ref resultData.regions, idx);
                }
                resultData.numRegions = idx;
                XmlSerializerHelper.WriteXML(resultData, $"{pred.saveDir}\\res{pred.id}.xml", typeof(ResultData));
                pred.Dispose();
            }
            // TODO: save clas result
            else if (pred.cls_category != IntPtr.Zero)
            {
                float[] score = new float[1];
                byte[] category = new byte[MaxStrLen];
                int[] category_id = new int[1];
                Marshal.Copy(pred.cls_score, score, 0, 1);
                Marshal.Copy(pred.cls_category, category, 0, MaxStrLen);
                Marshal.Copy(pred.cls_category_id, category_id, 0, 1);
                /* save action here */
                MatType srcType = MatType.CV_8UC1;
                if (pred.channel == 3) srcType = MatType.CV_8UC3;
                Mat src = new Mat(pred.height, pred.width, srcType, pred.src);
                Cv2.ImWrite($"{pred.saveDir}\\src{pred.id}.bmp", src);

                ResultData resultData = new ResultData();
                resultData.type = "clas";
                resultData.id = pred.id;
                resultData.regions = new Region[1];
                Region region = new Region();
                region.score = score[0];
                region.labelId = category_id[0];
                region.labelDesc = System.Text.Encoding.ASCII.GetString(category).Split('.')[0];
                resultData.regions[0] = region;
                resultData.numRegions = 1;
                XmlSerializerHelper.WriteXML(resultData, $"{pred.saveDir}\\res{pred.id}.xml", typeof(ResultData));
                pred.Dispose();
            }
            // TODO: save mask result
            else if (pred.mask_nBoxesNum != IntPtr.Zero)
            {
                float[] box_output = new float[MaxBoxNum * 6];
                int[] numBoxes = new int[1];
                byte[] labelList = new byte[MaxBoxNum * MaxStrLen];
                Marshal.Copy(pred.mask_box_output, box_output, 0, MaxBoxNum * 6);
                Marshal.Copy(pred.mask_nBoxesNum, numBoxes, 0, 1);
                Marshal.Copy(pred.mask_LabelList, labelList, 0, MaxBoxNum * MaxStrLen);
                /* save action here */
                MatType srcType = MatType.CV_8UC1;
                if (pred.channel == 3) srcType = MatType.CV_8UC3;
                Mat src = new Mat(pred.height, pred.width, srcType, pred.src);
                Mat map = new Mat(pred.height, pred.width, MatType.CV_8UC1, pred.mask_mask_output);
                Cv2.ImWrite($"{pred.saveDir}\\src{pred.id}.bmp", src);
                Cv2.ImWrite($"{pred.saveDir}\\map{pred.id}.bmp", map);

                ResultData resultData = new ResultData();
                resultData.type = "mask";
                resultData.id = pred.id;
                resultData.regions = new Region[numBoxes[0]];
                string[] labelList_ = System.Text.Encoding.ASCII.GetString(labelList).Trim().Split(' ');
                int idx = 0;
                for (int i = 0; i < numBoxes[0]; i++)
                {
                    if (box_output[i * 6 + 1] > 0.5) // score > 0.5
                    {
                        Region region = new Region();
                        region.labelId = (int)box_output[i * 6 + 0];
                        region.score = box_output[i * 6 + 1];
                        region.numPoints = 2;
                        region.labelDesc = labelList_[i];
                        region.xx = new float[region.numPoints];
                        region.yy = new float[region.numPoints];
                        for (int j = 0; j < 2; j++)
                        {
                            region.xx[j] = box_output[i * 6 + 2 + 2 * j];
                            region.yy[j] = box_output[i * 6 + 3 + 2 * j];
                        }
                        resultData.regions[idx] = region;
                        idx++;
                    }
                }
                if (idx > 0)
                {
                    Array.Resize<Region>(ref resultData.regions, idx);
                }
                resultData.numRegions = idx;
                XmlSerializerHelper.WriteXML(resultData, $"{pred.saveDir}\\res{pred.id}.xml", typeof(ResultData));
                pred.Dispose();
            }
        }



        private void menuClear_Click(object sender, RoutedEventArgs e)
        {
            metaInfos.Log = "";
        }

        private void menuExit_Click(object sender, RoutedEventArgs e)
        {
            ClearAllModels();
            System.Windows.Application.Current.Shutdown();
        }

        private void cmbPrecision_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            
            if (this.cmbPrecision.SelectedValue != null)
            {
                if (this.cmbPrecision.SelectedIndex >= 0)
                {
                    metaInfos.Precision = this.cmbPrecision.SelectedIndex;
                    if(metaInfos.UseTrt) metaInfos.Log = metaInfos.Log + $"[Info] TensorRT precision mode: {precisionModes[metaInfos.Precision]}.\n";

                }
            }
        }

        private void btnVis_Click(object sender, RoutedEventArgs e)
        {
            // 打开新窗口
            VisualizeWindow visWindow = new VisualizeWindow(metaInfos);
            visWindow.ShowDialog();
        }
    }
}
