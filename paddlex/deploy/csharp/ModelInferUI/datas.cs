using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace ModelInferUI
{
    public class BaseNotifyObj : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string name = null)
        {
            PropertyChangedEventHandler handler = PropertyChanged;
            handler?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }

    public class MetaInformation : BaseNotifyObj
    {
        // 日志信息
        private string log;
        public string Log { get { return log; } set { log = value; OnPropertyChanged(); } }
        // 推理图像文件夹
        private string imgDir;
        public string ImgDir { get { return imgDir; } set { imgDir = value; OnPropertyChanged(); } }
        
        // 模型根文件夹
        private string modelRootDir;
        public string ModelRootDir { get { return modelRootDir; } set { modelRootDir = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }

        // 多组模型文件夹
        public List<string> ModelDirs = new List<string>();
        // 多组模型文件列表
        public List<List<string>> ModelFiles = new List<List<string>>();
        
        // 图像目标宽度
        private int targetWidth;
        public int TargetWidth { get { return targetWidth; } set { targetWidth = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }
        // 图像目标高度
        private int targetHeight;
        public int TargetHeight { get { return targetHeight; } set { targetHeight = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }

        // 单模型线程个数
        private int numModels;
        public int NumModels { get { return numModels; } set { numModels = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }
        // 模型类型
        private string modelType;
        public string ModelType { get { return modelType; } set { modelType = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }
        // gpu号
        private int gpuId;
        public int GpuId { get { return gpuId; } set { gpuId = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }
        // 是否使用TensorRT
        private bool useTrt;
        public bool UseTrt { get { return useTrt; } set { useTrt = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }

        private int minW;
        public int MinW { get { return minW; } set { minW = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }
        private int minH;
        public int MinH { get { return minH; } set { minH = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }

        private int maxW;
        public int MaxW { get { return maxW; } set { maxW = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }
        private int maxH;
        public int MaxH { get { return maxH; } set { maxH = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }

        private int optW;
        public int OptW { get { return optW; } set { optW = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }
        private int optH;
        public int OptH { get { return optH; } set { optH = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }

        private int minSubgraphSize;
        public int MinSubgraphSize { get { return minSubgraphSize; } set { minSubgraphSize = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }

        private int precision;
        public int Precision { get { return precision; } set { precision = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }

        // 新增fine tune参数
        private string shapeRangeInfoPath;
        public string ShapeRangeInfoPath { get { return shapeRangeInfoPath; } set { shapeRangeInfoPath = value; OnPropertyChanged(); } }

        // 是否需要更新模型
        private bool needUpdate;
        public bool NeedUpdate { get { return needUpdate; } set { needUpdate = value; IsReady = ModelDirs.Count > 1 ? true : !value; OnPropertyChanged(); } }
        // 是否准备好进行推理
        private bool isReady;
        public bool IsReady { get { return isReady; } set { isReady = value; OnPropertyChanged(); } }
        // 是否进行模型预热
        private bool warmUp;
        public bool WarmUp { get { return warmUp; } set { warmUp = value; NeedUpdate = ModelDirs.Count > 1 ? false : true; OnPropertyChanged(); } }
        // 是否保存推理结果
        private bool saveResult;
        public bool SaveResult { get { return saveResult; } set { saveResult = value; OnPropertyChanged(); } }
        // 结果保存路径
        private string saveDir;
        public string SaveDir { get { return saveDir; } set { saveDir = value; OnPropertyChanged(); } }
        // 循环轮次
        private int cycleTime;
        public int CycleTime { get { return cycleTime; } set { cycleTime = value; OnPropertyChanged(); } }
        // 总耗时
        private long elapsedTime;
        public long ElapsedTime { get { return elapsedTime; } set { elapsedTime = value; OnPropertyChanged(); } }

        public MetaInformation()
        {
            IsReady = false;
            NumModels = 1;
            GpuId = 0;
            NeedUpdate = false;
            WarmUp = true;
            SaveResult = false;
            CycleTime = 1;
            UseTrt = false;
            TargetWidth = 512;
            TargetHeight = 512;
            MinSubgraphSize = 3;
            MinW = 0;MinH = 0;
            MaxW = 0;MaxH = 0;
            OptW = 0;OptH = 0;
        }
    }

    // 图像rescale恢复参数
    public class ResumeParam
    {
        public int src_width;
        public int src_height;
        public double scale_factor;
        public ResumeParam(int srcW, int srcH, double scaleF)
        {
            src_width = srcW;
            src_height = srcH;
            scale_factor = scaleF;
        }
        public ResumeParam()
        {
            src_width = 0;
            src_height = 0;
            scale_factor = 0.0;
        }
    }

    // 推理结果
    public class PredictResult
    {
        public int id;
        public int width;
        public int height;
        public int channel;
        public string saveDir;
        public ResumeParam resumeParam = null;
        public IntPtr src = IntPtr.Zero;
        public IntPtr seg_resultMap = IntPtr.Zero;
        public IntPtr det_output = IntPtr.Zero;
        public IntPtr det_nBoxesNum = IntPtr.Zero;
        public IntPtr det_LabelList = IntPtr.Zero;
        public IntPtr cls_score = IntPtr.Zero;
        public IntPtr cls_category = IntPtr.Zero;
        public IntPtr cls_category_id = IntPtr.Zero;
        public IntPtr mask_box_output = IntPtr.Zero;
        public IntPtr mask_mask_output = IntPtr.Zero;
        public IntPtr mask_nBoxesNum = IntPtr.Zero;
        public IntPtr mask_LabelList = IntPtr.Zero;

        public PredictResult(int id, int w, int h, int c, string saveDir, ResumeParam param = null, long src = 0, long seg_resultMap = 0, long det_output = 0,
            long det_nBoxesNum = 0, long det_LabelList = 0, long cls_score = 0, long cls_category = 0, long cls_category_id = 0,
            long mask_box_output = 0, long mask_mask_output = 0, long mask_nBoxesNum = 0, long mask_LabelList = 0)
        {
            this.id = id;
            this.width = w;
            this.height = h;
            this.channel = c;
            this.saveDir = saveDir;
            this.resumeParam = param;
            this.src = new IntPtr(src);
            this.seg_resultMap = new IntPtr(seg_resultMap);
            this.det_output = new IntPtr(det_output);
            this.det_nBoxesNum = new IntPtr(det_nBoxesNum);
            this.det_LabelList = new IntPtr(det_LabelList);
            this.cls_score = new IntPtr(cls_score);
            this.cls_category = new IntPtr(cls_category);
            this.cls_category_id = new IntPtr(cls_category_id);
            this.mask_box_output = new IntPtr(mask_box_output);
            this.mask_mask_output = new IntPtr(mask_mask_output);
            this.mask_nBoxesNum = new IntPtr(mask_nBoxesNum);
            this.mask_LabelList = new IntPtr(mask_LabelList);
        }

        public void Dispose()
        {
            // 注意：IntPtr src不能手动释放，因为该指针不是手动Marshal.AllocHGlobal()申请的，而是从Mat数据中获取的由Mat本身负责释放。
            //if(this.src != IntPtr.Zero) Marshal.FreeHGlobal(this.src);

            if (this.seg_resultMap != IntPtr.Zero) Marshal.FreeHGlobal(this.seg_resultMap);
            if(this.det_output != IntPtr.Zero) Marshal.FreeHGlobal(this.det_output);
            if(this.det_nBoxesNum != IntPtr.Zero) Marshal.FreeHGlobal(this.det_nBoxesNum);
            if(this.det_LabelList != IntPtr.Zero) Marshal.FreeHGlobal(this.det_LabelList);
            if(this.cls_score != IntPtr.Zero) Marshal.FreeHGlobal(this.cls_score);
            if(this.cls_category != IntPtr.Zero) Marshal.FreeHGlobal(this.cls_category);
            if(this.cls_category_id != IntPtr.Zero) Marshal.FreeHGlobal(this.cls_category_id);
            if(this.mask_box_output != IntPtr.Zero) Marshal.FreeHGlobal(this.mask_box_output);
            if(this.mask_mask_output != IntPtr.Zero) Marshal.FreeHGlobal(this.mask_mask_output);
            if(this.mask_nBoxesNum != IntPtr.Zero) Marshal.FreeHGlobal(this.mask_nBoxesNum);
            if(this.mask_LabelList != IntPtr.Zero) Marshal.FreeHGlobal(this.mask_LabelList);
        }
    }

    // 记录每个检测区域的信息：可以是det的box结果，也可以是seg的region结果
    [Serializable]
    public class Region
    {
        public float[] xx;          // x坐标列表
        public float[] yy;          // y坐标列表
        public int numPoints;       // 区域总点数
        public int labelId;         // 标签id
        public string labelDesc;    // 标签描述
        public float score;         // 预测得分
    }

    // 记录一张图片上所有检测region的结果
    [Serializable]
    public class ResultData
    {
        public string type;         // 模型类型[clas, det, seg, mask, paddlex]
        public int id;              // 图片编号
        public int numRegions;      // 区域个数
        public Region[] regions;    // 所有区域列表
    }


    // 可视化界面数据绑定: 代表一张图片的显示关联信息
    public class ShowData : BaseNotifyObj
    {
        //private string resultDir;
        //public string ResultDir { get { return resultDir; } set { resultDir = value; OnPropertyChanged(); } }

        private string srcPath;
        public string SrcPath { get { return srcPath; } set { srcPath = value; OnPropertyChanged(); } }

        private string resPath;
        public string ResPath { get { return resPath; } set { resPath = value; OnPropertyChanged(); } }

        private string mapPath;
        public string MapPath { get { return mapPath; } set { mapPath = value; OnPropertyChanged(); } }

        private int id;
        public int Id { get { return id; } set { id = value; OnPropertyChanged(); } }

        public ShowData(string srcPath, string resPath, string mapPath, int id)
        {
            this.SrcPath = srcPath;
            this.ResPath = resPath;
            this.MapPath = mapPath;
            this.Id = id;
        }
    }
}

