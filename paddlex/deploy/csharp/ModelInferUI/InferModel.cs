using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;   // Marshal, DllImport
using OpenCvSharp;

namespace ModelInferUI
{
    class InferModel
    {
        // model_infer v4接口
        //// 增加多线程二次封装接口
        //extern "C" __declspec(dllexport) ModelWrapper* ModelObjInit(const char* model_type, const char* model_dir, int gpu_id, bool use_trt, char* paddlex_model_type = NULL,
        //    const int* min_input_shape = NULL, const int* max_input_shape = NULL, const int* optim_input_shape = NULL, const int precision = 0, const int min_subgraph_size = 40,
        //    int target_width = 512, int target_height = 512, const char* shape_range_info_path = NULL);

        //extern "C" __declspec(dllexport) void ModelObjDestruct(ModelWrapper* modelObj);
        //extern "C" __declspec(dllexport) void ModelObjPredict_Seg(ModelWrapper* modelObj, unsigned char* imageData, int width, int height, int channels, unsigned char* resultMap);
        //extern "C" __declspec(dllexport) void ModelObjPredict_Det(ModelWrapper* modelObj, unsigned char* imageData, int width, int height, int channels, float* output, int* nBoxesNum, char* LabelList);
        //extern "C" __declspec(dllexport) void ModelObjPredict_Cls(ModelWrapper* modelObj, unsigned char* imageData, int width, int height, int channels, float* score, char* category, int* category_id);
        //extern "C" __declspec(dllexport) void ModelObjPredict_Mask(ModelWrapper* modelObj, unsigned char* imageData, int width, int height, int channels, float* box_output, unsigned char* mask_output, int* nBoxesNum, char* LabelList);

        // 模型初始化
        //[DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        //public extern static IntPtr ModelObjInit(string model_type, string model_dir, int gpu_id, bool use_trt,
        //    IntPtr paddlex_model_type = default(IntPtr), IntPtr min_input_shape = default(IntPtr), IntPtr max_input_shape = default(IntPtr), IntPtr opt_input_shape = default(IntPtr), int precision = 0, int min_subgraph_size = 40);

        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static IntPtr ModelObjInit(string model_type, string model_dir, int gpu_id, bool use_trt,
            IntPtr paddlex_model_type = default(IntPtr), IntPtr min_input_shape = default(IntPtr), IntPtr max_input_shape = default(IntPtr), IntPtr opt_input_shape = default(IntPtr), int precision = 0, int min_subgraph_size = 40,
            int target_width = 512, int target_height = 512, string shape_range_info_path = default(string));

        // seg推理: resultMap为跟原图一样大小(w*h)的单通道图，每个像素值代表一个类别号(0为背景，1-255为前景)
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjPredict_Seg(IntPtr modelObj, IntPtr imageData, int width, int height, int channels, IntPtr resultMap);

        // det推理：output为所有预测结果，长度为nbox*6(其中6分别表示id(int),score,x1,y1,x2,y2(float)，4*6=24个字节)，nBoxesNum表示box个数，LabelList为每个box的类别字符串描述，用空格隔开
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjPredict_Det(IntPtr modelObj, IntPtr imageData, int width, int height, int channels, IntPtr output, IntPtr nBoxesNum, IntPtr LabelList);
        
        // cls推理：score表示该预测类别得分(0-1)，category_id表示类别id，category表示类别的字符串描述
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjPredict_Cls(IntPtr modelObj, IntPtr imageData, int width, int height, int channels, IntPtr score, IntPtr category, IntPtr category_id);
        
        // mask推理：box_output表示每个box的预测结果，长度为nbox*6(其中6分别表示id,score,x1,y1,x2,y2)，mask_output表示原图大小的像素分类预测(即resultmap)，LabelList为每个box的类别字符串描述，用空格隔开，nBoxesNum表示box个数
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjPredict_Mask(IntPtr modelObj, IntPtr imageData, int width, int height, int channels, IntPtr box_output, IntPtr mask_output, IntPtr nBoxesNum, IntPtr LabelList);

        // 模型资源回收
        [DllImport("model_infer.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public extern static void ModelObjDestruct(IntPtr modelObj);

    }
}
