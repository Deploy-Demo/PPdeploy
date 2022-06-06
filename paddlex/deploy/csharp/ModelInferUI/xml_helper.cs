using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace ModelInferUI
{
    public class XmlSerializerHelper
    {
        public static bool WriteXML(object data, string XmlFilePath, Type type)
        {
            bool flag = true;
            StreamWriter writer = null;
            var serializer = new XmlSerializer(type); // 序列化器：需要指定被序列化对象类型
            try
            {
                writer = new StreamWriter(XmlFilePath, false); //构建文件写入器
                serializer.Serialize(writer, data); // 序列化结果写入文件
            }
            catch (Exception)
            {
                flag = false;
            }
            finally
            {
                if (writer != null)
                {
                    writer.Close();
                }
            }
            return flag;
        }

        public static object ReadXML(string XmlFilePath, Type type)
        {
            object object4Read = null;
            var serializer = new XmlSerializer(type);
            if (!File.Exists(XmlFilePath)) // 如果读取的文件不存在则返回新文件
            {
                return new object();
            }
            while (object4Read == null)
            {
                var stream = new FileStream(XmlFilePath, FileMode.Open); // 打开文件
                try
                {
                    object4Read = serializer.Deserialize(stream); // 尝试文件逆序列化解析到object
                }
                finally
                {
                    stream.Close();
                }
            }
            return object4Read;
        }
    }
}
