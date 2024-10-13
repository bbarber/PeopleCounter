using Emgu.CV;
using Yolov8Net;
using Emgu.CV.CvEnum;
using SixLabors.ImageSharp;
using System.Diagnostics;


using var yolo = YoloV8Predictor.Create("./Models/yolo11n.onnx");

var filename = "./Assets/webcam.jpg";
if (args.Length > 0) filename = args[0];
using var capture = new VideoCapture(0, VideoCapture.API.Any);

while (true)
{
    var sw = Stopwatch.StartNew();
    capture.Set(CapProp.AutoExposure, 20);
    var webcamImage = capture.QueryFrame();

    using var image = Image.Load("./Assets/webcam.jpg");
    var predictions = yolo.Predict(image);

    Console.WriteLine("Predictions:");
    foreach (var pred in predictions)
    {
        string text = $"{pred.Label.Name} [{pred.Score}]";
        Console.WriteLine(text);
    }

    Console.WriteLine($"Prediction took {sw.ElapsedMilliseconds}ms. (FPS: {1000 / sw.ElapsedMilliseconds})");
}
