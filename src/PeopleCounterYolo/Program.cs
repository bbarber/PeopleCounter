using Emgu.CV;
using Yolov8Net;
using Emgu.CV.CvEnum;
using SixLabors.ImageSharp;
using System.Diagnostics;

using var yolo = YoloV8Predictor.Create("./Models/yolo11n.onnx");
var filename = "./Assets/webcam.jpg";
using var capture = new VideoCapture(0, VideoCapture.API.Any);
var targetFps = 2;

while (true)
{
    var sw = Stopwatch.StartNew();

    // Take an image
    capture.Set(CapProp.AutoExposure, 20);
    var webcamImage = capture.QueryFrame();
    webcamImage.Save(filename);

    // Load the image and predict
    using var image = Image.Load("./Assets/webcam.jpg");
    var predictions = yolo.Predict(image);

    Console.WriteLine("Predictions:");
    Console.WriteLine(string.Join("\n", predictions.Select(p => $"{p.Label!.Name} [{p.Score}]")));

    // Throttle how many images / predictions we make per second
    var sleep = 1000 / targetFps - sw.ElapsedMilliseconds;
    Thread.Sleep(Math.Max((int)sleep, 0));
}
