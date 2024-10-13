using Emgu.CV;
using Yolov8Net;
using Emgu.CV.CvEnum;
using SixLabors.ImageSharp;
using System.Diagnostics;
using SixLabors.ImageSharp.PixelFormats;
using Emgu.CV.Structure;

var targetFps = 2;
var previewWebcam = true;
var previewWebcamImage = "./Assets/webcam.jpg";
var modelPath = "./Models/yolo11n.onnx";

using var yolo = YoloV8Predictor.Create(modelPath);
using var capture = new VideoCapture(0, VideoCapture.API.Any);

while (true)
{
    var sw = Stopwatch.StartNew();

    // Take an image
    capture.Set(CapProp.AutoExposure, 20);
    var webcamImage = capture.QueryFrame();
    var imageBytes = webcamImage.ToImage<Bgr, byte>().ToJpegData();

    // Optionally save the image to disk (for debugging)
    if (previewWebcam) webcamImage.Save(previewWebcamImage);

    // Load the image and predict
    using var image = Image.Load<Rgba32>(imageBytes); //Image.Load("./Assets/webcam.jpg");
    var predictions = yolo.Predict(image);

    Console.WriteLine("Predictions:");
    Console.WriteLine(string.Join("\n", predictions.Select(p => $"{p.Label!.Name} [{p.Score}]")));

    // Throttle how many images / predictions we make per second
    var sleep = 1000 / targetFps - sw.ElapsedMilliseconds;
    Thread.Sleep(Math.Max((int)sleep, 0));
}
