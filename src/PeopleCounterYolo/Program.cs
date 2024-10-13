using System.Diagnostics;
using Yolov8Net;

var sw = Stopwatch.StartNew();
using var yolo = YoloV8Predictor.Create("./Models/yolov8m.onnx");
Console.WriteLine($"Model loaded in {sw.ElapsedMilliseconds}ms");

sw.Restart();
using var image = SixLabors.ImageSharp.Image.Load("./Assets/person_01.jpg");
Console.WriteLine($"Image loaded in {sw.ElapsedMilliseconds}ms");

sw.Restart();
var predictions = yolo.Predict(image);
Console.WriteLine($"Predictions done in {sw.ElapsedMilliseconds}ms");


foreach (var pred in predictions)
{
    string text = $"{pred.Label.Name} [{pred.Score}]";
    Console.WriteLine(text);
}