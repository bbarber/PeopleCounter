using Yolov8Net;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;

using var yolo = YoloV8Predictor.Create("./Models/yolov8m.onnx");

// Provide an input image.  Image will be resized to model input if needed.
using var image = SixLabors.ImageSharp.Image.Load("./Assets/doggo.jpg");
var predictions = yolo.Predict(image);


foreach (var pred in predictions)
{
    var originalImageHeight = image.Height;
    var originalImageWidth = image.Width;

    var x = Math.Max(pred.Rectangle.X, 0);
    var y = Math.Max(pred.Rectangle.Y, 0);
    var width = Math.Min(originalImageWidth - x, pred.Rectangle.Width);
    var height = Math.Min(originalImageHeight - y, pred.Rectangle.Height);

    // Bounding Box Text
    string text = $"{pred.Label.Name} [{pred.Score}]";
    Console.WriteLine(text);
}