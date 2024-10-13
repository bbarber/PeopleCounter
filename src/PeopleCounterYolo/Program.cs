using Emgu.CV;
using Yolov8Net;



using var yolo = YoloV8Predictor.Create("./Models/yolov8m.onnx");

var filename = "./Assets/webcam.jpg";
if (args.Length > 0) filename = args[0];
using var capture = new VideoCapture(0, VideoCapture.API.Any);

while (true)
{
    capture.Set(Emgu.CV.CvEnum.CapProp.AutoExposure, 20); // Enable auto exposure
    var webcamImage = capture.QueryFrame();
    webcamImage.Save(filename);

    using var image = SixLabors.ImageSharp.Image.Load("./Assets/webcam.jpg");
    var predictions = yolo.Predict(image);

    Console.WriteLine("Predictions:");
    foreach (var pred in predictions)
    {
        string text = $"{pred.Label.Name} [{pred.Score}]";
        Console.WriteLine(text);
    }
}
