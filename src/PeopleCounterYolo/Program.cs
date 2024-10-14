using Emgu.CV;
using Yolov8Net;
using Emgu.CV.CvEnum;
using SixLabors.ImageSharp;
using System.Diagnostics;
using SixLabors.ImageSharp.PixelFormats;
using Emgu.CV.Structure;
using InfluxDB3.Client;
using System.Text.Json;
using InfluxDB3.Client.Write;


var hostUrl = "https://eastus-1.azure.cloud2.influxdata.com";
var secretsJson = File.ReadAllText("../../../secrets.json");
var secrets = JsonSerializer.Deserialize<Secrets>(secretsJson);
using var client = new InfluxDBClient(hostUrl, secrets.InfluxDbToken);

const string influxBucket = "People-QA";
const string location = "The Code Mine";


var targetFps = 2;
var previewWebcam = false;
var previewWebcamImage = "./Assets/webcam.jpg";

// When running on windows current dir is the bin
var modelPath = "../../../Models/yolo11n.onnx";

using var yolo = YoloV8Predictor.Create(modelPath);
using var capture = new VideoCapture(0, VideoCapture.API.Any);

var measurementStopwatch = Stopwatch.StartNew();
var measurementInterval = TimeSpan.FromSeconds(5);
var peopleCount = 0;

while (true)
{
    var frameStopwatch = Stopwatch.StartNew();

    // Take an image
    capture.Set(CapProp.AutoExposure, 20);
    var webcamImage = capture.QueryFrame();
    var imageBytes = webcamImage.ToImage<Bgr, byte>().ToJpegData();

    // Optionally save the image to disk (for debugging)
    if (previewWebcam) webcamImage.Save(previewWebcamImage);

    // Load the image and predict
    using var image = Image.Load<Rgba32>(imageBytes);
    var predictions = yolo.Predict(image);

    if (predictions.Any(p => p.Label.Name == "person"))
    {
        peopleCount++;
    }

    // Calculate fps
    Console.WriteLine($"FPS: {1000 / frameStopwatch.ElapsedMilliseconds}");
    Console.WriteLine("Predictions:");
    Console.WriteLine(string.Join("\n", predictions.Select(p => $"{p.Label!.Name} [{p.Score}]")));

    // Throttle how many images / predictions we make per second
    var sleep = 1000 / targetFps - frameStopwatch.ElapsedMilliseconds;
    Thread.Sleep(Math.Max((int)sleep, 0));


    // On each measurement interval, send the people count to InfluxDB
    if (measurementStopwatch.Elapsed > measurementInterval)
    {
        Console.WriteLine($"People count: {peopleCount}");
        var point = PointData.Measurement("people")
            .SetTag("location", "The Code Mine")
            .SetIntegerField("Count", peopleCount)
            .SetBooleanField("Detected", peopleCount > 0);

        await client.WritePointAsync(point, influxBucket);

        peopleCount = 0;
        measurementStopwatch.Restart();
    }
}
