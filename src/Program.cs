using Emgu.CV;
using Emgu.CV.CvEnum;

var win1 = "Test Window (Press any key to close)"; //The name of the window
CvInvoke.NamedWindow(win1); //Create the window using the specific name
using (Mat frame = new Mat(256, 256, DepthType.Cv8U, 3)) //Create a matrix to store the image)
using (VideoCapture capture = new VideoCapture())
    while (CvInvoke.WaitKey(1) == -1)
    {
        capture.Read(frame);
        CvInvoke.Imshow(win1, frame);
    }