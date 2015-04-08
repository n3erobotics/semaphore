// Cristóbal Carnero Liñán <grendel.ccl@gmail.com>

#include <iostream>
#include <iomanip>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


#if (defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__) || (defined(__APPLE__) & defined(__MACH__)))
#include <cv.h>
#include <highgui.h>
#else
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

#include <cvblob.h>
using namespace cv;
using namespace cvb;
using namespace std;

void my_handler(int s){

           cout << "GOT CTRL + C -> Exiting" << endl;
           exit(1); 

}

class Timer
{
public:
    Timer() { clock_gettime(CLOCK_REALTIME, &beg_); }

    double elapsed() {
        clock_gettime(CLOCK_REALTIME, &end_);
        return end_.tv_sec - beg_.tv_sec +
            (end_.tv_nsec - beg_.tv_nsec) / 1000000000.;
    }

    void reset() { clock_gettime(CLOCK_REALTIME, &beg_); }

private:
    timespec beg_, end_;
};


int main()
{
  CvTracks tracks;
	Mat cropped, mat_converted;
	Timer timer;

	double minx, miny, maxx, maxy, area;
	
	struct sigaction sigIntHandler;

	sigIntHandler.sa_handler = my_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;

   sigaction(SIGINT, &sigIntHandler, NULL);
	
  cvNamedWindow("red_object_tracking", CV_WINDOW_AUTOSIZE);

  CvCapture *capture = cvCaptureFromCAM(7);
  
	cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 640 );
	cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 480 );
  
  cvGrabFrame(capture);
  IplImage *img = cvRetrieveFrame(capture);


  CvSize imgSize = cvGetSize(img);

  IplImage *frame = cvCreateImage(imgSize, img->depth, img->nChannels);

  IplConvKernel* morphKernel = cvCreateStructuringElementEx(5, 5, 1, 1, CV_SHAPE_RECT, NULL);

  //unsigned int frameNumber = 0;
  unsigned int blobNumber = 0;

  bool quit = false;
  while (!quit&&cvGrabFrame(capture))
  {
    IplImage *img = cvRetrieveFrame(capture);

	timer.reset();

    cvConvertScale(img, frame, 1, 0);

    IplImage *segmentated = cvCreateImage(imgSize, 8, 1);


    // Detecting red pixels:
    // (This is very slow, use direct access better...)
    for (unsigned int j=0; j<imgSize.height; j++)
      for (unsigned int i=0; i<imgSize.width; i++)
      {
	CvScalar c = cvGet2D(frame, j, i);
	
	double b = ((double)c.val[0])/255.;
	double g = ((double)c.val[1])/255.;
	double r = ((double)c.val[2])/255.;
	unsigned char f = 255*((g>0.2+r)&&(g>0.2+b));

	cvSet2D(segmentated, j, i, CV_RGB(f, f, f));
      }


    cvMorphologyEx(segmentated, segmentated, NULL, morphKernel, CV_MOP_OPEN, 1);

    //cvShowImage("segmentated", segmentated);

    IplImage *labelImg = cvCreateImage(cvGetSize(frame), IPL_DEPTH_LABEL, 1);

    CvBlobs blobs;
    unsigned int result = cvLabel(segmentated, labelImg, blobs);
    cvFilterByArea(blobs, 500, 1000000);
    cvRenderBlobs(labelImg, blobs, frame, frame, CV_BLOB_RENDER_BOUNDING_BOX);
    cvUpdateTracks(blobs, tracks, 200., 5);
    cvRenderTracks(tracks, frame, frame, CV_TRACK_RENDER_ID|CV_TRACK_RENDER_BOUNDING_BOX);

    cvShowImage("red_object_tracking", frame);
    /*std::stringstream filename;
    filename << "redobject_" << std::setw(5) << std::setfill('0') << frameNumber << ".png";
    cvSaveImage(filename.str().c_str(), frame);*/

    cvReleaseImage(&labelImg);
    cvReleaseImage(&segmentated);

	mat_converted = Mat(frame,false);

	for (CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it)
	{
		minx = it->second->minx;
		miny = it->second->miny;
		maxx = it->second->maxx;
		maxy = it->second->maxy;
		area = it->second->area;

		//Printing the position information
		cout<<"X: "<<minx<<" Y: "<<miny<<"Xmax: "<<maxx<<" Ymax: "<<maxy<<endl;
		rectangle(mat_converted, Point( minx, miny ), Point( maxx, maxy), Scalar( 255, 255, 0 ), +1, 4 );
	}
	
	if (minx >= 0 && maxx >= 0 && miny >= 0 && maxy >= 0 && maxx >= minx && maxy>=miny){
	
		imshow("mat",mat_converted);
		cv::Rect myROI(minx, miny, maxx-minx, maxy-miny);
		//sleep(1);
	// Crop the full image to that image contained by the rectangle myROI
	// Note that this doesn't copy the data
		cv::Mat image(mat_converted);
		cv::Mat croppedImage = image(myROI);
	
		imshow("croppedImage",croppedImage);
	}
    char k = cvWaitKey(10)&0xff;
    switch (k)
    {
      case 27:
      case 'q':
      case 'Q':
        quit = true;
        break;
      case 's':
      case 'S':
        for (CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it)
        {
          std::stringstream filename;
          filename << "redobject_blob_" << std::setw(5) << std::setfill('0') << blobNumber << ".png";
          cvSaveImageBlob(filename.str().c_str(), img, it->second);
          blobNumber++;

          std::cout << filename.str() << " saved!" << std::endl;
        }
        break;
    }

    cvReleaseBlobs(blobs);

	cout << "Time: " << timer.elapsed() << endl;

    //frameNumber++;
  }

  cvReleaseStructuringElement(&morphKernel);
  cvReleaseImage(&frame);

  cvDestroyWindow("red_object_tracking");

  return 0;
}
