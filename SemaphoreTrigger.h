#define MODE RACE // RACE or PARKING
#define TIMEOUT_TIME 10 // Timeout se nao encontrar nada
#define CAMERA_NO 2 // Camera number

#define RACE 0
#define PARKING 1

#define STARTING 0
#define READY 1
#define GO 2

#include <iostream>
#include <iomanip>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <cv.h>
#include <highgui.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cvblob.h>

using namespace cv;
using namespace cvb;
using namespace std;

//void waitForIt();
int main();
