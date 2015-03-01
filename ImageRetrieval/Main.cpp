/**
* CS4185/CS5185 Multimedia Technologies and Applications
* Course Assignment
* Image Retrieval Project
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <time.h>
#include <stdio.h>
#include <direct.h>
#include <fstream>
#include <iostream>
#include <algorithm>
using namespace std;
using namespace cv;

#define IMAGE_LIST_FILE "inputimage.txt"

typedef struct image//自定义结构体
{
	double position_score;
	double size_score;
	double color_score;
	double total_score;
	char image_name[200];

}image;


Mat src_input, gray_input;
Mat db_img, db_gray_img;
Mat src_canny;
vector<image> images;

double score[1000];
int db_id = 0;
double maxscore = 1000000000;
double maxscore1 = 1000000000;
int maxscore_num;
char maximg_name[200];
Mat max_img;

char* window_name = "input";

Mat canny(int, void*, Mat source)
{
	Mat src_gray;
	Mat detectEdges;
	Mat dst;   //canny block
	int lowThreshold = 100;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;
	cvtColor(source, src_gray, CV_BGR2GRAY);
	blur(src_gray, detectEdges, Size(3, 3));
	Canny(detectEdges, detectEdges, lowThreshold, lowThreshold*ratio, kernel_size);
	dst = Scalar::all(0);
	source.copyTo(dst, detectEdges);

	return dst;
}

void cah(Mat img, Mat &img_color, Mat &nomalized)
{
	img_color = Mat::zeros(37, 8, CV_32S);
	Mat hsv_img;
	cvtColor(img, hsv_img, CV_BGR2HSV);
	int cols = hsv_img.cols, rows = hsv_img.rows;

	for (int i = 0; i < cols; i++)
	for (int j = 0; j < rows; j++)
	{
		Vec3b hsv = hsv_img.at<Vec3b>(j, i);
		double h = hsv.val[0] * 2;
		double s = hsv.val[1] / 255.00;
		double v = hsv.val[2] / 255.00;

		int color;
		if (v >= 0 && v < 0.2) color = 0;
		else if (s >= 0 && s < 0.2)
		{
			if (v >= 0.2 && v < 0.8)
				color = (int)floor((v - 0.2) * 10) + 1;
			else
				color = 36;
		}
		else
		{
			if (h >= 0 && h < 22) h = 2;
			else if (h >= 22 && h < 45) h = 1;
			else if (h >= 45 && h < 70) h = 0;
			else if (h >= 70 && h < 155) h = 5;
			else if (h >= 155 && h < 186) h = 4;
			else if (h >= 186 && h < 260) h = 6;
			else if (h >= 260 && h < 330) h = 3;
			else h = 2;

			if (s >= 0.2 && s < 0.65) s = 0;
			else s = 1;

			if (v >= 0.2 && v < 0.7) v = 0;
			else v = 1;

			color = 4 * h + 2 * s + v + 8;
		}

		int position;
		int x = i - cols / 2;
		int y = j - rows / 2;
		if (x == 0) {
			if (y >= 0) position = 2;
			else position = 6;
		}
		else{
			double gradient = double(y) / x;
			if (gradient >= 0 && gradient < 1) {
				if (x > 0) position = 0;
				else position = 4;
			}
			else if (gradient >= 1) {
				if (x > 0) position = 1;
				else position = 5;
			}
			else if (gradient < -1) {
				if (x < 0) position = 2;
				else position = 6;
			}
			else if (gradient >= -1 && gradient < 0) {
				if (x < 0) position = 3;
				else position = 7;
			}
		}

		img_color.at<int>(color, position)++;
	}
	nomalized = Mat::zeros(37, 8, CV_64F);
	int maxScore = 0;
	for (int i = 0; i < 37; i++)
	for (int j = 0; j < 8; j++)
		maxScore = maxScore > img_color.at<int>(i, j) ? maxScore : img_color.at<int>(i, j);
	for (int i = 0; i < 37; i++)
	for (int j = 0; j < 8; j++)
		nomalized.at<double>(i, j) = img_color.at<int>(i, j) / double(maxScore);
}

double compareColor(Mat img1, Mat img2)
{
	double sum = 0;
	for (int i = 0; i < 37; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if (img1.at<double>(i, j) == 0 || img2.at<double>(i, j) == 0)
				sum -= abs(img1.at<double>(i, j) - img2.at<double>(i, j));
			else sum += (1 - abs(img1.at<double>(i, j) - img2.at<double>(i, j))) * (img1.at<double>(i, j) < img2.at<double>(i, j) ? img1.at<double>(i, j) : img2.at<double>(i, j));
		}
	}
	return sum;
}

double compareImgs(Mat img1, Mat img2)
{
	int w = img1.cols, h = img2.rows;
	Mat new_img2;
	resize(img2, new_img2, img1.size());
	double sum = 0;
	for (int i = 0; i < w; i++)for (int j = 0; j < h; j++)
	{
		sum += (img1.at<uchar>(j, i) ^ new_img2.at<uchar>(j, i)) / 255;
	}
	return sum;
}

int comparePosition(Mat img1, Mat img2)
{
	cvtColor(img1, img1, COLOR_BGR2GRAY);
	cvtColor(img2, img2, COLOR_BGR2GRAY);

	resize(img2, img2, img1.size());
	Canny(img1, img1, 30, 90, 3);
	Canny(img2, img2, 30, 90, 3);

	return compareImgs(img1, img2);
}

bool comparison(const image& r, const image& s)
{
	return r.total_score < s.total_score;
}
int main(int argc, char** argv)
{
	string filename;
	int uplimit;
	cout << "Please type in the input picture name: ";
	cin >> filename;
	cout << "Please type in the quantity of output images: ";
	cin >> uplimit;
	src_input = imread(filename + ".jpg"); // read input image
	double min_color;
	if (!src_input.data)
	{
		printf("Cannot find the input image!\n");
		system("pause");
		return -1;
	}

	Mat src_color, src_color_nomed;
	cah(src_input, src_color, src_color_nomed);

	///Read Database
	FILE   *fp;
	char imagepath[200];
	fp = fopen("inputimage.txt", "r ");
	printf("Extracting features from input images...\n");

	//cvSplit(hsv_1, h_plane_1, s_plane_1, v_plane_1, 0);
	while (!feof(fp))
	{
		while (fscanf(fp, "%s ", imagepath)   >   0)
		{
			printf("%s\n", imagepath);
			char tempname[200];
			image tempimage;
			sprintf_s(tempimage.image_name, 200, "../%s", imagepath);

			db_img = imread(tempimage.image_name); // read database image
			if (!db_img.data)
			{
				printf("Cannot find the database image number %d!\n", db_id + 1);
				system("pause");
				return -1;
			}

			//color procedure
			Mat db_color, db_color_nomed;
			cah(db_img, db_color, db_color_nomed);

			tempimage.color_score = compareColor(src_color_nomed, db_color_nomed);

			tempimage.position_score = comparePosition(src_input, db_img);

			//tempimage.total_score = tempimage.color_score;// +tempimage.position_score;

			images.push_back(tempimage);
			if (db_id != 0)
				min_color = min_color < tempimage.color_score ? min_color : tempimage.color_score;
			else
				min_color = tempimage.color_score;
			db_id++;
		}
	}
	fclose(fp);
	double max_color;
	for (int i = 0; i < images.size(); i++)
		images[i].color_score -= min_color;
	for (int i = 0; i < images.size(); i++)
	{
		if (i != 0)
			max_color = max_color > images[i].color_score ? max_color : images[i].color_score;
		else
			max_color = images[i].color_score;
	}
	for (int i = 0; i < images.size(); i++)
		images[i].color_score /= max_color;

	for (int i = 0; i < images.size(); i++)
	{
		images[i].total_score = images[i].position_score*(1.1 - images[i].color_score);
		cout << images[i].total_score << endl;
	}

	sort(images.begin(), images.end(), comparison);

	for (int j = 0; j<uplimit; j++)
	{
		char name[20];
		itoa(j, name, 20);
		char path[50] = "set/";
		strcat(path, name);
		char suffix[] = ".jpg";
		strcat(path, suffix);
		Mat maximg = imread(images[j].image_name);
		IplImage* save = &maximg.operator IplImage();
		cvSaveImage(path, save);
	}

	//printf("the most similar image is %d, the pixel-by-pixel difference is %f\n",maxscore_num+1, images[1].score);

	printf("Done \n");
	// Wait for the user to press a key in the GUI window.
	//Press ESC to quit
	int keyValue = 0;
	while (keyValue >= 0)
	{
		keyValue = cvWaitKey(0);

		switch (keyValue)
		{
		case 27:keyValue = -1;
			break;
		}
	}

	return 0;
}
