#include <iostream>
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\calib3d.hpp"

using namespace std;
using namespace cv;

Mat read_image(string path) {
	return imread(path, IMREAD_COLOR);
}

void show_and_save(string name, Mat image, string save_path) {
	imshow(name, image);
	waitKey(0);
	imwrite(save_path, image);
}

// create a SIFT object
Ptr<Feature2D> sift = SIFT::create();
void detect_and_compute(Mat img, vector<KeyPoint>& keypoints, Mat& descriptors) {
	// create a mat object to hold the feature descriptors
	sift->detectAndCompute(img, Mat(), keypoints, descriptors);
}

vector<DMatch> match_descriptors(Mat descriptors1, Mat descriptors2) {
	// create a brute force matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	// create a vector of vector of DMatch to hold the matches
	vector<vector<DMatch>> matches;
	// match the descriptors
	matcher->knnMatch(descriptors1, descriptors2, matches, 4);
	// create a vector of DMatch to hold the good matches
	vector<DMatch> good_matches;
	// filter the matches
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.75 * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}
	return good_matches;
}

Mat Homography(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<DMatch> good_matches) {
	// find the homography matrix
	vector<Point2f> pts1, pts2;
	for (int i = 0; i < good_matches.size(); i++)
	{
		pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
	return findHomography(pts1, pts2 , RANSAC, 3);
}

void draw_bb(Mat& scene, Mat& temp, Mat H) {
	// create a vector of Point2f to hold the corners of the template image
	vector<Point2f> corners(4);
	// find the corners of the template image
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(temp.cols, 0);
	corners[2] = Point2f(temp.cols, temp.rows);
	corners[3] = Point2f(0, temp.rows);

	// create a vector of Point2f to hold the corners of the detected template image
	vector<Point2f> detected_corners(4);
	// find the corners of the detected template image
	perspectiveTransform(corners, detected_corners, H);
	// draw the bounding box
	line(scene, detected_corners[0], detected_corners[1], Scalar(0, 255, 0), 2);
	line(scene, detected_corners[1], detected_corners[2], Scalar(0, 255, 0), 2);
	line(scene, detected_corners[2], detected_corners[3], Scalar(0, 255, 0), 2);
	line(scene, detected_corners[3], detected_corners[0], Scalar(0, 255, 0), 2);
}

int main(int argv, char** argc)
{
	if (argv == 1) {
		cout << "The valid command is: 21127118.exe -sift <TemplateImagePath> <SceneImagePath> <OutputFilePath>";
		return 0;
	}

	// read the image
	Mat scene = read_image(argc[3]);
	Mat temp = read_image(argc[2]);

	// detect and compute the keypoints and descriptors
	vector<KeyPoint> keypoints_scene, keypoints_template;
	Mat descriptors_scene, descriptors_template;
	detect_and_compute(scene, keypoints_scene, descriptors_scene);
	detect_and_compute(temp, keypoints_template, descriptors_template);

	// filtering the matches
	vector<DMatch> good_matches = match_descriptors(descriptors_template, descriptors_scene);

	// find the homography matrix
	Mat H = Homography(keypoints_template, keypoints_scene, good_matches);

	// draw the matches
	//draw_matches(scene, temp, keypoints_scene, keypoints_template, good_matches);

	// draw the bounding box
	draw_bb(scene, temp, H);

	// show the result
	show_and_save("Result", scene, argc[4]);

	return (0);
}