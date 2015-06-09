#include <vl/generic.h>
#include <vl/dsift.h>
#include <windows.h>
#include <stdlib.h>
#include <time.h>

#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\nonfree\features2d.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

#include <glpk.h>

using namespace std;
using namespace cv;

#define gridNumX 16
#define gridNumY 16
//#define N_CLASS 101
#define N_CLASS 101
#define N_LABEL 15
#define N_TEST 15
String classes[N_CLASS];
String labeled_file[N_CLASS][N_LABEL];
String test_file[N_CLASS][N_TEST];
float** labeled_features[N_CLASS][N_LABEL];
cv::flann::Index* flann_index[N_CLASS];

int GetFileList(const char *searchkey, std::vector<std::string> &list)
{
    WIN32_FIND_DATA fd;
    HANDLE h = FindFirstFile(searchkey,&fd);
 
    if(h == INVALID_HANDLE_VALUE)
    {
        return 0; // no files found
    }
 
    while(1)
    {
        list.push_back(fd.cFileName);
 
        if(FindNextFile(h, &fd) == FALSE)
            break;
    }
    return list.size();
}

void setFolderFile(String caltechFolder) {
	//set class name
	vector<string> list;
	String topfolder = caltechFolder;
	topfolder.append("*");
	GetFileList(topfolder.c_str(), list);
	for(int i=2;i<N_CLASS+2;i++)
		classes[i-2] = list[i];
	//set file lists
	srand(time(NULL));
	for(int i=0;i<N_CLASS;i++) {
		list.clear();
		String subfolder = caltechFolder;
		subfolder.append(classes[i]).append("\\*");
		GetFileList(subfolder.c_str(), list);
		list.erase(list.begin());
		list.erase(list.begin());
		if(list.size() < N_LABEL + N_TEST) {
			cout << "Error in " << classes[i] << endl;
			exit(0);
		}
		for(int j=0;j<N_LABEL+N_TEST;j++) {
			int swapindex = rand() % (list.size()-j) + j;
			string temp = list[j];
			list[j] = list[swapindex];
			list[swapindex] = temp;
		}

		for(int j=0;j<N_LABEL;j++) {
			labeled_file[i][j] = list[j];
		}
		for(int j=0;j<N_TEST;j++) {
			test_file[i][j] = list[N_LABEL + j];
		}
	}
}

float** getDenseSift(String folder, String classname,  String filename) {
	float** ret = new float*[gridNumX * gridNumY];
	for(int i=0;i<gridNumX*gridNumY;i++) {
		ret[i] = new float[128];
	}
	//if file exists, read sift from file
	String path = folder.substr(0,folder.length()-1) + "_DSIFT\\" + classname + "\\" + filename + ".dat";
	ifstream fin(path, ios::binary);
	if(fin.is_open()) {
		for(int x=0;x<gridNumX*gridNumY;x++) {
			for(int y=0;y<128;y++) {
				fin.read((char*)&(ret[x][y]), sizeof(float));
			}
		}
		fin.close();
		return ret;
	}

	Mat image = imread(folder + classname + "\\" + filename, CV_LOAD_IMAGE_GRAYSCALE);
	std::vector<float> imgvec;
	for (int i = 0; i < image.rows; ++i){
		for (int j = 0; j < image.cols; ++j){
			imgvec.push_back(image.at<unsigned char>(i,j) / 255.0f);																															
		}
	}
	VlDsiftFilter* dsfilter = vl_dsift_new_basic(image.cols, image.rows, 5, 8);
	int minX = 0;
	int minY = 0;
	int maxX = image.cols - 1;
	int maxY = image.rows - 1;
	int stepX = (image.cols-24) / gridNumX;
	if ((image.cols-24) % gridNumX > 0) {
		stepX++;
		int gridnum = (image.cols-24) / stepX;
		if ((image.cols-24) % stepX > 0)
			gridnum++;
		if(gridnum < 16) {
			stepX--;
			int lastpoint = 12 + 15 * stepX;
			lastpoint += 13;
			maxX = lastpoint;
		}
	}
	int stepY = (image.rows-24) / gridNumY;
	if ((image.rows-24) % gridNumY > 0) {
		stepY++;
		int gridnum = (image.rows-24) / stepY;
		if ((image.rows-24) % stepY > 0)
			gridnum++;
		if(gridnum < 16) {
			stepY--;
			int lastpoint = 12 + 15 * stepY;
			lastpoint += 13;
			maxY = lastpoint;
		}
	}
	vl_dsift_set_bounds(dsfilter,minX,minY,maxX,maxY);
	vl_dsift_set_steps(dsfilter, stepX, stepY);	
	// call processing function of vl
	vl_dsift_process(dsfilter, &imgvec[0]);
	int keypointnum = vl_dsift_get_keypoint_num(dsfilter);
	int descriptornum = vl_dsift_get_descriptor_size(dsfilter);
	const float* features = vl_dsift_get_descriptors(dsfilter);
	// echo number of keypoints found
	//cout << folder << filename << ':' << image.cols << ',' << image.rows << ',' << keypointnum << "," << descriptornum << std::endl;
	if (keypointnum != gridNumX * gridNumY) {
		cout << "Error!";
		exit(0);
	}
	for(int i=0;i<gridNumX*gridNumY;i++) {
		memcpy(ret[i],features+i*128,sizeof(float) * 128);
	}
	
	vl_dsift_delete(dsfilter);
	return ret;
}

void calculateFeatures(String caltechFolder) {
	for(int i=0;i<N_CLASS;i++) {
		for(int j=0;j<N_LABEL;j++) {
			labeled_features[i][j] = getDenseSift(caltechFolder, classes[i], labeled_file[i][j]);
		}
	}
}

void makeIndex() {
	int nn = 1;
	for(int x=0;x<N_CLASS;x++)
	{
		cv::Mat* dataset = new cv::Mat(N_LABEL*gridNumX * gridNumY,128, CV_32F);
		for(int i=0;i<N_LABEL;i++) {
			for(int j=0;j<gridNumX*gridNumY;j++) {
				for(int k=0;k<128;k++) {
					(*dataset).at<float>(i*gridNumX*gridNumY+j,k) = labeled_features[x][i][j][k];
				}
			}
		}
		flann_index[x] = new cv::flann::Index(*dataset, flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
	}
}

double calculateDistance(float* a,float* b,int size) {
	double ret = 0;
	for(int i=0;i<size;i++)
		ret = ret + (b[i] - a[i]) * (b[i] - a[i]);
	ret = sqrt(ret);
	return ret;
}

int classify(float** features) {
	int result;
	int count[N_CLASS];
	for(int i=0;i<N_CLASS;i++)
		count[i] = 0;
	for(int x=0;x<gridNumX*gridNumY;x++) {
		result = 0;
		double min_distance = calculateDistance(features[x],labeled_features[0][0][0],128);
		for(int i=0;i<N_CLASS;i++) {
			for(int j=0;j<N_LABEL;j++) {
				for(int k=0;k<gridNumX*gridNumY;k++) {
					double distance = calculateDistance(features[x],labeled_features[i][j][k],128);
					if (distance < min_distance) {
						min_distance = distance;
						result = i;
					}
				}
			}
		}
		count[result]++;
	}
	int ret = 0;
	int max_count = count[0];
	for(int i=0;i<N_CLASS;i++) {
		if (count[i] > max_count) {
			max_count = count[i];
			ret = i;
		}
	}
	return ret;
}

int classify_flann(float** features) {
	int nn = 1;
	vector<int> index(128);
	vector<float> dist(128);
	float count[N_CLASS];
	for(int i=0;i<N_CLASS;i++)
		count[i] = 0;
	for(int x=0;x<gridNumX*gridNumY;x++) {
		for(int i=0;i<N_CLASS;i++) {
			vector<float> input(features[x], features[x] + 128);
			flann_index[i]->knnSearch(input, index, dist, nn, cv::flann::SearchParams(128));
			count[i] += dist[0] * dist[0];
		}
	}
	int ret = 0;
	float min_count = count[0];
	for(int i=0;i<N_CLASS;i++) {
		if (count[i] < min_count) {
			min_count = count[i];
			ret = i;
		}
	}
	return ret;
}

void queryImages(String caltechFolder) {
	float** features;
	int classified;
	int right, wrong;
	int sum_right = 0,sum_wrong = 0;
	for(int i=0;i<N_CLASS;i++) {
		right = 0;
		wrong = 0;
		for(int j=0;j<N_TEST;j++) {
			features = getDenseSift(caltechFolder, classes[i], test_file[i][j]);
			classified = classify(features);
			if(classified == i)
				right++;
			else
				wrong++;
			cout << "classified into " << classified << "(" << i << ")" << endl;
		}
		cout << classes[i] << ":" << (double)right / (right+wrong) << "%" << right << endl;
		sum_right += right;
		sum_wrong += wrong;
	}
	cout << "total:" << (double)sum_right / (sum_right+sum_wrong) * 100 << "%" << endl;
	cout << "right:" << sum_right << "wrong:" << sum_wrong << endl;
}

void queryImages_flann(String caltechFolder) {
	float** features;
	int classified;
	int right, wrong;
	int sum_right = 0,sum_wrong = 0;
	ofstream out("result.txt");
	for(int i=0;i<N_CLASS;i++) {
		right = 0;
		wrong = 0;
		cout << classes[i] << ":" << endl;
		for(int j=0;j<N_TEST;j++) {
			features = getDenseSift(caltechFolder, classes[i], test_file[i][j]);
			classified = classify_flann(features);
			if(classified == i)
				right++;
			else
				wrong++;
			cout << "classified into " << classified << "(" << i << ")" << endl;
		}
		cout << "accuracy:" << (double)right / (right+wrong) * 100 << "%(" << right << "/" << N_TEST << ")" << endl;
		out << classes[i] << "," << right << "," << (double)right / (right+wrong) * 100 << endl;
		sum_right += right;
		sum_wrong += wrong;
	}
	cout << "total:" << (double)sum_right / (sum_right+sum_wrong) * 100 << "%" << endl;
	cout << "right:" << sum_right << "wrong:" << sum_wrong << endl;
	out.close();
}

int main() {
	cout << glp_version() << endl;
	time_t start = time(NULL);
	setFolderFile("101_ObjectCategories\\");
	cout << "Setting folder and files is finished!" << endl;
	calculateFeatures("101_ObjectCategories\\");
	cout << "Calculating features of labeled images is finished!" << endl;
	makeIndex();
	cout << "Making indexes is finished!" << endl;
	time_t mid = time(NULL);
	//queryImages("101_ObjectCategories\\");
	queryImages_flann("101_ObjectCategories\\");
	time_t end = time(NULL);
	cout << "time_index : " << mid - start << endl;
	cout << "time_query : " << end - mid << endl;
}