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

using namespace std;
using namespace cv;

#define gridNumX 16
#define gridNumY 16

#define N_CLASS 101

float** getDenseSift2(String folder, String filename) {
	Mat image = imread(folder + filename, CV_LOAD_IMAGE_GRAYSCALE);
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
	cout << folder << filename << ':' << image.cols << ',' << image.rows << ',' << keypointnum << "," << descriptornum << std::endl;
	/*const VlDsiftKeypoint* keypoints = vl_dsift_get_keypoints(dsfilter);
	for(int i=0;i<keypointnum;i++)
		cout << keypoints[i].x << ' ' << keypoints[i].y << endl;*/
	if (keypointnum != gridNumX * gridNumY) {
		cout << "Error!";
		exit(0);
	}
	float** ret = new float*[gridNumX * gridNumY];
	for(int i=0;i<gridNumX*gridNumY;i++) {
		ret[i] = new float[128];
		memcpy(ret[i],features+i*128,sizeof(float) * 128);
	}
	
	vl_dsift_delete(dsfilter);
	return ret;
}

int GetFileList2(const char *searchkey, std::vector<std::string> &list)
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

void SaveFeaturesToFile(String caltechFolder) {
	//set class name
	String classes[N_CLASS];
	vector<string> list;
	String topfolder = caltechFolder;
	topfolder.append("*");
	GetFileList2(topfolder.c_str(), list);
	for(int i=2;i<N_CLASS+2;i++)
		classes[i-2] = list[i];
	//set file lists
	srand(time(NULL));
	for(int i=0;i<N_CLASS;i++) {
		list.clear();
		String subfolder = caltechFolder;
		subfolder.append(classes[i]).append("\\*");
		GetFileList2(subfolder.c_str(), list);
		list.erase(list.begin());
		list.erase(list.begin());
		for(int j=0;j<list.size();j++) {
			float** features = getDenseSift2(caltechFolder + classes[i] + "\\", list[j]);
			String dirname = "101_ObjectCategories_DSIFT\\" + classes[i];
			String filename = dirname + "\\" + list[j];
			CreateDirectory(dirname.c_str(), NULL);
			ofstream fout(filename.append(".dat"), ios::binary);
			for(int x=0;x<gridNumX*gridNumY;x++) {
				for(int y=0;y<128;y++) {
					fout.write((char*)&(features[x][y]), sizeof(float));
				}
			}
			fout.close();
		}
	}
}

int main() {
	time_t start = time(NULL);
	SaveFeaturesToFile("101_ObjectCategories\\");
	cout << "Saving features of all images is finished!" << endl;
	time_t end = time(NULL);
	cout << "time_feature : " << end - start << endl;
}