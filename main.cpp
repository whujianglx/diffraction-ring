#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>
#include <fstream>


using namespace std;
using namespace cv;
void mouse_callback(int event, int x, int y, int flags, void * param);//�����ص�����
int Distant(int x, int y, int x0, int y0);
int max(int x, int y, int z, int q);
void reDrCircle(Mat &);
int x_;
int y_;
int radius = 0;
Mat img;//��������ǰͼƬ�ĸ���
Mat temp;
int row=0;
int col=0;//����ͼ��ĳ��Ϳ�

int xnew;
int ynew;
int radNew;//Բ��ԭͼ��λ��
int main()
{
	cout << "��֪���Ĳ�����������Ϊ0" << endl;
	string s ;
	cout << "������ͼ���·��,·���������пո�" << endl;
	cin >> s;
	Mat bImage = imread(s.c_str());

	while(bImage.data == NULL){
		printf("�Ҳ���ͼƬ�����������룡\n");
		cout << "������ͼ���·����" << endl;
		cin >> s;
		bImage = imread(s);
	}//���·����Ч��
	row = bImage.rows;
	col = bImage.cols;
	float rowrate = row / 256.0;
	float colrate = col / 256.0;
	Mat aImage;
	resize(bImage, aImage, Size(256, 256), 0, 0, INTER_LINEAR);
	
	img = bImage.clone();//img��ͼƬ����������ˢ��Բ��λ��
	Mat srcImage,srcImageNoSize;
	cvtColor(aImage, srcImage, CV_RGB2GRAY);//RGB->GRAY
	cvtColor(bImage, srcImageNoSize, CV_RGB2GRAY);//srcImageNoSize�������Ȼ���
	//��Ե���
	Mat cannyImage;
	int edgeThresh = 50;
	Canny(srcImage, cannyImage, edgeThresh, edgeThresh * 3, 3);
	////��ʴ
	//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
	//Mat dstImage;
	//erode(cannyImage, dstImage, element);
	////����
	//Mat ssssImage;
	//dilate(dstImage, ssssImage, element);
	////��˹�˲�ƽ������
	//GaussianBlur(ssssImage, ssssImage, Size(9, 9), 2, 2);
	cout << "�������������ֵ��" << endl;
	int u;
	cin >> u;
	cout << "���ڼ��㣬��Ⱥ�........." << endl;
	vector<Vec3f>circles;
	if (u == 0)
	{
		//����Բ�任	
		HoughCircles(cannyImage, circles, CV_HOUGH_GRADIENT, 1, 1, 100, 20, 0, 0);
	}
	else
	{
		HoughCircles(cannyImage, circles, CV_HOUGH_GRADIENT, 1, 1, 100, u, 0, 0);
	}
	
	int n = 0;
	int sumx = 0;
	int sumy = 0;
	
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		radius = cvRound(circles[i][2]);
		sumx = sumx + center.x;
		sumy = sumy + center.y;
		n = n + 1;
	}

	Point center1(sumx / n, sumy / n);
	x_ = center1.x;
	y_ = center1.y;
	//reDrCircle(aImage);
	//imshow("Picture", aImage);//��ʾ�������任ͼ��

	xnew = x_ * rowrate;
	ynew = y_ * colrate;
	radNew = radius*(rowrate + colrate) / 2;
	center1.x = xnew;
	center1.y = ynew;//Բ����ԭͼ�ĳ�ʼλ��

	Mat m = img.clone();
	reDrCircle(m);
	cvNamedWindow("Picture", CV_WINDOW_KEEPRATIO);
	imshow("Picture", m);
	setMouseCallback("Picture", mouse_callback, 0);
	
	cout << "���ڿ�ʼ����Բ��λ�ã������������������������ESC������\n" << endl;
	while ((cvWaitKey(23) != 27));

	//�˹���������Բ�����

	reDrCircle(bImage);//��bImage�ϻ����趨�õ�Բ
	Point center2(xnew, ynew);
	
	//���Բ������
	ofstream outfile;
	outfile.open(s+"_data.txt");
	outfile << center1 << "," << center2 << endl;
	
	//
	//circle(aImage, center2, 5, Scalar(255, 255, 0), -1, 8, 0);

	/*if (xnew== 0 && ynew== 0)
	{
		center2 = center1;
	}*/

	

	//circle(bImage, cvPoint(xnew, ynew), 3, Scalar(0, 255, 0), -1, 8, 0);
	//circle(bImage, cvPoint(xnew, ynew), radNew, Scalar(155, 50, 255), 3, 8, 0);

	//������������
	double start_angle;
	double end_angle;
	cout << "��������ʼ��ĽǶ�" << endl;
	
	cin >> start_angle;
	cout << "��������ֹ��ĽǶ�" << endl;
	cin >> end_angle;
	cout << "���ڼ��㣬��Ⱥ�........." << endl;
	double angle;
	double x = 0;
	double y = 0;
	
	double cur_v = 0;
	int gray = 0;
	int a = 0;
	
	//��Ż��������Ⱥ�
	vector <Point>data;
	int r1 = Distant(xnew, ynew, 0, 0);
	int r2 = Distant(xnew, ynew, 0, bImage.rows);
	int r3 = Distant(xnew, ynew, bImage.rows, bImage.cols);
	int r4 = Distant(xnew, ynew, bImage.cols, 0);
	int maxjfr = max(r1, r2, r3, r4);

	for (double r = 1; r <= maxjfr; r++)
	{
		cur_v = 0;
		/* double r = 90;*/
		double o;
		o = 1 / r;
		char flag = 0;
		char flagfst = 1;
		int x1 = 0;
		int y1 = 0;
		for (angle = (360-start_angle); angle >= (360-end_angle); angle = angle - o)
		{
			int x2 = 0;
			int y2 = 0;
			x = xnew + r* cos(angle*3.14 / 180);
			y = ynew + r* sin(angle*3.14 / 180);
			x2 = (x / 10) * 10;
			y2 = (y / 10) * 10;
			//cout << x2 << ',' << y2 << endl;
			
			if (x2<0 || x2>bImage.cols || y2<0 || y2>bImage.rows){
				flag = 1;
				break;
			}
			else
			{
				
				
					if (x1 != x2 || y1 != y2)
					{
						
						if (abs(x1 - x2) > 1 || abs(y2 - y1) > 1 )
						{
							if (flagfst == 0)
								cout << "error" << "R= " << r << ',' << "X��ֵ��" << abs((x1 - x2)) << ',' << "Y��ֵ��"<<abs((y1 - y2)) << endl;
							
						}
	
							
						x1 = x2;
						y1 = y2;//

						//��ȡͼ������Ȳ����ۼ����
						//gray = *(aImage.data + aImage.step[0] * x1 + aImage.step[1] * y1);
						gray = srcImageNoSize.at<uchar>(x1, y1);
						//cout << gray << endl;
						cur_v = cur_v + gray;
						flagfst = 0;
					}
				
			}
		}
		if (flag == 0){
			data.push_back(Point(r, cur_v));
			outfile << Point(r, cur_v) << endl;
			ellipse(bImage, cvPoint(xnew, ynew), Size(r,r), 0, 360 - start_angle, 360 - end_angle, Scalar(0, 0, 255), 0.5, 8, 0);
		}
		else break;
		
		//������ֵ
		//if (cur_v > y_max) y_max = cur_v;
		/*cout << vec_r[3] << endl;*/
	}
	
	/*for (int r = 0; r <= radNew; r++)
	{
		Size axes;
		axes = Size(r, r);
		ellipse(bImage,cvPoint(xnew,ynew), axes, 0, 360-start_angle, 360-end_angle, Scalar(0, 0, 255), 2, 8, 0);
	}*/
	//cvNamedWindow("Result", CV_WINDOW_KEEPRATIO);
	//imshow("Result.jpg", bImage);
	imwrite(s + "_resut.jpg", bImage);
	cout << "�������н�����" << endl;
	cvWaitKey(1000);
	return 0;

}

void mouse_callback(int event, int x, int y, int flags, void* param)//����ص�����
{ 
	
	if (event == 1)
	{
		temp = img.clone();
		xnew=x;
		ynew=y;
		reDrCircle(temp);
		imshow("Picture", temp);
	}
}
void reDrCircle(Mat & M)
{
	circle(M, cvPoint(xnew, ynew), 3, Scalar(0, 255, 0), -1, 8, 0);
	circle(M, cvPoint(xnew, ynew), radNew, Scalar(155, 50, 255), 3, 8, 0);

}
int Distant(int x, int y, int x0, int y0){
	int a = abs(x - x0);
	int b = abs(y - y0);
	int c = sqrt(a*a + b*b);
	return c;
}
int max(int x, int y, int z, int q){
	int max = 0;
	if (x > y)max = x;
	else max = y;
	if (max > z){}
	else
		max = z;
	if (max > q){}
	else
		max = q;
	return max;
}