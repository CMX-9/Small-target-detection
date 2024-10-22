#include<stdio.h>
#include <conio.h >
#include <math.h>
#include<iostream>
#include<string.h>
#include<opencv2/opencv.hpp>

#define UP				 0
#define  LEFT			 1
#define  DOWN      2
#define  RIGHT       3

using namespace std;

using namespace cv;

//遍历像素法实现图像灰度化
int Gray(Mat &GrayImg,Mat OriginalImg)
{
	int height = OriginalImg.rows;
	int width = OriginalImg.cols;
	int step1 = OriginalImg.step;                                         //每行有多少个8bit
	
	int step2 =GrayImg.step;                                          //注意src和dst的step不一样
	int channels = OriginalImg.channels();                           //每个像素颜色通道数
	//彩色图像灰度化
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			GrayImg.data[i*step2 + j] = (int)
				(0.11*OriginalImg.data[i*step1 + channels * j] + 0.59*OriginalImg.data[i*step1 + channels * j + 1] + 0.3*OriginalImg.data[i*step1 + channels * j + 2]);
		}
	}
	return 1;
}
//灰度图像膨胀操作，结构元素为5*5
//flag==1,则进行膨胀，flag==0，则进行腐蚀
int GrayExpand_Erode(uchar *Expang_ErodeImg,Mat grayimg,int ma[5][5],int flag)
{
	int i, j, k, o,m,t=0;
	int tmp=0;
	int tmpdata[30] = { 0 };
    int rows = grayimg.rows;
	int cols = grayimg.cols*grayimg.channels();
    for (i = 2; i < rows-2 ; i++)
    {
	  for (j = 2; j < cols-2 ; j++)
	  {
		for (k = -2; k <= 2; k++)
		{
			for (o = -2; o <= 2; o++)
			{
					tmpdata[t] = grayimg.data[(i + k)*cols + j + o];
					t++;
			}
		}
		//计算局部最大值并作为该像素点的像素值。flag==1,则进行膨胀，flag==0，则进行腐蚀
		for (m = 0; m < t; m++)
		{
			if (flag == 1 && tmpdata[0] < tmpdata[m + 1])
			{
				tmp = tmpdata[0];
				tmpdata[0] = tmpdata[m + 1];
				tmpdata[m + 1] = tmp;
			}
			else if (flag == 0 && tmpdata[0] > tmpdata[m + 1])
			{
				tmp = tmpdata[0];
				tmpdata[0] = tmpdata[m + 1];
				tmpdata[m + 1] = tmp;
			}
		}
		*(Expang_ErodeImg+i * cols + j)= tmpdata[0];
		t = 0;
	  }
    }
return 1;
}

//基本测地线腐蚀（膨胀）
//flag==1,基本测地线膨胀，取最小值
//flag==0,基本测地线腐蚀，取最大值
//稳定条件，前一个结果图与当前结果图相同
int BaseGeodesicLine(Mat Dilate_ErodeImg,Mat OriginalImg, Mat Expang_ErodeImg, int ma[5][5], int flag)   // Exchange图像数据改变次数，为0时表示迭代结束。
{
	int i, j;			
	int rows = Expang_ErodeImg.rows;
	int cols = Expang_ErodeImg.cols*Expang_ErodeImg.channels();
	if (1 == flag)
	{
		
		//打开操作结果图再进行膨胀
		 GrayExpand_Erode(Dilate_ErodeImg.data,Expang_ErodeImg, ma, 1);
		//基本测地线膨胀，取最小值
		for (i = 2; i < rows-2; i++)
		{
			for (j = 2; j < cols-2; j++)
			{
				if(Dilate_ErodeImg.data[i * cols + j] >OriginalImg.data[i*cols + j])
				{
				Dilate_ErodeImg.data[i * cols + j] = OriginalImg.data[i*cols + j];
				}
			}
		}
		return  1;
	}

	if (0 == flag)
	{
		
		//关闭操作结果图再进行腐蚀
		  GrayExpand_Erode(Dilate_ErodeImg.data,Expang_ErodeImg, ma, 0);
		//基本测地线腐蚀，取最大值
		for (i = 2; i < rows-2; i++)
		{
			for (j = 2; j < cols-2; j++)
			{
				if(Dilate_ErodeImg.data[i * cols + j] < OriginalImg.data[i*cols + j])
				{
				Dilate_ErodeImg.data[i * cols + j] = OriginalImg.data[i*cols + j];
                }
			}
		}
		return  0;
	}
}

//迭代实现测地线腐蚀（膨胀）
int GeodesicLine(Mat GeodesicLineImg,Mat OriginalImg, Mat Expang_ErodeImg, int ma[5][5], int flag)
{
    int same=0;
	static int t=0;
	int i, j;
	BaseGeodesicLine(GeodesicLineImg, OriginalImg, Expang_ErodeImg, ma, flag);
	t++;
	for (i = 2; i < OriginalImg.rows - 2; i++)
	{
		for (j = 2; j < OriginalImg.cols - 2; j++)
		{
			if (GeodesicLineImg.data[i*OriginalImg.cols + j] != Expang_ErodeImg.data[i*OriginalImg.cols + j])
			same++;
		}
	}
	if (same>40)
	{
	  for (i = 2; i < OriginalImg.rows - 2; i++)
	  {
		 for (j = 2; j < OriginalImg.cols - 2; j++)
		 {
			Expang_ErodeImg.data[i*OriginalImg.cols + j] = GeodesicLineImg.data[i*OriginalImg.cols + j];
		 }
	  }
	    GeodesicLine(GeodesicLineImg,OriginalImg, Expang_ErodeImg, ma, flag);
	 }
	else if(same<=40)
	{
		return   1;
	}
}
//注意:
//通过亮舰的形态重建结果图CGMR得到暗舰的强度显著性特征图IFSMd
//通过暗舰的形态重建结果图OGMR得到亮舰的强度显著性特征图IFSMb
//通过CGMR得到BCSMb
//通过OGMR得到BCSMd 
//对于Intensity_brightness（）函数。最后两个数字：1x时得IFSM，01时得BCSMb，00时得BCSMd
int Intensity_brightness(Mat B_ISMImg,Mat GMR,  Mat OriginalImg, int ChooseFeature,int flag)
{
	int i, j;
	int ma[5][5]= { {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1} };
	int rows = GMR.rows;
	int cols = GMR.cols*GMR.channels();
	Mat buf1(rows, cols, CV_8UC1);
	Mat buf2(rows, cols, CV_8UC1);
	//ChooseFeature==1,表示用亮舰结果CGMR或暗舰结果OGMR得到强度显著性结果图IFSMd或IFSMb
	if(1== ChooseFeature)
	{
		for (i = 2; i < rows-2; i++)
		{
			for (j = 2; j < cols-2; j++)
			{
				if (GMR.data[i*cols + j] < OriginalImg.data[i*cols + j])
					B_ISMImg.data[i*cols + j] = (OriginalImg.data[i*cols + j] - GMR.data[i*cols + j])*(OriginalImg.data[i*cols + j] - GMR.data[i*cols + j]);
				if (GMR.data[i*cols + j] > OriginalImg.data[i*cols + j])
					B_ISMImg.data[i*cols + j] = (GMR.data[i*cols + j] - OriginalImg.data[i*cols + j])*(GMR.data[i*cols + j] - OriginalImg.data[i*cols + j]);
				if (GMR.data[i*cols + j] == OriginalImg.data[i*cols + j])
					B_ISMImg.data[i*cols + j] = 0;
			}
		}
		return 1;
	}
	//ChooseFeature==0,表示得到亮度显著性结果图BCSMb或BCSMd
	if(0== ChooseFeature)
	{
		//1==flag,表示用亮舰结果CGMR得到暗舰图像的BCSMb
		if(1==flag)
		{
			//打开操作
			GrayExpand_Erode(buf1.data,GMR,ma, 0);
			GrayExpand_Erode(buf2.data,buf1, ma, 1);
	  	   for (i = 2; i < rows-2; i++)
		   {
			  for (j = 2; j < cols-2; j++)
			  {
				  B_ISMImg.data[i*cols + j] = GMR.data[i*cols + j] - buf2.data[i*cols + j];
			  }
		   }
		   return 21;
		}

		//0==flag,表示用暗舰结果OGMR得到亮舰图像的BCSMd
		if (0 == flag)
		{
			//关闭操作
			 GrayExpand_Erode(buf1.data ,GMR, ma, 1);
			 GrayExpand_Erode(buf2.data ,buf1, ma, 0);
			for (i = 2; i < rows-2; i++)
			{
				for (j = 2; j < cols-2; j++)
				{
					B_ISMImg.data[i*cols + j] = buf2.data[i*cols + j]-GMR.data[i*cols + j] ;
				}
			}
			return 22;
		}
	}
}
//BCSM和ISFM归一化
int Normalization(float *NorImg,Mat BCSM_ISFMimg)
{
	int i, j;
	float t, m,g;
	int rows = BCSM_ISFMimg.rows;
	int cols = BCSM_ISFMimg.cols*BCSM_ISFMimg.channels();
	int Smax = BCSM_ISFMimg.data[0];
	int Smin= BCSM_ISFMimg.data[0];
	
	//求显著性图的最大值Smax和最小值Smin
	for (i = 2; i < rows-2; i++)
	{
		for (j = 2; j < cols-2; j++)
		{
			if (Smax < BCSM_ISFMimg.data[i*cols + j])
			{
				Smax = BCSM_ISFMimg.data[i*cols + j];
			}
			if (Smin > BCSM_ISFMimg.data[i*cols + j])
			{
				Smin = BCSM_ISFMimg.data[i*cols + j];
			}
		}
	}

	m = (float)(Smax - Smin);
	for (i = 2; i < rows-2; i++)
	{
		for (j = 2; j < cols-2; j++)
		{
			t = (float)(BCSM_ISFMimg.data[i*cols + j] - Smin);
			*(NorImg+i*cols + j)= t / m; //115;S:254,1
		}
	}
	return 1;
}
//显著性图融合
//显著性图BCSMb（BCSMd）和ISFMb（ISFMd）融合得SMb（SMd）
int Mix(float *MixImg,float *NormalBCSM, float *NormalISFMb,Mat Rows_ColsImg)
{
	int i, j;
	float t, m;
	int rows = Rows_ColsImg.rows;
	int cols = Rows_ColsImg.cols;

	for (i = 2; i < rows-2; i++)
	{
		for (j = 2; j < cols-2; j++)
		{
			t = *(NormalBCSM + i * cols + j);
			m = *(NormalISFMb + i * cols + j);
		    *(MixImg + i * cols + j) = t * m;
			
		}
	}
	return 1;
}

//阈值分割，计算阈值,求平方根：double sqrt(double x),头文件：<math.h>
int OtsuAlgThreshold(const Mat image) 
{
	if (image.channels() != 1)
	{ 
		cout << "Please input Gray-image!" << endl;		
		return 0;
	}	
	int T = 0;							 //Otsu算法阈值	
	double varValue=0;		//类间方差中间值保存	
	double w0=0;					//前景像素点数所占比例
	double w1=0;					//背景像素点数所占比例
	double u0=0;					//前景平均灰度	
	double u1=0;					//背景平均灰度	
	double Histogram[256]={0}; //灰度直方图，下标是灰度值，保存内容是灰度值对应的像素点总数	
	uchar *data=image.data;	
	double totalNum=image.rows*image.cols; //像素总数
   //计算灰度直方图分布，Histogram数组下标是灰度值，保存内容是灰度值对应像素点数	
	for(int i=0;i<image.rows;i++)   //为表述清晰，并没有把rows和cols单独提出来
	{		
		for(int j=0;j<image.cols;j++)	
		{			
			Histogram[data[i*image.step+j]]++;		
		}
	}
	for(int i=0;i<255;i++)
	{		
		//每次遍历之前初始化各变量
		w1=0;		u1=0;		w0=0;		u0=0;		
		//***********背景各分量值计算**************************	
		for(int j=0;j<=i;j++) //背景部分各值计算	
		{			
			w1+=Histogram[j];  //背景部分像素点总数			
			u1+=j*Histogram[j]; //背景部分像素总灰度和	
		}		
		if(w1==0) //背景部分像素点数为0时退出	
		{			
			continue;		
		}		
		u1=u1/w1; //背景像素平均灰度	
		w1=w1/totalNum; // 背景部分像素点数所占比例		
		//***********背景各分量值计算************************** 		
		//***********前景各分量值计算**************************		
		for(int k=i+1;k<255;k++)		
		{			
			w0+=Histogram[k];  //前景部分像素点总数		
			u0+=k*Histogram[k]; //前景部分像素总灰度和	
		}		
		if(w0==0) //前景部分像素点数为0时退出		
		{			
			break;
		}		
		u0=u0/w0; //前景像素平均灰度		
		w0=w0/totalNum; // 前景部分像素点数所占比例		
		//***********前景各分量值计算************************** 		
		//***********类间方差计算******************************		
		double varValueI=w0*w1*(u1-u0)*(u1-u0); //当前类间方差计算		
		if(varValue<varValueI)	
		{			
			varValue=varValueI;		
			T=i;	
		}	
	}	
	return T;
}

//阈值分割，计算阈值, 求平方根：double sqrt(double x), 头文件：<math.h>
//公式t=SM平均值+s*SM标准差，s取[10,15],s是实验数据
float Threshold(float *SM,float s,int rows,int cols)
{
	int i, j;
	float averSM=0.0,  sum = 0.0;
	float aver_stdSM=0.0, stdSM=0.0;
	float standard;
	for (i = 2; i < rows-2; i++)
	{
		for (j = 2; j < cols-2; j++)
		{
			sum += *(SM + i * cols + j);
		}
	}
	averSM = sum / (rows*cols-4*cols-4*rows+4*4);
	sum = 0.0;
	for (i = 2; i < rows-2; i++)
	{
		for (j = 2; j < cols-2; j++)
		{
			if(*(SM + i * cols + j)> averSM)
			sum += (*(SM + i * cols + j)- averSM)* (*(SM + i * cols + j) - averSM);
			if (*(SM + i * cols + j) < averSM)
			sum += (averSM -*(SM + i * cols + j) )* (averSM- *(SM + i * cols + j));
			if (*(SM + i * cols + j) ==averSM)
			sum += 0;
		}
	}
	aver_stdSM = sum / (rows*cols - 4 * cols - 4 * rows + 4 * 4);
	stdSM = sqrt(aver_stdSM);
	standard = averSM + s * stdSM;
	return standard;
}

//基于小数的阈值分割
int ThresholdOperate(Mat ThresholdImg, float *ptr, float threshold,int flag)
{
	int i, j;
	float data;
	int rows = ThresholdImg.rows;
	int cols= ThresholdImg.cols;
	//亮舰目标
	if(1==flag)
	{ 
		for (i = 2; i < rows - 2; i++)
		{
			for (j = 2; j < cols - 2; j++)
			{
				data = *(ptr + i * cols + j);
				if (data >= threshold)
				{
					ThresholdImg.data[i * cols + j] = 255;
				}
				if (data < threshold)
				{
					ThresholdImg.data[i * cols + j] = 0;
				}
			}
		}
	return 1;
	}
	//暗舰目标
	if (2== flag)
	{
		for (i = 2; i < rows - 2; i++)
		{
			for (j = 2; j < cols - 2; j++)
			{
				data = *(ptr + i * cols + j);
				if (data >= threshold)
				{
					ThresholdImg.data[i * cols + j] = 0;
				}
				if (data < threshold)
				{
					ThresholdImg.data[i * cols + j] = 255;
				}
			}
		}
		return 2;
	}
}

/*当前点为本类中的点时，判断其是否为该类边界点并更新边界*/
int UpdateBound(unsigned int pixelClass, int g_classCoordinate[257][4], unsigned int y_rows, unsigned int x_cols)
{
	if (x_cols < g_classCoordinate[pixelClass][LEFT])
	{
		g_classCoordinate[pixelClass][LEFT] = x_cols;             //更新左边界
	}
	else if (x_cols > g_classCoordinate[pixelClass][RIGHT])
	{
		g_classCoordinate[pixelClass][RIGHT] = x_cols;            //更新右边界
	}
	if (y_rows > g_classCoordinate[pixelClass][DOWN])
	{
		g_classCoordinate[pixelClass][DOWN] = y_rows;              //更新下界
	}
	return 1;
}


/*找到上下左右四邻域中的黑点，并将其标记为当前类中元素*/
int MarkerNeighbourhood(Mat ResultImg, int g_classCoordinate[257][4],unsigned int pixelClass, unsigned int y_rows, unsigned int x_cols,int *g_targetPixelUp,int *g_targetPixelLeft)
{
	    UpdateBound(pixelClass, g_classCoordinate, y_rows, x_cols);                      //更新边界

		if (255== ResultImg.data[(y_rows + 1)*ResultImg.cols + x_cols])        //目标点下侧
		{
			ResultImg.data[(y_rows + 1)*ResultImg.cols + x_cols] = pixelClass;
		}
		if (255 == ResultImg.data[(y_rows -1)*ResultImg.cols + x_cols])        //目标点上侧
		{
			ResultImg.data[(y_rows - 1)*ResultImg.cols + x_cols] = pixelClass;
			*g_targetPixelUp = 1;
		}
		if (255 == ResultImg.data[y_rows*ResultImg.cols + x_cols -1])          //目标点左侧
		{
			ResultImg.data[y_rows*ResultImg.cols + x_cols - 1] = pixelClass;
			*g_targetPixelLeft = 1;
		}
		if (255== ResultImg.data[y_rows*ResultImg.cols + x_cols +1])          //目标点右侧
		{
			ResultImg.data[y_rows*ResultImg.cols + x_cols +1] = pixelClass;
		}
		return 1;
}

/*将图像中各个连通的区域找出来*/
/*遍历顺序：从左至右，从下至上*/
//寻找船舰及非船舰类
//g_imageClass[]存该类的面积像素值，g_classCoordinate[][4]存该类的矩形边界，g_imageClass[0] 存找到的总类数，当前可检测的目标数目最多为255个
int FindClass(Mat ResultImg, int g_classCoordinate[257][4], int g_imageClass[257])
{
	unsigned int pixelClass = 0;
	unsigned int          x = 0;
	unsigned int          y = 0;
	unsigned int    HeaderX = 0;															 //各类起始点横坐标
	unsigned int    HeaderY = 0;															 //各类起始点纵坐标
	unsigned int  classFlag = 0;															 //0:尚未找到本类的第一个点；1：已找到本类的起始点，正准备找其联通点；2：已找到所有类
	int i, j;
	//注意，初始化边界应该取反向最大
	g_classCoordinate[pixelClass][UP] =2 ;                                        //初始化当前类区域上界
	g_classCoordinate[pixelClass][DOWN] = ResultImg.rows-2;                        //初始化当前类区域下界
	g_classCoordinate[pixelClass][LEFT] =2;										  //初始化当前类区域左边界
	g_classCoordinate[pixelClass][RIGHT] = ResultImg.cols - 2;       //初始化当前类区域右边界
	int g_targetPixelLeft = 0;
	int g_targetPixelUp = 0;
	while (classFlag != 2)
	{
		if (0 == classFlag)																			//查找当前类的起始点
		{
			pixelClass++;																			 //进入下一个类别的遍历
			if (pixelClass > 255)
			{
				g_imageClass[0] = 255;
				printf("当前可检测的目标数目最多为255个");
				exit(1);																					 //当前可检测的目标数目最多为255个
			}

			g_imageClass[pixelClass] = 0;                                                 //当前类像素点个数清零
			//假设周围宽为2的像素不存在目标
			g_classCoordinate[pixelClass][UP] = ResultImg.rows-2;      //初始化当前类区域上界
			g_classCoordinate[pixelClass][DOWN] = 2;						   //初始化当前类区域下界
			g_classCoordinate[pixelClass][LEFT] = ResultImg.cols-2;    //初始化当前类区域左边界
			g_classCoordinate[pixelClass][RIGHT] = 2;                           //初始化当前类区域右边界

			for (i = g_classCoordinate[pixelClass - 1][UP]; i < ResultImg.rows - 2; i++)           //每次纵轴从上个类的最上界开始搜寻
			{
				for (j = 2; j < ResultImg.cols - 2; j++)																	  //每次横轴从2开始搜寻
				{
					if (255 == ResultImg.data[i*ResultImg.cols + j])										     //找到该类的起始点
					{
						HeaderY = i;
						HeaderX = j;
						//g_imageClass[pixelClass]++;
						g_classCoordinate[pixelClass][UP] = i;													   //初始化起始点上界
						ResultImg.data[i*ResultImg.cols + j] = pixelClass;									    //置起始点为当前类的编号
						MarkerNeighbourhood(ResultImg, g_classCoordinate, pixelClass, HeaderY, HeaderX, &g_targetPixelUp, &g_targetPixelLeft);                  //搜寻起始点周围的目标点
						classFlag = 1;
						break;
					}
				}//for x = 1
				if (1 == classFlag)
				{
					break;
				}
			}//for y
			if (0 == classFlag)																											//图中所有黑点均已归类
			{
				pixelClass--;
				g_imageClass[0] = pixelClass;
				classFlag = 2;																													//图像已遍历完全
			}
		}//if
		else if (1 == classFlag)
		{
			//遍历全图：找到当前类的联通点
			for (i = g_classCoordinate[pixelClass][UP]; i < ResultImg.rows - 2; i++)
			{
				//g_rowTargetPixelNum = 0;																					  //当前行目标点数目清零
				for (j = 2; j < ResultImg.cols - 2; j++)
				{
					if (pixelClass == ResultImg.data[i*ResultImg.cols + j])										  //当前点为当前类的点
					{
						MarkerNeighbourhood(ResultImg, g_classCoordinate, pixelClass, i, j, &g_targetPixelUp, &g_targetPixelLeft);  //搜寻当前点邻域，查找并标记黑点
					}
					if (1 == g_targetPixelLeft)
					{
						j = 2;
						g_targetPixelLeft = 0;
					}
					if (1 == g_targetPixelUp)
					{
						i = i - 1;
						j = 2;
						if (i < 2)
						{
							i = 2;
						}
						g_targetPixelUp = 0;
					}
				}//for
			}//for 
			classFlag = 0;
		}//else if
	}//while
	return 1;
}//FindClass

//Ratiomami：长轴与短轴的比率，用来描述其最佳拟合椭圆。
//RecArea_BoatArea（矩形度）：通过假设在[0，1]范围内的值来表示矩形相似度，1表示理想的矩形区域。
int Feature(Mat ResultImg, int g_classCoordinate[257][4], int g_imageClass[257],float Ratiomami[255],float RecArea_BoatArea[255])
{
	int i, j,t;
	int rows = ResultImg.rows;
	int cols= ResultImg.cols*ResultImg.channels();   
	float Axis1 = 0.0;
    float Axis2= 0.0;
	float RectArea=0.0;
	float TargeArea = 0.0;

	for(t=1;t< g_imageClass[0]+1;t++)         
	{ 
	   for (i = g_classCoordinate[t][UP]; i <( g_classCoordinate[t][DOWN]+1); i++)
	   {
		  for (j = g_classCoordinate[t][LEFT]; j < (g_classCoordinate[t][RIGHT]+1); j++)
		  {
			  if (t == ResultImg.data[i*ResultImg.cols + j])
				  g_imageClass[t]++;
		  }
	   }
	}
	
	//计算每个类的Ratiomami和Rectangularity
	for (t = 1; t < g_imageClass[0]+1; t++)
	{
		Axis1 = (float)(g_classCoordinate[t][DOWN] - g_classCoordinate[t][UP]);
		Axis2 = (float)(g_classCoordinate[t][RIGHT] - g_classCoordinate[t][LEFT]);
		if(0!= Axis1&&0!= Axis2)
		{ 
		if (Axis1 > Axis2)
			Ratiomami[t] = Axis1 / Axis2;
		if (Axis1 < Axis2)
			Ratiomami[t] = Axis2 / Axis1;
		if (Axis1 == Axis2)
			Ratiomami[t] = 1;
		RectArea = Axis1 * Axis2;
		TargeArea = (float)g_imageClass[t];
		RecArea_BoatArea[t] = TargeArea / RectArea;
		}
		
	}
	return 1;
}

int Remark(Mat ResultImg, int Up, int Left, int Down, int Right,int flag)
{
	int i, j;
	int cols = ResultImg.cols*ResultImg.channels();
	if(1==flag)
	{ 
		for (j = (Left ); j < (Right+1 ); j++)
		{
			ResultImg.data[(Up )*cols + j] = 255;
			ResultImg.data[(Down )*cols + j] = 255;
		}
		for (i = (Up); i < (Down +1); i++)
		{
			ResultImg.data[i*cols + Left] = 255;
			ResultImg.data[i*cols + Right] = 255;
		}
		return 1;
	}
	if (0== flag)
	{
		for (j = (Left ); j < (Right + 1); j++)
		{
			ResultImg.data[(Up )*cols + j] = 0;
			ResultImg.data[(Down )*cols + j] = 0;
		}
		for (i = (Up ); i < (Down + 1); i++)
		{
			ResultImg.data[i*cols + Left ] = 0;
			ResultImg.data[i*cols + Right ] = 0;
		}
		return 0;
	}
}

int ShipsSurveys(Mat ResultImg, int g_classCoordinate[257][4], int g_imageClass[257], float Ratiomami[255], float RecArea_BoatArea[255],int flag)
{
	int i, j, t;
	int cols = ResultImg.cols*ResultImg.channels();
	if (0 == flag)
	{
		for (t = 1; t < g_imageClass[0] + 1; t++)
		{
			if ((1.2502 <= Ratiomami[t]) && (Ratiomami[t] <= 8.7857) && g_imageClass[t] < 5000)
			{
				Remark(ResultImg, g_classCoordinate[t][UP], g_classCoordinate[t][LEFT], g_classCoordinate[t][DOWN], g_classCoordinate[t][RIGHT], 1);
			}
			else 
				Remark(ResultImg, g_classCoordinate[t][UP], g_classCoordinate[t][LEFT], g_classCoordinate[t][DOWN], g_classCoordinate[t][RIGHT], 0);
		}
		return 0;
	}
	if (1 == flag)
	{
				for (t = 1; t < g_imageClass[0] + 1; t++)
				{
					if ((1.2502 <= Ratiomami[t]) && (Ratiomami[t] <= 8.7857) && g_imageClass[t] < 5000)
					{
						Remark(ResultImg, g_classCoordinate[t][UP], g_classCoordinate[t][LEFT], g_classCoordinate[t][DOWN], g_classCoordinate[t][RIGHT], 1);
					}
					else
						Remark(ResultImg, g_classCoordinate[t][UP], g_classCoordinate[t][LEFT], g_classCoordinate[t][DOWN], g_classCoordinate[t][RIGHT], 0);
				}
		return 1;
	}
}
//二值图像膨胀腐蚀
int BianryExpand_Erode(Mat ResultImg, int ma[5][5], int flag)
{
	int i, j;
	int rows = ResultImg.rows;
	int cols = ResultImg.cols*ResultImg.channels();
	int k, o,t=0;
	for (i = 2; i < rows - 2; i++)
	{
		for (j = 2; j < cols - 2; j++)
		{
			for (k = -2; k <= 2; k++)
			{
				for (o = -2; o <= 2; o++)
				{
					if ((ma[k + 2][o + 2] ==1)&&( ResultImg.data[(i + k)*cols + j + o]==255))
					{
						t++;
						
					}
				}
			}
			if (1 == flag && t > 3)
				ResultImg.data[i*cols + j] = 255;
			if (1 == flag && t == 0)
				ResultImg.data[i*cols + j] = 0;
			if (0 == flag && t == 25)
				ResultImg.data[i*cols + j] = 255;
			if (0 == flag && t < 25)
				ResultImg.data[i*cols + j] = 0;
			t = 0;
		}
		
	}
	return 1;
}

//二值膨胀
/*函数参数：
3     a——待膨胀的图像
4     b——膨胀后的结果
5     mat——结构元素
6 */
 void Bi_Expansion(Mat b, Mat a,  int mat[5][5])
 {
	    int i, j, k, o,t;
      int rows = a.rows;
	     int cols = a.cols*a.channels();
	     Mat tmp = a.clone();
	     uchar* src = tmp.data;
	    //膨胀是对图像中目标补集的腐蚀，因此先求输入图像数据的补集
		     for (i = 2; i < rows-2; i++)
		         for (j = 2; j < cols-2; j++)
		 * (src + i * cols + j) = 255 - *(src + i * cols + j);
	     //膨胀是结构元素的对称集对补集的腐蚀，此处求其反射
		     for (i = 0; i < 5; i++)
		         for (j = 0; j <= i; j++)
		             mat[i][j] = mat[j][i];
	     bool flag1;
	     uchar* dst = b.data;
	     //针对图像中每一个像素位置，判断是否结构元素能填入目标内部
		     for (i = 2; i < rows - 2; i++)
			 {
		         for (j = 2; j < cols - 2; j++) 
				 {
			             //判断结构元素是否可以在当前点填入目标内部，1为可以，0为不可以
				             flag1 = 1;
			             for (k = -2; k <=2/*&& flag*/; k++)
						 {
				                 for (o = -2; o <= 2; o++)
								 {
					                     //如果当前结构元素位置为1，判断与对应图像上的像素点是否为非0
						                     if (mat[k + 2][o + 2]) 
											 {
						                         if (!*(src + (i + k)*cols + j + o)) 
												 {//没击中
							                            flag1= 0;    break;
							
						                         }
				
					                         }
					
			                  	 }
			              }
			 * (dst + i * cols + j) = flag1 ? 255 : 0;
				 }
		
	         }
	       //用结构元素对称集对目标补集腐蚀后，还要对结构再求一次补集，才是膨胀结构输出
		     //赋值结构元素腐蚀漏掉的区域，使原图像恢复为二值图像
		  for (i = 2; i < rows-2; i++) 
		  {
             for (j = 2; j < cols-2; j++) 
			 {
			            * (dst + i * cols + j) = 255 - *(dst + i * cols + j);
			           if (*(dst + i * cols + j) != 255 && *(dst + i * cols + j) != 0/*&&1==flag*/)
			   	            * (dst + i * cols + j) = 0;
					   if (*(dst + i * cols + j) != 255 && *(dst + i * cols + j) != 0 /*&& 0 == flag*/)
						   * (dst + i * cols + j) = 255;
			
		      }
	      }
	
			  for (int i = 0; i < 30; i++)
			  {
				  Remark(b, i, i, rows - i - 1, cols - i - 1, 0);
			  }
}

 int N_Expansion(Mat b, Mat a, int mat[5][5], int np)
 {
	 
	   int i, j;
	    if (0 == np)
        { 
			return 0;
	    }
		Bi_Expansion(b, a, mat);
		 np--;
		 for (i = 2; i < b.rows - 2; i++)
		 {
			 for (j = 2; j < b.cols - 2; j++)
			 {
				 a.data[i*b.cols + j] = b.data[i*b.cols + j];
			 }
		 }
	  if ( 0 !=np)
	  {
		 N_Expansion(b, a, mat, np);
	  }
	
 }
 int OtsuTest(Mat OtsuResult,Mat Open_CloseImg,Mat OriginalImg,Mat grayimg,float threshold, int g_classCoordinate[257][4], int g_imageClass[257], float Ratiomami[255], float RecArea_BoatArea[255], int window[5][5],int flag)
 {
	 if(1==flag)
	 {
	   for (int i = 0; i < OriginalImg.rows; i++)
	   {
		 for (int j = 0; j < OriginalImg.cols; j++)
		 {
			 if (Open_CloseImg.data[i*OriginalImg.cols + j] >= threshold)
				 OtsuResult.data[i*OriginalImg.cols + j] = 255;
			 else OtsuResult.data[i*OriginalImg.cols + j] = 0;
		 }
	   }
	 }
	 if (2 == flag)
	 {
		 for (int i = 0; i < OriginalImg.rows; i++)
		 {
			 for (int j = 0; j < OriginalImg.cols; j++)
			 {
				 if (Open_CloseImg.data[i*OriginalImg.cols + j] >= threshold)
					 OtsuResult.data[i*OriginalImg.cols + j] = 0;
				 else OtsuResult.data[i*OriginalImg.cols + j] = 255;
			 }
		 }
	 }
	 for (int i = 0; i < 20; i++)
	 {
		
			 Remark(OtsuResult, i, i, OriginalImg.rows - i - 1, OriginalImg.cols - i - 1, 0);

	 }
	 N_Expansion(Open_CloseImg, OtsuResult, window, 3);
	 imshow("Otsu阈值分割膨胀结果图", Open_CloseImg);
	 FindClass(Open_CloseImg, g_classCoordinate, g_imageClass);
	 Feature(Open_CloseImg, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea);
	 for (int t = 1; t < g_imageClass[0] + 1; t++)
	 {
		 if ((1.2502 <= Ratiomami[t]) && (Ratiomami[t] <= 8.7857) && g_imageClass[t] < 8000)
		 {
			 Remark(grayimg, g_classCoordinate[t][UP], g_classCoordinate[t][LEFT], g_classCoordinate[t][DOWN], g_classCoordinate[t][RIGHT], 1);

		 }
		 else
			 Remark(grayimg, g_classCoordinate[t][UP], g_classCoordinate[t][LEFT], g_classCoordinate[t][DOWN], g_classCoordinate[t][RIGHT], 0);
	 }
	 return 1;
 }

 int MorpRecTest(Mat BCSMImg, Mat IFSMImg, Mat OriginalImg,Mat grayimg, float *ptr1, float *ptr2, float *ptr3, int g_classCoordinate[257][4], int g_imageClass[257], float Ratiomami[255], float RecArea_BoatArea[255],int window[5][5], int s,int n_expand)
 {
	 float theshold=0;
	 //BCSM和IFSM归一化并且融合
	 Normalization(ptr1, BCSMImg);
	 Normalization(ptr2, IFSMImg);
	 Mix(ptr3, ptr1, ptr2, OriginalImg);
	 //求取阈值,大的【10,15】，小的【25,35】
	 theshold = Threshold(ptr3, s, OriginalImg.rows, OriginalImg.cols);

	 /* 阈值分割 */
	 ThresholdOperate(IFSMImg, ptr3, theshold, 1);
	 imshow("基于小数的阈值分割结果图", IFSMImg);

	 /* 二值图像膨胀及原始图片边界周围赋值0 */
	 N_Expansion(BCSMImg, IFSMImg, window, n_expand);

	 for (int i = 0; i < 10; i++)
	 {
		 Remark(BCSMImg, i, i, OriginalImg.rows - i - 1, OriginalImg.cols - i - 1, 0);
	 }
	 imshow("阈值分割结果再膨胀结果图", BCSMImg);

	 /* 亮舰连通域查找、标识，多特征船舰检测及标注矩形框 */
	 FindClass(BCSMImg, g_classCoordinate, g_imageClass);


	 //特征提取及目标检测
	 Feature(BCSMImg, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea);
	 return 1;

 }
 int Thf_Bhf(Mat Thf_Bhfresult, Mat Open_CloseImg,Mat OrignalImg, int choose)
 {
	 int i, j;
	 int rows = Open_CloseImg.rows;
	 int cols = Open_CloseImg.cols*Open_CloseImg.channels();
	 if (1 == choose)
	 {
		 for (i = 2; i < rows - 2; i++)
		 {
			 for (j = 2; j < cols - 2; j++)
			 {
				 Thf_Bhfresult.data[i*cols + j] = OrignalImg.data[i*cols + j] - Open_CloseImg.data[i*cols + j];
			 }
		 }
		 imshow("THF亮舰", Thf_Bhfresult);
	 }
	 if (2== choose)
	 {
		 for (i = 2; i < rows - 2; i++)
		 {
			 for (j = 2; j < cols - 2; j++)
			 {
				 Thf_Bhfresult.data[i*cols + j] = Open_CloseImg.data[i*cols + j]- OrignalImg.data[i*cols + j] ;
			 }
		 }
		 imshow("BHF暗舰", Thf_Bhfresult);
	 }
	 return 1;
 }

 //注意：每个函数的第一个形参存放该函数的操作结果
int main()
{
	int SchenmeChoose = 0;																					//方案种类选择
	int ShipChoose=0;																								//船舰类型选择
	int s=14;																												//方案一阈值分割实验参数
	int n_expand=1;																										 //方案一膨胀系数
	int window[5][5] = { {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1} };  //结构元素
	int g_classCoordinate[257][4] = { 0 };                                                               //存储候选目标边界信息
	int g_imageClass[257] = { 0 };																	          //存储候选目标面积信息
	float Ratiomami[255] = { 0.0 };																			  //存储候选目标矩形边界长轴与短轴之比
	float RecArea_BoatArea[255] = { 0.0 };															  //存储候选目标面积与矩形边界面积之比
	float threshold1 = 0.0, threshold2 = 0.0;
	
	String path="C:\\Users\\HP\\Desktop\\byme\\rr.bmp";
	printf_s("Please enter numbers 1, 2 and 3 to select the scheme: scheme 1 is morphological reconstruction, scheme 2 is threshold segmentation, and scheme 3 is top/bottom cap transformation.\n");
	scanf_s("%d",&SchenmeChoose);
	if (1!=SchenmeChoose&&2!= SchenmeChoose&&3!= SchenmeChoose)
	{
		printf_s("Only 1, 2 and 3 should be selected for the scheme.");
		exit(1);
	}

	/*  读取原始图片 */
	Mat OriginalImg;
	OriginalImg = imread(path);
	if (OriginalImg.empty())
	{

		printf("读取图片错误，请确认目录下是否有imread函数指定图片存在！ \n ");
		exit(1);

	}

	/* 图像灰度化 */
	Mat graybmp(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
	Gray(graybmp, OriginalImg);

	/* 方案一：基于灰度重建及多特征分析检测弱小船舰目标 */
	if (1 == SchenmeChoose)
	{
		printf_s("Please choose a bright ship or a dark ship: 1 is a bright ship, 2 is a dark ship\n");
		scanf_s("%d", &ShipChoose);
		printf_s("Please enter threshold segmentation experimental coefficient s=");
		scanf_s("%d", &s);
		printf_s("Please enter expansion coefficient n_expand=");
		scanf_s("%d", &n_expand);
		if (1 != ShipChoose && 2 != ShipChoose )
		{
			printf_s("Only 1 and 2 should be selected for ship type.");
			exit(1);
		}

		////图像缓冲区，存储各图像处理过程的结果，
		Mat imgbuf1(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		Mat imgbuf2(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		float *ptr1, *ptr2, *ptr3;
		ptr1 = (float *)malloc((OriginalImg.rows*OriginalImg.cols) * sizeof(float));
		ptr2 = (float *)malloc((OriginalImg.rows*OriginalImg.cols) * sizeof(float));
		ptr3 = (float *)malloc((OriginalImg.rows*OriginalImg.cols) * sizeof(float));

		/*	 测地线腐蚀得CGMR */
		//关闭操作
		GrayExpand_Erode(&imgbuf1.data[0], graybmp, window, 1);
		GrayExpand_Erode(imgbuf2.data, imgbuf1, window, 0);
		//迭代基本测地线腐蚀
		GeodesicLine(imgbuf1, graybmp, imgbuf2, window, 0);
		imshow("测地线腐蚀结果图", imgbuf1);
		Mat  CGMRbuf = imgbuf1.clone();
		/* 测地线膨胀得OGMR */
		//打开操作
		GrayExpand_Erode(imgbuf1.data, graybmp, window, 0);
		GrayExpand_Erode(imgbuf2.data, imgbuf1, window, 1);
		//迭代基本测地线膨胀
		GeodesicLine(imgbuf1, graybmp, imgbuf2, window, 1);
		imshow("测地线膨胀结果图", imgbuf1);
		Mat  OGMRbuf = imgbuf1.clone();

		//亮舰目标检测
		if (1 == ShipChoose)
		{
				/* 求取亮度显著性特征图BCSMb */
				Intensity_brightness(imgbuf1, CGMRbuf, graybmp, 0, 1);
			imshow("亮舰亮度显著性特征图BCSMb", imgbuf1);
			/* 求取强度显著性特征图IFSMb */
			Intensity_brightness(imgbuf2, OGMRbuf, graybmp, 1, 1);
			imshow("亮舰强度显著性特征图IFSMb", imgbuf2);
			Mat  imgbuf3 = graybmp.clone();
			MorpRecTest(imgbuf1, imgbuf2, OriginalImg, graybmp, ptr1, ptr2, ptr3, g_classCoordinate, g_imageClass,Ratiomami,RecArea_BoatArea,window, s, n_expand);
			//标注矩形框
			ShipsSurveys(graybmp, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, 1);
			imshow("亮舰方案1检测结果", graybmp);
		}
		//暗舰目标检测
		if (2 == ShipChoose)
		{
			/* 求取亮度显著性特征图BCSMd */
			Intensity_brightness(imgbuf1, OGMRbuf, graybmp, 0, 0);
			imshow("暗舰亮度显著性特征图", imgbuf1);
			/* 求取强度显著性特征图IFSMd */
			Intensity_brightness(imgbuf2, CGMRbuf, graybmp, 1, 0);
			imshow("强度显著性特征图IFSMd", imgbuf2);
			Mat  imgbuf3 = graybmp.clone();
			MorpRecTest(imgbuf1, imgbuf2, OriginalImg, graybmp, ptr1, ptr2, ptr3, g_classCoordinate, g_imageClass,Ratiomami,RecArea_BoatArea,window, s, n_expand);
			/* 检测结果显示 */
			ShipsSurveys(graybmp, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea,1);
			imshow("暗舰方案1检测结果", graybmp);
		}
		waitKey(0);
		imgbuf1.release();
		imgbuf2.release();
		OGMRbuf.release();
		CGMRbuf.release();
		free(ptr1);
		free(ptr2);
		free(ptr3);
	}
	   


	/* 方案二：基于形态滤波及阈值分割检测弱小船舰目标 */
	if (2 == SchenmeChoose)
	{
		printf_s("Please choose a bright ship or a dark ship: 1 is a bright ship, 2 is a dark ship\n");
		scanf_s("%d", &ShipChoose);
		if (1 != ShipChoose && 2 != ShipChoose)
		{
			printf_s("Only 1 and 2 should be selected for ship type.");
			exit(1);
		}
		////图像缓冲区，存储各图像处理过程的结果，
		Mat imgbuf1(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		Mat imgbuf2(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		Mat imgbuf3(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		if(1 == ShipChoose)
		{
		   //打开操作,结果存放在imgbuf2
		   GrayExpand_Erode(imgbuf3.data, graybmp, window, 0);
		   GrayExpand_Erode(imgbuf2.data, imgbuf3, window, 1);

		   //Otsu求取阈值
		   threshold1 = (float)OtsuAlgThreshold(imgbuf2);
		   printf("threshold1 =%f    ", threshold1);
		   //方案二基于形态滤波及Otsu阈值分割检测弱小船舰目标。最终结果存放于函数的第一个参数
		   OtsuTest(imgbuf3, imgbuf2, OriginalImg, graybmp, threshold1, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, window, 1);
		   imshow("基于打开操作的Otsu阈值检测结果", graybmp);
		}
		if (2 == ShipChoose)
		{
			//关闭操作，结果存放在imgbuf3
			GrayExpand_Erode(&imgbuf2.data[0], graybmp, window, 1);
			GrayExpand_Erode(imgbuf3.data, imgbuf2, window, 0);
			
			//Otsu求取阈值
			threshold2 = (float)OtsuAlgThreshold(imgbuf3);

			printf("threshold2 =%f    ", threshold2);
			//方案二基于形态滤波及Otsu阈值分割检测弱小船舰目标。最终结果存放于函数的第一个参数
			OtsuTest(imgbuf2, imgbuf3, OriginalImg, graybmp, threshold2, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, window, 2);
			imshow("基于关闭操作的Otsu阈值检测结果", graybmp);
		}
		waitKey(0);
		imgbuf1.release();
		imgbuf2.release();
		imgbuf3.release();
	}




	/* 方案三：基于THF/BHF变换检测弱小船舰目标 */
	if (3 == SchenmeChoose)
	{
		int BiThreshold=0;
		printf_s("Please choose a bright ship or a dark ship: 1 is a bright ship, 2 is a dark ship\n");
		scanf_s("%d", &ShipChoose);
		printf_s("Please enter expansion coefficient n_expand=");
		scanf_s("%d", &n_expand);
		if (1 != ShipChoose && 2 != ShipChoose)
		{
			printf_s("Only 1 and 2 should be selected for ship type.");
			exit(1);
		}

		////图像缓冲区，存储各图像处理过程的结果，
		Mat imgbuf1(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		Mat imgbuf2(OriginalImg.rows, OriginalImg.cols, CV_8UC1);

		/* 亮舰目标检测 */
		if (1 == ShipChoose)
		{
			//打开操作
			GrayExpand_Erode(imgbuf1.data, graybmp, window, 0);//flag==0,腐蚀操作
			GrayExpand_Erode(imgbuf2.data, imgbuf1, window, 1); //得到打开操作图片，去除小白点
			//顶帽变换THF
			Thf_Bhf(imgbuf1, imgbuf2, graybmp, 1);
			BiThreshold = OtsuAlgThreshold(imgbuf1);
			printf("\nOtsuthreshold1=%d        ", BiThreshold);
			
			for (int i = 2; i < OriginalImg.rows - 2; i++)
			{
				for (int j = 2; j < OriginalImg.cols - 2; j++)
				{
					if (imgbuf1.data[i*OriginalImg.cols + j] >= BiThreshold)
						imgbuf2.data[i*OriginalImg.cols + j] = 255;
					if (imgbuf1.data[i*OriginalImg.cols + j] < BiThreshold)
						imgbuf2.data[i*OriginalImg.cols + j] = 0;
				}
			}
			imshow("亮舰阈值分割结果图", imgbuf2);
			N_Expansion(imgbuf1, imgbuf2, window, n_expand);
			FindClass(imgbuf1, g_classCoordinate, g_imageClass);
			Feature(imgbuf1, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea);
			/* 检测结果显示 */
			ShipsSurveys(graybmp, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, 1);
			imshow("亮舰方案3进行多特征约束检测结果", graybmp);
		}
		/* 暗舰目标检测 */
		if (2 == ShipChoose)
		{
			//关闭操作
			GrayExpand_Erode(&imgbuf1.data[0], graybmp, window, 1);										//flag==1,膨胀操作
			GrayExpand_Erode(imgbuf2.data, imgbuf1, window, 0);										    	//先膨胀后腐蚀得到关闭操作图片，去除小黑点
			//底帽变换BHF
			Thf_Bhf(imgbuf1, imgbuf2, graybmp, 2);
			BiThreshold = OtsuAlgThreshold(imgbuf1);
			printf("\nOtsuthreshold1=%d        ", BiThreshold);
			
			for (int i = 2; i < OriginalImg.rows - 2; i++)
			{
				for (int j = 2; j < OriginalImg.cols - 2; j++)
				{
					if (imgbuf1.data[i*OriginalImg.cols + j] >= BiThreshold)
						imgbuf2.data[i*OriginalImg.cols + j] = 255;
					if (imgbuf1.data[i*OriginalImg.cols + j] < BiThreshold)
						imgbuf2.data[i*OriginalImg.cols + j] = 0;
				}
			}
			imshow("暗舰阈值分割结果图", imgbuf2);
			N_Expansion(imgbuf1, imgbuf2, window, n_expand);
			FindClass(imgbuf1, g_classCoordinate, g_imageClass);
			Feature(imgbuf1, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea);
			/* 检测结果显示 */
			ShipsSurveys(graybmp, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, 1);
			imshow("暗舰方案3进行多特征约束检测结果", graybmp);
		}
		waitKey(0);
		imgbuf1.release();
		imgbuf2.release();
	}

	for (int t = 1; t < g_imageClass[0] + 1; t++)
	{
		printf("mianji:%d   ", g_imageClass[t]);
		printf("ChangBiDuan:%f   ", Ratiomami[t]);
		threshold1 = (float)(g_classCoordinate[t][DOWN] - g_classCoordinate[t][UP]);
		threshold2 = (float)(g_classCoordinate[t][RIGHT] - g_classCoordinate[t][LEFT]);
		printf("changzhou:%f   ", threshold1);
		printf("duanzhou:%f\n", threshold2);
	}

	
	graybmp.release();
	OriginalImg.release();
	return 0;
}