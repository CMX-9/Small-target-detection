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

//�������ط�ʵ��ͼ��ҶȻ�
int Gray(Mat &GrayImg,Mat OriginalImg)
{
	int height = OriginalImg.rows;
	int width = OriginalImg.cols;
	int step1 = OriginalImg.step;                                         //ÿ���ж��ٸ�8bit
	
	int step2 =GrayImg.step;                                          //ע��src��dst��step��һ��
	int channels = OriginalImg.channels();                           //ÿ��������ɫͨ����
	//��ɫͼ��ҶȻ�
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
//�Ҷ�ͼ�����Ͳ������ṹԪ��Ϊ5*5
//flag==1,��������ͣ�flag==0������и�ʴ
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
		//����ֲ����ֵ����Ϊ�����ص������ֵ��flag==1,��������ͣ�flag==0������и�ʴ
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

//��������߸�ʴ�����ͣ�
//flag==1,������������ͣ�ȡ��Сֵ
//flag==0,��������߸�ʴ��ȡ���ֵ
//�ȶ�������ǰһ�����ͼ�뵱ǰ���ͼ��ͬ
int BaseGeodesicLine(Mat Dilate_ErodeImg,Mat OriginalImg, Mat Expang_ErodeImg, int ma[5][5], int flag)   // Exchangeͼ�����ݸı������Ϊ0ʱ��ʾ����������
{
	int i, j;			
	int rows = Expang_ErodeImg.rows;
	int cols = Expang_ErodeImg.cols*Expang_ErodeImg.channels();
	if (1 == flag)
	{
		
		//�򿪲������ͼ�ٽ�������
		 GrayExpand_Erode(Dilate_ErodeImg.data,Expang_ErodeImg, ma, 1);
		//������������ͣ�ȡ��Сֵ
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
		
		//�رղ������ͼ�ٽ��и�ʴ
		  GrayExpand_Erode(Dilate_ErodeImg.data,Expang_ErodeImg, ma, 0);
		//��������߸�ʴ��ȡ���ֵ
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

//����ʵ�ֲ���߸�ʴ�����ͣ�
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
//ע��:
//ͨ����������̬�ؽ����ͼCGMR�õ�������ǿ������������ͼIFSMd
//ͨ����������̬�ؽ����ͼOGMR�õ�������ǿ������������ͼIFSMb
//ͨ��CGMR�õ�BCSMb
//ͨ��OGMR�õ�BCSMd 
//����Intensity_brightness��������������������֣�1xʱ��IFSM��01ʱ��BCSMb��00ʱ��BCSMd
int Intensity_brightness(Mat B_ISMImg,Mat GMR,  Mat OriginalImg, int ChooseFeature,int flag)
{
	int i, j;
	int ma[5][5]= { {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1} };
	int rows = GMR.rows;
	int cols = GMR.cols*GMR.channels();
	Mat buf1(rows, cols, CV_8UC1);
	Mat buf2(rows, cols, CV_8UC1);
	//ChooseFeature==1,��ʾ���������CGMR�򰵽����OGMR�õ�ǿ�������Խ��ͼIFSMd��IFSMb
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
	//ChooseFeature==0,��ʾ�õ����������Խ��ͼBCSMb��BCSMd
	if(0== ChooseFeature)
	{
		//1==flag,��ʾ���������CGMR�õ�����ͼ���BCSMb
		if(1==flag)
		{
			//�򿪲���
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

		//0==flag,��ʾ�ð������OGMR�õ�����ͼ���BCSMd
		if (0 == flag)
		{
			//�رղ���
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
//BCSM��ISFM��һ��
int Normalization(float *NorImg,Mat BCSM_ISFMimg)
{
	int i, j;
	float t, m,g;
	int rows = BCSM_ISFMimg.rows;
	int cols = BCSM_ISFMimg.cols*BCSM_ISFMimg.channels();
	int Smax = BCSM_ISFMimg.data[0];
	int Smin= BCSM_ISFMimg.data[0];
	
	//��������ͼ�����ֵSmax����СֵSmin
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
//������ͼ�ں�
//������ͼBCSMb��BCSMd����ISFMb��ISFMd���ںϵ�SMb��SMd��
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

//��ֵ�ָ������ֵ,��ƽ������double sqrt(double x),ͷ�ļ���<math.h>
int OtsuAlgThreshold(const Mat image) 
{
	if (image.channels() != 1)
	{ 
		cout << "Please input Gray-image!" << endl;		
		return 0;
	}	
	int T = 0;							 //Otsu�㷨��ֵ	
	double varValue=0;		//��䷽���м�ֵ����	
	double w0=0;					//ǰ�����ص�����ռ����
	double w1=0;					//�������ص�����ռ����
	double u0=0;					//ǰ��ƽ���Ҷ�	
	double u1=0;					//����ƽ���Ҷ�	
	double Histogram[256]={0}; //�Ҷ�ֱ��ͼ���±��ǻҶ�ֵ�����������ǻҶ�ֵ��Ӧ�����ص�����	
	uchar *data=image.data;	
	double totalNum=image.rows*image.cols; //��������
   //����Ҷ�ֱ��ͼ�ֲ���Histogram�����±��ǻҶ�ֵ�����������ǻҶ�ֵ��Ӧ���ص���	
	for(int i=0;i<image.rows;i++)   //Ϊ������������û�а�rows��cols���������
	{		
		for(int j=0;j<image.cols;j++)	
		{			
			Histogram[data[i*image.step+j]]++;		
		}
	}
	for(int i=0;i<255;i++)
	{		
		//ÿ�α���֮ǰ��ʼ��������
		w1=0;		u1=0;		w0=0;		u0=0;		
		//***********����������ֵ����**************************	
		for(int j=0;j<=i;j++) //�������ָ�ֵ����	
		{			
			w1+=Histogram[j];  //�����������ص�����			
			u1+=j*Histogram[j]; //�������������ܻҶȺ�	
		}		
		if(w1==0) //�����������ص���Ϊ0ʱ�˳�	
		{			
			continue;		
		}		
		u1=u1/w1; //��������ƽ���Ҷ�	
		w1=w1/totalNum; // �����������ص�����ռ����		
		//***********����������ֵ����************************** 		
		//***********ǰ��������ֵ����**************************		
		for(int k=i+1;k<255;k++)		
		{			
			w0+=Histogram[k];  //ǰ���������ص�����		
			u0+=k*Histogram[k]; //ǰ�����������ܻҶȺ�	
		}		
		if(w0==0) //ǰ���������ص���Ϊ0ʱ�˳�		
		{			
			break;
		}		
		u0=u0/w0; //ǰ������ƽ���Ҷ�		
		w0=w0/totalNum; // ǰ���������ص�����ռ����		
		//***********ǰ��������ֵ����************************** 		
		//***********��䷽�����******************************		
		double varValueI=w0*w1*(u1-u0)*(u1-u0); //��ǰ��䷽�����		
		if(varValue<varValueI)	
		{			
			varValue=varValueI;		
			T=i;	
		}	
	}	
	return T;
}

//��ֵ�ָ������ֵ, ��ƽ������double sqrt(double x), ͷ�ļ���<math.h>
//��ʽt=SMƽ��ֵ+s*SM��׼�sȡ[10,15],s��ʵ������
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

//����С������ֵ�ָ�
int ThresholdOperate(Mat ThresholdImg, float *ptr, float threshold,int flag)
{
	int i, j;
	float data;
	int rows = ThresholdImg.rows;
	int cols= ThresholdImg.cols;
	//����Ŀ��
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
	//����Ŀ��
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

/*��ǰ��Ϊ�����еĵ�ʱ���ж����Ƿ�Ϊ����߽�㲢���±߽�*/
int UpdateBound(unsigned int pixelClass, int g_classCoordinate[257][4], unsigned int y_rows, unsigned int x_cols)
{
	if (x_cols < g_classCoordinate[pixelClass][LEFT])
	{
		g_classCoordinate[pixelClass][LEFT] = x_cols;             //������߽�
	}
	else if (x_cols > g_classCoordinate[pixelClass][RIGHT])
	{
		g_classCoordinate[pixelClass][RIGHT] = x_cols;            //�����ұ߽�
	}
	if (y_rows > g_classCoordinate[pixelClass][DOWN])
	{
		g_classCoordinate[pixelClass][DOWN] = y_rows;              //�����½�
	}
	return 1;
}


/*�ҵ����������������еĺڵ㣬��������Ϊ��ǰ����Ԫ��*/
int MarkerNeighbourhood(Mat ResultImg, int g_classCoordinate[257][4],unsigned int pixelClass, unsigned int y_rows, unsigned int x_cols,int *g_targetPixelUp,int *g_targetPixelLeft)
{
	    UpdateBound(pixelClass, g_classCoordinate, y_rows, x_cols);                      //���±߽�

		if (255== ResultImg.data[(y_rows + 1)*ResultImg.cols + x_cols])        //Ŀ����²�
		{
			ResultImg.data[(y_rows + 1)*ResultImg.cols + x_cols] = pixelClass;
		}
		if (255 == ResultImg.data[(y_rows -1)*ResultImg.cols + x_cols])        //Ŀ����ϲ�
		{
			ResultImg.data[(y_rows - 1)*ResultImg.cols + x_cols] = pixelClass;
			*g_targetPixelUp = 1;
		}
		if (255 == ResultImg.data[y_rows*ResultImg.cols + x_cols -1])          //Ŀ������
		{
			ResultImg.data[y_rows*ResultImg.cols + x_cols - 1] = pixelClass;
			*g_targetPixelLeft = 1;
		}
		if (255== ResultImg.data[y_rows*ResultImg.cols + x_cols +1])          //Ŀ����Ҳ�
		{
			ResultImg.data[y_rows*ResultImg.cols + x_cols +1] = pixelClass;
		}
		return 1;
}

/*��ͼ���и�����ͨ�������ҳ���*/
/*����˳�򣺴������ң���������*/
//Ѱ�Ҵ������Ǵ�����
//g_imageClass[]�������������ֵ��g_classCoordinate[][4]�����ľ��α߽磬g_imageClass[0] ���ҵ�������������ǰ�ɼ���Ŀ����Ŀ���Ϊ255��
int FindClass(Mat ResultImg, int g_classCoordinate[257][4], int g_imageClass[257])
{
	unsigned int pixelClass = 0;
	unsigned int          x = 0;
	unsigned int          y = 0;
	unsigned int    HeaderX = 0;															 //������ʼ�������
	unsigned int    HeaderY = 0;															 //������ʼ��������
	unsigned int  classFlag = 0;															 //0:��δ�ҵ�����ĵ�һ���㣻1�����ҵ��������ʼ�㣬��׼��������ͨ�㣻2�����ҵ�������
	int i, j;
	//ע�⣬��ʼ���߽�Ӧ��ȡ�������
	g_classCoordinate[pixelClass][UP] =2 ;                                        //��ʼ����ǰ�������Ͻ�
	g_classCoordinate[pixelClass][DOWN] = ResultImg.rows-2;                        //��ʼ����ǰ�������½�
	g_classCoordinate[pixelClass][LEFT] =2;										  //��ʼ����ǰ��������߽�
	g_classCoordinate[pixelClass][RIGHT] = ResultImg.cols - 2;       //��ʼ����ǰ�������ұ߽�
	int g_targetPixelLeft = 0;
	int g_targetPixelUp = 0;
	while (classFlag != 2)
	{
		if (0 == classFlag)																			//���ҵ�ǰ�����ʼ��
		{
			pixelClass++;																			 //������һ�����ı���
			if (pixelClass > 255)
			{
				g_imageClass[0] = 255;
				printf("��ǰ�ɼ���Ŀ����Ŀ���Ϊ255��");
				exit(1);																					 //��ǰ�ɼ���Ŀ����Ŀ���Ϊ255��
			}

			g_imageClass[pixelClass] = 0;                                                 //��ǰ�����ص��������
			//������Χ��Ϊ2�����ز�����Ŀ��
			g_classCoordinate[pixelClass][UP] = ResultImg.rows-2;      //��ʼ����ǰ�������Ͻ�
			g_classCoordinate[pixelClass][DOWN] = 2;						   //��ʼ����ǰ�������½�
			g_classCoordinate[pixelClass][LEFT] = ResultImg.cols-2;    //��ʼ����ǰ��������߽�
			g_classCoordinate[pixelClass][RIGHT] = 2;                           //��ʼ����ǰ�������ұ߽�

			for (i = g_classCoordinate[pixelClass - 1][UP]; i < ResultImg.rows - 2; i++)           //ÿ��������ϸ�������Ͻ翪ʼ��Ѱ
			{
				for (j = 2; j < ResultImg.cols - 2; j++)																	  //ÿ�κ����2��ʼ��Ѱ
				{
					if (255 == ResultImg.data[i*ResultImg.cols + j])										     //�ҵ��������ʼ��
					{
						HeaderY = i;
						HeaderX = j;
						//g_imageClass[pixelClass]++;
						g_classCoordinate[pixelClass][UP] = i;													   //��ʼ����ʼ���Ͻ�
						ResultImg.data[i*ResultImg.cols + j] = pixelClass;									    //����ʼ��Ϊ��ǰ��ı��
						MarkerNeighbourhood(ResultImg, g_classCoordinate, pixelClass, HeaderY, HeaderX, &g_targetPixelUp, &g_targetPixelLeft);                  //��Ѱ��ʼ����Χ��Ŀ���
						classFlag = 1;
						break;
					}
				}//for x = 1
				if (1 == classFlag)
				{
					break;
				}
			}//for y
			if (0 == classFlag)																											//ͼ�����кڵ���ѹ���
			{
				pixelClass--;
				g_imageClass[0] = pixelClass;
				classFlag = 2;																													//ͼ���ѱ�����ȫ
			}
		}//if
		else if (1 == classFlag)
		{
			//����ȫͼ���ҵ���ǰ�����ͨ��
			for (i = g_classCoordinate[pixelClass][UP]; i < ResultImg.rows - 2; i++)
			{
				//g_rowTargetPixelNum = 0;																					  //��ǰ��Ŀ�����Ŀ����
				for (j = 2; j < ResultImg.cols - 2; j++)
				{
					if (pixelClass == ResultImg.data[i*ResultImg.cols + j])										  //��ǰ��Ϊ��ǰ��ĵ�
					{
						MarkerNeighbourhood(ResultImg, g_classCoordinate, pixelClass, i, j, &g_targetPixelUp, &g_targetPixelLeft);  //��Ѱ��ǰ�����򣬲��Ҳ���Ǻڵ�
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

//Ratiomami�����������ı��ʣ�������������������Բ��
//RecArea_BoatArea�����ζȣ���ͨ��������[0��1]��Χ�ڵ�ֵ����ʾ�������ƶȣ�1��ʾ����ľ�������
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
	
	//����ÿ�����Ratiomami��Rectangularity
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
//��ֵͼ�����͸�ʴ
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

//��ֵ����
/*����������
3     a���������͵�ͼ��
4     b�������ͺ�Ľ��
5     mat�����ṹԪ��
6 */
 void Bi_Expansion(Mat b, Mat a,  int mat[5][5])
 {
	    int i, j, k, o,t;
      int rows = a.rows;
	     int cols = a.cols*a.channels();
	     Mat tmp = a.clone();
	     uchar* src = tmp.data;
	    //�����Ƕ�ͼ����Ŀ�겹���ĸ�ʴ�������������ͼ�����ݵĲ���
		     for (i = 2; i < rows-2; i++)
		         for (j = 2; j < cols-2; j++)
		 * (src + i * cols + j) = 255 - *(src + i * cols + j);
	     //�����ǽṹԪ�صĶԳƼ��Բ����ĸ�ʴ���˴����䷴��
		     for (i = 0; i < 5; i++)
		         for (j = 0; j <= i; j++)
		             mat[i][j] = mat[j][i];
	     bool flag1;
	     uchar* dst = b.data;
	     //���ͼ����ÿһ������λ�ã��ж��Ƿ�ṹԪ��������Ŀ���ڲ�
		     for (i = 2; i < rows - 2; i++)
			 {
		         for (j = 2; j < cols - 2; j++) 
				 {
			             //�жϽṹԪ���Ƿ�����ڵ�ǰ������Ŀ���ڲ���1Ϊ���ԣ�0Ϊ������
				             flag1 = 1;
			             for (k = -2; k <=2/*&& flag*/; k++)
						 {
				                 for (o = -2; o <= 2; o++)
								 {
					                     //�����ǰ�ṹԪ��λ��Ϊ1���ж����Ӧͼ���ϵ����ص��Ƿ�Ϊ��0
						                     if (mat[k + 2][o + 2]) 
											 {
						                         if (!*(src + (i + k)*cols + j + o)) 
												 {//û����
							                            flag1= 0;    break;
							
						                         }
				
					                         }
					
			                  	 }
			              }
			 * (dst + i * cols + j) = flag1 ? 255 : 0;
				 }
		
	         }
	       //�ýṹԪ�ضԳƼ���Ŀ�겹����ʴ�󣬻�Ҫ�Խṹ����һ�β������������ͽṹ���
		     //��ֵ�ṹԪ�ظ�ʴ©��������ʹԭͼ��ָ�Ϊ��ֵͼ��
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
	 imshow("Otsu��ֵ�ָ����ͽ��ͼ", Open_CloseImg);
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
	 //BCSM��IFSM��һ�������ں�
	 Normalization(ptr1, BCSMImg);
	 Normalization(ptr2, IFSMImg);
	 Mix(ptr3, ptr1, ptr2, OriginalImg);
	 //��ȡ��ֵ,��ġ�10,15����С�ġ�25,35��
	 theshold = Threshold(ptr3, s, OriginalImg.rows, OriginalImg.cols);

	 /* ��ֵ�ָ� */
	 ThresholdOperate(IFSMImg, ptr3, theshold, 1);
	 imshow("����С������ֵ�ָ���ͼ", IFSMImg);

	 /* ��ֵͼ�����ͼ�ԭʼͼƬ�߽���Χ��ֵ0 */
	 N_Expansion(BCSMImg, IFSMImg, window, n_expand);

	 for (int i = 0; i < 10; i++)
	 {
		 Remark(BCSMImg, i, i, OriginalImg.rows - i - 1, OriginalImg.cols - i - 1, 0);
	 }
	 imshow("��ֵ�ָ��������ͽ��ͼ", BCSMImg);

	 /* ������ͨ����ҡ���ʶ��������������⼰��ע���ο� */
	 FindClass(BCSMImg, g_classCoordinate, g_imageClass);


	 //������ȡ��Ŀ����
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
		 imshow("THF����", Thf_Bhfresult);
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
		 imshow("BHF����", Thf_Bhfresult);
	 }
	 return 1;
 }

 //ע�⣺ÿ�������ĵ�һ���βδ�Ÿú����Ĳ������
int main()
{
	int SchenmeChoose = 0;																					//��������ѡ��
	int ShipChoose=0;																								//��������ѡ��
	int s=14;																												//����һ��ֵ�ָ�ʵ�����
	int n_expand=1;																										 //����һ����ϵ��
	int window[5][5] = { {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1} };  //�ṹԪ��
	int g_classCoordinate[257][4] = { 0 };                                                               //�洢��ѡĿ��߽���Ϣ
	int g_imageClass[257] = { 0 };																	          //�洢��ѡĿ�������Ϣ
	float Ratiomami[255] = { 0.0 };																			  //�洢��ѡĿ����α߽糤�������֮��
	float RecArea_BoatArea[255] = { 0.0 };															  //�洢��ѡĿ���������α߽����֮��
	float threshold1 = 0.0, threshold2 = 0.0;
	
	String path="C:\\Users\\HP\\Desktop\\byme\\rr.bmp";
	printf_s("Please enter numbers 1, 2 and 3 to select the scheme: scheme 1 is morphological reconstruction, scheme 2 is threshold segmentation, and scheme 3 is top/bottom cap transformation.\n");
	scanf_s("%d",&SchenmeChoose);
	if (1!=SchenmeChoose&&2!= SchenmeChoose&&3!= SchenmeChoose)
	{
		printf_s("Only 1, 2 and 3 should be selected for the scheme.");
		exit(1);
	}

	/*  ��ȡԭʼͼƬ */
	Mat OriginalImg;
	OriginalImg = imread(path);
	if (OriginalImg.empty())
	{

		printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���imread����ָ��ͼƬ���ڣ� \n ");
		exit(1);

	}

	/* ͼ��ҶȻ� */
	Mat graybmp(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
	Gray(graybmp, OriginalImg);

	/* ����һ�����ڻҶ��ؽ������������������С����Ŀ�� */
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

		////ͼ�񻺳������洢��ͼ������̵Ľ����
		Mat imgbuf1(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		Mat imgbuf2(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		float *ptr1, *ptr2, *ptr3;
		ptr1 = (float *)malloc((OriginalImg.rows*OriginalImg.cols) * sizeof(float));
		ptr2 = (float *)malloc((OriginalImg.rows*OriginalImg.cols) * sizeof(float));
		ptr3 = (float *)malloc((OriginalImg.rows*OriginalImg.cols) * sizeof(float));

		/*	 ����߸�ʴ��CGMR */
		//�رղ���
		GrayExpand_Erode(&imgbuf1.data[0], graybmp, window, 1);
		GrayExpand_Erode(imgbuf2.data, imgbuf1, window, 0);
		//������������߸�ʴ
		GeodesicLine(imgbuf1, graybmp, imgbuf2, window, 0);
		imshow("����߸�ʴ���ͼ", imgbuf1);
		Mat  CGMRbuf = imgbuf1.clone();
		/* ��������͵�OGMR */
		//�򿪲���
		GrayExpand_Erode(imgbuf1.data, graybmp, window, 0);
		GrayExpand_Erode(imgbuf2.data, imgbuf1, window, 1);
		//�����������������
		GeodesicLine(imgbuf1, graybmp, imgbuf2, window, 1);
		imshow("��������ͽ��ͼ", imgbuf1);
		Mat  OGMRbuf = imgbuf1.clone();

		//����Ŀ����
		if (1 == ShipChoose)
		{
				/* ��ȡ��������������ͼBCSMb */
				Intensity_brightness(imgbuf1, CGMRbuf, graybmp, 0, 1);
			imshow("������������������ͼBCSMb", imgbuf1);
			/* ��ȡǿ������������ͼIFSMb */
			Intensity_brightness(imgbuf2, OGMRbuf, graybmp, 1, 1);
			imshow("����ǿ������������ͼIFSMb", imgbuf2);
			Mat  imgbuf3 = graybmp.clone();
			MorpRecTest(imgbuf1, imgbuf2, OriginalImg, graybmp, ptr1, ptr2, ptr3, g_classCoordinate, g_imageClass,Ratiomami,RecArea_BoatArea,window, s, n_expand);
			//��ע���ο�
			ShipsSurveys(graybmp, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, 1);
			imshow("��������1�����", graybmp);
		}
		//����Ŀ����
		if (2 == ShipChoose)
		{
			/* ��ȡ��������������ͼBCSMd */
			Intensity_brightness(imgbuf1, OGMRbuf, graybmp, 0, 0);
			imshow("������������������ͼ", imgbuf1);
			/* ��ȡǿ������������ͼIFSMd */
			Intensity_brightness(imgbuf2, CGMRbuf, graybmp, 1, 0);
			imshow("ǿ������������ͼIFSMd", imgbuf2);
			Mat  imgbuf3 = graybmp.clone();
			MorpRecTest(imgbuf1, imgbuf2, OriginalImg, graybmp, ptr1, ptr2, ptr3, g_classCoordinate, g_imageClass,Ratiomami,RecArea_BoatArea,window, s, n_expand);
			/* �������ʾ */
			ShipsSurveys(graybmp, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea,1);
			imshow("��������1�����", graybmp);
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
	   


	/* ��������������̬�˲�����ֵ�ָ�����С����Ŀ�� */
	if (2 == SchenmeChoose)
	{
		printf_s("Please choose a bright ship or a dark ship: 1 is a bright ship, 2 is a dark ship\n");
		scanf_s("%d", &ShipChoose);
		if (1 != ShipChoose && 2 != ShipChoose)
		{
			printf_s("Only 1 and 2 should be selected for ship type.");
			exit(1);
		}
		////ͼ�񻺳������洢��ͼ������̵Ľ����
		Mat imgbuf1(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		Mat imgbuf2(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		Mat imgbuf3(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		if(1 == ShipChoose)
		{
		   //�򿪲���,��������imgbuf2
		   GrayExpand_Erode(imgbuf3.data, graybmp, window, 0);
		   GrayExpand_Erode(imgbuf2.data, imgbuf3, window, 1);

		   //Otsu��ȡ��ֵ
		   threshold1 = (float)OtsuAlgThreshold(imgbuf2);
		   printf("threshold1 =%f    ", threshold1);
		   //������������̬�˲���Otsu��ֵ�ָ�����С����Ŀ�ꡣ���ս������ں����ĵ�һ������
		   OtsuTest(imgbuf3, imgbuf2, OriginalImg, graybmp, threshold1, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, window, 1);
		   imshow("���ڴ򿪲�����Otsu��ֵ�����", graybmp);
		}
		if (2 == ShipChoose)
		{
			//�رղ�������������imgbuf3
			GrayExpand_Erode(&imgbuf2.data[0], graybmp, window, 1);
			GrayExpand_Erode(imgbuf3.data, imgbuf2, window, 0);
			
			//Otsu��ȡ��ֵ
			threshold2 = (float)OtsuAlgThreshold(imgbuf3);

			printf("threshold2 =%f    ", threshold2);
			//������������̬�˲���Otsu��ֵ�ָ�����С����Ŀ�ꡣ���ս������ں����ĵ�һ������
			OtsuTest(imgbuf2, imgbuf3, OriginalImg, graybmp, threshold2, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, window, 2);
			imshow("���ڹرղ�����Otsu��ֵ�����", graybmp);
		}
		waitKey(0);
		imgbuf1.release();
		imgbuf2.release();
		imgbuf3.release();
	}




	/* ������������THF/BHF�任�����С����Ŀ�� */
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

		////ͼ�񻺳������洢��ͼ������̵Ľ����
		Mat imgbuf1(OriginalImg.rows, OriginalImg.cols, CV_8UC1);
		Mat imgbuf2(OriginalImg.rows, OriginalImg.cols, CV_8UC1);

		/* ����Ŀ���� */
		if (1 == ShipChoose)
		{
			//�򿪲���
			GrayExpand_Erode(imgbuf1.data, graybmp, window, 0);//flag==0,��ʴ����
			GrayExpand_Erode(imgbuf2.data, imgbuf1, window, 1); //�õ��򿪲���ͼƬ��ȥ��С�׵�
			//��ñ�任THF
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
			imshow("������ֵ�ָ���ͼ", imgbuf2);
			N_Expansion(imgbuf1, imgbuf2, window, n_expand);
			FindClass(imgbuf1, g_classCoordinate, g_imageClass);
			Feature(imgbuf1, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea);
			/* �������ʾ */
			ShipsSurveys(graybmp, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, 1);
			imshow("��������3���ж�����Լ�������", graybmp);
		}
		/* ����Ŀ���� */
		if (2 == ShipChoose)
		{
			//�رղ���
			GrayExpand_Erode(&imgbuf1.data[0], graybmp, window, 1);										//flag==1,���Ͳ���
			GrayExpand_Erode(imgbuf2.data, imgbuf1, window, 0);										    	//�����ͺ�ʴ�õ��رղ���ͼƬ��ȥ��С�ڵ�
			//��ñ�任BHF
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
			imshow("������ֵ�ָ���ͼ", imgbuf2);
			N_Expansion(imgbuf1, imgbuf2, window, n_expand);
			FindClass(imgbuf1, g_classCoordinate, g_imageClass);
			Feature(imgbuf1, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea);
			/* �������ʾ */
			ShipsSurveys(graybmp, g_classCoordinate, g_imageClass, Ratiomami, RecArea_BoatArea, 1);
			imshow("��������3���ж�����Լ�������", graybmp);
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