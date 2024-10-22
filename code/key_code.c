关键代码
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
		
		//开运算结果图再进行膨胀
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
		
		//闭运算结果图再进行腐蚀
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