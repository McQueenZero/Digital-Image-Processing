%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   Image Basic Operations
%       ͼ��ĵ����㡢��������ͼ���������ϰ
%%-------------------------------------------------------------------------
%% ͼ��ĵ�����
clc, clear, close all
filename='Nogizaka_dark.jpg';
imSrc = imread(filename);
figure('Name','ԭʼͼƬ','NumberTitle','off'),imshow(imSrc);
% imwrite(imSrc,'�����任ǰ��ԭʼͼ��.png')
[hei, wid, len] = size(imSrc);
imDst=zeros(hei,wid,len);
Temp=mat2gray(log(1+double(imSrc)));    %�����任����һ��
imDst=im2uint8(Temp);   %����ת����uint8
figure('Name','�����任���ͼƬ','NumberTitle','off'),imshow(imDst);
% imwrite(imDst,'�����任���ͼ��.png')

%% ͼ��Ĵ�������
clc, clear, close all
filename1='Akihabara.jpg';
filename2='Ginza.jpg';
imSrc1 = imread(filename1);
imSrc2 = imread(filename2);
figure('Name','ԭʼͼƬ1','NumberTitle','off'),imshow(imSrc1);
figure('Name','ԭʼͼƬ2','NumberTitle','off'),imshow(imSrc2);
% imwrite(imSrc1,'��������ԭʼͼƬ1.png')
% imwrite(imSrc2,'��������ԭʼͼƬ2.png')
[hei1, wid1, len1] = size(imSrc1);
[hei2, wid2, len2] = size(imSrc2);
hei=(hei1+hei2)/2; wid=(wid1+wid2)/2; len=(len1+len2)/2;
imDst=zeros(hei,wid,len);

figure('Name','����������ͼƬ','NumberTitle','off')
subplot(2,2,1)
Temp=mat2gray(double(imSrc1)+double(imSrc2));   %���㲢��һ��
imDst=im2uint8(Temp);   %����ת����uint8
imshow(imDst)
title('��������')
% imwrite(imDst,'��������ͼƬ.png')
subplot(2,2,2)
Temp=mat2gray(abs(double(imSrc1)-double(imSrc2)));   %���㲢��һ��
imDst=im2uint8(Temp);   %����ת����uint8
imshow(imDst)
title('��������')
% imwrite(imDst,'��������ͼƬ.png')
subplot(2,2,3)
Temp=mat2gray(double(imSrc1).*double(imSrc2));   %���㲢��һ��
imDst=im2uint8(Temp);   %����ת����uint8
imshow(imDst)
title('��������')
% imwrite(imDst,'��������ͼƬ.png')
subplot(2,2,4)
Temp=mat2gray(double(imSrc1)./double(imSrc2)+ ...
    double(imSrc2)./double(imSrc1));   %���㲢��һ��
imDst=im2uint8(Temp);   %����ת����uint8
imshow(imDst)
title('��������')
% imwrite(imDst,'��������ͼƬ.png')

%% ͼ��ļ�������
clc, clear, close all
filename='Fuji Television Building Small.jpg';
imSrc = imread(filename);
figure('Name','����������ԭʼͼƬ','NumberTitle','off'),imshow(imSrc);
% imwrite(imSrc,'����������ԭʼͼƬ.png')
[hei, wid, len] = size(imSrc);
imDst=zeros(hei,wid,len);
imDst=imresize(imSrc,1.5,'bilinear'); 
figure('Name','�Ŵ�1.5����ͼƬ','NumberTitle','off'),imshow(imDst);
% imwrite(imDst,'�Ŵ�1.5����ͼƬ.png')
imDst=imresize(imSrc,0.8,'bilinear'); 
figure('Name','��С��0.8����ͼƬ','NumberTitle','off'),imshow(imDst);
% imwrite(imDst,'��С��0.8����ͼƬ.png')

filename='Tokyo University Yasuda Koudou.jpg';
imSrc = imread(filename);
figure('Name','��ת������ԭʼͼƬ','NumberTitle','off'),imshow(imSrc);
% imwrite(imSrc,'��ת������ԭʼͼƬ.png')
imDst=imrotate(imSrc,-45); 
figure('Name','˳ʱ����ת45���ͼƬ','NumberTitle','off'),imshow(imDst);
% imwrite(imDst,'˳ʱ����ת45���ͼƬ.png')
