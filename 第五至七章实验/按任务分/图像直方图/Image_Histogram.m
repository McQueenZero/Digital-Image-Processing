%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ͼ��ֱ��ͼ
% ע�⣺   ��С������
%%-------------------------------------------------------------------------
% ֱ��ͼ���⻯����ԭʼͼ��ֱ��ͼ�任Ϊ���ȷֲ�����ʽ��
% �������ػҶ�ֵ�Ķ�̬��Χ�����ͼ��Աȶ�
%%-------------------------------------------------------------------------
%% �Ҷ�ͼ��ֱ��ͼ
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = imSrc;
imGrayhisteq = histeq(imGray,256);   %ֱ��ͼ���⻯
figure('Name','ֱ��ͼ','NumberTitle','off')
subplot(1,2,1),imhist(imGray),title('ֱ��ͼ');    %ֱ��ͼ
subplot(1,2,2),imhist(imGrayhisteq),title('���⻯�������ֱ��ͼ');
figure('Name','�Ҷ�ͼ��','NumberTitle','off')
subplot(1,2,1),imshow(imGray),title('�Ҷ�ͼƬ');
subplot(1,2,2),imshow(imGrayhisteq),title('���⻯������ĻҶ�ͼ');

%% ��ɫͼ��ֱ��ͼ
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imRGBhisteq = histeq(imSrc,256);   %ֱ��ͼ���⻯
figure('Name','ֱ��ͼ','NumberTitle','off')
subplot(1,2,1),imhist(imSrc),title('ֱ��ͼ');    %ֱ��ͼ
subplot(1,2,2),imhist(imRGBhisteq),title('���⻯�������ֱ��ͼ');
figure('Name','��ɫͼ��','NumberTitle','off')
subplot(1,2,1),imshow(imSrc),title('��ɫͼƬ');
subplot(1,2,2),imshow(imRGBhisteq),title('���⻯������Ĳ�ɫͼ');

