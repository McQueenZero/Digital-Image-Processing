%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ͼ��ƽ������
% ע�⣺   ��С������
%%-------------------------------------------------------------------------
% ƽ���������ͨ�˲���
% �񻯣������ͨ�˲���
% �����˲�����ģ�塢�����ԭ��
% ��ֵ�˲������ذ��Ҷ�ֵ��С�������ĵ�ȡ����λ�þ��е����ػҶ�ֵ
% ������˹���ӣ����㵼�����ݶȣ������޲��ʵ��
%%-------------------------------------------------------------------------
%% �����˲�ͼ��ƽ��
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imNosSP=imnoise(imSrc,'salt & pepper',0.02);
imNosGAS=imnoise(imSrc,'gaussian');
figure('Name','ԭʼͼ������ͼ��','NumberTitle','off')
subplot(1,3,1),imshow(imSrc),title('ԭʼͼƬ');
subplot(1,3,2),imshow(imNosSP),title('��������ͼƬ');
subplot(1,3,3),imshow(imNosGAS),title('��˹����ͼƬ');

imDstSP3x3=medfilt2(imNosSP,[3 3]);
imDstSP5x5=medfilt2(imNosSP,[5 5]);
imDstGAS3x3=medfilt2(imNosSP,[3 3]);
imDstGAS5x5=medfilt2(imNosSP,[5 5]);
figure('Name','��ֵ�˲�ȥ��Ա�','NumberTitle','off')
subplot(2,2,1),imshow(imDstSP3x3),title('����3x3��ֵ�˲�ȥ��������ͼƬ');
subplot(2,2,2),imshow(imDstSP5x5),title('����5x5��ֵ�˲�ȥ��������ͼƬ');
subplot(2,2,3),imshow(imDstGAS3x3),title('����3x3��ֵ�˲�ȥ��˹����ͼƬ');
subplot(2,2,4),imshow(imDstGAS5x5),title('����5x5��ֵ�˲�ȥ��˹����ͼƬ');

%% ������˹����ͼ����
clc, clear, close all
filename = 'cameraman.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
m4 = fspecial('laplacian',0);   %����Ϊ-4��������˹�˲���
m8 = [1 1 1; 1 -8 1; 1 1 1];    %����Ϊ-8��������˹�˲���
imf = im2double(imSrc);
imDstm4 = imf-imfilter(imf,m4,'symmetric');
imDstm8 = imf-imfilter(imf,m8,'symmetric');
figure('Name','ͼ����','NumberTitle','off')
subplot(1,3,1),imshow(imSrc),title('ԭʼͼƬ');
subplot(1,3,2),imshow(imDstm4),title('����Ϊ-4��������˹�˲�����������ͼƬ');
subplot(1,3,3),imshow(imDstm8),title('����Ϊ-8��������˹�˲�����������ͼƬ');
