%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ͼ���Ե���
%%-------------------------------------------------------------------------
%% ͼ���Ե��⣨������˹.MATLAB������ͼ����,P.289��
clc, clear, close all
filename = 'building.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
% Sobel��Ե�����
[gS, tS] = edge(imSrc, 'sobel');
% Prewitt��Ե�����
[gP, tP] = edge(imSrc, 'prewitt');
% Roberts��Ե�����
[gR, tR] = edge(imSrc, 'roberts');
% LoG��Ե�����
[gL, tL] = edge(imSrc, 'log');
% Zero crossings��Ե�����
[gZ, tZ] = edge(imSrc, 'zerocross');
% Canny��Ե�����
[gC, tC] = edge(imSrc, 'canny');
% ���
figure('Name','ԭʼͼƬ','NumberTitle','off')
imshow(imSrc)
figure('Name','��ͬ��Ե�����������','NumberTitle','off')
subplot(2,3,1)
imshow(gS),title('Sobel��Ե���������ͼƬ')
subplot(2,3,2)
imshow(gP),title('Prewitt��Ե���������ͼƬ')
subplot(2,3,3)
imshow(gR),title('Roberts��Ե���������ͼƬ')
subplot(2,3,4)
imshow(gL),title('LoG��Ե���������ͼƬ')
subplot(2,3,5)
imshow(gZ),title('Zero crossings��Ե���������ͼƬ')
subplot(2,3,6)
imshow(gC),title('Canny��Ե���������ͼƬ')

