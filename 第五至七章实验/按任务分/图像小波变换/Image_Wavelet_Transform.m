%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ͼ��С���任
%%-------------------------------------------------------------------------
% Reference: ������˹.����ͼ����,P.290С���Ͷ�ֱ��ʴ���
%% ��ȡͼ��任������������˹.MATLAB������ͼ����,P.184С���任��
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
imSrc = rgb2gray(imSrc);
[hei, wid, dim] = size(imSrc);
% ��ά���߶�HaarС���ֽ�
[cA1, cH1, cV1, cD1] = dwt2(imSrc, 'haar');
figure('Name','ԭʼͼƬ','NumberTitle','off')
imshow(imSrc)
figure('Name','��ά���߶�HaarС���ֽ�ͼ','NumberTitle','off')
subplot(2,2,1), imshow(cA1,[]),title('ͼ��Ľ��ƣ���Ƶ���֣�')
subplot(2,2,2), imshow(cH1,[]),title('ͼ���������ˮƽ��')
subplot(2,2,3), imshow(cV1,[]),title('ͼ�����������ֱ��')
subplot(2,2,4), imshow(cD1,[]),title('ͼ����������Խǣ�')
% ��upcoef2������ϵ�����ع����ƺ�ϸ��
A1 = upcoef2('a', cA1, 'bior3.7', 1);
H1 = upcoef2('d', cH1, 'bior3.7', 1);
V1 = upcoef2('v', cV1, 'bior3.7', 1);
D1 = upcoef2('d', cD1, 'bior3.7', 1);
figure('Name','upcoef2�����ع��Ľ���ϸ��ͼ','NumberTitle','off')
subplot(2,2,1), imshow(A1, []),title('ͼ��Ľ��ƣ���Ƶ���֣�')
subplot(2,2,2), imshow(H1, []),title('ͼ���������ˮƽ��')
subplot(2,2,3), imshow(V1, []),title('ͼ�����������ֱ��')
subplot(2,2,4), imshow(D1, []),title('ͼ����������Խǣ�')
% ����waverec2��������С���ع�
[C, S] = wavedec2(imSrc, 2, 'bior3.7');
I_wrec = waverec2(C, S, 'bior3.7');
figure('Name','waverec2�����ع����ͼ','NumberTitle','off')
imshow(I_wrec, [])
% ����appcoef2������ȡ��һ���Ƶ����ϵ���͵ڶ����Ƶ����ϵ��
wcA1 = appcoef2(C, S, 'bior3.7', 1);
wcA2 = appcoef2(C, S, 'bior3.7', 2);
figure('Name','appcoef2������ȡ�ĵ�һ��͵ڶ������ϵ��','NumberTitle','off')
subplot(1,2,1),imshow(wcA1, []),title('��һ�����ϵ��')
subplot(1,2,2),imshow(wcA2, []),title('�ڶ������ϵ��')
% ʹ��upcoef�����ع�
I_wup1 = upcoef2('a', wcA1, 'bior3.7', 1);
I_wup2 = upcoef2('a', wcA2, 'bior3.7', 2);
figure('Name','upcoef�����ع��ĵ�һ��͵ڶ�����ƽ��','NumberTitle','off')
subplot(1,2,1),imshow(I_wup1, []),title('��һ���ع����ƽ��')
subplot(1,2,2),imshow(I_wup2, []),title('�ڶ����ع����ƽ��')
% ʹ��detcoef2������ȡ��һ��ϸ��
[chd1, cvd1, cdd1] = detcoef2('all', C, S, 1);
figure('Name','ʹ��detcoef2������ȡ�ĵ�һ��ϸ��','NumberTitle','off')
subplot(1,3,1), imshow(chd1, []),title('ͼ���������ˮƽ��')
subplot(1,3,2), imshow(cvd1, []),title('ͼ�����������ֱ��')
subplot(1,3,3), imshow(cdd1, []),title('ͼ����������Խǣ�')

