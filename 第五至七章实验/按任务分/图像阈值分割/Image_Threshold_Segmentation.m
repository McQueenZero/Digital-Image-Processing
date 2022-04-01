%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ͼ����ֵ�ָ�
%%-------------------------------------------------------------------------
%% ��ֵ�ָ������˹.MATLAB������ͼ����,P.305��
clc, clear, close all
filename = 'building.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);

T = 0.5*(double(min(imSrc(:))) + double(max(imSrc(:))));
done = false;
while ~done
    g_ILPF = imSrc >= T;
    Tnext = 0.5*(mean(imSrc(g_ILPF)) + mean(imSrc(~g_ILPF)));
    done = abs(T - Tnext) < 0.5;
    T = Tnext;
end
Tg = graythresh(imSrc)*255;
imDst_manual = zeros(hei,wid,dim);
imDst_graythresh = zeros(hei,wid,dim);
for ver = 1:hei   %�±�i���к�
    for hor = 1:wid   %�±�j���к�
        if imSrc(ver,hor) > T
            imDst_manual(ver,hor) = 0;
        else 
            imDst_manual(ver,hor) = 1;
        end
        if imSrc(ver,hor) > Tg
            imDst_graythresh(ver,hor) = 0;
        else
            imDst_graythresh(ver,hor) = 1;
        end
    end
end
figure('Name','��ֵ�ָ�','NumberTitle','off')
subplot(1,3,1)
imshow(imSrc),title('ԭʼͼƬ')
subplot(1,3,2)
imshow(imDst_manual),title('�Ա෽��������ֵ�ķָ�ͼƬ')
subplot(1,3,3)
imshow(imDst_graythresh),title('Otsu����(graythresh)������ֵ�ķָ�ͼƬ')
disp(['�Ա෽����ֵ��', num2str(T)])
disp(['graythresh������ֵ��',num2str(Tg)])

