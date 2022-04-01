%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像阈值分割
%%-------------------------------------------------------------------------
%% 阈值分割（冈萨雷斯.MATLAB版数字图像处理,P.305）
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
for ver = 1:hei   %下标i，行号
    for hor = 1:wid   %下标j，列号
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
figure('Name','阈值分割','NumberTitle','off')
subplot(1,3,1)
imshow(imSrc),title('原始图片')
subplot(1,3,2)
imshow(imDst_manual),title('自编方法计算阈值的分割图片')
subplot(1,3,3)
imshow(imDst_graythresh),title('Otsu方法(graythresh)计算阈值的分割图片')
disp(['自编方法阈值：', num2str(T)])
disp(['graythresh采用阈值：',num2str(Tg)])

