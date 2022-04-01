%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年3月
% 说明：   Image Demosaic
%       输入一幅RGB图像，先进行RGB2Bayer转换，再进行Bayer2RGB转换。
%       用边缘自适应插值的方法进行去马赛克处理
%%-------------------------------------------------------------------------
%% 选择去马赛克方法
clc,clear,close all;
while 1
    disp("MATLAB自带去马赛克函数，输入'1'")
    disp("3x3窗口插值，输入'2'")
    disp("边缘自适应插值，输入'3'")
    Method = input('输入去马赛克方法：');
    switch Method
        case 1
            break
        case 2
            break
        case 3
            break
        otherwise
            disp('非法，请重新输入')
    end
end

%% 读取原始图像
tic;
% 读真彩色图
filename = 'Tokyo University Yasuda Koudou';
imSrc = imread([filename,'.jpg']);
figure('Name','原始图片','NumberTitle','off'),imshow(imSrc);
% imwrite(imSrc,'原始真彩色图片.png')
[hei, wid, len] = size(imSrc);
bayer = uint8(zeros(hei,wid));
for k=1:3
    bayer_mosaic(:,:,k) = uint8(zeros(hei,wid));
end

%% RGB图像转RGGB模式Bayer马赛克图像
% RGGB模式如下
% R G R G R G
% G B G B G B
% R G R G R G
% G B G B G B
% -----------
for ver = 1:hei
    for hor = 1:wid
        if((1 == mod(ver,2)) && (1 == mod(hor,2)))
            bayer(ver,hor) = imSrc(ver,hor,1);  %R
            bayer_mosaic(ver,hor,1) = imSrc(ver,hor,1);
        elseif((0 == mod(ver,2)) && (0 == mod(hor,2)))
            bayer(ver,hor) = imSrc(ver,hor,3);  %B
            bayer_mosaic(ver,hor,3) = imSrc(ver,hor,3);
        else
            bayer(ver,hor) = imSrc(ver,hor,2);  %G
            bayer_mosaic(ver,hor,2) = imSrc(ver,hor,2);
        end
    end
end

% figure('Name','RGGB灰度图片','NumberTitle','off'),imshow(bayer);
figure('Name','RGGB马赛克图片','NumberTitle','off'),imshow(bayer_mosaic);
% imwrite(bayer_mosaic,'RGGB的Bayer格式马赛克图片.png')

%% MATLAB自带去马赛克函数（线性插值）
if Method == 1
    imDst = demosaic(bayer,'rggb');
    figure('Name','MATLAB自带去马赛克函数所得图片','NumberTitle','off'),imshow(imDst);
    % imwrite(imDst,'MATLAB自带去马赛克函数所得图片.png');
    toc;
end
%% 3x3窗口插值去马赛克
if Method == 2
    bayerPadding = zeros(hei+2,wid+2);
    bayerPadding(2:hei+1,2:wid+1) = bayer;
    bayerPadding(1,:) = bayerPadding(3,:);
    bayerPadding(hei+2,:) = bayerPadding(hei,:);
    bayerPadding(:,1) = bayerPadding(:,3);
    bayerPadding(:,wid+2) = bayerPadding(:,wid);
    imDst = zeros(hei+2, wid+2, len);

    for ver = 2:hei+1
        for hor = 2:wid+1
            if(1 == mod(ver-1,2))   %纵向R
                if(1 == mod(hor-1,2))   %横向R
                    imDst(ver,hor,1) = bayerPadding(ver,hor);
                    imDst(ver,hor,3) = (bayerPadding(ver-1,hor-1) + bayerPadding(ver-1,hor+1) + bayerPadding(ver+1,hor-1) + bayerPadding(ver+1,hor+1)) / 4;
                    imDst(ver,hor,2) = (bayerPadding(ver-1,hor) + bayerPadding(ver,hor-1) + bayerPadding(ver,hor+1) + bayerPadding(ver+1,hor)) / 4;
                else                    %横向G
                    imDst(ver,hor,2) = bayerPadding(ver,hor);
                    imDst(ver,hor,3) = (bayerPadding(ver-1,hor) + bayerPadding(ver+1,hor)) / 2;
                    imDst(ver,hor,1) = (bayerPadding(ver,hor-1) + bayerPadding(ver,hor+1)) / 2;
                end
            else                    %纵向G
                if(1 == mod(hor-1,2))   %横向G
                    imDst(ver,hor,2) = bayerPadding(ver,hor);
                    imDst(ver,hor,3) = (bayerPadding(ver,hor-1) + bayerPadding(ver,hor+1)) / 2;
                    imDst(ver,hor,1) = (bayerPadding(ver-1,hor) + bayerPadding(ver+1,hor)) / 2;
                else                    %横向B
                    imDst(ver,hor,3) = bayerPadding(ver,hor);
                    imDst(ver,hor,2) = (bayerPadding(ver-1,hor) + bayerPadding(ver,hor-1) + bayerPadding(ver,hor+1) + bayerPadding(ver+1,hor)) / 4;
                    imDst(ver,hor,1) = (bayerPadding(ver-1,hor-1) + bayerPadding(ver-1,hor+1) + bayerPadding(ver+1,hor-1) + bayerPadding(ver+1,hor+1)) / 4;
                end
            end
        end
    end

    imDst = uint8(imDst(2:hei+1,2:wid+1,:));
    figure('Name','3x3窗口插值去马赛克图片','NumberTitle','off'),imshow(imDst);
    % imwrite(imDst,'3x3窗口插值去马赛克图片.png');
    toc;
end
%% 边缘自适应插值去马赛克
% RGGB模式如下
% R G R G R G
% G B G B G B
% R G R G R G
% G B G B G B
% 四种模型如下
% B G B   R G R   G R G   G B G
% G R G   G B G   B G B   R G R
% B G B   R G R   G R G   G B G
%  (a)     (b)     (c)     (d)
% -----------
if Method == 3
    bayerPadding = zeros(hei+4,wid+4);
    bayerPadding(3:hei+2,3:wid+2) = bayer;
    bayerPadding(1,:) = bayerPadding(5,:);
    bayerPadding(2,:) = bayerPadding(4,:);
    bayerPadding(hei+4,:) = bayerPadding(hei,:);
    bayerPadding(hei+3,:) = bayerPadding(hei+1,:);
    bayerPadding(:,1) = bayerPadding(:,5);
    bayerPadding(:,2) = bayerPadding(:,4);
    bayerPadding(:,wid+4) = bayerPadding(:,wid);
    bayerPadding(:,wid+3) = bayerPadding(:,wid+1);
    imDst = zeros(hei+4, wid+4, len);

    for ver = 3:hei+2   %下标i，行号
        for hor = 3:wid+2   %下标j，列号
            if mod(ver-2,2)==1  %奇数行
                if mod(hor-2,2)==1  %奇数列，R位置判断
                    imDst(ver,hor,1) = bayerPadding(ver,hor);
                    if delta_H(ver,hor,bayerPadding)<delta_V(ver,hor,bayerPadding)
                        imDst(ver,hor,2) = GH(ver,hor,bayerPadding);
                    elseif delta_H(ver,hor,bayerPadding)>delta_V(ver,hor,bayerPadding)
                        imDst(ver,hor,2) = GV(ver,hor,bayerPadding);
                    else
                        imDst(ver,hor,2) = GM(ver,hor,bayerPadding);
                    end
                else                %偶数列，G位置判断
                    imDst(ver,hor,2) = bayerPadding(ver,hor);
                end
            else                %偶数行
                if mod(hor-2,2)==0  %偶数列，B位置判断
                    imDst(ver,hor,3) = bayerPadding(ver,hor);
                    if delta_H(ver,hor,bayerPadding)<delta_V(ver,hor,bayerPadding)
                        imDst(ver,hor,2) = GH(ver,hor,bayerPadding);
                    elseif delta_H(ver,hor,bayerPadding)>delta_V(ver,hor,bayerPadding)
                        imDst(ver,hor,2) = GV(ver,hor,bayerPadding);
                    else
                        imDst(ver,hor,2) = GM(ver,hor,bayerPadding);
                    end
                else                %奇数列，G位置判断
                    imDst(ver,hor,2) = bayerPadding(ver,hor);
                end
            end
        end
    end

    for ver = 3:hei+2   %下标i，行号
        for hor = 3:wid+2   %下标j，列号
            if mod(ver-2,2)==1  %奇数行
                if mod(hor-2,2)==1  %奇数列，R位置判断
                    if D_45(ver,hor,bayerPadding,imDst)<D_135(ver,hor,bayerPadding,imDst)
                        imDst(ver,hor,3) = RB45(ver,hor,bayerPadding,imDst);
                    elseif D_45(ver,hor,bayerPadding,imDst)>D_135(ver,hor,bayerPadding,imDst)
                        imDst(ver,hor,3) = RB135(ver,hor,bayerPadding,imDst);
                    else
                        imDst(ver,hor,3) = RBM(ver,hor,bayerPadding,imDst);
                    end
                else
                    imDst(ver,hor,3) = RBV(ver,hor,bayerPadding,imDst);
                    imDst(ver,hor,1) = RBH(ver,hor,bayerPadding,imDst);
                end
            else                %偶数行
                if mod(hor-2,2)==0  %偶数列，B位置判断
                    if D_45(ver,hor,bayerPadding,imDst)<D_135(ver,hor,bayerPadding,imDst)
                        imDst(ver,hor,1) = RB45(ver,hor,bayerPadding,imDst);
                    elseif D_45(ver,hor,bayerPadding,imDst)>D_135(ver,hor,bayerPadding,imDst)
                        imDst(ver,hor,1) = RB135(ver,hor,bayerPadding,imDst);
                    else
                        imDst(ver,hor,1) = RBM(ver,hor,bayerPadding,imDst);
                    end
                else
                    imDst(ver,hor,1) = RBV(ver,hor,bayerPadding,imDst);
                    imDst(ver,hor,3) = RBH(ver,hor,bayerPadding,imDst);
                end
            end
        end
    end

    imDst = uint8(imDst(3:hei+2,3:wid+2,:));
    figure('Name','边缘自适应插值去马赛克图片','NumberTitle','off'),imshow(imDst);
    % imwrite(imDst,'边缘自适应插值去马赛克图片.png');
    toc;
end
%% 边缘自适应插值内嵌函数
% 绿色分量重建
function HO=delta_H(ii,jj,bayer)  %水平检测算子
HO=abs(bayer(ii,jj-1)-bayer(ii,jj+1))+ ...
    abs(2*bayer(ii,jj)-bayer(ii,jj-2)-bayer(ii,jj+2));
end

function VO=delta_V(ii,jj,bayer)  %垂直检测算子
VO=abs(bayer(ii-1,jj)-bayer(ii+1,jj))+ ...
    abs(2*bayer(ii,jj)-bayer(ii-2,jj)-bayer(ii+2,jj));
end

function g=GH(ii,jj,bayer)        %绿色分量沿水平方向计算
g=(bayer(ii,jj-1)+bayer(ii,jj+1))/2+ ...
    (2*bayer(ii,jj)-bayer(ii,jj-2)-bayer(ii,jj+2))/4;
end

function g=GV(ii,jj,bayer)        %绿色分量沿垂直方向计算
g=(bayer(ii-1,jj)+bayer(ii+1,jj))/2+ ...
    (2*bayer(ii,jj)-bayer(ii-2,jj)-bayer(ii+2,jj))/4;
end

function g=GM(ii,jj,bayer)        %绿色分量为两方向计算的平均值
g=(GH(ii,jj,bayer)+GV(ii,jj,bayer))/2;
end

% 绿色采样点处的红色和蓝色分量重建
function rb=RBV(ii,jj,bayer,rgb)      %红/蓝色分量使用上下两点R/B-G空间的线性插值
rb=(bayer(ii-1,jj)+bayer(ii+1,jj))/2+ ...
    (2*bayer(ii,jj)-rgb(ii-1,jj,2)-rgb(ii+1,jj,2))/2;
end

function rb=RBH(ii,jj,bayer,rgb)    %红/蓝色分量使用左右两点R/B-G空间的线性插值
rb=(bayer(ii,jj-1)+bayer(ii,jj+1))/2+ ...
    (2*bayer(ii,jj)-rgb(ii,jj-1,2)-rgb(ii,jj+1,2))/2;
end

% 红色(蓝色)采样点处的蓝色(红色)分量的重建
function grad=D_45(ii,jj,bayer,rgb)   %左下右上梯度计算
grad=abs(bayer(ii-1,jj+1)-bayer(ii+1,jj-1))+ ...
    abs(2*rgb(ii,jj,2)-rgb(ii-1,jj+1,2)-rgb(ii+1,jj-1,2));
end

function grad=D_135(ii,jj,bayer,rgb)  %左上右下梯度计算
grad=abs(bayer(ii-1,jj-1)-bayer(ii+1,jj+1))+ ...
    abs(2*rgb(ii,jj,2)-rgb(ii-1,jj-1,2)-rgb(ii+1,jj+1,2));
end

function rb=RB45(ii,jj,bayer,rgb)     %D_45<D_135
rb=(bayer(ii-1,jj+1)+bayer(ii+1,jj-1))/2+ ...
    (2*rgb(ii,jj,2)-rgb(ii-1,jj+1,2)-rgb(ii+1,jj-1,2))/2;
end

function rb=RB135(ii,jj,bayer,rgb)    %D_45>D_135
rb=(bayer(ii-1,jj-1)+bayer(ii+1,jj+1))/2+ ...
    (2*rgb(ii,jj,2)-rgb(ii-1,jj-1,2)-rgb(ii+1,jj+1,2))/2;
end

function rb=RBM(ii,jj,bayer,rgb)      %D_45=D_135
rb=(RB45(ii,jj,bayer,rgb)+RB135(ii,jj,bayer,rgb))/2;
end