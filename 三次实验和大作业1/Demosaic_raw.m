%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   Raw Image Demosaic
%       用近邻插值去偏振马赛克，主要用3x3窗口插值方法进行去贝尔马赛克
%%-------------------------------------------------------------------------
%% 方法选择
clc,clear,close all;
% 选择彩色去马赛克方法
disp('选择彩色去马赛克方法：')
while 1
    disp("  MATLAB自带去马赛克函数，输入'1'")
    disp("  3x3窗口插值，输入'2'")
    disp("  边缘自适应插值，输入'3'")
    CDM = input('输入去马赛克方法：');
    switch CDM
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
% 选择偏振去马赛克方法
disp('选择偏振去马赛克方法：')
while 1
    disp("  近邻插值，输入'1'")
    disp("  3x3窗口插值，输入'2'")
    disp("  边缘自适应插值，输入'3'")
    PDM = input('输入去马赛克方法：');
    switch PDM
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
% 选择偏振去马赛克方法
disp('选择去噪方法：')
while 1
    disp("  空域高斯，输入'1'")
    disp("  空域中值，输入'2'")
    disp("  频域高斯，输入'3'")
    disp("  频域巴特沃斯，输入'4'")
    DNM = input('输入去马赛克方法：');
    switch DNM
        case 1
            break
        case 2
            break
        case 3
            break
        case 4
            break
        otherwise
            disp('非法，请重新输入')
    end
end

%% 读取原始马赛克（彩色RGGB+偏振）图像
tic;
fidimame='可见光人像马赛克';
imSrc = imread([fidimame,'.jpg']);
% figure('Name','原始图片','NumberTitle','off'),imshow(imSrc);
[hei, wid, dim] = size(imSrc);

% 图像最外圈处理
imSrcPadding = zeros(hei+8,wid+8);
imSrcPadding(5:hei+4,5:wid+4) = imSrc;
imSrcPadding(1,:) = imSrcPadding(9,:);
imSrcPadding(2,:) = imSrcPadding(10,:);
imSrcPadding(3,:) = imSrcPadding(7,:);
imSrcPadding(4,:) = imSrcPadding(8,:);
imSrcPadding(hei+8,:) = imSrcPadding(hei,:);
imSrcPadding(hei+7,:) = imSrcPadding(hei-1,:);
imSrcPadding(hei+6,:) = imSrcPadding(hei+2,:);
imSrcPadding(hei+5,:) = imSrcPadding(hei+1,:);
imSrcPadding(:,1) = imSrcPadding(:,9);
imSrcPadding(:,2) = imSrcPadding(:,10);
imSrcPadding(:,3) = imSrcPadding(:,7);
imSrcPadding(:,4) = imSrcPadding(:,8);
imSrcPadding(:,wid+8) = imSrcPadding(:,wid);
imSrcPadding(:,wid+7) = imSrcPadding(:,wid-1);
imSrcPadding(:,wid+6) = imSrcPadding(:,wid+2);
imSrcPadding(:,wid+5) = imSrcPadding(:,wid+1);

imSSamp90 = zeros((hei+8)/2, (wid+8)/2, dim);
imSSamp45 = zeros((hei+8)/2, (wid+8)/2, dim);
imSSamp135 = zeros((hei+8)/2, (wid+8)/2, dim);
imSSamp0 = zeros((hei+8)/2, (wid+8)/2, dim);
imCDmsk = zeros((hei+8)/2, (wid+8)/2, dim);

imSSR = zeros(hei+8, wid+8, 3);
imDstI0 = zeros(hei+8, wid+8, 3);
imDstI45 = zeros(hei+8, wid+8, 3);
imDstI90 = zeros(hei+8, wid+8, 3);
imDstI135 = zeros(hei+8, wid+8, 3);
toc;

%% 按偏振滤镜分四个角度亚采样
for ver = 5:hei+4   %下标i，行号
    for hor = 5:wid+4   %下标j，列号
        if mod(ver-4,2) == 1  %纵向奇数行
            if mod(hor-4,2) == 1    %90°
                imSSamp90(ceil(ver/2),ceil(hor/2)) = imSrcPadding(ver,hor);
            else                    %45°
                imSSamp45(ceil(ver/2),ceil(hor/2)) = imSrcPadding(ver,hor);
            end
        else                    %纵向偶数行
            if mod(hor-4,2) == 1    %135°
                imSSamp135(ceil(ver/2),ceil(hor/2)) = imSrcPadding(ver,hor);
            else                    %0°
                imSSamp0(ceil(ver/2),ceil(hor/2)) = imSrcPadding(ver,hor);
            end
        end
    end
end
toc;

%% MATLAB自带去彩色马赛克函数（线性插值）
if CDM == 1 
    imCDmsk0 = demosaic(uint32(imSSamp0),'rggb');
    imCDmsk45 = demosaic(uint32(imSSamp45),'rggb');
    imCDmsk90 = demosaic(uint32(imSSamp90),'rggb');
    imCDmsk135 = demosaic(uint32(imSSamp135),'rggb');
    toc;
end
% 查看仅彩色去马赛克效果------------------------------------
% imCDmsk0 = im2uint8(mat2gray(log(1+double(imCDmsk0))));
% imCDmsk45 = im2uint8(mat2gray(log(1+double(imCDmsk45))));
% imCDmsk90 = im2uint8(mat2gray(log(1+double(imCDmsk90))));
% imCDmsk135 = im2uint8(mat2gray(log(1+double(imCDmsk135))));
% ---------------------------------------------------------
% figure('Name','MATLAB自带去马赛克函数所得图片','NumberTitle','off')
% subplot(2,2,1)
% imshow(uint8(imCDmsk90))
% title('90°偏振滤镜')
% subplot(2,2,2)
% imshow(uint8(imCDmsk45))
% title('45°偏振滤镜')
% subplot(2,2,3)
% imshow(uint8(imCDmsk135))
% title('135°偏振滤镜')
% subplot(2,2,4)
% imshow(uint8(imCDmsk0))
% title('0°偏振滤镜')

%% 3x3窗口插值去彩色马赛克
if CDM == 2
    for PF = 1:4    %偏振滤镜4个通道
        switch PF
            case 1
                imSSamp = imSSamp0;
            case 2
                imSSamp = imSSamp45;
            case 3
                imSSamp = imSSamp90;
            case 4
                imSSamp = imSSamp135;
        end
        for ver = ceil(5/2):ceil((hei+4)/2)   %下标i，行号
            for hor = ceil(5/2):ceil((wid+4)/2)   %下标j，列号
                if mod(ver-4/2,2) == 1  %纵向奇数行
                    if mod(hor-4/2,2) == 1    %R
                        imCDmsk(ver,hor,1) = imSSamp(ver,hor);
                        imCDmsk(ver,hor,3) = (imSSamp(ver-1,hor-1) + imSSamp(ver-1,hor+1) + imSSamp(ver+1,hor-1) + imSSamp(ver+1,hor+1)) / 4;
                        imCDmsk(ver,hor,2) = (imSSamp(ver-1,hor) + imSSamp(ver,hor-1) + imSSamp(ver,hor+1) + imSSamp(ver+1,hor)) / 4;
                    else                    %G
                        imCDmsk(ver,hor,2) = imSSamp(ver,hor);
                        imCDmsk(ver,hor,3) = (imSSamp(ver-1,hor) + imSSamp(ver+1,hor)) / 2;
                        imCDmsk(ver,hor,1) = (imSSamp(ver,hor-1) + imSSamp(ver,hor+1)) / 2;
                    end
                else                    %纵向偶数行
                    if mod(hor-4/2,2) == 1    %G
                        imCDmsk(ver,hor,2) = imSSamp(ver,hor);
                        imCDmsk(ver,hor,3) = (imSSamp(ver,hor-1) + imSSamp(ver,hor+1)) / 2;
                        imCDmsk(ver,hor,1) = (imSSamp(ver-1,hor) + imSSamp(ver+1,hor)) / 2;
                    else                    %B
                        imCDmsk(ver,hor,3) = imSSamp(ver,hor);
                        imCDmsk(ver,hor,2) = (imSSamp(ver-1,hor) + imSSamp(ver,hor-1) + imSSamp(ver,hor+1) + imSSamp(ver+1,hor)) / 4;
                        imCDmsk(ver,hor,1) = (imSSamp(ver-1,hor-1) + imSSamp(ver-1,hor+1) + imSSamp(ver+1,hor-1) + imSSamp(ver+1,hor+1)) / 4;
                    end
                end
            end
        end
        switch PF
            case 1
                imCDmsk0 = imCDmsk;
            case 2
                imCDmsk45 = imCDmsk;
            case 3
                imCDmsk90 = imCDmsk;
            case 4
                imCDmsk135 = imCDmsk;
        end
    end
    toc;
end

%% 边缘自适应插值去彩色马赛克
if CDM == 3
    for PF = 1:4
        switch PF
            case 1
                imSSamp = imSSamp0;
            case 2
                imSSamp = imSSamp45;
            case 3
                imSSamp = imSSamp90;
            case 4
                imSSamp = imSSamp135;
        end
        for ver = ceil(5/2):ceil((hei+4)/2)   %下标i，行号
            for hor = ceil(5/2):ceil((wid+4)/2)   %下标j，列号
                if mod(ver-4/2,2)==1  %奇数行
                    if mod(hor-4/2,2)==1  %奇数列，R位置判断
                        imCDmsk(ver,hor,1) = imSSamp(ver,hor);
                        if delta_H(ver,hor,imSSamp)<delta_V(ver,hor,imSSamp)
                            imCDmsk(ver,hor,2) = GH(ver,hor,imSSamp);
                        elseif delta_H(ver,hor,imSSamp)>delta_V(ver,hor,imSSamp)
                            imCDmsk(ver,hor,2) = GV(ver,hor,imSSamp);
                        else
                            imCDmsk(ver,hor,2) = GM(ver,hor,imSSamp);
                        end
                    else                %偶数列，G位置判断
                        imCDmsk(ver,hor,2) = imSSamp(ver,hor);
                    end
                else                %偶数行
                    if mod(hor-4/2,2)==0  %偶数列，B位置判断
                        imCDmsk(ver,hor,3) = imSSamp(ver,hor);
                        if delta_H(ver,hor,imSSamp)<delta_V(ver,hor,imSSamp)
                            imCDmsk(ver,hor,2) = GH(ver,hor,imSSamp);
                        elseif delta_H(ver,hor,imSSamp)>delta_V(ver,hor,imSSamp)
                            imCDmsk(ver,hor,2) = GV(ver,hor,imSSamp);
                        else
                            imCDmsk(ver,hor,2) = GM(ver,hor,imSSamp);
                        end
                    else                %奇数列，G位置判断
                        imCDmsk(ver,hor,2) = imSSamp(ver,hor);
                    end
                end
            end
        end

        for ver = ceil(5/2):ceil((hei+4)/2)   %下标i，行号
            for hor = ceil(5/2):ceil((wid+4)/2)   %下标j，列号
                if mod(ver-4/2,2)==1  %奇数行
                    if mod(hor-4/2,2)==1  %奇数列，R位置判断
                        if D_45(ver,hor,imSSamp,imCDmsk)<D_135(ver,hor,imSSamp,imCDmsk)
                            imCDmsk(ver,hor,3) = RB45(ver,hor,imSSamp,imCDmsk);
                        elseif D_45(ver,hor,imSSamp,imCDmsk)>D_135(ver,hor,imSSamp,imCDmsk)
                            imCDmsk(ver,hor,3) = RB135(ver,hor,imSSamp,imCDmsk);
                        else
                            imCDmsk(ver,hor,3) = RBM(ver,hor,imSSamp,imCDmsk);
                        end
                    else
                        imCDmsk(ver,hor,3) = RBV(ver,hor,imSSamp,imCDmsk);
                        imCDmsk(ver,hor,1) = RBH(ver,hor,imSSamp,imCDmsk);
                    end
                else                %偶数行
                    if mod(hor-4/2,2)==0  %偶数列，B位置判断
                        if D_45(ver,hor,imSSamp,imCDmsk)<D_135(ver,hor,imSSamp,imCDmsk)
                            imCDmsk(ver,hor,1) = RB45(ver,hor,imSSamp,imCDmsk);
                        elseif D_45(ver,hor,imSSamp,imCDmsk)>D_135(ver,hor,imSSamp,imCDmsk)
                            imCDmsk(ver,hor,1) = RB135(ver,hor,imSSamp,imCDmsk);
                        else
                            imCDmsk(ver,hor,1) = RBM(ver,hor,imSSamp,imCDmsk);
                        end
                    else
                        imCDmsk(ver,hor,1) = RBV(ver,hor,imSSamp,imCDmsk);
                        imCDmsk(ver,hor,3) = RBH(ver,hor,imSSamp,imCDmsk);
                    end
                end
            end
        end
        switch PF
            case 1
                imCDmsk0 = imCDmsk;
            case 2
                imCDmsk45 = imCDmsk;
            case 3
                imCDmsk90 = imCDmsk;
            case 4
                imCDmsk135 = imCDmsk;
        end
    end
    toc;
end

%% 按偏振滤镜分四个角度逆亚采样
for ver = 5:hei+4   %下标i，行号
    for hor = 5:wid+4   %下标j，列号
        for ch = 1:3    %色彩通道号
            if mod(ver-4,2) == 1  %纵向奇数行
                if mod(hor-4,2) == 1    %90°
                    imSSR(ver,hor,ch) = imCDmsk90(ceil(ver/2),ceil(hor/2),ch);
                else                    %45°
                    imSSR(ver,hor,ch) = imCDmsk45(ceil(ver/2),ceil(hor/2),ch);
                end
            else                    %纵向偶数行
                if mod(hor-4,2) == 1    %135°
                    imSSR(ver,hor,ch) = imCDmsk135(ceil(ver/2),ceil(hor/2),ch);
                else                    %0°
                    imSSR(ver,hor,ch) = imCDmsk0(ceil(ver/2),ceil(hor/2),ch);
                end
            end
        end
    end
end

% 查看仅彩色去马赛克效果------------------------------------
% imSSR = im2uint8(mat2gray(log(1+double(imSSR))));
% ---------------------------------------------------------
% figure('Name','逆亚采样所得图片','NumberTitle','off')
% imshow(uint8(imSSR))
toc;

%% 近邻插值去偏振马赛克
if PDM == 1
    for ch = 1:3    %色彩通道号
        for ver = 5:hei+4   %下标i，行号
            for hor = 5:wid+4   %下标j，列号
                if mod(ver-4,2) == 1  %纵向奇数行
                    if mod(hor-4,2) == 1    %90°
                        imDstI90(ver,hor,ch) = imSSR(ver,hor,ch);
                        imDstI0(ver,hor,ch) = imSSR(ver+1,hor+1,ch);
                        imDstI45(ver,hor,ch) = imSSR(ver,hor+1,ch);
                        imDstI135(ver,hor,ch) = imSSR(ver+1,hor,ch);
                    else                    %45°
                        imDstI45(ver,hor,ch) = imSSR(ver,hor,ch);
                        imDstI0(ver,hor,ch) = imSSR(ver+1,hor,ch);
                        imDstI90(ver,hor,ch) = imSSR(ver,hor-1,ch);
                        imDstI135(ver,hor,ch) = imSSR(ver+1,hor-1,ch);
                    end
                else                    %纵向偶数行
                    if mod(hor-4,2) == 1    %135°
                        imDstI135(ver,hor,ch) = imSSR(ver,hor,ch);
                        imDstI0(ver,hor,ch) = imSSR(ver,hor+1,ch);
                        imDstI45(ver,hor,ch) = imSSR(ver-1,hor+1,ch);
                        imDstI90(ver,hor,ch) = imSSR(ver-1,hor,ch);
                    else                    %0°
                        imDstI0(ver,hor,ch) = imSSR(ver,hor,ch);
                        imDstI45(ver,hor,ch) = imSSR(ver-1,hor,ch);
                        imDstI90(ver,hor,ch) = imSSR(ver-1,hor-1,ch);
                        imDstI135(ver,hor,ch) = imSSR(ver,hor-1,ch);
                    end
                end
            end
        end
    end
    toc;
end

%% 3x3窗口插值去偏振马赛克
if PDM == 2
    for ch = 1:3    %色彩通道号
        for ver = 5:hei+4   %下标i，行号
            for hor = 5:wid+4   %下标j，列号
                if mod(ver-4,2) == 1  %纵向奇数行
                    if mod(hor-4,2) == 1    %90°
                        imDstI90(ver,hor,ch) = imSSR(ver,hor,ch);
                        imDstI0(ver,hor,ch) = 1/4*(imSSR(ver+1,hor+1,ch) + imSSR(ver-1,hor+1,ch) + imSSR(ver+1,hor-1,ch) + imSSR(ver-1,hor-1,ch));
                        imDstI45(ver,hor,ch) = 1/2*(imSSR(ver,hor+1,ch) + imSSR(ver,hor-1,ch));
                        imDstI135(ver,hor,ch) = 1/2*(imSSR(ver+1,hor,ch) + imSSR(ver-1,hor,ch));
                    else                    %45°
                        imDstI45(ver,hor,ch) = imSSR(ver,hor,ch);
                        imDstI0(ver,hor,ch) = 1/2*(imSSR(ver+1,hor,ch) + imSSR(ver-1,hor,ch));
                        imDstI90(ver,hor,ch) = 1/2*(imSSR(ver,hor+1,ch) + imSSR(ver,hor-1,ch));
                        imDstI135(ver,hor,ch) = 1/4*(imSSR(ver+1,hor+1,ch) + imSSR(ver-1,hor+1,ch) + imSSR(ver+1,hor-1,ch) + imSSR(ver-1,hor-1,ch));
                    end
                else                    %纵向偶数行
                    if mod(hor-4,2) == 1    %135°
                        imDstI135(ver,hor,ch) = imSSR(ver,hor,ch);
                        imDstI0(ver,hor,ch) = 1/2*(imSSR(ver,hor+1,ch) + imSSR(ver,hor-1,ch));
                        imDstI45(ver,hor,ch) = 1/4*(imSSR(ver+1,hor+1,ch)  + imSSR(ver-1,hor+1,ch) + imSSR(ver+1,hor-1,ch)  + imSSR(ver-1,hor-1,ch));
                        imDstI90(ver,hor,ch) = 1/2*(imSSR(ver+1,hor,ch) + imSSR(ver-1,hor,ch));
                    else                    %0°
                        imDstI0(ver,hor,ch) = imSSR(ver,hor,ch);
                        imDstI45(ver,hor,ch) = 1/2*(imSSR(ver+1,hor,ch) + imSSR(ver-1,hor,ch));
                        imDstI90(ver,hor,ch) = 1/4*(imSSR(ver+1,hor+1,ch) + imSSR(ver-1,hor+1,ch) + imSSR(ver+1,hor-1,ch) + imSSR(ver-1,hor-1,ch));
                        imDstI135(ver,hor,ch) = 1/2*(imSSR(ver,hor+1,ch) + imSSR(ver,hor-1,ch));
                    end
                end
            end
        end
    end
    toc;
end

%% 边缘自适应插值去偏振马赛克
if PDM == 3
    for ch = 1:3    %色彩通道号
        switch ch
            case 1
                imSSRch = imSSR(:,:,1);
            case 2 
                imSSRch = imSSR(:,:,2);
            case 3
                imSSRch = imSSR(:,:,3);
        end
        for ver = 5:hei+4   %下标i，行号
            for hor = 5:wid+4   %下标j，列号
                if mod(ver-4,2) == 1  %纵向奇数行
                    if mod(hor-4,2) == 1    %90°)
                        imDstI90(ver,hor,ch) = imSSRch(ver,hor);
                        if Pdelta_H(ver,hor,imSSRch)<Pdelta_V(ver,hor,imSSRch)
                            imDstI45(ver,hor,ch) = PH(ver,hor,imSSRch);
                            imDstI135(ver,hor,ch) = PHX(ver,hor,imSSRch);
                        elseif Pdelta_H(ver,hor,imSSRch)>Pdelta_V(ver,hor,imSSRch)
                            imDstI135(ver,hor,ch) = PV(ver,hor,imSSRch);
                            imDstI45(ver,hor,ch) = PVX(ver,hor,imSSRch);
                        else
                            imDstI45(ver,hor,ch) = PH(ver,hor,imSSRch);
                            imDstI135(ver,hor,ch) = PV(ver,hor,imSSRch);
                        end
                        if Pdelta_DLUR(ver,hor,imSSRch)<Pdelta_ULDR(ver,hor,imSSRch)
                            imDstI0(ver,hor,ch) = PDLUR(ver,hor,imSSRch);
                        elseif Pdelta_DLUR(ver,hor,imSSRch)>Pdelta_ULDR(ver,hor,imSSRch)
                            imDstI0(ver,hor,ch) = PULDR(ver,hor,imSSRch);
                        else
                            imDstI0(ver,hor,ch) = PXM(ver,hor,imSSRch);
                        end
                    else                    %45°
                        imDstI45(ver,hor,ch) = imSSRch(ver,hor);
                        if Pdelta_H(ver,hor,imSSRch)<Pdelta_V(ver,hor,imSSRch)
                            imDstI90(ver,hor,ch) = PH(ver,hor,imSSRch);
                            imDstI0(ver,hor,ch) = PHX(ver,hor,imSSRch);
                        elseif Pdelta_H(ver,hor,imSSRch)>Pdelta_V(ver,hor,imSSRch)
                            imDstI0(ver,hor,ch) = PV(ver,hor,imSSRch);
                            imDstI90(ver,hor,ch) = PVX(ver,hor,imSSRch);
                        else
                            imDstI90(ver,hor,ch) = PH(ver,hor,imSSRch);
                            imDstI0(ver,hor,ch) = PV(ver,hor,imSSRch);
                        end
                        if Pdelta_DLUR(ver,hor,imSSRch)<Pdelta_ULDR(ver,hor,imSSRch)
                            imDstI135(ver,hor,ch) = PDLUR(ver,hor,imSSRch);
                        elseif Pdelta_DLUR(ver,hor,imSSRch)>Pdelta_ULDR(ver,hor,imSSRch)
                            imDstI135(ver,hor,ch) = PULDR(ver,hor,imSSRch);
                        else
                            imDstI135(ver,hor,ch) = PXM(ver,hor,imSSRch);
                        end
                    end
                else                    %纵向偶数行
                    if mod(hor-4,2) == 1    %135°
                        imDstI135(ver,hor,ch) = imSSRch(ver,hor);
                        if Pdelta_H(ver,hor,imSSRch)<Pdelta_V(ver,hor,imSSRch)
                            imDstI0(ver,hor,ch) = PH(ver,hor,imSSRch);
                            imDstI90(ver,hor,ch) = PHX(ver,hor,imSSRch);
                        elseif Pdelta_H(ver,hor,imSSRch)>Pdelta_V(ver,hor,imSSRch)
                            imDstI90(ver,hor,ch) = PV(ver,hor,imSSRch);
                            imDstI0(ver,hor,ch) = PVX(ver,hor,imSSRch);
                        else
                            imDstI0(ver,hor,ch) = PH(ver,hor,imSSRch);
                            imDstI90(ver,hor,ch) = PV(ver,hor,imSSRch);
                        end
                        if Pdelta_DLUR(ver,hor,imSSRch)<Pdelta_ULDR(ver,hor,imSSRch)
                            imDstI45(ver,hor,ch) = PDLUR(ver,hor,imSSRch);
                        elseif Pdelta_DLUR(ver,hor,imSSRch)>Pdelta_ULDR(ver,hor,imSSRch)
                            imDstI45(ver,hor,ch) = PULDR(ver,hor,imSSRch);
                        else
                            imDstI45(ver,hor,ch) = PXM(ver,hor,imSSRch);
                        end
                    else                    %0°
                        imDstI0(ver,hor,ch) = imSSRch(ver,hor);
                        if Pdelta_H(ver,hor,imSSRch)<Pdelta_V(ver,hor,imSSRch)
                            imDstI135(ver,hor,ch) = PH(ver,hor,imSSRch);
                            imDstI45(ver,hor,ch) = PHX(ver,hor,imSSRch);
                        elseif Pdelta_H(ver,hor,imSSRch)>Pdelta_V(ver,hor,imSSRch)
                            imDstI45(ver,hor,ch) = PV(ver,hor,imSSRch);
                            imDstI135(ver,hor,ch) = PVX(ver,hor,imSSRch);
                        else
                            imDstI135(ver,hor,ch) = PH(ver,hor,imSSRch);
                            imDstI45(ver,hor,ch) = PV(ver,hor,imSSRch);
                        end
                        if Pdelta_DLUR(ver,hor,imSSRch)<Pdelta_ULDR(ver,hor,imSSRch)
                            imDstI90(ver,hor,ch) = PDLUR(ver,hor,imSSRch);
                        elseif Pdelta_DLUR(ver,hor,imSSRch)>Pdelta_ULDR(ver,hor,imSSRch)
                            imDstI90(ver,hor,ch) = PULDR(ver,hor,imSSRch);
                        else
                            imDstI90(ver,hor,ch) = PXM(ver,hor,imSSRch);
                        end
                    end
                end
            end
        end
    end
    toc;
end

%% 计算Stokes参量、偏振度DOP和偏振角AOP并输出  
S0 = (imDstI0+imDstI90+imDstI135+imDstI45)/2;
S1 = imDstI0-imDstI90;
S2 = imDstI45-imDstI135; 

% 计算I
imDstI = S0;
imDstI_Light = S0*4;
% 计算DOP、AOP
imDstDOP = sqrt(S1.^2+S2.^2)./S0;
imDstAOP = 0.5.*atan(S2./S1);
% 灰度级映射
imDstI = uint8(imDstI);
imDstI_Light = uint8(imDstI_Light);
imDstDOP_histeq = histeq(im2uint8(imDstDOP),256); %DOP作对比度变换
imDstAOP_rescale = uint8(rescale(imDstAOP,0,255)); %AOP作灰阶映射

% 去除插值补充的多余边缘
imDstI = imDstI(5:hei+4,5:wid+4,:);
imDstDOP = imDstDOP(5:hei+4,5:wid+4,:);
imDstAOP = imDstAOP(5:hei+4,5:wid+4,:);
imDstI_Light = imDstI_Light(5:hei+4,5:wid+4,:);
imDstDOP_histeq = imDstDOP_histeq(5:hei+4,5:wid+4,:);
imDstAOP_rescale = imDstAOP_rescale(5:hei+4,5:wid+4,:);

% 输出
% figure('Name','去偏振马赛克光强图片','NumberTitle','off')
% imshow(imDstI)
% figure('Name','去偏振马赛克偏振度图片','NumberTitle','off')
% imshow(imDstDOP)
% figure('Name','去偏振马赛克偏振角图片','NumberTitle','off')
% imshow(imDstAOP)

figure('Name','去偏振马赛克光强增强图片','NumberTitle','off')
imshow(imDstI_Light)
figure('Name','去偏振马赛克DOP对比度增强图片','NumberTitle','off')
imshow(imDstDOP_histeq)
figure('Name','去偏振马赛克AOP颜色空间映射图片','NumberTitle','off')
imshow(rgb2hsv(imDstAOP_rescale))
toc;
disp('去马赛克处理完毕')

% 保存
% Demosaic_method = '边缘自适应插值+边缘自适应插值';
% imwrite(imDstI,[Demosaic_method,'光强.png']);
% imwrite(imDstDOP,[Demosaic_method,'DOP.png']);
% imwrite(imDstAOP,[Demosaic_method,'AOP.png']);

% imwrite(imDstI_Light,'raw去马赛克含噪声光强.png');
% imwrite(imDstDOP_histeq,'raw去马赛克含噪声DOP.png');
% imwrite(rgb2hsv(imDstAOP_rescale),'raw去马赛克含噪声AOP.png');

% imwrite(imDstI_Light,'raw去马赛克含噪声光强.bmp');
% imwrite(imDstDOP_histeq,'raw去马赛克含噪声DOP.bmp');
% imwrite(imDstAOP_rescale,'raw去马赛克含噪声AOP.bmp');
% imwrite(imDstI_Light,'raw去马赛克含噪声光强.jpg');
% imwrite(imDstDOP_histeq,'raw去马赛克含噪声DOP.jpg');
% imwrite(imDstAOP_rescale,'raw去马赛克含噪声AOP.jpg');
% toc;
% disp('保存完毕')

%% 空域高斯滤波
if DNM == 1
    imDstDOPdenoise = zeros(hei,wid,dim);
    imDstAOPdenoise = zeros(hei,wid,dim);
    for k=1:3
        imDstDOPdenoise(:,:,k)=imgaussfilt(imDstDOP_histeq(:,:,k),1);
        imDstAOPdenoise(:,:,k)=imgaussfilt(imDstAOP_rescale(:,:,k),1);
    end
    figure('Name','空域高斯滤波DOP图像去噪图片','NumberTitle','off')
    imshow(uint8(imDstDOPdenoise))
    figure('Name','空域高斯滤波AOP图像去噪图片','NumberTitle','off')
    imshow(rgb2hsv(uint8(imDstAOPdenoise)))
    % imwrite(uint8(imDstDOPdenoise),'空域高斯去噪DOP.png');
    % imwrite(rgb2hsv(uint8(imDstAOPdenoise)),'空域高斯去噪AOP.png');
    toc;
end

%% 空域中值滤波
if DNM == 2
    imDstDOPdenoise = zeros(hei,wid,dim);
    imDstAOPdenoise = zeros(hei,wid,dim);
    for k=1:3
        imDstDOPdenoise(:,:,k)=medfilt2(imDstDOP_histeq(:,:,k),[5 5]);
        imDstAOPdenoise(:,:,k)=medfilt2(imDstAOP_rescale(:,:,k),[5 5]);
    end
    figure('Name','空域中值滤波DOP图像去噪图片','NumberTitle','off')
    imshow(uint8(imDstDOPdenoise))
    figure('Name','空域中值滤波AOP图像去噪图片','NumberTitle','off')
    imshow(rgb2hsv(uint8(imDstAOPdenoise)))
    % imwrite(uint8(imDstDOPdenoise),'空域中值去噪DOP.png');
    % imwrite(rgb2hsv(uint8(imDstAOPdenoise)),'空域中值去噪AOP.png');
    toc;
end

%% 频域高斯低通滤波
if DNM == 3
    for PS = 1:2
        for k=1:3
            switch PS
                case 1
                    f = imDstDOP_histeq(:,:,k);
                case 2
                    f = imDstAOP_rescale(:,:,k);
            end
            PQ = paddedsize(size(f));
            H = lpfilter('gaussian', PQ(1), PQ(2), 0.1*PQ(2));
            F = fft2(f, PQ(1), PQ(2));
            g(:,:,k) = dftfilt(f, H);
            
            %     switch k
            %         case 1
            %             figure('Name','噪声图像的频谱','NumberTitle','off')
            %             subplot(1,3,1)
            %             imshow(log(1 + abs(fftshift(F))), [ ])
            %             title('R通道')
            %         case 2
            %             subplot(1,3,2)
            %             imshow(log(1 + abs(fftshift(F))), [ ])
            %             title('G通道')
            %         case 3
            %             subplot(1,3,3)
            %             imshow(log(1 + abs(fftshift(F))), [ ])
            %             title('B通道')
            %     end
        end
        switch PS
            case 1
                imDstDOPdenoise = uint8(g);
            case 2
                imDstAOPdenoise = uint8(g);
        end
    end
    figure('Name','频域高斯低通滤波DOP图像去噪图片','NumberTitle','off')
    imshow(imDstDOPdenoise)
    figure('Name','频域高斯低通滤波AOP图像去噪图片','NumberTitle','off')
    imshow(rgb2hsv(imDstAOPdenoise))
    % imwrite(imDstDOPdenoise,'频域高斯去噪DOP.png');
    % imwrite(rgb2hsv(imDstAOPdenoise),'频域高斯去噪AOP.png');
    toc;
end

%% 频域巴特沃斯低通滤波
if DNM == 4
    for PS = 1:2
        for k=1:3
            switch PS
                case 1
                    f = imDstDOP_histeq(:,:,k);
                case 2
                    f = imDstAOP_rescale(:,:,k);
            end
            PQ = paddedsize(size(f));
            H = lpfilter('btw', PQ(1), PQ(2), 0.1*PQ(2), 1);
            F = fft2(f, PQ(1), PQ(2));
            g(:,:,k) = dftfilt(f, H);

            %     switch k
            %         case 1
            %             figure('Name','噪声图像的频谱','NumberTitle','off')
            %             subplot(1,3,1)
            %             imshow(log(1 + abs(fftshift(F))), [ ])
            %             title('R通道')
            %         case 2
            %             subplot(1,3,2)
            %             imshow(log(1 + abs(fftshift(F))), [ ])
            %             title('G通道')
            %         case 3
            %             subplot(1,3,3)
            %             imshow(log(1 + abs(fftshift(F))), [ ])
            %             title('B通道')
            %     end
        end
        switch PS
            case 1
                imDstDOPdenoise = uint8(g);
            case 2
                imDstAOPdenoise = uint8(g);
        end
    end
    figure('Name','频域巴特沃斯低通滤波DOP图像去噪图片','NumberTitle','off')
    imshow(imDstDOPdenoise)
    figure('Name','频域巴特沃斯低通滤波AOP图像去噪图片','NumberTitle','off')
    imshow(rgb2hsv(imDstAOPdenoise))
    % imwrite(imDstDOPdenoise,'频域巴特沃斯去噪DOP.png');
    % imwrite(rgb2hsv(imDstAOPdenoise),'频域巴特沃斯去噪AOP.png');
    toc;
end

%% 彩色边缘自适应插值内嵌函数
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

%% 偏振边缘自适应插值内嵌函数
% 水平竖直方向分量重建
function HO=Pdelta_H(ii,jj,Pbayer)      %水平检测算子
HO=abs(Pbayer(ii,jj-1)-Pbayer(ii,jj+1))+ ...
    abs(2*Pbayer(ii,jj)-Pbayer(ii,jj-2)-Pbayer(ii,jj+2));
end

function VO=Pdelta_V(ii,jj,Pbayer)      %垂直检测算子
VO=abs(Pbayer(ii-1,jj)-Pbayer(ii+1,jj))+ ...
    abs(2*Pbayer(ii,jj)-Pbayer(ii-2,jj)-Pbayer(ii+2,jj));
end
    
function c=PH(ii,jj,Pbayer)             %中心分量沿水平方向计算（8邻域）
c=(Pbayer(ii,jj-1)+Pbayer(ii,jj+1))/2;
end

function c=PHX(ii,jj,Pbayer)            %中心分量沿水平方向计算（24邻域）
c=((PH(ii-1,jj-1,Pbayer)+PH(ii+1,jj-1,Pbayer))/2 ...
    +(PH(ii-1,jj+1,Pbayer)+PH(ii+1,jj+1,Pbayer))/2)/2;
end

function c=PV(ii,jj,Pbayer)             %中心分量沿垂直方向计算（8邻域）
c=(Pbayer(ii-1,jj)+Pbayer(ii+1,jj))/2;
end

function c=PVX(ii,jj,Pbayer)             %中心分量沿垂直方向计算（24邻域）
c=((PV(ii-1,jj-1,Pbayer)+PV(ii-1,jj+1,Pbayer))/2 ...
    +(PV(ii+1,jj-1,Pbayer)+PV(ii+1,jj+1,Pbayer))/2)/2;
end

% 45°（左下右上DLUR）和135°（左上右下ULDR）方向分量重建
function DLURO=Pdelta_DLUR(ii,jj,Pbayer) %45°检测算子
DLURO=abs(Pbayer(ii-1,jj+1)-Pbayer(ii+1,jj-1))+ ...
    abs(2*Pbayer(ii,jj)-Pbayer(ii-2,jj+2)-Pbayer(ii+2,jj-2));
end

function ULDRO=Pdelta_ULDR(ii,jj,Pbayer) %135°检测算子
ULDRO=abs(Pbayer(ii-1,jj-1)-Pbayer(ii+1,jj+1))+ ...
    abs(2*Pbayer(ii,jj)-Pbayer(ii-2,jj-2)-Pbayer(ii+2,jj+2));
end

function c=PDLUR(ii,jj,Pbayer)           %中心分量沿45°方向计算
c=(Pbayer(ii-1,jj+1)+Pbayer(ii+1,jj-1))/2;
end

function c=PULDR(ii,jj,Pbayer)           %中心分量沿135°方向计算
c=(Pbayer(ii-1,jj-1)+Pbayer(ii+1,jj+1))/2;
end

function c=PXM(ii,jj,Pbayer)             %中心分量为两方向计算的平均值
c=(PDLUR(ii,jj,Pbayer)+PULDR(ii,jj,Pbayer))/2;
end

%% 频域滤波函数
% DFTFILT Performs frequency domain filtering
function g = dftfilt(f,H)
%   G = DFTFILT(F, H) filters F in the frequency domain using the
%   filter transfer function H. The output, G, is the filtered 
%   image, which has the same size as F. DFTFILT automatically pads
%   F to be the same size as H. Function PADDEDSIZE can be used
%   to determine an appropriate size for H.
% 
%   DFTFILT assumes that F is real and that H is a real, uncentered,
%   circularly-symmetric filter function.

% Obtain the FFT of the padded input.
F = fft2(f, size(H,1), size(H,2));
% Perform filtering.
g = real(ifft2(H.*F));
% Crop to original size
g = g(1:size(f,1), 1:size(f,2));

end

% PADDEDSIZE Computes padded sizes useful for FFT-based filtering.
function PQ = paddedsize(AB, CD, PARAM)
%   PQ = PADDEDSIZE(AB), where AB is a two-element size vector,
%   computes the two-element size vector PQ = 2*AB.
%   
%   PQ = PADDEDSIZE(AB, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX(AB).
% 
%   PQ = PADDEDSIZE(AB, CD), where AB and CD are two-element size
%   vectors, computes the two-element size vector PQ. The elements
%   of PQ are the smallest even integers greater than or equal to 
%   AB + CD - 1.
% 
%   PQ = PADDEDSIZE(AB, CD, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX([AB CD]).

if nargin == 1
    PQ = 2*AB;
elseif nargin == 2 & ~ischar(CD)
    PQ = AB + CD -1;
    PQ = 2 * ceil(PQ / 2);
elseif nargin == 2
    m = max(AB); % Maximum dimension,
    
    % Find power-of-2 at least twice m.
    P = 2^nextpow2(2*m);
    PQ = [P, P];
elseif nargin == 3
    m = max([AB CD]); % Maximum dimension.
    P = 2^nextpow2(2*m);
    PQ = [P, P];
else
    error('Wrong number of inputs.')
end

end

% DFTUV Computes meshgrid frequency matrices.
function [U, V] = dftuv(M, N)
%   [U, V] = DFTUV(M, N) computes meshgrid frequency matrices U and
%   V.  U and V are useful for computing frequency-domain filter
%   functions that can be used with DFTFILT.  U and V are both
%   M-by-N.

% Set up range of variables.
u = 0:(M - 1);
v = 0:(N - 1);

% Compute the indices for use in meshgrid.
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;

% Compute the meshgrid arrays.
[V, U] = meshgrid(v, u);

end

% LPFILTER Computes frequency domain lowpass filters
function [H, D] = lpfilter(type, M, N, D0, n)
%   H = LPFILTER(TYPE, M, N, D0, n) creates the transfer function of 
%   a lowpass filter, H, of the specified TYPE and size (M-by-N). To
%   view the filter as an image or mesh plot, it should be centered 
%   using H = fftshift(H).
% 
%   Valid values for TYPE, D0, and n are:
% 
%   'ideal'     Ideal lowpass filter with cutoff frequency D0. n need
%               not be supplied. D0 must be positive.
% 
%   'btw'       Butterworth lowpass filter of order n, and cutoff
%               D0. The default value for n is 1.0. D0 must be
%               positive.
% 
%   'gaussian'  Gaussian lowpass filter with cutoff (standard
%               deviation) D0. n need not be supplied. D0 must be
%               positive.

% Use function dftuv to set up the meshgrid arrays needed for
% computing the required distances.
[U, V] = dftuv(M, N);

% Compute the distance D(U, V).
D = sqrt(U.^2 + V.^2);

% Begin filter computations.
switch type
    case 'ideal'
        H = double(D <=D0);
    case 'btw'
        if nargin == 4
            n = 1;
        end
        H = 1./(1 + (D./D0).^(2*n));
    case 'gaussian'
        H = exp(-(D.^2)./(2*(D0^2)));
    otherwise
        error('Unknown filter type.')
end

end