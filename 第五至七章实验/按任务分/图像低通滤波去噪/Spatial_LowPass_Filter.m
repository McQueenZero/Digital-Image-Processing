%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像空域低通滤波去噪
%%-------------------------------------------------------------------------
%% 图像加噪声
clc, clear, close all
filename = 'eight.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imNosSP=imnoise(imSrc,'salt & pepper');
imNosGAS=imnoise(imSrc,'gaussian');
figure('Name','原始图片与加噪声图片','NumberTitle','off')
subplot(1,3,1)
imshow(imSrc),title('原始图片')
subplot(1,3,2)
imshow(imNosSP),title('加椒盐噪声图片')
subplot(1,3,3)
imshow(imNosGAS),title('加高斯噪声图片')

%% 不同填充方法低通滤波 （冈萨雷斯数字图像处理MATLAB版，P.71）
fo=fspecial('average');   %指定滤波算子（filter operator）
imDstSP0 = imfilter(imNosSP,fo);
imDstSPre = imfilter(imNosSP,fo,'replicate');
imDstSPsy = imfilter(imNosSP,fo,'symmetric');
imDstSPci = imfilter(imNosSP,fo,'circular');
imDstGAS0 = imfilter(imNosGAS,fo);
imDstGASre = imfilter(imNosGAS,fo,'replicate');
imDstGASsy = imfilter(imNosGAS,fo,'symmetric');
imDstGASci = imfilter(imNosGAS,fo,'circular');
figure('Name','不同填充方法均值滤波处理后的图片','NumberTitle','off')
subplot(2,4,1)
imshow(imDstSP0),title('去椒盐噪声，零填充')
subplot(2,4,2)
imshow(imDstSPre),title('去椒盐噪声，边界复制填充')
subplot(2,4,3)
imshow(imDstSPsy),title('去椒盐噪声，边界镜像填充')
subplot(2,4,4)
imshow(imDstSPci),title('去椒盐噪声，二维周期函数扩展填充')
subplot(2,4,5)
imshow(imDstGAS0),title('去高斯噪声，零填充')
subplot(2,4,6)
imshow(imDstGASre),title('去高斯噪声，边界复制填充')
subplot(2,4,7)
imshow(imDstGASsy),title('去高斯噪声，边界镜像填充')
subplot(2,4,8)
imshow(imDstGASci),title('去高斯噪声，二维周期函数扩展填充')

%% 椒盐噪声去噪对比
fo=fspecial('average');   %指定滤波算子（filter operator）
imDstSPavrg = imfilter(imNosSP,fo,'symmetric');
imDstSPmed = medfilt2(imNosSP);
figure('Name','去除椒盐噪声结果图片','NumberTitle','off')
subplot(1,2,1)
imshow(imDstSPavrg),title('均值滤波')
subplot(1,2,2)
imshow(imDstSPmed),title('中值滤波')

%% 设计平滑空间滤波器（冈萨雷斯.MATLAB版数字图像处理,P.117）
imDstSPamean = spfilt(imNosSP,'amean');
imDstSPgmean = spfilt(imNosSP,'gmean');
imDstSPhmean = spfilt(imNosSP,'hmean');
imDstSPchmean = spfilt(imNosSP,'chmean');
imDstSPmedian = spfilt(imNosSP,'median');
imDstSPmax = spfilt(imNosSP,'max');
imDstSPmin = spfilt(imNosSP,'min');
imDstSPmidpoint = spfilt(imNosSP,'midpoint');
imDstSPatrimmed = spfilt(imNosSP,'atrimmed');

figure('Name','设计的空域滤波器去除椒盐噪声结果图片','NumberTitle','off')
subplot(3,3,1)
imshow(imDstSPamean),title('算术平均')
subplot(3,3,2)
imshow(imDstSPgmean),title('几何平均')
subplot(3,3,3)
imshow(imDstSPhmean),title('调和均值')
subplot(3,3,4)
imshow(imDstSPchmean),title('反调和均值')
subplot(3,3,5)
imshow(imDstSPmedian),title('中值')
subplot(3,3,6)
imshow(imDstSPmax),title('最大')
subplot(3,3,7)
imshow(imDstSPmin),title('最小')
subplot(3,3,8)
imshow(imDstSPmidpoint),title('中点')
subplot(3,3,9)
imshow(imDstSPatrimmed),title('顺序-平衡均值')

%% 平滑空间滤波器内嵌函数
function f = spfilt(g, type, m, n, parameter)
% SPFILT Performs linear and nonlinear spatial filtering
%   F = SPFILT(G, TYPE, M, N, PARAMETER) performss spatial filtering
%   of image G using a TYPE filter of size M-by-N. Valid calls to
%   SPFILT are as follows
%
%       F = SPFILT(G, 'amean', M, N)        Arithmetic mean filtering.
%       F = SPFILT(G, 'gmean', M, N)        Geometric mean filtering.
%       F = SPFILT(G, 'hmean', M, N)        Harmonic mean filtering.
%       F = SPFILT(G, 'chmean', M, N, Q)    Contraharmonic mean
%                                           filtering of order Q. The
%                                           default is Q = 1.5.
%       F = SPFILT(G, 'median', M, N)       Median filtering.
%       F = SPFILT(G, 'max', M, N)          Max filtering.
%       F = SPFILT(G, 'min', M, N)          Min filtering.
%       F = SPFILT(G, 'midpoint', M, N)     Midpoint filtering.
%       F = SPFILT(G, 'atrimmed', M, N, D)  Alpha-trimmed mean filtering.
%                                           Parameter D must be a nonnegative even
%                                           integer; its default value
%                                           is D = 2.
%
%   The default values when only G and TYPE are input are M=N=3,
%   Q = 1.5, and D = 2.
% Process inputs.
if nargin == 2
    m = 3; n = 3; Q = 1.5; d = 2;
elseif nargin == 5
    Q = parameter; d = parameter;
elseif nargin == 4
    Q = 1.5; d = 2;
else
    error('Wrong number of inputs.')
end
% Do the filtering.
switch type
    case 'amean'
        w = fspecial('average', [m n]);
        f = imfilter(g, w, 'replicate');
    case 'gmean'
        f = gmean(g, m, n);
    case 'hmean'
        f = harmean(g, m, n);
    case 'chmean'
        f = charmean(g, m, n, Q);
    case 'median'
        f = medfilt2(g, [m n], 'symmetric');
    case 'max'
        f = ordfilt2(g, m*n, ones(m,n), 'symmetric');
    case 'min'
        f = ordfilt2(g, 1, ones(m,n), 'symmetric');
    case 'midpoint'
        f1 = ordfilt2(g, 1, ones(m,n), 'symmetric');
        f2 = ordfilt2(g, m*n, ones(m,n), 'symmetric');
        f = imlincomb(0.5, f1, 0.5, f2);
    case 'atrimmed'
        if d < 0 || d/2~=round(d/2)
            error('d must be a nonnegative, even integer.')
        end
        f = alphatrim(g, m, n, d);
    otherwise
        error('Unknown filter type.')
end
end
%-------------------------------------------------------------------------%
function f = gmean(g, m, n)
%   Implements a geometric mean filter.
inclass = class(g);
g = im2double(g);
% Disable log(0) warning.
warning off;
f = exp(imfilter(log(g), ones(m,n), 'replicate')).^(1/m/n);
warning on;
f = changeclass(inclass, f);
end
%-------------------------------------------------------------------------%
function f = harmean(g, m, n)
%   Implements a harmonic mean filter.
inclass = class(g);
g = im2double(g);
f = m*n./imfilter(1./(g+eps), ones(m,n), 'replicate');
f = changeclass(inclass, f);
end
%-------------------------------------------------------------------------%
function f = charmean(g, m, n, q)
%   Implements a contraharmonic mean filter.
inclass = class(g);
g = im2double(g);
f = imfilter(g.^(q+1), ones(m,n), 'replicate');
f = f./(imfilter(g.^q, ones(m,n), 'replicate')+eps);
f = changeclass(inclass, f);
end
%-------------------------------------------------------------------------%
function f = alphatrim(g, m, n, d)
%   Implements an alpha-trimmed mean filter.
inclass = class(g);
g = im2double(g);
f = imfilter(g, ones(m,n), 'symmetric');
for k = 1:d/2
    f = imsubtract(f, ordfilt2(g, k, ones(m,n), 'symmetric'));
end
for k = (m*n-d/2+1):m*n
    f = imsubtract(f, ordfilt2(g, k, ones(m,n), 'symmetric'));
end
f = f/(m*n-d);
f = changeclass(inclass, f);
end
%-------------------------------------------------------------------------%
function B = imlincomb(c1, A1, c2, A2)
B = c1*A1 + c2*A2;
end
%-------------------------------------------------------------------------%
function image = changeclass(class, varargin)
% CHANGECLASS changes the storage class of an image.
%   I2 = CHANGECLASS(CLASS, I);
%   RGB2 = CHANGECLASS(CLASS, RGB);
%   BW2 = CHANGECLASS(CLASS, BW);
%   X2 = CHANGECLASS(CLASS, X, 'Indexed');
switch class
    case 'uint8'
        image = im2uint8(varargin{:});
    case 'uint16'
        image = im2uint16(varargin{:});
    case 'double'
        image = im2double(varargin{:});
    otherwise
        error('Unsupported IPT data class.');
end
end
