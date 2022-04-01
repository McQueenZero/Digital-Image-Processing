%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像复原和重建
%       图像滤波、纹理特征提取等
% 注意：   分小节运行
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

%% 不同填充方法低通滤波
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

%% 提取纹理特征（冈萨雷斯.MATLAB版数字图像处理,P.350）
clc, clear, close all
filename = 'lena.png';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
imGray = rgb2gray(imSrc);
imGray1 = imGray(1:ceil(hei/2),1:ceil(wid/2));
imGray2 = imGray(1:ceil(hei/2),ceil(wid/2):wid);
imGray3 = imGray(ceil(hei/2):hei,1:ceil(wid/2));
imGray4 = imGray(ceil(hei/2):hei,ceil(wid/2):wid);
figure('Name','局部图','NumberTitle','off')
subplot(2,2,1)
imshow(imGray1),title('第1幅图')
subplot(2,2,2)
imshow(imGray2),title('第2幅图')
subplot(2,2,3)
imshow(imGray3),title('第3幅图')
subplot(2,2,4)
imshow(imGray4),title('第4幅图')

% 基于亮度直方图统计属性（灰度共生矩阵）的纹理描绘子
figure('Name','局部图的直方图','NumberTitle','off')
subplot(2,2,1)
imhist(imGray1),title('第1幅图的直方图')
subplot(2,2,2)
imhist(imGray2),title('第2幅图的直方图')
subplot(2,2,3)
imhist(imGray3),title('第3幅图的直方图')
subplot(2,2,4)
imhist(imGray4),title('第4幅图的直方图')
DuLiang(1,:) = statxture(imGray1);
DuLiang(2,:) = statxture(imGray2);
DuLiang(3,:) = statxture(imGray3);
DuLiang(4,:) = statxture(imGray4);
disp('基于亮度直方图统计属性（灰度共生矩阵）的纹理描绘子如下：')
disp('  平均亮度   平均对比度   平滑度    三阶矩    一致性     熵')
disp(DuLiang)

% 基于傅里叶频谱的纹理描绘子
[srad1, sang1, S1] = specxture(imGray1);
[srad2, sang2, S2] = specxture(imGray2);
[srad3, sang3, S3] = specxture(imGray3);
[srad4, sang4, S4] = specxture(imGray4);
figure('Name','频谱','NumberTitle','off')
subplot(2,2,1)
imshow(S1),title('第1幅图的频谱')
subplot(2,2,2)
imshow(S2),title('第2幅图的频谱')
subplot(2,2,3)
imshow(S3),title('第3幅图的频谱')
subplot(2,2,4)
imshow(S4),title('第4幅图的频谱')
figure('Name','S(r)和S(theta)曲线','NumberTitle','off')
subplot(2,4,1)
plot(srad1),title('第1幅图的S(r)曲线')
axis tight
subplot(2,4,2)
plot(sang1),title('第1幅图的S(\theta)曲线')
axis tight
subplot(2,4,3)
plot(srad2),title('第2幅图的S(r)曲线')
axis tight
subplot(2,4,4)
plot(sang2),title('第2幅图的S(\theta)曲线')
axis tight
subplot(2,4,5)
plot(srad3),title('第3幅图的S(r)曲线')
axis tight
subplot(2,4,6)
plot(sang3),title('第3幅图的S(\theta)曲线')
axis tight
subplot(2,4,7)
plot(srad4),title('第4幅图的S(r)曲线')
axis tight
subplot(2,4,8)
plot(sang4),title('第4幅图的S(\theta)曲线')
axis tight

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

%% 纹理特征提取内嵌函数
function [t] = statxture(f, scale)
% STATXTURE Computes statistical measures of texture in an image.
%   T = STATXTURE(F, SCALE) computes six measures of texture from an
%   image (region) F. Parameter SCALE is a 6-dim row vector whose
%   elements multiply the 6 corresponding elements of T for scaling
%   purposes. If SCALE is not provided it defaults to all 1s. The
%   output T is 6-by-1 vector with the following elements:
%       T(1) = Average gray level
%       T(2) = Average contrast
%       T(3) = Measure of smoothness
%       T(4) = Third moment
%       T(5) = Measure of uniformity
%       T(6) = Entropy
if nargin == 1
    scale(1:6) = 1;
else % Make sure it's a row vector
    scale = scale(:)';
end
% Obtain histogram and normalize it.
p = imhist(f);
p = p./numel(f);
L = length(p);
% Compute the three moments. We need the unnormalized ones
% from function statmoments. These are in vector mu.
[v, mu] = statmoments(p,3);
% Compute the six texture measures:
% Average gray level.
t(1) = mu(1);
% Standard deviation.
t(2) = mu(2).^0.5;
% Smoothness.
% First normalize the variance to [0 1] by
% dividing it by (L-1)^2.
varn = mu(2)/(L-1)^2;
t(3) = 1-1/(1+varn);
% Third moment (normalized by (L-1)^2 also).
t(4) = mu(3)/(L-1)^2;
% Uniformity.
t(5) = sum(p.^2);
% Entropy
t(6) = -sum(p.*(log2(p + eps)));
% Scale the values.
t = t.*scale;
end

function [v, unv] = statmoments(p, n)
% STATMOMENTS Computes statistical central moments of image histogram.
%   [V, UNV] = STATMOMENTS(P, N) computes up to the Nth statistical
%   central moment of a histogram whose components are in vector
%   P. The length of P must equal 256 or 65536.
% 
%   The program outputs a vector V with V(1) = mean, V(2) = variance,
%   V(3) = 3rd moment, ... V(N) = Nth central moment. The random
%   variable values are normalized to the range [0, 1], so all 
%   moments also are in this range.
% 
%   The program also outputs a vector UNV comtaining the same moments
%   as V, but using un-normalized random variable values(e.g., 0 to
%   255 if length(P) = 2^8). For example, if length(P) = 256 and V(1)
%   = 0.5, then UNV(1) would have the value UNV(1) = 127.5 （half of 
%   the [0 255] range.
Lp = length(p);
if Lp~=256 && Lp~=65536
    error('P must be a 256- or 65536-element vector.');
end
G = Lp - 1;
% Make sure the histogram has unit area, and convert it to a
% column vector.
p = p/sum(p);p = p(:);
% Form a vector of all the possible values of the
% random variable.
z = 0:G;
% Now normalize the z's to the range [0,1].
z = z./G;
% The mean.
m = z*p;
% Center random variables about the mean.
z = z-m;
% Compute the central moments.
v = zeros(1,n);
v(1) = m;
for j = 2:n
    v(j) = (z.^j)*p;
end
if nargout > 1
    % Compute the uncentralized moments.
    unv = zeros(1,n);
    unv(1) = m.*G;
    for j =2:n
        unv(j) = ((z*G).^j)*p;
    end
end
end

function [srad, sang, S] = specxture(f)
% SPECXTURE Computes spectral texture of an image.
%   [SRAD, SANG, S] = SPECXTURE(F) computes SRAD, the spectral energy
%   distribution as a function of radius from the center of the
%   spectrum, SANG, the spectral energy distribution as a function of
%   angle for 0 to 180 degrees in increments of 1 degree, and S = 
%   log(1 + spectrum of f), normalized to the range [0, 1]. The
%   maximum vaalue of radius is min(M,N), where M and N are the number
%   of rows and columns of image (region) f. Thus, SRAD is a row 
%   vector of length = (min(M, N)/2) - 1; and SANG is a row vector of
%   length 180.
% Obtain the centered spectrum, S, of f. The variables of S are
% (u, v), running from 1:M and 1:N, with the center (zero frequency)
% at (M/2 + 1, N/2 + 1)
S = fftshift(fft2(f));
S = abs(S);
[M, N] = size(S);
x0 = floor(M/2 + 1);
y0 = floor(N/2 + 1);
% Maximum radius that guarantees a circle centered at (x0, y0) that
% does not exceed the boundaries of S.
rmax = floor(min(M, N)/2 - 1);
% Compute srad.
srad = zeros(1, rmax);
srad(1) = S(x0, y0);
for r = 2:rmax
    [xc, yc] = halfcircle(r, x0, y0);
    srad(r) = sum(S(sub2ind(size(S), xc, yc)));
end
% Compute sang.
[xc, yc] = halfcircle(rmax, x0, y0);
sang = zeros(1, length(xc));
for a = 1:length(xc)
    [xr, yr] = radial(x0, y0, xc(a), yc(a));
    sang(a) = sum(S(sub2ind(size(S), xr, yr)));
end
% Output the log of the spectrum for easier viewing, scaled to the
% range [0, 1].
S = mat2gray(log(1 + S));
end
%-------------------------------------------------------------------------%
function [xc, yc] = halfcircle(r, x0, y0)
%   Computes the integer coordinates of a half circle of radius r and
%   center at (x0,y0) using one degree increments.
% 
%   Goes from 91 to 270 because we want the half circle to be in the
%   region defined by top right and top left quadrants, in the
%   standard image coordinates.

theta = 91:270;
theta = theta*pi/180;
[xc, yc] = pol2cart(theta, r);
xc = round(xc)' + x0; % Column vector.
yc = round(yc)' + y0;
end
%-------------------------------------------------------------------------%
function [xr,yr] = radial(x0, y0, x, y)
%   Computes the coordinates of a straight line segment extending
%   from (x0, y0) to (x, y).
% 
%   Based on function intline.m xr and yr are
%   returned as column vectors.
[xr, yr] = intline(x0, x, y0, y);
end

function [x,y] = intline(x1, x2, y1 ,y2)
% INTLINE Integer-coordinate line drawing algorithm.
%   [X, Y] = INTLINE(X1, X2, Y1, Y2) computes an
%   approximation to the line segment joining (X1, Y1) and
%   (X2, Y2) with integer coordinates. X1, X2, Y1, and Y2
%   should be integers. INTLINE is reversible; that is,
%   INTLINE(X1, X2, Y1, Y2) produces the same results as
%   FLIPUD(INTLINE(X2, X1, Y2, Y1)).
dx = abs(x2 - x1);
dy = abs(y2 - y1);
% Check for degenerate case.
if dx == 0 && dy == 0
    x = x1;
    y = y1;
    return;
end
flip = 0;
if dx >= dy
    if x1 > x2
        % Always "draw" from left to right.
        t = x1; x1 = x2; x2 = t;
        t = y1; y1 = y2; y2 = t;
        flip = 1;
    end
    m = (y2 - y1)/(x2 - x1);
    x = (x1:x2).';
    y = round(y1 + m*(x - x1));
else
    if y1 > y2
        % Always "draw" from bottom to top.
        t = x1; x1 = x2; x2 = t;
        t = y1; y1 = y2; y2 = t;
        flip = 1;
    end
    m = (x2 - x1)/(y2 - y1);
    y = (y1:y2).';
    x = round(x1 + m*(y - y1));
end
if flip
    x = flipud(x);
    y = flipud(y);
end
end