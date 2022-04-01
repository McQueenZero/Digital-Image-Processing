%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   提取图像纹理特征
%%-------------------------------------------------------------------------
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

% 基于傅里叶频谱的纹理描绘子(r:幅值, theta:辐角)
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
plot(srad2),title('第2幅图的S(r)曲线')
axis tight
subplot(2,4,3)
plot(sang1),title('第1幅图的S(\theta)曲线')
axis tight
subplot(2,4,4)
plot(sang2),title('第2幅图的S(\theta)曲线')
axis tight
subplot(2,4,5)
plot(srad3),title('第3幅图的S(r)曲线')
axis tight
subplot(2,4,6)
plot(srad4),title('第4幅图的S(r)曲线')
axis tight
subplot(2,4,7)
plot(sang3),title('第3幅图的S(\theta)曲线')
axis tight
subplot(2,4,8)
plot(sang4),title('第4幅图的S(\theta)曲线')
axis tight

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