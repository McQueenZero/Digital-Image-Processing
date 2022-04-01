%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   Image Recovery
%       图像的去噪和去模糊练习
%%-------------------------------------------------------------------------
%% 读取原始图像并加椒盐/高斯白噪声
clc,clear,close all;
filename='Rainbow Bridge';
imSrc = imread([filename,'.jpg']);
figure('Name','原始图片','NumberTitle','off'),imshow(imSrc);
% imNos=imnoise(imSrc,'salt & pepper',0.02);
imNos=imnoise(imSrc,'gaussian');
figure('Name','噪声图片','NumberTitle','off'),imshow(imNos);

%% 空域均值滤波
fo=fspecial('average');   %指定滤波算子（filter operator）
for k=1:3
    imDst(:,:,k)=uint8(filter2(fo,imNos(:,:,k)));
end
figure('Name','空域均值滤波去噪图片','NumberTitle','off'),imshow(imDst);

%% 空域高斯滤波
for k=1:3
    imDst(:,:,k)=imgaussfilt(imNos(:,:,k),1);
end
figure('Name','空域高斯滤波去噪图片','NumberTitle','off'),imshow(imDst);

%% 空域中值滤波
for k=1:3
    imDst(:,:,k)=medfilt2(imNos(:,:,k));
end
figure('Name','空域中值滤波去噪图片','NumberTitle','off'),imshow(imDst);

%% 频域理想低通滤波
for k=1:3
    f = imNos(:,:,k);  %处理的是噪声图像
    PQ = paddedsize(size(f));
    H = lpfilter('ideal', PQ(1), PQ(2), 0.05*PQ(2));
    F = fft2(f, PQ(1), PQ(2));
    g(:,:,k) = dftfilt(f, H);
    
    switch k
        case 1
            figure('Name','噪声图像的频谱','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('R通道')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('G通道')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('B通道')
    end
end
figure('Name','以图像形式显示的理想低通滤波器','NumberTitle','off')
imshow(fftshift(H), [ ])

imDst = uint8(g);
figure('Name','频域理想低通滤波去噪图片','NumberTitle','off')
imshow(imDst, [ ])

%% 频域高斯低通滤波
for k=1:3
    f = imNos(:,:,k);  %处理的是噪声图像
    PQ = paddedsize(size(f));
    H = lpfilter('gaussian', PQ(1), PQ(2), 0.05*PQ(2));
    F = fft2(f, PQ(1), PQ(2));
    g(:,:,k) = dftfilt(f, H);
    
    switch k
        case 1
            figure('Name','噪声图像的频谱','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('R通道')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('G通道')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('B通道')
    end
end
figure('Name','以图像形式显示的高斯滤波器','NumberTitle','off')
imshow(fftshift(H), [ ])

imDst = uint8(g);
figure('Name','频域高斯低通滤波去噪图片','NumberTitle','off')
imshow(imDst, [ ])

%% 频域巴特沃斯低通滤波
for k=1:3
    f = imNos(:,:,k);  %处理的是噪声图像
    PQ = paddedsize(size(f));
    H = lpfilter('btw', PQ(1), PQ(2), 0.05*PQ(2), 1);
    F = fft2(f, PQ(1), PQ(2));
    g(:,:,k) = dftfilt(f, H);
    
    switch k
        case 1
            figure('Name','噪声图像的频谱','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('R通道')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('G通道')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F))), [ ])
            title('B通道')
    end
end
figure('Name','以图像形式显示的巴特沃斯滤波器','NumberTitle','off')
imshow(fftshift(H), [ ])

imDst = uint8(g);
figure('Name','频域巴特沃斯低通滤波去噪图片','NumberTitle','off')
imshow(imDst, [ ])

%% 读取运动模糊图像后进行去运动模糊操作
clc,clear,close all;
filename='运动模糊原图';
imSrc = imread([filename,'.jpg']);
figure('Name','原始模糊图片','NumberTitle','off'),imshow(imSrc);
fo=fspecial('motion',20,180);   %指定模糊滤波算子（filter operator）
% 第一个参数为运动的像素，第二个参数为运动的角度
for k=1:3
    imDst(:,:,k)=uint8(filter2(fo,imSrc(:,:,k)));
end
figure('Name','去模糊图片','NumberTitle','off'),imshow(imDst);

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
