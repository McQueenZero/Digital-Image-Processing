%%-------------------------------------------------------------------------
% 作者：   赵敏琨
% 日期：   2021年4月
% 说明：   图像频域低通滤波去噪
%%-------------------------------------------------------------------------
%% 图像频域滤波
% 读取原始图片并加噪声
clc,clear,close all;
filename='lena.png';
imSrc = imread(filename);
figure('Name','原始图片','NumberTitle','off'),imshow(imSrc);
imNos=imnoise(imSrc,'gaussian');
figure('Name','噪声图片','NumberTitle','off'),imshow(imNos);

% 频域理想低通滤波
for k=1:3
    f_ILPF = imNos(:,:,k);  %处理的是噪声图像
    PQ_ILPF = paddedsize(size(f_ILPF));
    H_ILPF = lpfilter('ideal', PQ_ILPF(1), PQ_ILPF(2), 0.05*PQ_ILPF(2));
    F_ILPF = fft2(f_ILPF, PQ_ILPF(1), PQ_ILPF(2));
    g_ILPF(:,:,k) = dftfilt(f_ILPF, H_ILPF);
    
    switch k
        case 1
            figure('Name','ILPF下处理时噪声图像的频谱','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F_ILPF))), [ ])
            title('R通道')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F_ILPF))), [ ])
            title('G通道')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F_ILPF))), [ ])
            title('B通道')
    end
end
figure('Name','以图像形式显示的理想低通滤波器','NumberTitle','off')
imshow(fftshift(H_ILPF), [ ])

imDst_ILPF = uint8(g_ILPF);
figure('Name','频域理想低通滤波去噪图片','NumberTitle','off')
imshow(imDst_ILPF, [ ])

% 频域高斯低通滤波
for k=1:3
    f_GLPF = imNos(:,:,k);  %处理的是噪声图像
    PQ_GLPF = paddedsize(size(f_GLPF));
    H_GLPF = lpfilter('gaussian', PQ_GLPF(1), PQ_GLPF(2), 0.05*PQ_GLPF(2));
    F_GLPF = fft2(f_GLPF, PQ_GLPF(1), PQ_GLPF(2));
    g_GLPF(:,:,k) = dftfilt(f_GLPF, H_GLPF);
    
    switch k
        case 1
            figure('Name','GLPF下处理时噪声图像的频谱','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F_GLPF))), [ ])
            title('R通道')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F_GLPF))), [ ])
            title('G通道')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F_GLPF))), [ ])
            title('B通道')
    end
end
figure('Name','以图像形式显示的高斯滤波器','NumberTitle','off')
imshow(fftshift(H_GLPF), [ ])

imDst_GLPF = uint8(g_GLPF);
figure('Name','频域高斯低通滤波去噪图片','NumberTitle','off')
imshow(imDst_GLPF, [ ])

% 频域巴特沃斯低通滤波
for k=1:3
    f_BLPF = imNos(:,:,k);  %处理的是噪声图像
    PQ_BLPF = paddedsize(size(f_BLPF));
    H_BLPF = lpfilter('btw', PQ_BLPF(1), PQ_BLPF(2), 0.05*PQ_BLPF(2), 1);
    F_BLPF = fft2(f_BLPF, PQ_BLPF(1), PQ_BLPF(2));
    g_BLPF(:,:,k) = dftfilt(f_BLPF, H_BLPF);
    
    switch k
        case 1
            figure('Name','BLPF下处理时噪声图像的频谱','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F_BLPF))), [ ])
            title('R通道')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F_BLPF))), [ ])
            title('G通道')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F_BLPF))), [ ])
            title('B通道')
    end
end
figure('Name','以图像形式显示的巴特沃斯滤波器','NumberTitle','off')
imshow(fftshift(H_BLPF), [ ])

imDst_BLPF = uint8(g_BLPF);
figure('Name','频域巴特沃斯低通滤波去噪图片','NumberTitle','off')
imshow(imDst_BLPF, [ ])

%% 频域滤波内嵌函数
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
