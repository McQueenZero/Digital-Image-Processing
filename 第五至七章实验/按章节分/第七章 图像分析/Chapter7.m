%%-------------------------------------------------------------------------
% ���ߣ�   ������
% ���ڣ�   2021��4��
% ˵����   ͼ�����
% ͼ���Ե��⡢��ȡ��״��������ȡ�任������ͼ����ֵ�ָͼ��Ƶ���˲���
% ע�⣺   ��С������
%%-------------------------------------------------------------------------
%% ͼ���Ե��⣨������˹.MATLAB������ͼ����,P.289��
clc, clear, close all
filename = 'building.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);
% Sobel��Ե�����
[gS, tS] = edge(imSrc, 'sobel');
% Prewitt��Ե�����
[gP, tP] = edge(imSrc, 'prewitt');
% Roberts��Ե�����
[gR, tR] = edge(imSrc, 'roberts');
% LoG��Ե�����
[gL, tL] = edge(imSrc, 'log');
% Zero crossings��Ե�����
[gZ, tZ] = edge(imSrc, 'zerocross');
% Canny��Ե�����
[gC, tC] = edge(imSrc, 'canny');
% ���
figure('Name','ԭʼͼƬ','NumberTitle','off')
imshow(imSrc)
figure('Name','��ͬ��Ե�����������','NumberTitle','off')
subplot(2,3,1)
imshow(gS),title('Sobel��Ե���������ͼƬ')
subplot(2,3,2)
imshow(gP),title('Prewitt��Ե���������ͼƬ')
subplot(2,3,3)
imshow(gR),title('Roberts��Ե���������ͼƬ')
subplot(2,3,4)
imshow(gL),title('LoG��Ե���������ͼƬ')
subplot(2,3,5)
imshow(gZ),title('Zero crossings��Ե���������ͼƬ')
subplot(2,3,6)
imshow(gC),title('Canny��Ե���������ͼƬ')

%% ��ȡͼ����״����
clc, clear, close all
filename = 'building.tif';
imSrc = imread(filename);
[hei, wid, dim] = size(imSrc);

% Hu�أ�����أ�������������˹.MATLAB������ͼ����,P.353��
phi = invmoments(imSrc);
phi = abs(log(phi));
disp('����ͼȡ��������ֵ��Hu�����£�')
disp('    phi_1    phi_2     phi_3     phi_4     phi_5     phi_6     phi_7')
disp(phi)

% ROI������������˹.MATLAB������ͼ����,P.114��
[B, c, r] = roipoly(imSrc); %����ʽѡ�����Ȥ��ROI������
imDst = uint8(double(imSrc).*double(B));
figure('Name','����Ȥ����','NumberTitle','off')
imshow(imDst)

[p, npix] = histroi(imSrc, c, r);
figure('Name','��ģB�����ֱ��ͼ','NumberTitle','off')
bar(p,1) %��ģB�����ֱ��ͼ

phi = invmoments(imDst);
phi = abs(log(phi));
disp('����Ȥ����ȡ��������ֵ��Hu�����£�')
disp('    phi_1    phi_2     phi_3     phi_4     phi_5     phi_6     phi_7')
disp(phi)

%% ��ȡͼ��任������������˹.MATLAB������ͼ����,P.179С���任��
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
imshow(imDst_manual),title('Gonzalez����������ֵ�ķָ�ͼƬ')
subplot(1,3,3)
imshow(imDst_graythresh),title('Otsu����������ֵ�ķָ�ͼƬ')
disp(['�Ա෽����ֵ��', num2str(T)])
disp(['graythresh������ֵ��',num2str(Tg)])

%% ͼ��Ƶ���˲�
% ��ȡԭʼͼƬ��������
clc,clear,close all;
filename='lena.png';
imSrc = imread(filename);
figure('Name','ԭʼͼƬ','NumberTitle','off'),imshow(imSrc);
imNos=imnoise(imSrc,'gaussian');
figure('Name','����ͼƬ','NumberTitle','off'),imshow(imNos);

% Ƶ�������ͨ�˲�
for k=1:3
    f_ILPF = imNos(:,:,k);  %�����������ͼ��
    PQ_ILPF = paddedsize(size(f_ILPF));
    H_ILPF = lpfilter('ideal', PQ_ILPF(1), PQ_ILPF(2), 0.05*PQ_ILPF(2));
    F_ILPF = fft2(f_ILPF, PQ_ILPF(1), PQ_ILPF(2));
    g_ILPF(:,:,k) = dftfilt(f_ILPF, H_ILPF);
    
    switch k
        case 1
            figure('Name','ILPF�´���ʱ����ͼ���Ƶ��','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F_ILPF))), [ ])
            title('Rͨ��')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F_ILPF))), [ ])
            title('Gͨ��')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F_ILPF))), [ ])
            title('Bͨ��')
    end
end
figure('Name','��ͼ����ʽ��ʾ�������ͨ�˲���','NumberTitle','off')
imshow(fftshift(H_ILPF), [ ])

imDst_ILPF = uint8(g_ILPF);
figure('Name','Ƶ�������ͨ�˲�ȥ��ͼƬ','NumberTitle','off')
imshow(imDst_ILPF, [ ])

% Ƶ���˹��ͨ�˲�
for k=1:3
    f_GLPF = imNos(:,:,k);  %�����������ͼ��
    PQ_GLPF = paddedsize(size(f_GLPF));
    H_GLPF = lpfilter('gaussian', PQ_GLPF(1), PQ_GLPF(2), 0.05*PQ_GLPF(2));
    F_GLPF = fft2(f_GLPF, PQ_GLPF(1), PQ_GLPF(2));
    g_GLPF(:,:,k) = dftfilt(f_GLPF, H_GLPF);
    
    switch k
        case 1
            figure('Name','GLPF�´���ʱ����ͼ���Ƶ��','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F_GLPF))), [ ])
            title('Rͨ��')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F_GLPF))), [ ])
            title('Gͨ��')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F_GLPF))), [ ])
            title('Bͨ��')
    end
end
figure('Name','��ͼ����ʽ��ʾ�ĸ�˹�˲���','NumberTitle','off')
imshow(fftshift(H_GLPF), [ ])

imDst_GLPF = uint8(g_GLPF);
figure('Name','Ƶ���˹��ͨ�˲�ȥ��ͼƬ','NumberTitle','off')
imshow(imDst_GLPF, [ ])

% Ƶ�������˹��ͨ�˲�
for k=1:3
    f_BLPF = imNos(:,:,k);  %�����������ͼ��
    PQ_BLPF = paddedsize(size(f_BLPF));
    H_BLPF = lpfilter('btw', PQ_BLPF(1), PQ_BLPF(2), 0.05*PQ_BLPF(2), 1);
    F_BLPF = fft2(f_BLPF, PQ_BLPF(1), PQ_BLPF(2));
    g_BLPF(:,:,k) = dftfilt(f_BLPF, H_BLPF);
    
    switch k
        case 1
            figure('Name','BLPF�´���ʱ����ͼ���Ƶ��','NumberTitle','off')
            subplot(1,3,1)
            imshow(log(1 + abs(fftshift(F_BLPF))), [ ])
            title('Rͨ��')
        case 2
            subplot(1,3,2)
            imshow(log(1 + abs(fftshift(F_BLPF))), [ ])
            title('Gͨ��')
        case 3
            subplot(1,3,3)
            imshow(log(1 + abs(fftshift(F_BLPF))), [ ])
            title('Bͨ��')
    end
end
figure('Name','��ͼ����ʽ��ʾ�İ�����˹�˲���','NumberTitle','off')
imshow(fftshift(H_BLPF), [ ])

imDst_BLPF = uint8(g_BLPF);
figure('Name','Ƶ�������˹��ͨ�˲�ȥ��ͼƬ','NumberTitle','off')
imshow(imDst_BLPF, [ ])

%% ��ȡͼ����״������Ƕ����
function phi = invmoments(F)
% INVMOMENTS Compute invariant moments of image.
%   PHI = INVMOMENTS(F) computes the moment invariants of the image
%   F. PHI is a seven-element row vector containing the moment
%   invariants as defined in equations (11.3-11.7) through (11.3-23) of
%   Gonzalez and Woods, Digital Image Processing, 2nd Ed.
% 
%   F must be a 2-D, real, nonsparse, numeric or logical matrix.
if ndims(F)~=2 || issparse(F) || ~isreal(F) || ~(isnumeric(F) || islogical(F))
    error('F must be a 2-D, real, nonsparse, numeric or logical matrix.');
end
F = double(F);
phi = compute_phi(compute_eta(compute_m(F)));
end
%-------------------------------------------------------------------------%
function m = compute_m(F)
[M, N] = size(F);
[x, y] = meshgrid(1:N, 1:M);
% Turn x, y, and F into column vectors to make the summations a bit
% easier to compute in the following.
x = x(:);
y = y(:);
F = F(:);
% DIP equation (11.3-12)
m.m00 = sum(F);
% Protect against divide-by-zero warnings.
if m.m00 == 0
    m.m00 = eps;
end
% The other central moments:
m.m10 = sum(x.*F);
m.m01 = sum(y.*F);
m.m11 = sum(x.*y.*F);
m.m20 = sum(x.^2.*F);
m.m02 = sum(y.^2.*F);
m.m30 = sum(x.^3.*F);
m.m03 = sum(y.^3.*F);
m.m12 = sum(x.*y.^2.*F);
m.m21 = sum(x.^2.*y.*F);
end
%-------------------------------------------------------------------------%
function e = compute_eta(m)
% DIP equations (11.3-14) through (11.3-16).
xbar = m.m10/m.m00;
ybar = m.m01/m.m00;
e.eta11 = (m.m11-ybar*m.m10)/m.m00^2;
e.eta20 = (m.m20-xbar*m.m10)/m.m00^2;
e.eta02 = (m.m02-ybar*m.m01)/m.m00^2;
e.eta30 = (m.m30-3*xbar*m.m20+2*xbar^2*m.m10)/m.m00^2.5;
e.eta03 = (m.m03-3*ybar*m.m02+2*ybar^2*m.m01)/m.m00^2.5;
e.eta21 = (m.m21-2*xbar*m.m11-ybar*m.m20+2*xbar^2*m.m01)/m.m00^2.5;
e.eta12 = (m.m12-2*ybar*m.m11-xbar*m.m02+2*ybar^2*m.m10)/m.m00^2.5;
end
%-------------------------------------------------------------------------%
function phi = compute_phi(e)
% DIP equations (11.3-17) through (11.3-23)
phi(1) = e.eta20 + e.eta02;
phi(2) = (e.eta20 - e.eta02)^2 + 4*e.eta11^2;
phi(3) = (e.eta30 - 3*e.eta12)^2 + (3*e.eta21 - e.eta03)^2;
phi(4) = (e.eta30 + e.eta12)^2 + (e.eta21 + e.eta03)^2;
phi(5) = (e.eta30 - 3*e.eta12)*(e.eta30 + e.eta12)*((e.eta30 + e.eta12)^2 - 3*(e.eta21 + e.eta03)^2) + ...
         (3*e.eta21 - e.eta03)*(e.eta21 + e.eta03)*(3*(e.eta30 + e.eta12)^2 - (e.eta21 + e.eta03)^2);
phi(6) = (e.eta20 - e.eta02)*((e.eta30 + e.eta12)^2 - (e.eta21 + e.eta03)^2) + ...
         4*e.eta11*(e.eta30 + e.eta12)*(e.eta21 + e.eta03);
phi(7) = (3*e.eta21 - e.eta03)*(e.eta30 + e.eta12)*((e.eta30 + e.eta12)^2 - 3*(e.eta21 + e.eta03)^2) + ...
         (3*e.eta12 - e.eta30)*(e.eta21 + e.eta03)*(3*(e.eta30 + e.eta12)^2 - (e.eta21 + e.eta03)^2);
end

function [p, npix] = histroi(f, c, r)
% HISTROI Computes the histogram of an ROI in an image.
%   [P, NPIX] = HISTROI(F, C, R) computes the histogram, P, of a 
%   polygonal region of interesst (ROI) in image F. The polygonal
%   region is defined by the column and row coordinates of its
%   vertices, which are speciried (sequentially) in vectors C and R,
%   respectively. All pixels of F must be >= 0. Parameter NPIX is the 
%   number of pixxels in the polygonal region.
% Generate the binary mask image.
B = roipoly(f, c, r);
% Compute the histogram of the pixels in the ROI.
p = imhist(f(B));
% Obtain the number of pixels in the ROI if requested in the output.
if nargout > 1
    npix = sum(B(:));
end
end

%% Ƶ���˲���Ƕ����
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